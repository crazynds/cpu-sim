// The SIMT pass: calls, atomics, warp operations (shuffles, ballots),
// math and tensor core operations of the warp code.
#include "compiler/simt_internal.h"

namespace cpusim {
namespace simt_internal {

// ---------------------------------------------------------------- calls

// `__cpusim_fn<W>_<name>(ctx, mask, args...)`: every argument and the result
// as W-lane vectors. Internal, fastcc: the vectors travel in registers.
Function* WarpBuilder::vector_fn(Function* f) {
    std::string name = "__cpusim_fn" + std::to_string(W) + "_" + f->getName().str();
    if (Function* v = m_.getFunction(name)) return v;
    auto lane_type = [](Type* t) { return t->isIntegerTy() || t->isFloatingPointTy() || t->isPointerTy(); };
    Type* ret = f->getReturnType();
    if (!ret->isVoidTy() && !lane_type(ret))
        return fail("a function that is not inlined returns an aggregate", f), nullptr;
    SmallVector<Type*, 8> params{PointerType::get(ctx_, 0), mask_type_};
    for (Argument& a : f->args()) {
        if (!lane_type(a.getType()) || a.hasByValAttr() || a.hasStructRetAttr())
            return fail("a function that is not inlined takes an aggregate argument", f), nullptr;
        params.push_back(vec_type(a.getType()));
    }
    Function* v = Function::Create(FunctionType::get(ret->isVoidTy() ? ret : vec_type(ret), params, false),
                                   GlobalValue::InternalLinkage, name, m_);
    v->setCallingConv(CallingConv::Fast);
    v->addFnAttr(Attribute::NoUnwind);
    for (const char* a : {"target-cpu", "target-features"})
        if (f->hasFnAttribute(a)) v->addFnAttr(f->getFnAttribute(a));
    pending_functions().push_back({v, f, W});
    return v;
}

Value* WarpBuilder::call_vector_fn(Function* f, CallInst* call, Value* mask) {
    Function* v = vector_fn(f);
    if (!v) return nullptr;
    SmallVector<Value*, 8> args{ctx_arg_, mask};
    for (unsigned i = 0; i < call->arg_size(); ++i) args.push_back(vector(call->getArgOperand(i)));
    CallInst* r = b_.CreateCall(v, args);
    r->setCallingConv(CallingConv::Fast);
    return r;
}

// A call that was not inlined runs once for all the active lanes (the block
// it is in only runs when some lane is: that is what ends a recursion). A call
// through a pointer runs once per distinct target among the active lanes,
// each with the lanes that point at it; the targets are the device functions
// of that type whose address is taken.
bool WarpBuilder::emit_call(CallInst* call, Value* mask) {
    Type* rt = call->getType()->isVoidTy() ? nullptr : vec_type(call->getType());
    auto result = [&](Value* v) {
        if (!rt) return true;
        if (!v) return false;
        (uni_->is_varying(call) ? vector_map_ : scalar_map_)[call] = v;
        return true;
    };
    if (Function* f = call->getCalledFunction()) {
        Value* r = call_vector_fn(f, call, mask);
        return error_.empty() && result(r);
    }

    std::vector<Function*> targets;
    for (Function& f : m_)
        if (!f.isDeclaration() && f.hasFnAttribute(Attribute::AlwaysInline) &&
            f.getFunctionType() == call->getFunctionType() && f.hasAddressTaken())
            targets.push_back(&f);
    // One target with its lanes: compare against each candidate in turn.
    auto dispatch = [&](Value* t, Value* m) -> Value* {
        BasicBlock* join = BasicBlock::Create(ctx_, "icall.join", w_);
        std::vector<std::pair<Value*, BasicBlock*>> incoming;
        for (Function* f : targets) {
            BasicBlock* yes = BasicBlock::Create(ctx_, "icall." + f->getName(), w_);
            BasicBlock* no = BasicBlock::Create(ctx_, "icall.next", w_);
            b_.CreateCondBr(b_.CreateICmpEQ(t, f), yes, no);
            b_.SetInsertPoint(yes);
            Value* r = call_vector_fn(f, call, m);
            if (rt) incoming.push_back({r ? r : PoisonValue::get(rt), b_.GetInsertBlock()});
            b_.CreateBr(join);
            b_.SetInsertPoint(no);
        }
        if (rt) incoming.push_back({PoisonValue::get(rt), b_.GetInsertBlock()});
        b_.CreateBr(join);
        b_.SetInsertPoint(join);
        if (!rt) return nullptr;
        PHINode* phi = b_.CreatePHI(rt, unsigned(incoming.size()));
        for (auto [v, bb] : incoming) phi->addIncoming(v, bb);
        return phi;
    };
    Value* target = call->getCalledOperand();
    if (!uni_->is_varying(target)) return result(dispatch(scalar(target), mask)) && error_.empty();

    // Per distinct target: the first remaining lane's, and every lane that
    // has the same.
    Value* tv = vector(target);
    BasicBlock* pre = b_.GetInsertBlock();
    BasicBlock* loop = BasicBlock::Create(ctx_, "icall.loop", w_);
    BasicBlock* done = BasicBlock::Create(ctx_, "icall.done", w_);
    b_.CreateBr(loop);
    b_.SetInsertPoint(loop);
    PHINode* rem = b_.CreatePHI(mask_type_, 2, "icall.rem");
    rem->addIncoming(mask, pre);
    PHINode* acc = rt ? b_.CreatePHI(rt, 2, "icall.acc") : nullptr;
    if (acc) acc->addIncoming(PoisonValue::get(rt), pre);
    Value* bits = b_.CreateBitCast(rem, b_.getIntNTy(W));
    Value* first = b_.CreateBinaryIntrinsic(Intrinsic::cttz, bits, b_.getTrue());
    Value* t = b_.CreateExtractElement(tv, b_.CreateZExtOrTrunc(first, b_.getInt32Ty()));
    Value* same = b_.CreateAnd(rem, b_.CreateICmpEQ(tv, b_.CreateVectorSplat(W, t)));
    Value* r = dispatch(t, same);
    Value* acc2 = acc ? b_.CreateSelect(same, r, acc) : nullptr;
    Value* rem2 = b_.CreateAnd(rem, b_.CreateNot(same));
    rem->addIncoming(rem2, b_.GetInsertBlock());
    if (acc) acc->addIncoming(acc2, b_.GetInsertBlock());
    b_.CreateCondBr(any_lane(rem2), loop, done);
    b_.SetInsertPoint(done);
    return result(acc2) && error_.empty();
}

// ---------------------------------------------------------------- atomics

// Every lane updates the same address: the active lanes' operands combined
// (inactive lanes contribute the identity), one atomic with the total, and
// each lane's old value = the value before the warp's update combined with
// the lanes before it (an exclusive scan). CUDA leaves the order of a warp's
// atomics unspecified, so any order of the lanes is one the GPU could pick.
Value* WarpBuilder::emit_uniform_rmw(AtomicRMWInst* rmw, Value* mask, Value* ptr) {
    using Op = AtomicRMWInst::BinOp;
    Type* t = rmw->getType();
    const Op op = rmw->getOperation();
    Constant* id = nullptr;
    Intrinsic::ID reduce = Intrinsic::not_intrinsic, combine_id = Intrinsic::not_intrinsic;
    Instruction::BinaryOps combine = Instruction::BinaryOpsEnd;
    if (t->isIntegerTy()) {
        const unsigned bits = t->getIntegerBitWidth();
        switch (op) {
            case Op::Add: case Op::Sub:
                id = ConstantInt::get(t, 0), reduce = Intrinsic::vector_reduce_add, combine = Instruction::Add;
                break;
            case Op::Or: id = ConstantInt::get(t, 0), reduce = Intrinsic::vector_reduce_or, combine = Instruction::Or; break;
            case Op::Xor: id = ConstantInt::get(t, 0), reduce = Intrinsic::vector_reduce_xor, combine = Instruction::Xor; break;
            case Op::And:
                id = ConstantInt::get(t, APInt::getAllOnes(bits)), reduce = Intrinsic::vector_reduce_and,
                combine = Instruction::And;
                break;
            case Op::Max:
                id = ConstantInt::get(t, APInt::getSignedMinValue(bits)), reduce = Intrinsic::vector_reduce_smax,
                combine_id = Intrinsic::smax;
                break;
            case Op::Min:
                id = ConstantInt::get(t, APInt::getSignedMaxValue(bits)), reduce = Intrinsic::vector_reduce_smin,
                combine_id = Intrinsic::smin;
                break;
            case Op::UMax:
                id = ConstantInt::get(t, 0), reduce = Intrinsic::vector_reduce_umax, combine_id = Intrinsic::umax;
                break;
            case Op::UMin:
                id = ConstantInt::get(t, APInt::getAllOnes(bits)), reduce = Intrinsic::vector_reduce_umin,
                combine_id = Intrinsic::umin;
                break;
            default:
                return nullptr;
        }
    } else if (op == Op::FAdd) {
        id = ConstantFP::getNegativeZero(t), combine = Instruction::FAdd;
    } else {
        return nullptr;
    }
    auto comb = [&](Value* a, Value* b) {
        return combine_id != Intrinsic::not_intrinsic ? b_.CreateBinaryIntrinsic(combine_id, a, b)
                                                      : b_.CreateBinOp(combine, a, b);
    };
    Constant* ids = ConstantVector::getSplat(ElementCount::getFixed(W), id);
    Value* v = b_.CreateSelect(mask, vector(rmw->getValOperand()), ids);
    Value* total = op == Op::FAdd ? b_.CreateFAddReduce(id, v)
                                  : b_.CreateCall(Intrinsic::getDeclaration(&m_, reduce, {v->getType()}), {v});
    if (auto* fp = dyn_cast<Instruction>(total); fp && op == Op::FAdd) fp->setHasAllowReassoc(true);
    AtomicRMWInst* one = b_.CreateAtomicRMW(op, ptr ? ptr : scalar(rmw->getPointerOperand()), total, rmw->getAlign(),
                                            rmw->getOrdering(), rmw->getSyncScopeID());
    one->setVolatile(rmw->isVolatile());
    if (rmw->use_empty()) return one;
    // Inclusive scan in log2(W) steps, then shifted by one lane.
    Value* x = v;
    for (unsigned d = 1; d < W; d *= 2) {
        SmallVector<int, 512> idx;
        for (unsigned i = 0; i < W; ++i) idx.push_back(i < d ? int(W + i) : int(i - d));
        x = comb(x, b_.CreateShuffleVector(x, ids, idx));
    }
    SmallVector<int, 512> idx;
    for (unsigned i = 0; i < W; ++i) idx.push_back(i == 0 ? int(W) : int(i - 1));
    Value* before = b_.CreateShuffleVector(x, ids, idx);
    Value* old = b_.CreateVectorSplat(W, one);
    return op == Op::Sub ? b_.CreateSub(old, before) : comb(old, before);
}

// Atomic read-modify-write, compare-and-swap, atomic load or store: the
// per-thread instruction once per active lane, lowest lane first, with that
// lane's operands; the results gathered in a stack array.
bool WarpBuilder::emit_atomic(Instruction& inst, Value* mask) {
    // An atomic load of one address for the whole warp: one load (the value
    // is uniform, as the analysis says).
    if (auto* ld = dyn_cast<LoadInst>(&inst); ld && !uni_->is_varying(ld->getPointerOperand())) {
        Instruction* one = inst.clone();
        one->setOperand(0, scalar(ld->getPointerOperand()));
        b_.Insert(one);
        (uni_->is_varying(&inst) ? vector_map_ : scalar_map_)[&inst] =
            uni_->is_varying(&inst) ? b_.CreateVectorSplat(W, one) : one;
        return true;
    }
    if (auto* rmw = dyn_cast<AtomicRMWInst>(&inst); rmw && !uni_->is_varying(rmw->getPointerOperand()))
        if (Value* v = emit_uniform_rmw(rmw, mask)) {
            if (!inst.use_empty()) vector_map_[&inst] = v;
            return true;
        }
    // Lanes updating the same address (a histogram's hot bins): one atomic
    // per distinct address among the active lanes, each with the reduction
    // of its lanes' operands, instead of one locked instruction per lane on
    // the same cache line.
    if (auto* rmw = dyn_cast<AtomicRMWInst>(&inst); rmw && groupable(rmw)) {
        Type* rt = vec_type(rmw->getType());
        Value* ptrs = vector(rmw->getPointerOperand());
        BasicBlock* pre = b_.GetInsertBlock();
        BasicBlock* loop = BasicBlock::Create(ctx_, "atomic.group", w_);
        BasicBlock* done = BasicBlock::Create(ctx_, "atomic.groups.done", w_);
        b_.CreateCondBr(any_lane(mask), loop, done);
        b_.SetInsertPoint(loop);
        PHINode* rem = b_.CreatePHI(mask_type_, 2, "atomic.rem");
        rem->addIncoming(mask, pre);
        PHINode* acc = b_.CreatePHI(rt, 2, "atomic.acc");
        acc->addIncoming(PoisonValue::get(rt), pre);
        Value* first = b_.CreateBinaryIntrinsic(Intrinsic::cttz, b_.CreateBitCast(rem, b_.getIntNTy(W)), b_.getTrue());
        Value* p = b_.CreateExtractElement(ptrs, b_.CreateZExtOrTrunc(first, b_.getInt32Ty()));
        Value* same = b_.CreateAnd(rem, b_.CreateICmpEQ(ptrs, b_.CreateVectorSplat(W, p)));
        Value* r = emit_uniform_rmw(rmw, same, p);
        Value* acc2 = inst.use_empty() ? static_cast<Value*>(acc) : b_.CreateSelect(same, r, acc);
        Value* rem2 = b_.CreateAnd(rem, b_.CreateNot(same));
        rem->addIncoming(rem2, b_.GetInsertBlock());
        acc->addIncoming(acc2, b_.GetInsertBlock());
        b_.CreateCondBr(any_lane(rem2), loop, done);
        b_.SetInsertPoint(done);
        if (!inst.use_empty()) {
            PHINode* res = b_.CreatePHI(rt, 2);
            res->addIncoming(PoisonValue::get(rt), pre);
            res->addIncoming(acc2, loop);
            vector_map_[&inst] = res;
        }
        return true;
    }
    auto* cx = dyn_cast<AtomicCmpXchgInst>(&inst);
    Type* t = cx ? cx->getCompareOperand()->getType() : inst.getType();
    const DataLayout& dl = m_.getDataLayout();
    IRBuilder<> ab(alloca_block_->getTerminator());
    AllocaInst* results = nullptr;
    AllocaInst* success = nullptr;
    if (!t->isVoidTy() && !inst.use_empty()) {
        results = ab.CreateAlloca(ArrayType::get(t, W), nullptr, "atomic.results");
        results->setAlignment(Align(64));
        if (cx) success = ab.CreateAlloca(ArrayType::get(b_.getInt8Ty(), W), nullptr, "atomic.success");
    }
    BasicBlock* pre = b_.GetInsertBlock();
    BasicBlock* loop = BasicBlock::Create(ctx_, "atomic.lane", w_);
    BasicBlock* done = BasicBlock::Create(ctx_, "atomic.done", w_);
    Value* bits = b_.CreateBitCast(mask, b_.getIntNTy(W));
    b_.CreateCondBr(b_.CreateICmpNE(bits, b_.getIntN(W, 0)), loop, done);
    b_.SetInsertPoint(loop);
    PHINode* rem = b_.CreatePHI(bits->getType(), 2, "atomic.rem");
    rem->addIncoming(bits, pre);
    Value* lane = b_.CreateZExtOrTrunc(b_.CreateBinaryIntrinsic(Intrinsic::cttz, rem, b_.getTrue()), b_.getInt32Ty());
    Instruction* one = inst.clone();
    for (Use& u : one->operands())
        u.set(uni_->is_varying(u.get()) ? b_.CreateExtractElement(vector(u.get()), lane) : scalar(u.get()));
    b_.Insert(one);
    if (results) {
        Value* r = cx ? b_.CreateExtractValue(one, 0) : one;
        b_.CreateStore(r, b_.CreateGEP(results->getAllocatedType(), results, {b_.getInt32(0), lane}));
        if (success)
            b_.CreateStore(b_.CreateZExt(b_.CreateExtractValue(one, 1), b_.getInt8Ty()),
                           b_.CreateGEP(success->getAllocatedType(), success, {b_.getInt32(0), lane}));
    }
    Value* next = b_.CreateAnd(rem, b_.CreateSub(rem, ConstantInt::get(rem->getType(), 1)));
    rem->addIncoming(next, b_.GetInsertBlock());
    b_.CreateCondBr(b_.CreateICmpNE(next, ConstantInt::get(rem->getType(), 0)), loop, done);
    b_.SetInsertPoint(done);
    if (!results) return true;
    Value* v = b_.CreateAlignedLoad(vec_type(t), results, Align(std::min<uint64_t>(64, dl.getTypeAllocSize(t) * W)));
    if (!cx) {
        vector_map_[&inst] = v;
        return true;
    }
    Value* ok = b_.CreateAlignedLoad(FixedVectorType::get(b_.getInt8Ty(), W), success, Align(64));
    cmpxchg_parts_[&inst] = {v, b_.CreateTrunc(ok, mask_type_)};
    return true;
}

// threadIdx for a known block shape. Lane l of the group starting at thread
// `base` (= warp_index * 32) is thread base + l. When the group covers the
// whole block, base is 0 and every index is a constant vector; when the width
// is a multiple of blockDim.x and groups stay inside one z-plane, x is a
// constant vector and y, z are a constant vector plus a uniform value.
void WarpBuilder::emit_specialized_tid() {
    const uint32_t bx = k_.spec_block[0], by = k_.spec_block[1], bz = k_.spec_block[2];
    const uint32_t threads = bx * by * bz;
    auto* i32 = b_.getInt32Ty();
    auto lanes = [&](auto f) {
        SmallVector<Constant*, 64> c;
        for (unsigned l = 0; l < W; ++l) c.push_back(b_.getInt32(f(l)));
        return ConstantVector::get(c);
    };
    if (threads <= W) {
        special_[(int)SpecialReg::TidX] = lanes([&](unsigned l) { return l < threads ? l % bx : 0; });
        special_[(int)SpecialReg::TidY] = lanes([&](unsigned l) { return l < threads ? l / bx % by : 0; });
        special_[(int)SpecialReg::TidZ] = lanes([&](unsigned l) { return l < threads ? l / (bx * by) : 0; });
        SmallVector<Constant*, 64> m;
        for (unsigned l = 0; l < W; ++l) m.push_back(b_.getInt1(l < threads));
        spec_lane_mask_ = ConstantVector::get(m);
        return;
    }
    // base = warp_index * warp size, a multiple of W here (groups are W lanes).
    Value* warp = b_.CreateAlignedLoad(i32, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_WARP_INDEX),
                                       Align(4), "warp");
    cast<Instruction>(warp)->setMetadata(
        LLVMContext::MD_range, MDNode::get(ctx_, {ConstantAsMetadata::get(b_.getInt32(0)),
                                                  ConstantAsMetadata::get(b_.getInt32((threads + k_.warp_size - 1) / k_.warp_size))}));
    Value* base = b_.CreateMul(warp, b_.getInt32(k_.warp_size), "base", true, true);
    auto* v = FixedVectorType::get(i32, W);
    if (W % bx == 0 && (bx * by) % W == 0) {
        Value* q = b_.CreateUDiv(base, b_.getInt32(bx));  // first row of the group
        special_[(int)SpecialReg::TidX] = lanes([&](unsigned l) { return l % bx; });
        special_[(int)SpecialReg::TidY] =
            b_.CreateAdd(b_.CreateVectorSplat(W, b_.CreateURem(q, b_.getInt32(by))), lanes([&](unsigned l) { return l / bx; }),
                         "tid.y", true, true);
        special_[(int)SpecialReg::TidZ] = b_.CreateVectorSplat(W, b_.CreateUDiv(q, b_.getInt32(by)), "tid.z");
    } else {
        Value* t = b_.CreateAdd(b_.CreateVectorSplat(W, base), lanes([](unsigned l) { return l; }));
        special_[(int)SpecialReg::TidX] = b_.CreateURem(t, ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt32(bx)));
        Value* row = b_.CreateUDiv(t, ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt32(bx)));
        special_[(int)SpecialReg::TidY] = b_.CreateURem(row, ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt32(by)));
        special_[(int)SpecialReg::TidZ] = b_.CreateUDiv(row, ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt32(by)));
    }
    (void)v;
    if (threads % W == 0) {
        spec_lane_mask_ = Constant::getAllOnesValue(mask_type_);
    } else {
        Value* t = b_.CreateAdd(b_.CreateVectorSplat(W, base), lanes([](unsigned l) { return l; }));
        spec_lane_mask_ = b_.CreateICmpULT(t, ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt32(threads)));
    }
}

// threadIdx for a class of block shapes (cpusim_shape_class, 1D blocks):
// lane l of the group starting at thread `base` (= warp_index * warp size)
// is thread base + l; blockDim.x is a uniform read from the context.
void WarpBuilder::emit_class_tid() {
    const uint32_t kind = k_.spec_class & 0xff;
    auto* i32 = b_.getInt32Ty();
    SmallVector<Constant*, 64> lane;
    for (unsigned l = 0; l < W; ++l) lane.push_back(b_.getInt32(l));
    auto field = [&](unsigned offset) { return b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, offset); };
    Value* warp = b_.CreateAlignedLoad(i32, field(CPUSIM_CTX_WARP_INDEX), Align(4), "warp");
    Value* base = b_.CreateMul(warp, b_.getInt32(k_.warp_size), "base", true, true);
    Value* t = b_.CreateAdd(b_.CreateVectorSplat(W, base), ConstantVector::get(lane), "tid.x", true, true);
    special_[(int)SpecialReg::TidX] = t;
    special_[(int)SpecialReg::TidY] = special_[(int)SpecialReg::TidZ] = Constant::getNullValue(t->getType());
    // A multiple of 128 (or of 32 with 32-lane groups): every lane a thread.
    if (kind == 1 || W <= 32) {
        spec_lane_mask_ = Constant::getAllOnesValue(mask_type_);
    } else {
        Value* bx = b_.CreateAlignedLoad(i32, field(CPUSIM_CTX_BLOCK_DIM), Align(4), "ntid.x");
        spec_lane_mask_ = b_.CreateICmpULT(t, b_.CreateVectorSplat(W, bx));
    }
}

// A varying math call as calls to its libmvec version, one per 16 floats or
// 8 doubles: _ZGVeN16v_expf(<16 x float>), _ZGVeN8vv_pow(<8 x double>, ...).
// Every lane is computed (inactive ones on whatever they hold; the functions
// have no side effects).
Value* WarpBuilder::emit_libmvec(CallInst& call, const std::string& name, unsigned arity) {
    Type* t = call.getType();
    const unsigned chunk = t->isFloatTy() ? 16 : 8;
    auto* vt = FixedVectorType::get(t, chunk);
    SmallVector<Type*, 2> params(arity, vt);
    std::string mangled = "_ZGVeN" + std::to_string(chunk) + std::string(arity, 'v') + "_" + name;
    FunctionCallee fn = m_.getOrInsertFunction(mangled, FunctionType::get(vt, params, false));
    if (auto* f = dyn_cast<Function>(fn.getCallee())) {
        f->setDoesNotThrow();
        f->setDoesNotAccessMemory();
    }
    std::vector<Value*> wide;
    for (unsigned i = 0; i < arity; ++i) wide.push_back(vector(call.getArgOperand(i)));
    std::vector<Value*> parts;
    for (unsigned c = 0; c < W / chunk; ++c) {
        SmallVector<int, 16> idx;
        for (unsigned i = 0; i < chunk; ++i) idx.push_back(int(c * chunk + i));
        SmallVector<Value*, 2> args;
        for (Value* w : wide) args.push_back(b_.CreateShuffleVector(w, idx));
        CallInst* r = b_.CreateCall(fn, args);
        r->setDoesNotThrow();
        parts.push_back(r);
    }
    return concat_chunks(parts);
}

// 32-bit division (and remainder) of a varying value by a uniform, not
// constant, divisor: x86 has no vector integer division, so LLVM would split
// it into one scalar `div` per lane (heat2d: 160 per 32-lane call). Instead,
// a multiplier computed once from the divisor (Granlund-Montgomery,
// "round-up" form, exact for every unsigned 32-bit divisor):
//     l = ceil(log2 d), m = floor(2^32 (2^l - d) / d) + 1
//     t = mulhi(m, n), q = (t + ((n - t) >> min(l, 1))) >> max(l - 1, 0)
// Signed division divides the magnitudes and fixes the sign (rounding toward
// zero, as C). A zero divisor (undefined anyway, maybe only in code no lane
// runs) is treated as 1 so the scalar setup cannot trap.
Value* WarpBuilder::emit_uniform_division(BinaryOperator* bin) {
    const auto op = bin->getOpcode();
    if (op != Instruction::UDiv && op != Instruction::URem && op != Instruction::SDiv && op != Instruction::SRem)
        return nullptr;
    Value* divisor = bin->getOperand(1);
    if (!bin->getType()->isIntegerTy(32) || uni_->is_varying(divisor) || isa<Constant>(divisor)) return nullptr;
    const bool is_signed = op == Instruction::SDiv || op == Instruction::SRem;
    auto* i32 = b_.getInt32Ty();
    auto* i64 = b_.getInt64Ty();
    auto* v32 = FixedVectorType::get(i32, W);
    auto* v64 = FixedVectorType::get(i64, W);

    // Scalar setup (uniform; leaves loops through LICM).
    Value* d = scalar(divisor);
    Value* neg_d = is_signed ? b_.CreateICmpSLT(d, b_.getInt32(0)) : nullptr;
    Value* ad = is_signed ? b_.CreateSelect(neg_d, b_.CreateNeg(d), d) : d;  // |d| as unsigned
    ad = b_.CreateSelect(b_.CreateICmpEQ(ad, b_.getInt32(0)), b_.getInt32(1), ad);
    Value* ctlz = b_.CreateBinaryIntrinsic(Intrinsic::ctlz, b_.CreateSub(ad, b_.getInt32(1)), b_.getTrue());
    // ctlz(0) is 32 (poison flag off below): d = 1 gives l = 0.
    ctlz = b_.CreateSelect(b_.CreateICmpEQ(ad, b_.getInt32(1)), b_.getInt32(32), ctlz);
    Value* l = b_.CreateSub(b_.getInt32(32), ctlz);
    Value* l64 = b_.CreateZExt(l, i64);
    Value* ad64 = b_.CreateZExt(ad, i64);
    Value* top = b_.CreateShl(b_.CreateSub(b_.CreateShl(b_.getInt64(1), l64), ad64), b_.getInt64(32));
    Value* m = b_.CreateTrunc(b_.CreateAdd(b_.CreateUDiv(top, ad64), b_.getInt64(1)), i32);
    Value* sh1 = b_.CreateSelect(b_.CreateICmpUGT(l, b_.getInt32(0)), b_.getInt32(1), b_.getInt32(0));
    Value* sh2 = b_.CreateSub(l, sh1);

    // Per lane.
    Value* n = vector(bin->getOperand(0));
    Value* neg_n = nullptr;
    Value* un = n;
    if (is_signed) {
        neg_n = b_.CreateICmpSLT(n, Constant::getNullValue(v32));
        un = b_.CreateSelect(neg_n, b_.CreateNeg(n), n);
    }
    Value* t = b_.CreateTrunc(
        b_.CreateLShr(b_.CreateMul(b_.CreateZExt(un, v64), b_.CreateVectorSplat(W, b_.CreateZExt(m, i64))),
                      ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt64(32))),
        v32);
    Value* q = b_.CreateLShr(b_.CreateAdd(t, b_.CreateLShr(b_.CreateSub(un, t), b_.CreateVectorSplat(W, sh1))),
                             b_.CreateVectorSplat(W, sh2));
    if (is_signed) {
        Value* flip = b_.CreateXor(neg_n, b_.CreateVectorSplat(W, neg_d));  // signs differ
        q = b_.CreateSelect(flip, b_.CreateNeg(q), q);
    }
    if (op == Instruction::UDiv || op == Instruction::SDiv) return q;
    return b_.CreateSub(n, b_.CreateMul(q, b_.CreateVectorSplat(W, d)), bin->getName());
}

// __cpusim_shfl: lane l gets v from lane src[l] of its own warp (the 32-lane
// segment it is in). One vpermi2d per 16 lanes, picking from the warp's two
// halves.
Value* WarpBuilder::emit_shfl(Value* v, Value* src) {
    // Each warp's lanes are 1, 2 or 4 vectors of 16 (S of them present: the
    // warp size, or fewer when the block is smaller than a warp; a source lane
    // that does not exist reads the caller's own value).
    Function* perm = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_vpermi2var_d_512);
    auto slice = [&](Value* x, unsigned first) {
        SmallVector<int, 16> idx;
        for (unsigned i = 0; i < 16; ++i) idx.push_back(int(first + i));
        return b_.CreateShuffleVector(x, idx);
    };
    auto splat = [&](uint32_t c) { return ConstantVector::getSplat(ElementCount::getFixed(16), b_.getInt32(c)); };
    const unsigned L = k_.warp_size, S = std::min(L, W);
    std::vector<Value*> parts;
    for (unsigned w = 0; w < W; w += S) {
        std::vector<Value*> c;
        for (unsigned i = 0; i < S; i += 16) c.push_back(slice(v, w + i));
        for (unsigned h = 0; h < S / 16; ++h) {
            Value* idx = b_.CreateAnd(slice(src, w + 16 * h), splat(L - 1));
            Value* r;
            if (S == 16) {
                r = b_.CreateCall(perm, {c[0], idx, c[0]});
            } else if (S == 32) {
                r = b_.CreateCall(perm, {c[0], idx, c[1]});
            } else {  // 64: two pairs, then bit 5 of the index picks one
                Value* lo = b_.CreateCall(perm, {c[0], idx, c[1]});
                Value* hi = b_.CreateCall(perm, {c[2], idx, c[3]});
                Value* upper = b_.CreateICmpNE(b_.CreateAnd(idx, splat(32)), splat(0));
                r = b_.CreateSelect(upper, hi, lo);
            }
            if (S < L) r = b_.CreateSelect(b_.CreateICmpUGE(idx, splat(S)), c[h], r);
            parts.push_back(r);
        }
    }
    return concat_chunks(parts);
}

// __cpusim_ballot: every lane of a warp gets the 64-bit mask of its warp's
// lanes that are active here with pred != 0.
// The next threads of the call's range (refill_next .. refill_end) for the
// lanes in `mask`, in lane order: lane l gets next + the number of lanes of
// the mask below l. The first call gives every lane one (mask: all lanes).
// Past the end: ~0.
Value* WarpBuilder::emit_refill(bool first, Value* mask) {
    auto* i64 = b_.getInt64Ty();
    Value* next_p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_REFILL_NEXT);
    Value* end = b_.CreateAlignedLoad(i64, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_REFILL_END),
                                      Align(8), "refill.end");
    Value* next = b_.CreateAlignedLoad(i64, next_p, Align(8), "refill.next");
    Value* rank;
    Value* count;
    if (first) {
        SmallVector<Constant*, 64> l;
        for (unsigned i = 0; i < W; ++i) l.push_back(b_.getInt64(i));
        rank = ConstantVector::get(l);
        count = b_.getInt64(W);
    } else {
        // Lanes of the mask below each lane: popcount(bits & ((1 << l) - 1)).
        Value* bits = b_.CreateZExt(b_.CreateBitCast(mask, b_.getIntNTy(W)), i64);
        SmallVector<Constant*, 64> below;
        for (unsigned i = 0; i < W; ++i) below.push_back(b_.getInt64(i == 64 ? ~0ull : (1ull << i) - 1));
        Value* masked = b_.CreateAnd(b_.CreateVectorSplat(W, bits), ConstantVector::get(below));
        rank = b_.CreateUnaryIntrinsic(Intrinsic::ctpop, masked);
        count = b_.CreateUnaryIntrinsic(Intrinsic::ctpop, bits);
    }
    Value* lin = b_.CreateAdd(b_.CreateVectorSplat(W, next), rank, "refill.lin");
    b_.CreateAlignedStore(b_.CreateAdd(next, count), next_p, Align(8));
    Value* past = b_.CreateICmpUGE(lin, b_.CreateVectorSplat(W, end));
    return b_.CreateSelect(past, ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt64(~0ull)), lin);
}

Value* WarpBuilder::emit_ballot(Value* pred, Value* mask) {
    const unsigned S = std::min(k_.warp_size, W);  // a warp's lanes present in the vector
    Value* bits = b_.CreateAnd(b_.CreateICmpNE(pred, Constant::getNullValue(pred->getType())), mask);
    Value* words = b_.CreateBitCast(bits, FixedVectorType::get(b_.getIntNTy(S), W / S));
    words = b_.CreateZExt(words, FixedVectorType::get(b_.getInt64Ty(), W / S));
    SmallVector<int, 64> idx;
    for (unsigned l = 0; l < W; ++l) idx.push_back(int(l / S));
    return b_.CreateShuffleVector(words, idx, "ballot");
}

// __cpusim_mma_<types>(d, a, b, c): one warp's 16x16x16 multiply-accumulate
// on fragments in per-thread arrays. Lane-major, element j of the warp's 32
// lanes is contiguous, so each warp of the vector hands its part of the four
// arrays to the native routine (libcpusim_tensor), with the vector width and
// its 32 bits of the mask: a warp with no lane here leaves its fragments alone.
bool WarpBuilder::emit_mma(CallInst* call, Value* mask) {
    auto* ptr = PointerType::get(ctx_, 0);
    auto* i32 = b_.getInt32Ty();
    auto* i64 = b_.getInt64Ty();
    const std::string native =
        kTensorPrefix + call->getCalledFunction()->getName().drop_front(StringRef(kMmaPrefix).size()).str();
    FunctionCallee fn = m_.getOrInsertFunction(native, FunctionType::get(b_.getVoidTy(), {ptr, ptr, ptr, ptr, i64, i32}, false));
    Value* base[4];
    uint64_t elem[4];
    for (unsigned i = 0; i < 4; ++i) {
        Value* p = call->getArgOperand(i);
        auto* a = dyn_cast<AllocaInst>(getUnderlyingObject(p, 12));
        if (!a || !private_elem_.count(a)) return fail("tensor core fragment not in a per-thread array", call), false;
        SplitAddress s = split_address(p);
        // The same fragment in every lane: each lane's copy at l * elem.
        elem[i] = private_elem_[a];
        auto* lanes = dyn_cast<Constant>(s.offsets);
        for (unsigned l = 0; lanes && l < W; ++l) {
            auto* e = dyn_cast_or_null<ConstantInt>(lanes->getAggregateElement(l));
            if (!e || e->getZExtValue() != l * elem[i]) lanes = nullptr;
        }
        if (!lanes) return fail("tensor core fragment chosen per lane (an array of fragments indexed by thread)", call), false;
        base[i] = s.base;
    }
    Value* bits = b_.CreateBitCast(mask, b_.getIntNTy(W));
    for (unsigned w = 0; w < W / CPUSIM_WARP_SIZE; ++w) {
        SmallVector<Value*, 6> args;
        for (unsigned i = 0; i < 4; ++i)
            args.push_back(b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), base[i], w * CPUSIM_WARP_SIZE * elem[i]));
        args.push_back(ConstantInt::get(i64, W));
        args.push_back(b_.CreateTrunc(b_.CreateLShr(bits, uint64_t(w * CPUSIM_WARP_SIZE)), i32));
        b_.CreateCall(fn, args);
    }
    return true;
}

// __cpusim_mmav_<in>_<acc>(j, a, b..., c...): element j of one warp's
// D = A * B + C, with B and C as values and A by pointer (its fragment, a
// per-thread array). Element e of a fragment, over a warp's 32 lanes, is
// 32 / C whole rows of its tile, C its columns (mma.h), so C's values are
// already accumulators, a row per C lanes, and B's hold rows of k. A's values
// are broadcast from its fragment in memory, which the load units do; loading
// A one step ahead (mma.h's note) keeps those loads off stores still in
// flight. The first call computes all of D; the others take their element.
//
// Per type, one dot-product instruction per (row, group of k):
//   bf16: vdpbf16ps, pairs of k. B's two rows interleaved with vpermw; A's
//         pair broadcast as the third operand ({1to16} memory form).
//   s8, u8: vpdpbusd, quads of k, unsigned bytes x signed bytes. A goes in as
//         it is, a broadcast load per quad, on the side of its signedness;
//         B's four rows interleaved with vpermb and xor 0x80 for the other
//         side (B + 128 unsigned, B - 128 signed); the 128 * (row sum of A)
//         that adds or removes comes from vpsadbw and is applied at the end
//         with broadcast loads. (A xor 0x80 with a column-sum fix would need
//         a vector xor per vpdpbusd, and a per-row broadcast out of a register
//         a shuffle: both on vpdpbusd's pipes.)
//   fp16: no fp16 dot product without AVX512-FP16: A converted to fp32 once
//         per mma into a scratch area, B's rows converted, then one FMA per
//         (row, k) with A broadcast from the scratch area.
//   tf32 (16 x 16 x 8): the same with fp32 values whose 13 low mantissa bits
//         are cleared (the tensor core ignores them).
//   f64 (8 x 8 x 4): one FMA of 8 doubles per (row, k), A broadcast from its
//         fragment.
//   s4, u4 (8 x 8 x 32): nibbles expanded to bytes, then as int8, with ymm.
//   b1 (8 x 8 x 128): xor or and, then vpopcntd, per (row, word of k).
Value* WarpBuilder::emit_mmav(CallInst* call) {
    std::vector<Value*> key{b_.GetInsertBlock()};
    for (unsigned i = 1; i < call->arg_size(); ++i) key.push_back(call->getArgOperand(i));
    std::array<Value*, 8>& out = mmav_done_[key];
    const unsigned j = unsigned(cast<ConstantInt>(call->getArgOperand(0))->getZExtValue());
    if (out[0]) return out[j];

    const StringRef types = call->getCalledFunction()->getName().drop_front(StringRef(kMmavPrefix).size());
    auto* i8 = b_.getInt8Ty();
    auto* i32 = b_.getInt32Ty();
    auto slice = [&](Value* v, unsigned first, unsigned n) {
        SmallVector<int, 64> idx;
        for (unsigned i = 0; i < n; ++i) idx.push_back(int(first + i));
        return b_.CreateShuffleVector(v, idx);
    };
    Value* pa = call->getArgOperand(1);
    auto* fa = dyn_cast<AllocaInst>(getUnderlyingObject(pa, 12));
    if (!fa || !private_elem_.count(fa)) return fail("tensor core fragment not in a per-thread array", call), nullptr;
    const uint64_t ea = private_elem_[fa];  // bytes per element of A
    SplitAddress sa = split_address(pa);
    if (auto* lanes = dyn_cast<Constant>(sa.offsets)) {
        for (unsigned l = 0; l < W; ++l) {
            auto* e = dyn_cast_or_null<ConstantInt>(lanes->getAggregateElement(l));
            if (!e || e->getZExtValue() != l * ea) return fail("tensor core fragment chosen per lane", call), nullptr;
        }
    } else {
        return fail("tensor core fragment chosen per lane", call), nullptr;
    }
    // The type and the tile shape: "<in>_<acc>" is 16 x 16 x 16 (tf32: 16 x
    // 16 x 8, f64: 8 x 8 x 4), "_m32n8k16" and "_m8n32k16" the other two.
    StringRef kind = types;
    unsigned M = 16, N = 16, K = 16;
    if (kind.consume_back("_m32n8k16")) M = 32, N = 8;
    else if (kind.consume_back("_m8n32k16")) M = 8, N = 32;
    else if (kind == "tf32_f32") K = 8;
    else if (kind == "f64_f64") M = 8, N = 8, K = 4;
    // Sub-byte types: 8 x 8 x 32 (4 bits) or 8 x 8 x 128 (bits), one 32-bit
    // word of A and of B per lane: word l % 4 of row (A) or column (B) l / 4.
    const bool subbyte = kind == "s4_s32" || kind == "u4_s32" || kind.starts_with("b1");
    if (subbyte) M = 8, N = 8, K = kind.starts_with("b1") ? 128 : 32;
    const unsigned nb = subbyte ? 1 : K * N / 32, nc = M * N / 32;  // elements of B and C per lane
    if (call->arg_size() != 2 + nb + nc) return fail("tensor core call with the wrong arguments", call), nullptr;
    Value *bv[16], *c[8];
    for (unsigned e = 0; e < nb; ++e) bv[e] = vector(call->getArgOperand(2 + e));
    for (unsigned e = 0; e < nc; ++e) c[e] = vector(call->getArgOperand(2 + nb + e));
    // An accumulator row is `segs` vectors of `vl` lanes: one zmm for 16
    // columns (8 for doubles), two for 32, one ymm for 8. Two ways to give
    // m32n8k16 zmm rows were slower: storing its tiles transposed (an m8n32k16
    // product, but loading them from row-major memory is then a gather per
    // element), and two rows interleaved per zmm (A's operand is then a
    // 64-bit broadcast, which does not fold into the instruction).
    const unsigned vl = kind == "f64_f64" ? 8 : std::min(N, 16u), segs = N / vl;
    // Accumulator rows done together (all groups of k for these rows, then
    // the next ones): at most 16 registers of accumulators live, so that
    // they, B and a broadcast fit in the 32 vector registers (m32n8k16 has
    // 32 rows of one ymm).
    const unsigned rows_per_block = std::min(M, std::max(1u, 16 / segs));
    // A's fragment and the scratch area are read through pointers passed
    // through an empty asm: LLVM no longer knows what they point to, so it
    // neither forwards the stores that filled them nor promotes A's array to
    // registers. Each value is then a broadcast load (a {1to16} operand
    // where the instruction has one); broadcast out of a register it would
    // be a shuffle per instruction, on the pipes that do the arithmetic.
    // (The scratch area is written through its own pointer and read through
    // one laundered after the stores: through the same pointer, LLVM would
    // still forward them.)
    Value* abase = launder(sa.base);
    // A fragment element holds 32 / C rows of its tile, C its columns.
    auto a_at = [&](unsigned l0, unsigned r, unsigned k) {  // A's row r, column k: byte address
        const unsigned per = 32 / K;
        return b_.CreateConstInBoundsGEP1_64(i8, abase, ((r / per) * W + l0 + (r % per) * K + k) * ea);
    };
    auto a_word = [&](unsigned l0, unsigned i) {  // sub-byte types: A's word in lane l0 + i
        return b_.CreateConstInBoundsGEP1_64(i8, abase, (l0 + i) * ea);
    };
    auto b_seg = [&](unsigned l0, unsigned k, unsigned s) {  // B's row k, segment s
        const unsigned per = 32 / N;
        return slice(bv[k / per], l0 + (k % per) * N + s * vl, vl);
    };
    auto splat_load = [&](Type* t, Value* at, unsigned n) {
        return b_.CreateVectorSplat(n, b_.CreateAlignedLoad(t, at, Align(t->getPrimitiveSizeInBits() / 8)));
    };

    std::array<std::vector<Value*>, 8> parts;  // per element of C: its row segments, in lane order
    for (unsigned w = 0; w < W / CPUSIM_WARP_SIZE; ++w) {
        const unsigned l0 = w * CPUSIM_WARP_SIZE;
        const unsigned per = 32 / N;  // accumulator rows per element
        std::vector<std::array<Value*, 2>> acc(M);
        for (unsigned r = 0; r < M; ++r)
            for (unsigned s = 0; s < segs; ++s) acc[r][s] = slice(c[r / per], l0 + (r % per) * N + s * vl, vl);
        if (kind == "bf16_f32") {
            // vdpbf16ps per (row, pair of k): B's rows 2p and 2p + 1
            // interleaved with vpermw; A's pair broadcast (the third operand,
            // which has the {1to16} memory form; pair products are symmetric).
            auto* vbf = FixedVectorType::get(b_.getBFloatTy(), 2 * vl);
            SmallVector<uint16_t, 32> pairs;
            for (unsigned n = 0; n < vl; ++n) pairs.push_back(uint16_t(n)), pairs.push_back(uint16_t(vl + n));
            Constant* pair_idx = ConstantDataVector::get(ctx_, ArrayRef<uint16_t>(pairs));
            Function* dp = Intrinsic::getDeclaration(&m_, vl == 16 ? Intrinsic::x86_avx512bf16_dpbf16ps_512
                                                                   : Intrinsic::x86_avx512bf16_dpbf16ps_256);
            Function* perm = Intrinsic::getDeclaration(&m_, vl == 16 ? Intrinsic::x86_avx512_permvar_hi_512
                                                                     : Intrinsic::x86_avx512_permvar_hi_256);
            std::vector<std::array<Value*, 2>> bp(K / 2);
            for (unsigned p = 0; p < K / 2; ++p)
                for (unsigned s = 0; s < segs; ++s)
                    bp[p][s] = b_.CreateBitCast(
                        b_.CreateCall(perm, {concat_chunks({b_seg(l0, 2 * p, s), b_seg(l0, 2 * p + 1, s)}), pair_idx}), vbf);
            // With AMX (the target has amx-bf16, a 32-lane warp, 16 x 16 x
            // 16): one tdpbf16ps on tiles instead of 128 vdpbf16ps. A's
            // fragment is its tile row by row (rows 32 bytes apart), B's
            // pairs above are already the interleaved (VNNI) rows the tile
            // operand wants, and the accumulator goes through the scratch
            // area as a 16 x 16 float tile (13 cycles per tdpbf16ps on a
            // Xeon, but see amx_bf16(): the round trip costs more). LLVM
            // configures the tiles at the function's entry (ldtilecfg); the
            // executor asks the OS for the tile state once (arch_prctl).
            if (amx_bf16() && W == 32 && M == 16 && N == 16 && K == 16 && segs == 1) {
                auto* i16t = b_.getInt16Ty();
                auto* i64t = b_.getInt64Ty();
                Value* scratch = mma_scratch();
                Value* cbuf = scratch;                                               // 16 rows x 64 bytes
                Value* bbuf = b_.CreateConstInBoundsGEP1_64(i8, scratch, 1024);      // 8 rows x 64 bytes
                for (unsigned r = 0; r < 16; ++r)
                    b_.CreateAlignedStore(acc[r][0], b_.CreateConstInBoundsGEP1_64(i8, cbuf, r * 64), Align(64));
                for (unsigned p = 0; p < 8; ++p)
                    b_.CreateAlignedStore(bp[p][0], b_.CreateConstInBoundsGEP1_64(i8, bbuf, p * 64), Align(64));
                Function* load = Intrinsic::getDeclaration(&m_, Intrinsic::x86_tileloadd64_internal);
                Function* dpt = Intrinsic::getDeclaration(&m_, Intrinsic::x86_tdpbf16ps_internal);
                Function* store = Intrinsic::getDeclaration(&m_, Intrinsic::x86_tilestored64_internal);
                auto c16 = [&](unsigned v) { return ConstantInt::get(i16t, v); };
                Value* tc = b_.CreateCall(load, {c16(16), c16(64), cbuf, ConstantInt::get(i64t, 64)});
                Value* ta = b_.CreateCall(load, {c16(16), c16(32), a_at(l0, 0, 0), ConstantInt::get(i64t, 32)});
                Value* tb = b_.CreateCall(load, {c16(8), c16(64), bbuf, ConstantInt::get(i64t, 64)});
                Value* td = b_.CreateCall(dpt, {c16(16), c16(64), c16(32), tc, ta, tb});
                b_.CreateCall(store, {c16(16), c16(64), cbuf, ConstantInt::get(i64t, 64), td});
                Value* back = launder(cbuf);
                for (unsigned r = 0; r < 16; ++r)
                    acc[r][0] = b_.CreateAlignedLoad(acc[r][0]->getType(), b_.CreateConstInBoundsGEP1_64(i8, back, r * 64),
                                                     Align(64));
            } else
            for (unsigned r0 = 0; r0 < M; r0 += rows_per_block)
                for (unsigned p = 0; p < K / 2; ++p)
                    for (unsigned r = r0; r < r0 + rows_per_block; ++r) {
                            Value* pr = b_.CreateVectorSplat(vl, b_.CreateAlignedLoad(i32, a_at(l0, r, 2 * p), Align(4)));
                        for (unsigned s = 0; s < segs; ++s)
                            acc[r][s] = b_.CreateCall(dp, {acc[r][s], bp[p][s], b_.CreateBitCast(pr, vbf)});
                    }
        } else if (kind == "s8_s32" || kind == "u8_s32") {
            const bool sgn = kind == "s8_s32";
            auto* vi32 = FixedVectorType::get(i32, vl);
            auto* v64i8 = FixedVectorType::get(i8, 64);
            auto* v32i8 = FixedVectorType::get(i8, 32);
            SmallVector<uint8_t, 64> quads;
            for (unsigned n = 0; n < vl; ++n)
                for (unsigned t = 0; t < 4; ++t) quads.push_back(uint8_t(vl * t + n));
            Constant* quad_idx = ConstantDataVector::get(ctx_, ArrayRef<uint8_t>(quads));
            Function* dp = Intrinsic::getDeclaration(&m_, vl == 16 ? Intrinsic::x86_avx512_vpdpbusd_512
                                                                   : Intrinsic::x86_avx512_vpdpbusd_256);
            Function* perm = Intrinsic::getDeclaration(&m_, vl == 16 ? Intrinsic::x86_avx512_permvar_qi_512
                                                                     : Intrinsic::x86_avx512_permvar_qi_256);
            Function* sad = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_psad_bw_512);
            // A's row sums: vpsadbw (unsigned; s8 goes in as A + 128 and
            // 16 * 128 comes off), 4 rows per register, two 8-byte halves
            // each added with a qword swap, stored as they are: row r's fix
            // is the dword at (r / 4) * 64 + (r % 4) * 16 of the scratch area.
            Value* scratch = mma_scratch();
            auto* v8i64 = FixedVectorType::get(b_.getInt64Ty(), 8);
            for (unsigned r = 0; r < M; r += 4) {
                Value* ab = concat_chunks({b_.CreateAlignedLoad(v32i8, a_at(l0, r, 0), Align(1)),
                                           b_.CreateAlignedLoad(v32i8, a_at(l0, r + 2, 0), Align(1))});
                if (sgn) ab = b_.CreateXor(ab, ConstantInt::get(v64i8, 0x80));
                Value* sums = b_.CreateCall(sad, {ab, Constant::getNullValue(v64i8)});
                Value* rows = b_.CreateAdd(sums, b_.CreateShuffleVector(sums, {1, 0, 3, 2, 5, 4, 7, 6}));
                if (sgn) rows = b_.CreateSub(rows, ConstantInt::get(v8i64, 16 * 128));
                b_.CreateAlignedStore(b_.CreateShl(rows, 7), b_.CreateConstInBoundsGEP1_64(i8, scratch, r / 4 * 64), Align(64));
            }
            // vpdpbusd per (row, quad of k): A's quad as it is, a broadcast
            // load, on the side of its signedness; B's four rows interleaved
            // with vpermb and xor 0x80 for the other side (B + 128 unsigned,
            // B - 128 signed), which adds or removes 128 * (row sum of A).
            Value* sread = launder(scratch);
            Value* flip = ConstantInt::get(vi32, 0x80808080);
            std::vector<std::array<Value*, 2>> bq(K / 4);
            for (unsigned q = 0; q < K / 4; ++q)
                for (unsigned s = 0; s < segs; ++s) {
                    Value* rows = concat_chunks({b_seg(l0, 4 * q, s), b_seg(l0, 4 * q + 1, s), b_seg(l0, 4 * q + 2, s),
                                                 b_seg(l0, 4 * q + 3, s)});
                    bq[q][s] = b_.CreateXor(b_.CreateBitCast(b_.CreateCall(perm, {rows, quad_idx}), vi32), flip);
                }
            for (unsigned r0 = 0; r0 < M; r0 += rows_per_block)
                for (unsigned q = 0; q < K / 4; ++q)
                    for (unsigned r = r0; r < r0 + rows_per_block; ++r) {
                        Value* quad = splat_load(i32, a_at(l0, r, 4 * q), vl);
                        for (unsigned s = 0; s < segs; ++s)
                            acc[r][s] = sgn ? b_.CreateCall(dp, {acc[r][s], bq[q][s], quad})
                                            : b_.CreateCall(dp, {acc[r][s], quad, bq[q][s]});
                    }
            // The fix, at the end: by then the store above is done.
            for (unsigned r = 0; r < M; ++r) {
                Value* fix = splat_load(i32, b_.CreateConstInBoundsGEP1_64(i8, sread, r / 4 * 64 + r % 4 * 16), vl);
                for (unsigned s = 0; s < segs; ++s)
                    acc[r][s] = sgn ? b_.CreateSub(acc[r][s], fix) : b_.CreateAdd(acc[r][s], fix);
            }
        } else if (kind == "f16_f32" || kind == "tf32_f32") {
            // One FMA per (row, k), A in fp32 in the scratch area, row-major,
            // broadcast from there. fp16: A and B converted (no fp16 dot
            // product without AVX512-FP16). tf32: the 13 low mantissa bits of
            // A and B cleared, as the tensor core ignores them.
            const bool tf = kind == "tf32_f32";
            auto* f32 = b_.getFloatTy();
            auto* vf = FixedVectorType::get(f32, vl);
            Function* fma = Intrinsic::getDeclaration(&m_, Intrinsic::fma, {vf});
            auto to_f32 = [&](Value* v) {  // a vector of B or A as loaded: half bits, or float
                auto* vt = cast<FixedVectorType>(v->getType());
                const unsigned n = vt->getNumElements();
                if (!tf) return b_.CreateFPExt(b_.CreateBitCast(v, FixedVectorType::get(b_.getHalfTy(), n)),
                                               FixedVectorType::get(f32, n));
                auto* it = FixedVectorType::get(i32, n);
                return b_.CreateBitCast(b_.CreateAnd(b_.CreateBitCast(v, it), ConstantInt::get(it, 0xffffe000u)), vt);
            };
            Value* scratch = mma_scratch();
            auto* ta = FixedVectorType::get(tf ? f32 : b_.getInt16Ty(), 32);
            for (unsigned e = 0; e < M * K / 32; ++e)  // element e: 32 / K rows of A
                b_.CreateAlignedStore(to_f32(b_.CreateAlignedLoad(ta, a_at(l0, e * (32 / K), 0), Align(2))),
                                      b_.CreateConstInBoundsGEP1_64(i8, scratch, e * 128), Align(64));
            Value* sread = launder(scratch);
            std::vector<std::array<Value*, 2>> bk(K);
            for (unsigned k = 0; k < K; ++k)
                for (unsigned s = 0; s < segs; ++s) bk[k][s] = to_f32(b_seg(l0, k, s));
            for (unsigned r0 = 0; r0 < M; r0 += rows_per_block)
                for (unsigned k = 0; k < K; ++k)
                    for (unsigned r = r0; r < r0 + rows_per_block; ++r) {
                        Value* av = splat_load(f32, b_.CreateConstInBoundsGEP1_64(i8, sread, (r * K + k) * 4), vl);
                        for (unsigned s = 0; s < segs; ++s) acc[r][s] = b_.CreateCall(fma, {av, bk[k][s], acc[r][s]});
                    }
        } else if (kind == "f64_f64") {
            // One FMA of 8 doubles per (row, k), A broadcast from its fragment.
            auto* f64 = b_.getDoubleTy();
            Function* fma = Intrinsic::getDeclaration(&m_, Intrinsic::fma, {FixedVectorType::get(f64, 8)});
            for (unsigned k = 0; k < K; ++k) {
                Value* bk = b_seg(l0, k, 0);
                for (unsigned r = 0; r < M; ++r)
                    acc[r][0] = b_.CreateCall(fma, {splat_load(f64, a_at(l0, r, k), 8), bk, acc[r][0]});
            }
        } else if (kind == "s4_s32" || kind == "u4_s32") {
            // Nibbles expanded to bytes once per mma, then vpdpbusd (ymm: the
            // 8 columns of D) per (row, quad of k), as for int8. A goes on the
            // unsigned side: u4 as it is, s4 xor 8 (= A + 8, 0..15); B on the
            // signed side, s4 sign-extended. s4 then has 8 * (column sum of
            // B) too many: one vector, the same for every row.
            const bool sgn = kind == "s4_s32";
            auto* v8i32 = FixedVectorType::get(i32, 8);
            Function* dp = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_vpdpbusd_256);
            // 4 nibbles (half h of each word) into the 4 bytes of a dword.
            auto expand = [&](Value* words, unsigned h) {
                auto* t = cast<FixedVectorType>(words->getType());
                auto k = [&](uint32_t v) { return ConstantInt::get(t, v); };
                Value* x = h ? b_.CreateLShr(words, k(16)) : words;
                Value* r = b_.CreateAnd(x, k(0xf));
                r = b_.CreateOr(r, b_.CreateShl(b_.CreateAnd(x, k(0xf0)), k(4)));
                r = b_.CreateOr(r, b_.CreateShl(b_.CreateAnd(x, k(0xf00)), k(8)));
                return b_.CreateOr(r, b_.CreateShl(b_.CreateAnd(x, k(0xf000)), k(12)));
            };
            auto pick = [&](Value* v, unsigned w) {  // word w of each of the 8 rows or columns
                SmallVector<int, 8> idx;
                for (unsigned n = 0; n < 8; ++n) idx.push_back(int(4 * n + w));
                return b_.CreateShuffleVector(v, idx);
            };
            Value* bw = slice(bv[0], l0, 32);
            Value* be[2] = {expand(bw, 0), expand(bw, 1)};
            Value* bq[8];
            for (unsigned q = 0; q < 8; ++q) {
                bq[q] = pick(be[q % 2], q / 2);
                if (sgn) {  // sign-extend each byte's nibble: (b ^ 8) - 8
                    auto* v32i8 = FixedVectorType::get(i8, 32);
                    Value* by = b_.CreateBitCast(bq[q], v32i8);
                    by = b_.CreateSub(b_.CreateXor(by, ConstantInt::get(v32i8, 8)), ConstantInt::get(v32i8, 8));
                    bq[q] = b_.CreateBitCast(by, v8i32);
                }
            }
            // A, expanded, in the scratch area: row m, quad q at (8 m + q) * 4.
            Value* aw = b_.CreateAlignedLoad(FixedVectorType::get(i32, 32), a_word(l0, 0), Align(4));
            Value* ae[2] = {expand(aw, 0), expand(aw, 1)};
            if (sgn)
                for (Value*& v : ae) v = b_.CreateXor(v, ConstantInt::get(v->getType(), 0x08080808));
            SmallVector<int, 64> inter;
            for (unsigned m = 0; m < 8; ++m)
                for (unsigned q = 0; q < 8; ++q) inter.push_back(int((q % 2) * 32 + 4 * m + q / 2));
            Value* scratch = mma_scratch();
            b_.CreateAlignedStore(b_.CreateShuffleVector(ae[0], ae[1], inter), scratch, Align(64));
            Value* sread = launder(scratch);
            if (sgn) {
                Value* ones = ConstantInt::get(v8i32, 0x01010101);
                Value* colsum = Constant::getNullValue(v8i32);
                for (unsigned q = 0; q < 8; ++q) colsum = b_.CreateCall(dp, {colsum, ones, bq[q]});
                for (unsigned r = 0; r < 8; ++r) acc[r][0] = b_.CreateSub(acc[r][0], b_.CreateShl(colsum, 3));
            }
            for (unsigned q = 0; q < 8; ++q)
                for (unsigned r = 0; r < 8; ++r) {
                    Value* quad = splat_load(i32, b_.CreateConstInBoundsGEP1_64(i8, sread, (8 * r + q) * 4), 8);
                    acc[r][0] = b_.CreateCall(dp, {acc[r][0], quad, bq[q]});
                }
        } else if (kind == "b1xor_s32" || kind == "b1and_s32") {
            // D[m][n] = C + popcount(A row m op B column n), 128 bits: per
            // row, 4 words of A broadcast against the 8 columns' word w, then
            // vpopcntd (AVX512-VPOPCNTDQ).
            const bool xor_op = kind == "b1xor_s32";
            auto* v8i32 = FixedVectorType::get(i32, 8);
            Function* popc = Intrinsic::getDeclaration(&m_, Intrinsic::ctpop, {v8i32});
            Value* bw = slice(bv[0], l0, 32);
            Value* bcol[4];
            for (unsigned w = 0; w < 4; ++w) {
                SmallVector<int, 8> idx;
                for (unsigned n = 0; n < 8; ++n) idx.push_back(int(4 * n + w));
                bcol[w] = b_.CreateShuffleVector(bw, idx);
            }
            for (unsigned r = 0; r < 8; ++r)
                for (unsigned w = 0; w < 4; ++w) {
                    Value* aword = splat_load(i32, a_word(l0, 4 * r + w), 8);
                    Value* x = xor_op ? b_.CreateXor(aword, bcol[w]) : b_.CreateAnd(aword, bcol[w]);
                    acc[r][0] = b_.CreateAdd(acc[r][0], b_.CreateCall(popc, {x}));
                }
        } else {
            return fail("tensor core types not supported inline: " + types.str(), call), nullptr;
        }
        for (unsigned e = 0; e < nc; ++e)
            for (unsigned i = 0; i < per; ++i)
                for (unsigned s = 0; s < segs; ++s) parts[e].push_back(acc[e * per + i][s]);
    }
    for (unsigned e = 0; e < nc; ++e) out[e] = concat_chunks(parts[e]);
    return out[j];
}


}  // namespace simt_internal
}  // namespace cpusim
