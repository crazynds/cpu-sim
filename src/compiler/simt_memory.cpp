// The SIMT pass: loads, stores, gathers and scatters of the warp code.
#include "compiler/simt_internal.h"

namespace cpusim {
namespace simt_internal {

// ---------------------------------------------------------------- memory

// The W-lane vector made of 16-lane parts, joined pairwise (16+16, 32+32,
// ...): one shuffle per pair instead of one W-wide shuffle per part, which
// made the optimizer's work quadratic in W.
Value* WarpBuilder::concat_chunks(const std::vector<Value*>& parts) {
    std::vector<Value*> level = parts;
    while (level.size() > 1) {
        const int n = int(cast<FixedVectorType>(level[0]->getType())->getNumElements());
        SmallVector<int, 512> idx;
        for (int i = 0; i < 2 * n; ++i) idx.push_back(i);
        std::vector<Value*> next;
        for (size_t i = 0; i + 1 < level.size(); i += 2) next.push_back(b_.CreateShuffleVector(level[i], level[i + 1], idx));
        level = std::move(next);
    }
    return level[0];
}

// Lanes [16h, 16h + 16) of a 32-lane vector.
Value* WarpBuilder::half(Value* v, unsigned h) {
    SmallVector<int, 16> idx;
    for (int i = 0; i < 16; ++i) idx.push_back(int(h) * 16 + i);
    return b_.CreateShuffleVector(v, idx);
}

// Splits an integer index into (uniform scalar i64, varying <32 x i64>) whose
// sum is the index widened to 64 bits. Distributes over additions that cannot
// wrap (nsw for a sign extension, nuw for a zero extension).
std::pair<Value*, Value*> WarpBuilder::split_index(Value* idx, bool is_signed) {
    auto* i64 = b_.getInt64Ty();
    auto* v64 = FixedVectorType::get(i64, W);
    if (!uni_->is_varying(idx)) {
        Value* s = scalar(idx);
        return {is_signed ? b_.CreateSExtOrTrunc(s, i64) : b_.CreateZExtOrTrunc(s, i64), Constant::getNullValue(v64)};
    }
    if (auto* ext = dyn_cast<CastInst>(idx); ext && (isa<SExtInst>(ext) || isa<ZExtInst>(ext)))
        return split_index(ext->getOperand(0), isa<SExtInst>(ext));
    if (auto* add = dyn_cast<BinaryOperator>(idx);
        add && (add->getOpcode() == Instruction::Add ||
                (add->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(add)->isDisjoint()))) {
        // `or disjoint` (what instcombine makes of t + threadIdx.x when the
        // bits cannot overlap) is an addition that cannot wrap.
        bool disjoint = add->getOpcode() == Instruction::Or;
        // Under a sign extension, an add without flags (CUDA indexes often
        // mix int and unsigned threadIdx, so the add is unsigned) is split too:
        // it can only differ from the exact result when the signed sum leaves
        // the int range, and then the original index is ~2^31 elements away,
        // outside any allocation. Not under a zero extension, where unsigned
        // wrap-around can be legitimate.
        bool no_wrap = disjoint || idx->getType()->isIntegerTy(64) || is_signed || add->hasNoUnsignedWrap();
        if (no_wrap) {
            auto [u0, v0] = split_index(add->getOperand(0), is_signed);
            auto [u1, v1] = split_index(add->getOperand(1), is_signed);
            return {b_.CreateAdd(u0, u1), b_.CreateAdd(v0, v1)};
        }
    }
    // (u + v) * c and (u + v) << c with c uniform: both parts scale (same
    // assumption about wrapping as for additions).
    if (auto* bo = dyn_cast<BinaryOperator>(idx);
        bo && (bo->getOpcode() == Instruction::Mul || bo->getOpcode() == Instruction::Shl) &&
        (is_signed || bo->hasNoUnsignedWrap() || idx->getType()->isIntegerTy(64))) {
        Value* a = bo->getOperand(0);
        Value* c = bo->getOperand(1);
        if (bo->getOpcode() == Instruction::Mul && uni_->is_varying(c) && !uni_->is_varying(a)) std::swap(a, c);
        if (!uni_->is_varying(c) && uni_->is_varying(a)) {
            auto [u, v] = split_index(a, is_signed);
            Value* cs = scalar(c);
            Value* c64 = is_signed && bo->getOpcode() == Instruction::Mul ? b_.CreateSExtOrTrunc(cs, i64)
                                                                          : b_.CreateZExtOrTrunc(cs, i64);
            if (bo->getOpcode() == Instruction::Mul)
                return {b_.CreateMul(u, c64), b_.CreateMul(v, b_.CreateVectorSplat(W, c64))};
            return {b_.CreateShl(u, c64), b_.CreateShl(v, b_.CreateVectorSplat(W, c64))};
        }
    }
    Value* v = vector(idx);
    v = is_signed ? b_.CreateSExtOrTrunc(v, v64) : b_.CreateZExtOrTrunc(v, v64);
    return {ConstantInt::get(i64, 0), v};
}

// Writes a varying pointer as uniform base + varying byte offsets, looking
// through GEPs. The offsets usually depend only on the thread index, so
// everything computed from them alone (the access pattern tests below) is
// loop invariant and gets hoisted.
namespace {

// Shared memory or a per-thread array: an object far below 2 GiB.
bool small_object(const Value* p) {
    const Value* o = getUnderlyingObject(p, 0);
    if (isa<AllocaInst>(o)) return true;
    auto* c = dyn_cast<CallInst>(o);
    return c && c->getCalledFunction() && c->getCalledFunction()->getName() == kSharedBaseFn;
}

}  // namespace

WarpBuilder::SplitAddress WarpBuilder::split_address(Value* ptr) {
    auto* i64 = b_.getInt64Ty();
    auto* v64 = FixedVectorType::get(i64, W);
    if (auto* a = dyn_cast<AllocaInst>(ptr); a && private_elem_.count(a)) {
        // Lane l's copy of the array starts at base + l * elem.
        SmallVector<Constant*, 64> lanes;
        for (unsigned l = 0; l < W; ++l) lanes.push_back(ConstantInt::get(i64, l * private_elem_[a]));
        Value* base = private_base(a);
        return {base, ConstantVector::get(lanes), W, base, ConstantInt::get(i64, 0), true};
    }
    if (!uni_->is_varying(ptr)) {
        Value* base = scalar(ptr);
        return {base, Constant::getNullValue(v64), 1, base, ConstantInt::get(i64, 0), small_object(ptr)};
    }
    if (auto* gep = dyn_cast<GetElementPtrInst>(ptr)) {
        const DataLayout& dl = m_.getDataLayout();
        SplitAddress inner = split_address(gep->getPointerOperand());
        Value* uniform = ConstantInt::get(i64, 0);
        Value* varying = inner.offsets;
        for (gep_type_iterator it = gep_type_begin(gep), end = gep_type_end(gep); it != end; ++it) {
            if (StructType* st = it.getStructTypeOrNull()) {
                uint64_t field = cast<ConstantInt>(it.getOperand())->getZExtValue();
                uniform = b_.CreateAdd(uniform, ConstantInt::get(i64, dl.getStructLayout(st)->getElementOffset(field) *
                                                                          inner.scale));
                continue;
            }
            uint64_t size = it.getSequentialElementStride(dl) * inner.scale;
            auto [u, v] = split_index(it.getOperand(), true);
            uniform = b_.CreateAdd(uniform, b_.CreateMul(u, ConstantInt::get(i64, size)));
            varying = b_.CreateAdd(varying, b_.CreateMul(v, ConstantVector::getSplat(ElementCount::getFixed(W),
                                                                                    ConstantInt::get(i64, size))));
        }
        SplitAddress out{b_.CreateGEP(b_.getInt8Ty(), inner.base, uniform), varying, inner.scale};
        if (inner.root) {
            out.root = inner.root;
            out.uoff = b_.CreateAdd(inner.uoff, uniform);
            out.small = inner.small;
        }
        return out;
    }
    // Opaque varying pointer: offsets from address zero, no root.
    Value* addrs = b_.CreatePtrToInt(vector(ptr), v64);
    return {ConstantPointerNull::get(cast<PointerType>(ptr->getType()->getScalarType())), addrs};
}

namespace {

Constant* lane_steps(IRBuilder<>& b, uint64_t size) {
    SmallVector<Constant*, 16> steps;
    for (unsigned i = 0; i < 16; ++i) steps.push_back(ConstantInt::get(b.getInt64Ty(), i * size));
    return ConstantVector::get(steps);
}

}  // namespace

bool WarpBuilder::linear_form(Value* v, LinearForm& out, int ext) {
    out.terms.clear();
    if (auto* c = dyn_cast<Constant>(v)) {
        std::vector<int64_t> lanes(W);
        for (unsigned i = 0; i < W; ++i) {
            auto* e = dyn_cast_or_null<ConstantInt>(c->getAggregateElement(i));
            if (!e) return false;
            lanes[i] = ext == 1 ? int64_t(e->getZExtValue()) : e->getSExtValue();
        }
        out.terms.push_back({nullptr, std::move(lanes)});
        return true;
    }
    if (Value* s = getSplatValue(v)) {
        if (ext) s = ext == 1 ? b_.CreateZExt(s, b_.getInt64Ty()) : b_.CreateSExt(s, b_.getInt64Ty());
        out.terms.push_back({s, std::vector<int64_t>(W, 1)});
        return true;
    }
    if (auto* cast = dyn_cast<CastInst>(v); cast && !ext && (isa<ZExtInst>(cast) || isa<SExtInst>(cast)))
        return linear_form(cast->getOperand(0), out, isa<ZExtInst>(cast) ? 1 : 2);
    auto* bo = dyn_cast<BinaryOperator>(v);
    if (!bo) return false;
    // Under an extension, the operation must not wrap in the narrow type for
    // the extension to distribute over it. Under a sign extension the same
    // assumption as split_index: an int index that wraps would point ~2^31
    // elements away, outside any allocation (CUDA indexes mix int and the
    // unsigned threadIdx, so their adds carry no flags).
    bool disjoint = bo->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(bo)->isDisjoint();
    if (ext == 1 && !disjoint) {
        auto* ob = dyn_cast<OverflowingBinaryOperator>(bo);
        if (!ob) return false;
        if (!ob->hasNoUnsignedWrap()) {
            // No flag, but the value ranges may still rule out wrapping
            // (blockIdx.y * 16 + threadIdx.y with size_t indexes).
            SimplifyQuery q(m_.getDataLayout());
            OverflowResult r = OverflowResult::MayOverflow;
            if (bo->getOpcode() == Instruction::Add)
                r = computeOverflowForUnsignedAdd(WithCache<const Value*>(bo->getOperand(0)),
                                                  WithCache<const Value*>(bo->getOperand(1)), q);
            else if (bo->getOpcode() == Instruction::Mul)
                r = computeOverflowForUnsignedMul(bo->getOperand(0), bo->getOperand(1), q);
            if (r != OverflowResult::NeverOverflows) return false;
        }
    }
    LinearForm a, b;
    switch (bo->getOpcode()) {
        case Instruction::Or:
            if (!disjoint) return false;
            [[fallthrough]];
        case Instruction::Add:
        case Instruction::Sub:
            if (!linear_form(bo->getOperand(0), a, ext) || !linear_form(bo->getOperand(1), b, ext)) return false;
            if (bo->getOpcode() == Instruction::Sub)
                for (auto& t : b.terms)
                    for (int64_t& x : t.second) x = -x;
            out.terms = std::move(a.terms);
            out.terms.insert(out.terms.end(), b.terms.begin(), b.terms.end());
            return true;
        case Instruction::Mul:
        case Instruction::Shl: {
            if (!linear_form(bo->getOperand(0), a, ext) || !linear_form(bo->getOperand(1), b, ext)) return false;
            auto constant = [](const LinearForm& f) { return f.terms.size() == 1 && !f.terms[0].first; };
            if (bo->getOpcode() == Instruction::Shl) {
                if (!constant(b)) return false;
                for (int64_t& x : b.terms[0].second) {
                    if (x < 0 || x > 62) return false;
                    x = int64_t(1) << x;
                }
            } else if (!constant(b)) {
                std::swap(a, b);
                if (!constant(b)) {
                    // a * splat(s): every coefficient times s (a scalar product).
                    auto splat_of = [](const LinearForm& f) -> Value* {
                        if (f.terms.size() != 1 || !f.terms[0].first) return nullptr;
                        for (int64_t x : f.terms[0].second)
                            if (x != 1) return nullptr;
                        return f.terms[0].first;
                    };
                    Value* s = splat_of(b);
                    if (!s) std::swap(a, b), s = splat_of(b);
                    if (!s) return false;
                    for (auto& t : a.terms) t.first = t.first ? b_.CreateMul(t.first, s) : s;
                    out.terms = std::move(a.terms);
                    return true;
                }
            }
            // a * constant lanes, term by term (splat(s) * c is the term s * c).
            const std::vector<int64_t>& k = b.terms[0].second;
            for (auto& t : a.terms)
                for (unsigned i = 0; i < W; ++i) t.second[i] *= k[i];
            out.terms = std::move(a.terms);
            return true;
        }
        default:
            return false;
    }
}

std::vector<char> WarpBuilder::static_chunks(const LinearForm& f, uint64_t size, std::vector<Value*>& firsts) {
    const unsigned chunks = W / 16;
    std::vector<char> kind(chunks, 0);
    firsts.assign(chunks, nullptr);
    auto* i64 = b_.getInt64Ty();
    for (unsigned h = 0; h < chunks; ++h) {
        const unsigned l0 = h * 16;
        bool uniform_terms = true;  // the uniform coefficients add the same to every lane of the chunk
        std::vector<int64_t> delta(16, 0);
        Value* first = ConstantInt::get(i64, 0);
        int64_t first_const = 0;
        for (const auto& [coef, lanes] : f.terms) {
            for (unsigned i = 0; i < 16; ++i) {
                if (!coef) delta[i] += lanes[l0 + i] - lanes[l0];
                else if (lanes[l0 + i] != lanes[l0]) uniform_terms = false;
            }
            if (!coef) first_const += lanes[l0];
            else if (lanes[l0]) first = b_.CreateAdd(first, b_.CreateMul(coef, ConstantInt::get(i64, lanes[l0])));
        }
        if (!uniform_terms) continue;
        firsts[h] = b_.CreateAdd(first, ConstantInt::get(i64, first_const));
        bool contig = true, same = true;
        for (unsigned i = 0; i < 16; ++i) {
            contig &= delta[i] == int64_t(i * size);
            same &= delta[i] == 0;
        }
        kind[h] = contig ? 'c' : same ? 'b' : 0;
    }
    return kind;
}

unsigned WarpBuilder::static_groups(const LinearForm& f, uint64_t size, std::vector<Value*>& firsts) {
    auto* i64 = b_.getInt64Ty();
    for (unsigned g : {8u, 4u, 2u}) {
        bool fits = true;
        for (unsigned s = 0; s < W && fits; s += g)
            for (unsigned i = 1; i < g && fits; ++i) {
                int64_t delta = 0;
                for (const auto& [coef, lanes] : f.terms) {
                    if (coef) fits &= lanes[s + i] == lanes[s];
                    else delta += lanes[s + i] - lanes[s];
                }
                fits &= delta == int64_t(i * size);
            }
        if (!fits) continue;
        firsts.assign(W / g, nullptr);
        for (unsigned s = 0; s < W; s += g) {
            Value* first = ConstantInt::get(i64, 0);
            int64_t k = 0;
            for (const auto& [coef, lanes] : f.terms) {
                if (!coef) k += lanes[s];
                else if (lanes[s]) first = b_.CreateAdd(first, b_.CreateMul(coef, ConstantInt::get(i64, lanes[s])));
            }
            firsts[s / g] = b_.CreateAdd(first, ConstantInt::get(i64, k));
        }
        return g;
    }
    return 0;
}

// A per-thread array the lane-major layout can hold: a fixed size, one
// scalar size throughout, and only element accesses (through GEPs) as uses.
bool WarpBuilder::check_private_array(AllocaInst* a) {
    const DataLayout& dl = m_.getDataLayout();
    if (!a->isStaticAlloca()) return fail("per-thread arrays of variable size are not supported", a), false;
    // The one scalar size of the type (arrays, vectors and structs of it).
    std::function<uint64_t(Type*)> elem = [&](Type* t) -> uint64_t {
        if (auto* at = dyn_cast<ArrayType>(t)) return elem(at->getElementType());
        if (auto* vt = dyn_cast<FixedVectorType>(t)) return elem(vt->getElementType());
        if (auto* st = dyn_cast<StructType>(t)) {
            uint64_t size = 0;
            for (Type* f : st->elements()) {
                uint64_t e = elem(f);
                if (!e || (size && e != size)) return 0;
                size = e;
            }
            return size;
        }
        if (t->isIntegerTy() || t->isFloatingPointTy() || t->isPointerTy()) return dl.getTypeStoreSize(t);
        return 0;
    };
    // An array the lane-major layout cannot hold (mixed element sizes, or
    // read with another size: bytes of a long, a struct copied field by
    // field) gets a real copy per lane instead, as one passed to a function.
    auto per_lane = [&]() {
        escaping_.insert(a);
        return true;
    };
    const uint64_t size = elem(a->getAllocatedType());
    if (!size) return per_lane();
    SmallVector<Value*, 8> work{a};
    while (!work.empty()) {
        Value* v = work.pop_back_val();
        for (User* u : v->users()) {
            if (auto* gep = dyn_cast<GetElementPtrInst>(u); gep && gep->getPointerOperand() == v) {
                work.push_back(gep);
            } else if (auto* ld = dyn_cast<LoadInst>(u)) {
                if (dl.getTypeStoreSize(ld->getType()) != size)
                    return per_lane();
            } else if (auto* st = dyn_cast<StoreInst>(u); st && st->getPointerOperand() == v) {
                if (dl.getTypeStoreSize(st->getValueOperand()->getType()) != size)
                    return per_lane();
            } else if (auto* ii = dyn_cast<IntrinsicInst>(u); ii && ii->isLifetimeStartOrEnd()) {
            } else if (is_mma(u) || is_mmav(u)) {  // a tensor core fragment, handed over whole (emit_mma)
            } else {
                return per_lane();
            }
        }
    }
    private_elem_[a] = size;
    return true;
}

// Where the lanes' copies of a per-thread array live: in the warp's frame
// when the variant suspends at barriers (the stack is gone on resume),
// otherwise in this call's stack.
Value* WarpBuilder::private_base(const AllocaInst* a) {
    const DataLayout& dl = m_.getDataLayout();
    const uint64_t bytes = dl.getTypeAllocSize(a->getAllocatedType()) * W;
    if (frame_) {
        auto [it, fresh] = private_frame_offset_.try_emplace(a, 0);
        if (fresh) {
            it->second = alignTo(cache_end_, 64);
            cache_end_ = it->second + bytes;
            note_frame_size();
        }
        return b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, it->second, a->getName());
    }
    Value*& stack = private_stack_[a];
    if (!stack) {
        IRBuilder<> ab(alloca_block_->getTerminator());
        auto* s = ab.CreateAlloca(ArrayType::get(b_.getInt8Ty(), bytes), nullptr, a->getName());
        s->setAlignment(Align(64));
        stack = s;
    }
    return stack;
}

// Re-creates `v` in the warp-start block if it only depends on the prologue
// (context fields, parameters) and constants through pure operations.
Value* WarpBuilder::clone_invariant(Value* v, DenseMap<Value*, Value*>& memo) {
    if (isa<Constant>(v) || isa<Argument>(v)) return v;
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst) return nullptr;
    if (inst->getParent() == entry_ || inst->getParent() == alloca_block_) return v;
    if (auto it = memo.find(v); it != memo.end()) return it->second;
    if (!(isa<BinaryOperator>(inst) || isa<CastInst>(inst) || isa<CmpInst>(inst) || isa<SelectInst>(inst) ||
          isa<GetElementPtrInst>(inst) || isa<ShuffleVectorInst>(inst) || isa<ExtractElementInst>(inst) ||
          isa<InsertElementInst>(inst) || isa<FreezeInst>(inst) || isa<UnaryOperator>(inst)))
        return memo[v] = nullptr;
    switch (inst->getOpcode()) {
        case Instruction::UDiv: case Instruction::SDiv: case Instruction::URem: case Instruction::SRem:
            return memo[v] = nullptr;
        default:
            break;
    }
    Instruction* c = inst->clone();
    for (Use& op : c->operands()) {
        Value* o = clone_invariant(op.get(), memo);
        if (!o) {
            c->deleteValue();
            return memo[v] = nullptr;
        }
        op.set(o);
    }
    c->insertBefore(cache_block_->getTerminator()->getIterator());
    return memo[v] = c;
}

// The value of `v`, from the warp's frame if it is warp-invariant.
Value* WarpBuilder::cached(Value* v) {
    if (!cache_block_ || isa<Constant>(v)) return v;
    DenseMap<Value*, Value*> memo;
    Value* at_start = clone_invariant(v, memo);
    if (!at_start || at_start == v) return v;
    const DataLayout& dl = m_.getDataLayout();
    Type* t = v->getType();
    uint64_t align = std::min<uint64_t>(64, dl.getPrefTypeAlign(t).value());
    uint64_t offset = alignTo(cache_end_, align);
    cache_end_ = offset + dl.getTypeAllocSize(t);
    note_frame_size();
    IRBuilder<> sb(cache_block_->getTerminator());
    sb.CreateAlignedStore(at_start, sb.CreateConstInBoundsGEP1_64(sb.getInt8Ty(), frame_, offset), Align(align));
    LoadInst* ld = b_.CreateAlignedLoad(t, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset), Align(align),
                                        v->getName() + ".cached");
    return ld;
}

// Gathers and scatters with 32-bit indices. AVX-512 has them in two forms:
// 64-bit indices (vpgatherqd: 8 lanes per instruction, the offsets in two
// zmm per 16 lanes) and 32-bit ones (vpgatherdd: 16 lanes, one zmm). The
// memory side costs the same per element on Zen 4; the 32-bit form halves
// the instructions, the registers holding offsets and the arithmetic on them.
// Indices count bytes from addr.root, so they fit whenever root's object is
// under 2 GiB: always for shared memory and per-thread arrays, and for
// global memory when root is outside the span of the allocations of 2 GiB or
// more that the runtime passes (cpusim_warp_ctx::huge_lo/hi).
bool WarpBuilder::narrow_fits(Type* elem, Align align) {
    const uint64_t size = m_.getDataLayout().getTypeStoreSize(elem);
    return size == 4 || size == 8 || size == 1 || (size == 2 && align.value() >= 2);
}

// root is outside the span of the huge objects (cpusim_warp_ctx::huge_lo/hi),
// so its object, and every offset from it, is under 2 GiB. Uniform, and
// usually loop invariant (root is a parameter): it leaves the loops.
Value* WarpBuilder::root_is_small(Value* root) {
    auto field = [&](uint64_t off) {
        Value* p = b_.CreateConstGEP1_64(b_.getInt8Ty(), ctx_arg_, off);
        LoadInst* v = b_.CreateAlignedLoad(b_.getInt64Ty(), p, Align(8));
        v->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(ctx_, {}));
        return v;
    };
    Value* r = b_.CreatePtrToInt(root, b_.getInt64Ty());
    return b_.CreateOr(b_.CreateICmpULT(r, field(CPUSIM_CTX_HUGE_LO)), b_.CreateICmpUGT(r, field(CPUSIM_CTX_HUGE_HI)));
}

// root + index = base + offs, as 32-bit byte indices (exact while the true
// distance from root fits; the wrapping arithmetic gives it mod 2^32).
Value* WarpBuilder::narrow_index(const SplitAddress& a, Value* offs) {
    auto* v32 = FixedVectorType::get(b_.getInt32Ty(), 16);
    Value* u = b_.CreateTrunc(a.uoff, b_.getInt32Ty());
    return b_.CreateAdd(b_.CreateTrunc(offs, v32), b_.CreateVectorSplat(16, u));
}

namespace {

Value* low_half(IRBuilder<>& b, Value* v, unsigned part) {
    SmallVector<int, 8> idx;
    for (int i = 0; i < 8; ++i) idx.push_back(int(part) * 8 + i);
    return b.CreateShuffleVector(v, idx);
}

}  // namespace

Value* WarpBuilder::gather16_narrow(Type* v16, const SplitAddress& a, Value* offs, Value* mask) {
    Type* et = cast<VectorType>(v16)->getElementType();
    const uint64_t size = m_.getDataLayout().getTypeStoreSize(et);
    auto* i32 = b_.getInt32Ty();
    auto* v32 = FixedVectorType::get(i32, 16);
    Value* idx = narrow_index(a, offs);
    auto from_bits = [&](Value* bits) -> Value* {
        if (et->isPointerTy()) return b_.CreateIntToPtr(bits, v16);
        return bits->getType() == v16 ? bits : b_.CreateBitCast(bits, v16);
    };
    if (size == 4) {
        Function* f = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_mask_gather_dpi_512);
        return from_bits(b_.CreateCall(f, {Constant::getNullValue(v32), a.root, idx, mask, b_.getInt32(1)}));
    }
    if (size == 8) {
        // vpgatherdq: 8 lanes per instruction, indices from a ymm.
        auto* v64 = FixedVectorType::get(b_.getInt64Ty(), 8);
        Function* f = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_mask_gather_dpq_512);
        Value* lo = b_.CreateCall(f, {Constant::getNullValue(v64), a.root, low_half(b_, idx, 0), low_half(b_, mask, 0),
                                      b_.getInt32(1)});
        Value* hi = b_.CreateCall(f, {Constant::getNullValue(v64), a.root, low_half(b_, idx, 1), low_half(b_, mask, 1),
                                      b_.getInt32(1)});
        return from_bits(concat_chunks({lo, hi}));
    }
    // A byte, or an aligned 16-bit value: the aligned 32-bit word holding it
    // (as masked_gather does), counted from root rounded down to 4 bytes.
    Value* rlow = b_.CreateAnd(b_.CreatePtrToInt(a.root, b_.getInt64Ty()), b_.getInt64(3));
    Value* word_root = b_.CreateGEP(b_.getInt8Ty(), a.root, b_.CreateNeg(rlow));
    Value* at = b_.CreateAdd(idx, b_.CreateVectorSplat(16, b_.CreateTrunc(rlow, i32)));
    Value* word = b_.CreateAnd(at, ConstantInt::get(v32, ~3u));
    Value* shift = b_.CreateShl(b_.CreateAnd(at, ConstantInt::get(v32, 3)), ConstantInt::get(v32, 3));
    Function* f = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_mask_gather_dpi_512);
    Value* w = b_.CreateCall(f, {Constant::getNullValue(v32), word_root, word, mask, b_.getInt32(1)});
    Value* v = b_.CreateTrunc(b_.CreateLShr(w, shift), FixedVectorType::get(b_.getIntNTy(unsigned(size * 8)), 16));
    return et->isIntegerTy() ? v : b_.CreateBitCast(v, v16);
}

void WarpBuilder::scatter16_narrow(Value* v, const SplitAddress& a, Value* offs, Value* mask) {
    auto* vt = cast<FixedVectorType>(v->getType());
    Type* et = vt->getElementType();
    const uint64_t size = m_.getDataLayout().getTypeStoreSize(et);
    auto* bits_ty = FixedVectorType::get(b_.getIntNTy(unsigned(size * 8)), 16);
    Value* bits = et->isPointerTy() ? b_.CreatePtrToInt(v, bits_ty) : b_.CreateBitCast(v, bits_ty);
    Value* idx = narrow_index(a, offs);
    if (size == 4) {
        Function* f = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_scatter_dpi_512);
        b_.CreateCall(f, {a.root, b_.CreateBitCast(mask, b_.getInt16Ty()), idx, bits, b_.getInt32(1)});
        return;
    }
    // size 8: two vpscatterdq, low lanes first (the last active lane wins).
    Function* f = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_scatter_dpq_512);
    for (unsigned part = 0; part < 2; ++part)
        b_.CreateCall(f, {a.root, b_.CreateBitCast(low_half(b_, mask, part), b_.getInt8Ty()), low_half(b_, idx, part),
                          low_half(b_, bits, part), b_.getInt32(1)});
}

Value* WarpBuilder::gather16(Type* v16, const SplitAddress& a, Value* offs, Align align, Value* mask) {
    auto wide = [&] { return masked_gather(v16, b_.CreateGEP(b_.getInt8Ty(), a.base, offs), align, mask); };
    if (!a.root || !narrow_fits(cast<VectorType>(v16)->getElementType(), align)) return wide();
    if (a.small) return gather16_narrow(v16, a, offs, mask);
    // Global memory: 32-bit unless root points into a huge object.
    BasicBlock* narrow = BasicBlock::Create(ctx_, "g.narrow", w_);
    BasicBlock* wide_bb = BasicBlock::Create(ctx_, "g.wide", w_);
    BasicBlock* join = BasicBlock::Create(ctx_, "g.join", w_);
    b_.CreateCondBr(root_is_small(a.root), narrow, wide_bb);
    b_.SetInsertPoint(narrow);
    Value* vn = gather16_narrow(v16, a, offs, mask);
    BasicBlock* narrow_end = b_.GetInsertBlock();
    b_.CreateBr(join);
    b_.SetInsertPoint(wide_bb);
    Value* vw = wide();
    BasicBlock* wide_end = b_.GetInsertBlock();
    b_.CreateBr(join);
    b_.SetInsertPoint(join);
    PHINode* phi = b_.CreatePHI(v16, 2);
    phi->addIncoming(vn, narrow_end);
    phi->addIncoming(vw, wide_end);
    return phi;
}

void WarpBuilder::scatter16(Value* v, const SplitAddress& a, Value* offs, Align align, Value* mask) {
    auto wide = [&] { b_.CreateMaskedScatter(v, b_.CreateGEP(b_.getInt8Ty(), a.base, offs), align, mask); };
    const uint64_t size = m_.getDataLayout().getTypeStoreSize(cast<VectorType>(v->getType())->getElementType());
    if (!a.root || (size != 4 && size != 8)) return wide();  // no scatter of bytes (LLVM stores them one by one)
    if (a.small) return scatter16_narrow(v, a, offs, mask);
    BasicBlock* narrow = BasicBlock::Create(ctx_, "s.narrow", w_);
    BasicBlock* wide_bb = BasicBlock::Create(ctx_, "s.wide", w_);
    BasicBlock* join = BasicBlock::Create(ctx_, "s.join", w_);
    b_.CreateCondBr(root_is_small(a.root), narrow, wide_bb);
    b_.SetInsertPoint(narrow);
    scatter16_narrow(v, a, offs, mask);
    b_.CreateBr(join);
    b_.SetInsertPoint(wide_bb);
    wide();
    b_.CreateBr(join);
    b_.SetInsertPoint(join);
}

// Loads through varying addresses. The access pattern is found at run time
// from the offsets alone (usually invariant, so the tests leave loops or are
// cached per warp), first for the whole access, then per 16 lanes:
//   - every 16-lane chunk contiguous and starting at the same address (e.g.
//     Bs[k][threadIdx.x] in a 16x16 block): ONE vector load, replicated;
//   - every chunk contiguous: one masked load per chunk;
//   - every chunk on a single address: one broadcast per chunk;
//   - groups of 2, 4 or 8 contiguous lanes (rows of a tile with a run-time
//     stride), known at compile time: one masked load per group;
//   - otherwise each chunk picks contiguous / broadcast / gather by itself.
// Inactive lanes are never dereferenced: masked loads only touch active lanes,
// and a chunk's broadcast only happens if it has an active lane.
Value* WarpBuilder::gather(Type* elem, Value* ptr, Align align, Value* mask, const Twine& name) {
    SplitAddress addr = split_address(ptr);
    uint64_t size = m_.getDataLayout().getTypeStoreSize(elem);
    const unsigned chunks = W / 16;
    auto* v16 = FixedVectorType::get(elem, 16);
    auto* i16 = b_.getInt16Ty();

    // Pattern known at compile time (constant thread indices): straight-line
    // code, no tests, no vector of 64-bit offsets.
    if (LinearForm form; linear_form(addr.offsets, form)) {
        std::vector<Value*> firsts;
        std::vector<char> kind = static_chunks(form, size, firsts);
        if (std::all_of(kind.begin(), kind.end(), [](char k) { return k != 0; })) {
            // Every chunk reads the same 16 elements (Bs[k][threadIdx.x] in a
            // 16-wide block): one load, replicated.
            bool one_row = chunks > 1 && std::all_of(kind.begin(), kind.end(), [](char k) { return k == 'c'; });
            for (const auto& t : form.terms)
                for (unsigned h = 1; h < chunks && one_row; ++h) one_row = t.second[h * 16] == t.second[0];
            if (one_row) {
                Value* any_mask = half(mask, 0);
                for (unsigned h = 1; h < chunks; ++h) any_mask = b_.CreateOr(any_mask, half(mask, h));
                Value* row = b_.CreateMaskedLoad(v16, byte_ptr(addr.base, firsts[0]), align, any_mask,
                                                 PoisonValue::get(v16));
                SmallVector<int, 64> rep;
                for (int i = 0; i < int(W); ++i) rep.push_back(i % 16);
                return b_.CreateShuffleVector(row, rep, name);
            }
            std::vector<Value*> parts(chunks);
            for (unsigned h = 0; h < chunks; ++h) {
                Value* m = half(mask, h);
                Value* base = byte_ptr(addr.base, firsts[h]);
                if (kind[h] == 'c') {
                    parts[h] = b_.CreateMaskedLoad(v16, base, align, m, PoisonValue::get(v16));
                } else {
                    // As in the run-time broadcast path: a chunk without active
                    // lanes reads the (always readable) context instead.
                    Value* any = b_.CreateICmpNE(b_.CreateBitCast(m, i16), b_.getInt16(0));
                    Value* p = b_.CreateSelect(any, base, b_.CreatePointerCast(ctx_arg_, base->getType()));
                    parts[h] = b_.CreateVectorSplat(16, b_.CreateAlignedLoad(elem, p, Align(1)));
                }
            }
            return concat_chunks(parts);
        }
        // Groups of contiguous lanes (rows with a run-time stride): one
        // masked load per group, instead of a gather.
        if (unsigned g = static_groups(form, size, firsts)) {
            auto* vg = FixedVectorType::get(elem, g);
            std::vector<Value*> parts;
            for (unsigned s = 0; s < W; s += g) {
                SmallVector<int, 8> idx;
                for (unsigned i = 0; i < g; ++i) idx.push_back(int(s + i));
                parts.push_back(b_.CreateMaskedLoad(vg, byte_ptr(addr.base, firsts[s / g]), align,
                                                    b_.CreateShuffleVector(mask, idx), PoisonValue::get(vg)));
            }
            return concat_chunks(parts);
        }
    }

    std::vector<Value*> offs(chunks), firsts(chunks), bases(chunks), masks(chunks), cbits(chunks), sbits(chunks);
    Value* all_contig = b_.getTrue();
    Value* all_bcast = b_.getTrue();
    Value* same_first = b_.getTrue();
    for (unsigned h = 0; h < chunks; ++h) {
        offs[h] = half(addr.offsets, h);
        masks[h] = half(mask, h);
        Value* first = b_.CreateExtractElement(offs[h], uint64_t(0));
        Value* delta = b_.CreateSub(offs[h], b_.CreateVectorSplat(16, first));
        cbits[h] = b_.CreateFreeze(b_.CreateBitCast(b_.CreateICmpEQ(delta, lane_steps(b_, size)), i16));
        sbits[h] = b_.CreateFreeze(b_.CreateBitCast(b_.CreateICmpEQ(delta, Constant::getNullValue(delta->getType())), i16));
        all_contig = b_.CreateAnd(all_contig, b_.CreateICmpEQ(cbits[h], b_.getInt16(-1)));
        all_bcast = b_.CreateAnd(all_bcast, b_.CreateICmpEQ(sbits[h], b_.getInt16(-1)));
        if (h > 0) same_first = b_.CreateAnd(same_first, b_.CreateICmpEQ(first, firsts[0]));
        firsts[h] = first;
    }
    all_contig = cached(all_contig);
    all_bcast = cached(all_bcast);
    same_first = cached(same_first);
    for (unsigned h = 0; h < chunks; ++h) {
        bases[h] = byte_ptr(addr.base, cached(firsts[h]));
        cbits[h] = cached(cbits[h]);
        sbits[h] = cached(sbits[h]);
    }

    auto concat = [&](const std::vector<Value*>& parts) { return concat_chunks(parts); };
    BasicBlock* done = BasicBlock::Create(ctx_, "ld.done", w_);
    std::vector<std::pair<Value*, BasicBlock*>> results;

    BasicBlock* contig = BasicBlock::Create(ctx_, "ld.contig", w_);
    BasicBlock* not_contig = BasicBlock::Create(ctx_, "ld.notcontig", w_);
    b_.CreateCondBr(all_contig, contig, not_contig);

    b_.SetInsertPoint(contig);
    if (chunks > 1) {
        BasicBlock* shared_row = BasicBlock::Create(ctx_, "ld.onerow", w_);
        BasicBlock* rows = BasicBlock::Create(ctx_, "ld.rows", w_);
        b_.CreateCondBr(same_first, shared_row, rows);
        b_.SetInsertPoint(shared_row);
        Value* any_mask = masks[0];
        for (unsigned h = 1; h < chunks; ++h) any_mask = b_.CreateOr(any_mask, masks[h]);
        Value* row = b_.CreateMaskedLoad(v16, bases[0], align, any_mask, PoisonValue::get(v16));
        SmallVector<int, 64> rep;
        for (int i = 0; i < int(W); ++i) rep.push_back(i % 16);
        results.push_back({b_.CreateShuffleVector(row, rep), b_.GetInsertBlock()});
        b_.CreateBr(done);
        b_.SetInsertPoint(rows);
    }
    {
        std::vector<Value*> parts(chunks);
        for (unsigned h = 0; h < chunks; ++h)
            parts[h] = b_.CreateMaskedLoad(v16, bases[h], align, masks[h], PoisonValue::get(v16));
        results.push_back({concat(parts), b_.GetInsertBlock()});
        b_.CreateBr(done);
    }

    b_.SetInsertPoint(not_contig);
    BasicBlock* bcast = BasicBlock::Create(ctx_, "ld.bcast", w_);
    BasicBlock* mixed = BasicBlock::Create(ctx_, "ld.mixed", w_);
    b_.CreateCondBr(all_bcast, bcast, mixed);

    b_.SetInsertPoint(bcast);
    {
        auto* v1 = FixedVectorType::get(elem, 1);
        std::vector<Value*> parts(chunks);
        (void)v1;
        for (unsigned h = 0; h < chunks; ++h) {
            // A chunk without active lanes may hold any address: read from the
            // (always readable) context instead and ignore the value. No branch.
            Value* any = b_.CreateICmpNE(b_.CreateBitCast(masks[h], i16), b_.getInt16(0));
            Value* p = b_.CreateSelect(any, bases[h], b_.CreatePointerCast(ctx_arg_, bases[h]->getType()));
            parts[h] = b_.CreateVectorSplat(16, b_.CreateAlignedLoad(elem, p, Align(1)));
        }
        results.push_back({concat(parts), b_.GetInsertBlock()});
        b_.CreateBr(done);
    }

    b_.SetInsertPoint(mixed);
    {
        // Rare path: rebuild the offsets here rather than keep them alive.
        {
            SplitAddress again = split_address(ptr);
            for (unsigned h = 0; h < chunks; ++h) offs[h] = half(again.offsets, h);
            addr = again;
        }
        std::vector<Value*> parts(chunks);
        for (unsigned h = 0; h < chunks; ++h) {
            // Wider than a warp: one gather per chunk. Testing each chunk (3
            // paths each) multiplies the code by the chunks: the generic
            // fdtd3d module took 24 s to compile instead of 7. Only the
            // module without a known block shape gets here.
            if (chunks > 2) {
                parts[h] = gather16(v16, addr, offs[h], align, masks[h]);
                continue;
            }
            Value* m_bits = b_.CreateBitCast(masks[h], i16);
            auto all = [&](Value* bits) { return b_.CreateICmpEQ(b_.CreateAnd(m_bits, b_.CreateNot(bits)), b_.getInt16(0)); };
            BasicBlock* bb_cont = BasicBlock::Create(ctx_, "ld.c.contig", w_);
            BasicBlock* bb_test = BasicBlock::Create(ctx_, "ld.c.test", w_);
            BasicBlock* bb_same = BasicBlock::Create(ctx_, "ld.c.bcast", w_);
            BasicBlock* bb_gather = BasicBlock::Create(ctx_, "ld.c.gather", w_);
            BasicBlock* bb_done = BasicBlock::Create(ctx_, "ld.c.done", w_);
            b_.CreateCondBr(all(cbits[h]), bb_cont, bb_test);
            b_.SetInsertPoint(bb_cont);
            Value* v_cont = b_.CreateMaskedLoad(v16, bases[h], align, masks[h], PoisonValue::get(v16));
            b_.CreateBr(bb_done);
            b_.SetInsertPoint(bb_test);
            b_.CreateCondBr(all(sbits[h]), bb_same, bb_gather);
            b_.SetInsertPoint(bb_same);
            Value* v_same = b_.CreateVectorSplat(16, b_.CreateAlignedLoad(elem, bases[h], align));
            b_.CreateBr(bb_done);
            b_.SetInsertPoint(bb_gather);
            Value* v_gather = gather16(v16, addr, offs[h], align, masks[h]);
            BasicBlock* gather_end = b_.GetInsertBlock();  // gather16 may branch
            b_.CreateBr(bb_done);
            b_.SetInsertPoint(bb_done);
            PHINode* phi = b_.CreatePHI(v16, 3);
            phi->addIncoming(v_cont, bb_cont);
            phi->addIncoming(v_same, bb_same);
            phi->addIncoming(v_gather, gather_end);
            parts[h] = phi;
        }
        results.push_back({concat(parts), b_.GetInsertBlock()});
        b_.CreateBr(done);
    }

    b_.SetInsertPoint(done);
    PHINode* out = b_.CreatePHI(vec_type(elem), unsigned(results.size()), name);
    for (auto& [v, bb] : results) out->addIncoming(v, bb);
    return out;
}

// One contiguous chunk of 16 lanes. A non-temporal store (__stcs) has no
// masked form on x86 (vmovntps), so it is used when all 16 lanes are active
// and the chunk is 64-byte aligned, and the masked store otherwise.
void WarpBuilder::store_chunk(Value* v, Value* base, Align align, Value* mask, bool nontemporal) {
    if (!nontemporal) {
        b_.CreateMaskedStore(v, base, align, mask);
        return;
    }
    Value* full = b_.CreateICmpEQ(b_.CreateBitCast(mask, b_.getInt16Ty()), b_.getInt16(-1));
    Value* aligned = b_.CreateICmpEQ(b_.CreateAnd(b_.CreatePtrToInt(base, b_.getInt64Ty()), b_.getInt64(63)),
                                     b_.getInt64(0));
    BasicBlock* bb_nt = BasicBlock::Create(ctx_, "st.nt", w_);
    BasicBlock* bb_masked = BasicBlock::Create(ctx_, "st.masked", w_);
    BasicBlock* bb_done = BasicBlock::Create(ctx_, "st.nt.done", w_);
    b_.CreateCondBr(b_.CreateAnd(full, aligned), bb_nt, bb_masked);
    b_.SetInsertPoint(bb_nt);
    StoreInst* st = b_.CreateAlignedStore(v, base, Align(64));
    st->setMetadata(LLVMContext::MD_nontemporal, MDNode::get(ctx_, ConstantAsMetadata::get(b_.getInt32(1))));
    b_.CreateBr(bb_done);
    b_.SetInsertPoint(bb_masked);
    b_.CreateMaskedStore(v, base, align, mask);
    b_.CreateBr(bb_done);
    b_.SetInsertPoint(bb_done);
}

// Same idea for stores: when every chunk is contiguous, one masked store per
// chunk; otherwise each chunk tests by itself (and scatters if needed, which
// also keeps "the last lane wins" when lanes share an address).
void WarpBuilder::scatter(Value* vals, Value* ptr, Align align, Value* mask, bool nontemporal) {
    SplitAddress addr = split_address(ptr);
    Type* elem = cast<VectorType>(vals->getType())->getElementType();
    uint64_t size = m_.getDataLayout().getTypeStoreSize(elem);
    const unsigned chunks = W / 16;
    auto* i16 = b_.getInt16Ty();
    if (LinearForm form; linear_form(addr.offsets, form)) {
        std::vector<Value*> firsts;
        std::vector<char> kind = static_chunks(form, size, firsts);
        if (std::all_of(kind.begin(), kind.end(), [](char k) { return k == 'c'; })) {
            for (unsigned h = 0; h < chunks; ++h)
                store_chunk(half(vals, h), byte_ptr(addr.base, firsts[h]), align, half(mask, h), nontemporal);
            return;
        }
        if (unsigned g = static_groups(form, size, firsts)) {  // rows with a run-time stride
            for (unsigned s = 0; s < W; s += g) {
                SmallVector<int, 8> idx;
                for (unsigned i = 0; i < g; ++i) idx.push_back(int(s + i));
                b_.CreateMaskedStore(b_.CreateShuffleVector(vals, idx), byte_ptr(addr.base, firsts[s / g]), align,
                                     b_.CreateShuffleVector(mask, idx));
            }
            return;
        }
    }
    std::vector<Value*> offs(chunks), bases(chunks), cbits(chunks);
    Value* all_contig = b_.getTrue();
    for (unsigned h = 0; h < chunks; ++h) {
        offs[h] = half(addr.offsets, h);
        Value* first = b_.CreateExtractElement(offs[h], uint64_t(0));
        Value* delta = b_.CreateSub(offs[h], b_.CreateVectorSplat(16, first));
        cbits[h] = b_.CreateFreeze(b_.CreateBitCast(b_.CreateICmpEQ(delta, lane_steps(b_, size)), i16));
        all_contig = b_.CreateAnd(all_contig, b_.CreateICmpEQ(cbits[h], b_.getInt16(-1)));
        bases[h] = byte_ptr(addr.base, cached(first));
    }
    all_contig = cached(all_contig);
    for (unsigned h = 0; h < chunks; ++h) cbits[h] = cached(cbits[h]);

    BasicBlock* contig = BasicBlock::Create(ctx_, "st.contig", w_);
    BasicBlock* mixed = BasicBlock::Create(ctx_, "st.mixed", w_);
    BasicBlock* done = BasicBlock::Create(ctx_, "st.done", w_);
    b_.CreateCondBr(all_contig, contig, mixed);

    b_.SetInsertPoint(contig);
    for (unsigned h = 0; h < chunks; ++h) store_chunk(half(vals, h), bases[h], align, half(mask, h), nontemporal);
    b_.CreateBr(done);

    b_.SetInsertPoint(mixed);
    {
        SplitAddress again = split_address(ptr);
        for (unsigned h = 0; h < chunks; ++h) offs[h] = half(again.offsets, h);
        addr = again;
    }
    for (unsigned h = 0; h < chunks; ++h) {
        Value* m = half(mask, h);
        Value* v = half(vals, h);
        if (chunks > 2) {  // as for loads: one scatter per chunk (lanes in order, the last one wins)
            scatter16(v, addr, offs[h], align, m);
            continue;
        }
        Value* m_bits = b_.CreateBitCast(m, i16);
        Value* ok = b_.CreateICmpEQ(b_.CreateAnd(m_bits, b_.CreateNot(cbits[h])), b_.getInt16(0));
        BasicBlock* bb_cont = BasicBlock::Create(ctx_, "st.c.contig", w_);
        BasicBlock* bb_scatter = BasicBlock::Create(ctx_, "st.c.scatter", w_);
        BasicBlock* bb_done = BasicBlock::Create(ctx_, "st.c.done", w_);
        b_.CreateCondBr(ok, bb_cont, bb_scatter);
        b_.SetInsertPoint(bb_cont);
        b_.CreateMaskedStore(v, bases[h], align, m);
        b_.CreateBr(bb_done);
        b_.SetInsertPoint(bb_scatter);
        scatter16(v, addr, offs[h], align, m);
        b_.CreateBr(bb_done);
        b_.SetInsertPoint(bb_done);
    }
    b_.CreateBr(done);
    b_.SetInsertPoint(done);
}

// c ? *p : other, per lane; only lanes in mask & c touch memory.
Value* WarpBuilder::emit_cond_load(CallInst* call, Value* mask) {
    Value* ptr = call->getArgOperand(0);
    Value* c = call->getArgOperand(1);
    Value* other = call->getArgOperand(2);
    Type* t = call->getType();
    Value* m = uni_->is_varying(c) ? b_.CreateAnd(mask, vector(c)) : b_.CreateSelect(scalar(c), mask, no_lanes());
    if (!uni_->is_varying(call)) {
        // Everything uniform: one guarded scalar load.
        auto* v1 = FixedVectorType::get(t, 1);
        Value* go = b_.CreateAnd(scalar(c), any_lane(mask));
        Value* v = b_.CreateMaskedLoad(v1, scalar(ptr), Align(1), b_.CreateVectorSplat(1, go),
                                       b_.CreateVectorSplat(1, scalar(other)));
        return b_.CreateExtractElement(v, uint64_t(0));
    }
    const DataLayout& dl = m_.getDataLayout();
    Value* v = gather(t, ptr, dl.getABITypeAlign(t), m, call->getName());
    return b_.CreateSelect(m, v, vector(other));
}


}  // namespace simt_internal
}  // namespace cpusim
