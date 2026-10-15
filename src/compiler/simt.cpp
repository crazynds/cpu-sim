// Turns a per-thread kernel into warp code.
//
//  1. Cross-block SSA values and phis are demoted to allocas. Each lane then
//     communicates between blocks only through those slots, and a slot write
//     under a partial mask is a blend, so per-lane program order is kept no
//     matter in which order the blocks run.
//  2. The CFG is linearized: blocks run in a topological order of each loop
//     level, each with a lane mask (OR of its incoming edge masks). A block
//     whose mask is empty is skipped. A loop repeats while any lane took a
//     back edge; lanes that leave add themselves to the exit block's mask.
//  3. Uniform values stay scalar; varying values become <32 x T>. Memory
//     accesses through varying pointers become masked gathers/scatters.
#include "compiler/simt_internal.h"

namespace cpusim {
namespace simt_internal {

std::string describe(const Value* v) {
    std::string s;
    raw_string_ostream os(s);
    v->print(os);
    return s;
}

// A call to a device function that was not inlined (recursive), or through
// a pointer.
bool is_device_call(const CallInst* call) {
    const Function* f = call->getCalledFunction();
    if (!f) return !call->isInlineAsm();
    return !f->isIntrinsic() && !f->isDeclaration() && !f->isVarArg();
}

// A function of the process the kernel runs in (malloc, free, the device
// printf calls): called once per active lane with that lane's arguments.
bool is_external_call(const CallInst* call) {
    const Function* f = call->getCalledFunction();
    if (!f || !f->isDeclaration() || f->isIntrinsic() || f->isVarArg()) return false;
    StringRef n = f->getName();
    if (n.starts_with("__cpusim_")) return n.starts_with("__cpusim_printf_");
    for (Type* t : f->getFunctionType()->params())
        if (!(t->isIntegerTy() || t->isFloatingPointTy() || t->isPointerTy())) return false;
    return true;
}

std::vector<PendingFunction>& pending_functions() {
    static std::vector<PendingFunction> list;
    return list;
}

bool is_barrier(const Instruction& inst) {
    auto* call = dyn_cast<IntrinsicInst>(&inst);
    if (!call) return false;
    switch (call->getIntrinsicID()) {
        case Intrinsic::nvvm_barrier0:
        case Intrinsic::nvvm_bar_sync:
        case Intrinsic::nvvm_barrier:
        case Intrinsic::nvvm_barrier_n:
            return true;
        default:
            return false;
    }
}

// Math functions with an AVX-512 version in glibc's libmvec: the name of a
// libm function or of the LLVM intrinsic Clang made of it ("expf", "exp";
// llvm.exp.f32 -> "expf"), the libmvec base name, and the number of arguments.
// Empty: not one of them.
std::string libmvec_name(const CallInst& call, unsigned* arity) {
    static const std::pair<const char*, unsigned> kFuncs[] = {
        {"exp", 1},   {"exp2", 1},  {"exp10", 1}, {"expm1", 1}, {"log", 1},   {"log2", 1},   {"log10", 1},
        {"log1p", 1}, {"sin", 1},   {"cos", 1},   {"tan", 1},   {"asin", 1},  {"acos", 1},   {"atan", 1},
        {"sinh", 1},  {"cosh", 1},  {"tanh", 1},  {"asinh", 1}, {"acosh", 1}, {"atanh", 1},  {"erf", 1},
        {"erfc", 1},  {"cbrt", 1},  {"pow", 2},   {"atan2", 2}, {"hypot", 2}};
    Function* f = call.getCalledFunction();
    Type* t = call.getType();
    if (!f || !(t->isFloatTy() || t->isDoubleTy())) return "";
    // libmvec's double versions err by up to ~4 ulp, CUDA's by 1 or 2: a
    // double function stays a call of the C library's, per lane. Unless it is
    // a float function computed in double (the device header's way: float
    // arguments widened, the result rounded back to float), where those ulp
    // of a double vanish in the rounding.
    if (t->isDoubleTy()) {
        const bool from_float = all_of(call.args(), [](const Use& u) {
            auto* e = dyn_cast<FPExtInst>(u.get());
            return e && e->getSrcTy()->isFloatTy();
        });
        const bool to_float = call.hasOneUse() && isa<FPTruncInst>(*call.user_begin()) &&
                              call.user_back()->getType()->isFloatTy();
        if (!from_float || !to_float) return "";
    }
    std::string base;
    if (f->isIntrinsic()) {
        // llvm.exp.f32 -> exp
        StringRef n = f->getName();
        if (!n.consume_front("llvm.")) return "";
        base = n.split('.').first.str();
    } else {
        StringRef n = f->getName();
        if (t->isFloatTy() && n.ends_with("f")) n = n.drop_back();
        base = n.str();
    }
    for (auto [name, args] : kFuncs)
        if (base == name && call.arg_size() == args) {
            *arity = args;
            return base + (t->isFloatTy() ? "f" : "");
        }
    return "";
}

// ---------------------------------------------------------------- preparation

// Candidate software prefetches for global memory streams in loops: an access
// in an innermost loop whose per-thread address moves by a fixed step each
// iteration (constant or uniform, e.g. n * 4 down a column) gets a prefetch of
// the address a few iterations ahead. They are marked "cpusim.auto" and kept
// at vectorization only where the lanes' addresses are spread out (one line
// per lane: hardware prefetchers, which learn per instruction, do not follow
// 16 streams of one gather); for lanes next to each other they are dropped,
// the hardware already handles those. matrix-rotate: 5.45 -> 4.55 ms.
bool WarpBuilder::insert_prefetches() {
    bool inserted = false;
    TargetLibraryInfoImpl tlii(Triple(m_.getTargetTriple()));
    TargetLibraryInfo tli(tlii);
    AssumptionCache ac(f_);
    ScalarEvolution se(f_, tli, ac, *dt_, *li_);
    SCEVExpander expander(se, m_.getDataLayout(), "pf");
    Loop* cur = nullptr;
    auto* i64 = Type::getInt64Ty(ctx_);
    // Bytes the address moves from one lane to the next (threadIdx.x + 1),
    // through sums, constant factors, extensions and recurrences (the
    // start's). Unknown values that are uniform do not move; anything else
    // varying makes it unknown (null).
    std::function<const SCEV*(const SCEV*)> lane_step = [&](const SCEV* x) -> const SCEV* {
        if (auto* u = dyn_cast<SCEVUnknown>(x)) {
            if (special_reg(u->getValue()) == SpecialReg::TidX) return se.getOne(i64);
            return uni_->is_varying(u->getValue()) ? nullptr : se.getZero(i64);
        }
        if (isa<SCEVConstant>(x)) return se.getZero(i64);
        if (auto* r = dyn_cast<SCEVAddRecExpr>(x)) {
            if (!r->isAffine()) return nullptr;
            const SCEV* st = lane_step(r->getStepRecurrence(se));
            if (!st || !st->isZero()) return nullptr;  // the step itself varies across lanes
            return lane_step(r->getStart());
        }
        if (auto* a = dyn_cast<SCEVAddExpr>(x)) {
            const SCEV* sum = se.getZero(i64);
            for (const SCEV* op : a->operands()) {
                const SCEV* st = lane_step(op);
                if (!st) return nullptr;
                sum = se.getAddExpr(sum, st);
            }
            return sum;
        }
        if (auto* m = dyn_cast<SCEVMulExpr>(x)) {
            // Product rule with at most one factor that moves.
            const SCEV* moving = nullptr;
            const SCEV* rest = se.getOne(i64);
            for (const SCEV* op : m->operands()) {
                const SCEV* st = lane_step(op);
                if (!st) return nullptr;
                if (st->isZero()) {
                    rest = se.getMulExpr(rest, se.getTruncateOrSignExtend(op, i64));
                } else {
                    if (moving) return nullptr;
                    moving = st;
                }
            }
            return moving ? se.getMulExpr(rest, moving) : se.getZero(i64);
        }
        if (auto* c = dyn_cast<SCEVCastExpr>(x)) {
            const SCEV* st = lane_step(c->getOperand(0));
            return st ? se.getTruncateOrSignExtend(st, i64) : nullptr;
        }
        return nullptr;
    };
    // Bytes the address moves per iteration of `l`, as an i64 SCEV: through
    // sums, constant factors and extensions (CUDA indexes are int: the
    // recurrence sits inside a sext). Ignoring wrap-around is fine for a
    // prefetch, a hint that cannot fault. Null if unknown.
    std::function<const SCEV*(const SCEV*)> byte_step = [&](const SCEV* x) -> const SCEV* {
        if (se.isLoopInvariant(x, cur)) return se.getZero(i64);
        if (auto* r = dyn_cast<SCEVAddRecExpr>(x)) {
            if (r->getLoop() != cur || !r->isAffine()) return nullptr;
            return se.getTruncateOrSignExtend(r->getStepRecurrence(se), i64);
        }
        if (auto* a = dyn_cast<SCEVAddExpr>(x)) {
            const SCEV* sum = se.getZero(i64);
            for (const SCEV* op : a->operands()) {
                const SCEV* st = byte_step(op);
                if (!st) return nullptr;
                sum = se.getAddExpr(sum, st);
            }
            return sum;
        }
        if (auto* m = dyn_cast<SCEVMulExpr>(x); m && m->getNumOperands() == 2 && isa<SCEVConstant>(m->getOperand(0))) {
            const SCEV* st = byte_step(m->getOperand(1));
            return st ? se.getMulExpr(se.getTruncateOrSignExtend(m->getOperand(0), i64), st) : nullptr;
        }
        if (auto* c = dyn_cast<SCEVCastExpr>(x)) {
            const SCEV* st = byte_step(c->getOperand(0));
            return st ? se.getTruncateOrSignExtend(st, i64) : nullptr;
        }
        return nullptr;
    };
    Function* pf = nullptr;
    for (Loop* l : li_->getLoopsInPreorder()) {
        if (!l->isInnermost()) continue;
        cur = l;
        uint64_t insts = 0;
        for (BasicBlock* bb : l->blocks()) insts += bb->size();
        // About a thousand instructions ahead, 8 to 64 iterations.
        const int64_t ahead = std::clamp<int64_t>(int64_t(1000 / std::max<uint64_t>(insts, 1)), 8, 64);
        SmallPtrSet<const SCEV*, 8> done;
        for (BasicBlock* bb : l->blocks()) {
            if (li_->getLoopFor(bb) != l) continue;
            for (Instruction& inst : make_early_inc_range(*bb)) {
                Value* ptr = nullptr;
                bool write = false;
                if (auto* ld = dyn_cast<LoadInst>(&inst)) ptr = ld->getPointerOperand();
                else if (auto* st = dyn_cast<StoreInst>(&inst)) ptr = st->getPointerOperand(), write = true;
                if (!ptr || ptr->getType()->getPointerAddressSpace() != 0) continue;
                // Not shared memory or per-thread arrays: those stay in L1.
                const Value* obj = getUnderlyingObject(ptr, 12);
                if (isa<AllocaInst>(obj)) continue;
                if (auto* c = dyn_cast<CallInst>(obj); c && c->getCalledFunction() &&
                                                        c->getCalledFunction()->getName() == kSharedBaseFn)
                    continue;
                const SCEV* step = byte_step(se.getSCEV(ptr));
                if (!step || step->isZero() || !se.isLoopInvariant(step, l)) continue;
                const SCEV* offset = se.getMulExpr(se.getConstant(step->getType(), ahead), step);
                // Only where neighbouring lanes are a cache line or more
                // apart (or a run-time size apart, as down a column): there
                // the hardware prefetchers do not follow. Adjacent or equal
                // addresses across lanes (coalesced, broadcast): left to them.
                const SCEV* across = lane_step(se.getSCEV(ptr));
                if (!across || across->isZero()) continue;
                if (auto* c = dyn_cast<SCEVConstant>(across); c && c->getAPInt().abs().ult(64)) continue;
                if (!done.insert(se.getAddExpr(se.getSCEV(ptr), offset)).second) continue;
                if (!expander.isSafeToExpandAt(offset, &inst)) continue;
                IRBuilder<> eb(&inst);
                Value* at = eb.CreateGEP(eb.getInt8Ty(), ptr,
                                         expander.expandCodeFor(offset, eb.getInt64Ty(), inst.getIterator()), "pf.addr");
                if (!pf) pf = Intrinsic::getDeclaration(&m_, Intrinsic::prefetch, {ptr->getType()});
                IRBuilder<> b(&inst);
                CallInst* call = b.CreateCall(pf, {at, b.getInt32(write ? 1 : 0), b.getInt32(3), b.getInt32(1)});
                call->setMetadata("cpusim.auto", MDNode::get(ctx_, {}));
                inserted = true;
            }
        }
    }
    return inserted;
}

bool WarpBuilder::check_supported() {
    for (BasicBlock& bb : f_) {
        Instruction* term = bb.getTerminator();
        if (!isa<BranchInst>(term) && !isa<ReturnInst>(term) && !isa<UnreachableInst>(term))
            fail("unsupported terminator", term);
        for (Instruction& inst : bb) {
            if (fn_ && (is_barrier(inst) || special_reg(&inst) == SpecialReg::SharedBase))
                fail("__syncthreads and __shared__ in a function that is not inlined (recursive or called "
                     "through a pointer) are not supported", &inst);
            if (fn_)
                for (Value* op : inst.operands()) {
                    SmallPtrSet<const Value*, 8> seen;
                    std::function<bool(const Value*)> shared = [&](const Value* v) {
                        if (!seen.insert(v).second) return false;
                        if (auto* g = dyn_cast<GlobalVariable>(v)) return g->getAddressSpace() == 3;
                        if (auto* ce = dyn_cast<ConstantExpr>(v)) return any_of(ce->operands(), shared);
                        return false;
                    };
                    if (shared(op))
                        fail("__shared__ in a function that is not inlined (recursive or called through a "
                             "pointer) is not supported", &inst);
                }
            if (auto* call = dyn_cast<CallInst>(&inst)) {
                Function* callee = call->getCalledFunction();
                if (!callee) {
                    if (call->getFunctionType()->isVarArg()) fail("variadic calls through a pointer", call);
                    continue;  // through a pointer: emit_call
                } else if (callee->getName() == kSharedBaseFn || is_cond_load(call)) continue;
                else if (callee->getName() == kShflFn || callee->getName() == kBallotFn || is_mma(call) || is_mmav(call) ||
                         callee->getName() == kYieldFn || callee->getName() == kRefillFirstFn ||
                         callee->getName() == kRefillNextFn)
                    continue;
                else if (unsigned n; !libmvec_name(*call, &n).empty()) continue;
                else if (!callee->isIntrinsic() && !callee->isDeclaration() && !callee->isVarArg()) continue;  // emit_call
                else if (is_external_call(call)) continue;  // emit_atomic: once per lane
                else if (!callee->isIntrinsic()) fail("call to an external device function", call);
                else if (callee->getIntrinsicID() == Intrinsic::nvvm_barrier0_popc ||
                         callee->getIntrinsicID() == Intrinsic::nvvm_barrier0_and ||
                         callee->getIntrinsicID() == Intrinsic::nvvm_barrier0_or)
                    fail("__syncthreads_count/_and/_or are not supported yet");
            } else if (auto* a = dyn_cast<AllocaInst>(&inst)) {
                if (escapes(a)) {
                    // Each lane needs a real address for it: lane-major
                    // layout will not do. Lives in this call's stack.
                    // (In the warp's frame when the variant suspends at barriers.)
                    if (!a->isStaticAlloca()) fail("per-thread arrays of variable size are not supported", a);
                    escaping_.insert(a);
                } else if (!check_private_array(a)) {
                    return false;
                }
            } else if (auto* cx = dyn_cast<AtomicCmpXchgInst>(&inst)) {
                // Its fields are read where it is (a { T, i1 } is not a value
                // a lane can carry): a field read in another block is moved
                // next to it.
                SmallVector<ExtractValueInst*, 4> far;
                for (User* u : cx->users()) {
                    auto* ev = dyn_cast<ExtractValueInst>(u);
                    if (!ev) fail("compare-and-swap result used other than by its fields", u);
                    else if (ev->getParent() != cx->getParent()) far.push_back(ev);
                }
                for (ExtractValueInst* ev : far) ev->moveAfter(cx);
            }
        }
    }
    for (Loop* l : li_->getLoopsInPreorder())
        if (!l->getHeader()) fail("malformed loop");
    return error_.empty();
}

// Puts every barrier alone in a block (barrier + branch), so that whatever is
// live across it crosses a block boundary and gets a slot.
bool WarpBuilder::split_barriers() {
    std::vector<Instruction*> barriers;
    for (BasicBlock& bb : f_)
        for (Instruction& inst : bb)
            if (is_barrier(inst) || is_yield(inst)) barriers.push_back(&inst);
    for (Instruction* bar : barriers) {
        BasicBlock* bb = bar->getParent();
        if (&*bb->getFirstNonPHIIt() != bar) bb = bb->splitBasicBlock(bar, "barrier");
        bb->splitBasicBlock(std::next(bar->getIterator()), "after.barrier");
        barrier_blocks_.insert(bb);
        if (is_yield(*bar)) yield_blocks_.insert(bb);
    }
    return !barriers.empty();
}

// ---------------------------------------------------------------- if-conversion

// A side of a divergent branch that can run with all lanes: small, no stores,
// calls or barriers, nothing that can trap except loads (which become guarded
// loads, masked per lane).
bool WarpBuilder::convertible(BasicBlock* bb) {
    unsigned n = 0;
    for (Instruction& inst : *bb) {
        if (inst.isTerminator()) break;
        if (++n > 24 || isa<PHINode>(inst)) return false;
        if (auto* ld = dyn_cast<LoadInst>(&inst)) {
            if (!ld->isSimple()) return false;
            continue;
        }
        if (auto* call = dyn_cast<CallInst>(&inst)) {
            if (special_reg(call) != SpecialReg::None || is_cond_load(call)) continue;
            Function* f = call->getCalledFunction();
            if (f && f->isIntrinsic() && isTriviallyVectorizable(f->getIntrinsicID())) continue;
            return false;
        }
        if (isa<BinaryOperator>(inst)) {
            switch (inst.getOpcode()) {
                case Instruction::UDiv: case Instruction::SDiv: case Instruction::URem: case Instruction::SRem:
                    return false;
                default:
                    continue;
            }
        }
        if (isa<CastInst>(inst) || isa<CmpInst>(inst) || isa<SelectInst>(inst) || isa<GetElementPtrInst>(inst) ||
            isa<UnaryOperator>(inst) || isa<FreezeInst>(inst))
            continue;
        return false;
    }
    return true;
}

// Moves `from`'s instructions to the end of `into`, turning loads into loads
// guarded by `guard`.
void WarpBuilder::hoist_guarded(BasicBlock* from, BasicBlock* into, Value* guard) {
    Instruction* at = into->getTerminator();
    std::vector<Instruction*> insts;
    for (Instruction& inst : *from)
        if (!inst.isTerminator()) insts.push_back(&inst);
    for (Instruction* inst : insts) {
        auto* ld = dyn_cast<LoadInst>(inst);
        if (!ld) {
            inst->moveBefore(at);
            continue;
        }
        Type* t = ld->getType();
        std::string tn;
        raw_string_ostream os(tn);
        t->print(os);
        FunctionCallee fn = m_.getOrInsertFunction(
            kCondLoadPrefix + tn, FunctionType::get(t, {ld->getPointerOperandType(), Type::getInt1Ty(ctx_), t}, false));
        if (auto* f = dyn_cast<Function>(fn.getCallee())) {
            f->setDoesNotThrow();
            f->setOnlyReadsMemory();
            f->setWillReturn();
        }
        auto* call = CallInst::Create(fn, {ld->getPointerOperand(), guard, Constant::getNullValue(t)}, ld->getName(),
                                      at->getIterator());
        ld->replaceAllUsesWith(call);
        ld->eraseFromParent();
    }
}

// Divergent if-then and if-then-else whose sides are small become straight
// code: both sides run with the parent's lanes, loads are guarded per lane,
// and the merge phis become selects. No extra blocks, masks or slots.
bool WarpBuilder::if_convert() {
    bool any = false, changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock& b : f_) {
            auto* br = dyn_cast<BranchInst>(b.getTerminator());
            if (!br || !br->isConditional() || !uni_->is_varying(br->getCondition())) continue;
            if (barrier_blocks_.contains(&b)) continue;
            BasicBlock* s0 = br->getSuccessor(0);
            BasicBlock* s1 = br->getSuccessor(1);
            Loop* l = li_->getLoopFor(&b);
            auto side = [&](BasicBlock* s, BasicBlock* join) {
                return s->getSinglePredecessor() == &b && s->getSingleSuccessor() == join &&
                       li_->getLoopFor(s) == l && !barrier_blocks_.contains(s) && convertible(s);
            };
            auto joins_exactly = [&](BasicBlock* j, BasicBlock* p0, BasicBlock* p1) {
                if (li_->getLoopFor(j) != l || (l && l->getHeader() == j)) return false;
                SmallPtrSet<BasicBlock*, 4> preds(pred_begin(j), pred_end(j));
                return preds.size() == 2 && preds.contains(p0) && preds.contains(p1) &&
                       pred_size(j) == 2;
            };
            Value* c = br->getCondition();
            BasicBlock *t = nullptr, *e = nullptr, *j = nullptr;
            bool t_is_true = true;
            if (s0->getSingleSuccessor() && side(s0, s0->getSingleSuccessor()) && s0->getSingleSuccessor() == s1 &&
                joins_exactly(s1, &b, s0)) {
                t = s0, j = s1;  // if (c) { t } j
            } else if (s1->getSingleSuccessor() && side(s1, s1->getSingleSuccessor()) &&
                       s1->getSingleSuccessor() == s0 && joins_exactly(s0, &b, s1)) {
                t = s1, j = s0, t_is_true = false;  // if (!c) { t } j
            } else if (s0->getSingleSuccessor() && s0->getSingleSuccessor() == s1->getSingleSuccessor() &&
                       side(s0, s0->getSingleSuccessor()) && side(s1, s0->getSingleSuccessor()) &&
                       joins_exactly(s0->getSingleSuccessor(), s0, s1)) {
                t = s0, e = s1, j = s0->getSingleSuccessor();  // if (c) { t } else { e } j
            } else {
                continue;
            }

            IRBuilder<> ib(b.getTerminator());
            Value* not_c = ib.CreateNot(c, c->getName() + ".not");
            uni_->set_varying(not_c);
            hoist_guarded(t, &b, t_is_true ? c : not_c);
            if (e) hoist_guarded(e, &b, not_c);
            BasicBlock* other = e ? e : &b;  // where the "condition false" value comes from
            ib.SetInsertPoint(b.getTerminator());
            std::vector<PHINode*> phis;
            for (PHINode& phi : j->phis()) phis.push_back(&phi);
            for (PHINode* phi : phis) {
                Value* vt = phi->getIncomingValueForBlock(t);
                Value* vo = phi->getIncomingValueForBlock(other);
                Value* sel = t_is_true ? ib.CreateSelect(c, vt, vo, phi->getName()) : ib.CreateSelect(c, vo, vt, phi->getName());
                uni_->set_varying(sel);
                phi->replaceAllUsesWith(sel);
                phi->eraseFromParent();
            }
            b.getTerminator()->eraseFromParent();
            BranchInst::Create(j, &b);
            t->eraseFromParent();
            if (e) e->eraseFromParent();
            // Values moved here keep their classification; recompute later.
            changed = any = true;
            break;  // the block list changed
        }
    }
    return any;
}

// True if no path that a single lane can take from `ld` to `use` (without
// executing `ld` again) stores to `slot`.
bool WarpBuilder::reload_is_safe(LoadInst* ld, AllocaInst* slot, Instruction* use) {
    BasicBlock* from = ld->getParent();
    BasicBlock* to = use->getParent();
    auto stores_in = [&](BasicBlock* bb, BasicBlock::iterator begin, BasicBlock::iterator end) {
        for (auto it = begin; it != end; ++it)
            if (auto* st = dyn_cast<StoreInst>(&*it); st && st->getPointerOperand() == slot) return true;
        return false;
    };
    if (stores_in(from, std::next(ld->getIterator()), from->end())) return false;

    // Blocks reachable from `from` without going through it again...
    DenseSet<BasicBlock*> reach;
    SmallVector<BasicBlock*, 16> work(succ_begin(from), succ_end(from));
    while (!work.empty()) {
        BasicBlock* bb = work.pop_back_val();
        if (bb == from || !reach.insert(bb).second) continue;
        for (BasicBlock* s : successors(bb)) work.push_back(s);
    }
    if (!reach.contains(to)) return false;
    // ...that can still reach the use, inside that set.
    DenseSet<BasicBlock*> on_path;
    work.push_back(to);
    while (!work.empty()) {
        BasicBlock* bb = work.pop_back_val();
        if (!reach.contains(bb) || !on_path.insert(bb).second) continue;
        for (BasicBlock* p : predecessors(bb)) work.push_back(p);
    }
    for (BasicBlock* bb : on_path) {
        // In the use's own block only what comes before the use matters,
        // unless the block can be re-entered on the way (then all of it).
        bool reentered = false;
        for (BasicBlock* s : successors(bb)) reentered |= s == to;
        BasicBlock::iterator end = (bb == to && !reentered) ? use->getIterator() : bb->end();
        if (stores_in(bb, bb->begin(), end)) return false;
    }
    return true;
}

// Pure, cheap, and computable anywhere: depends only on arguments, special
// registers (threadIdx, blockIdx, ...), the shared memory base and constants.
bool WarpBuilder::is_invariant(Value* v, unsigned depth) {
    if (isa<Constant>(v) || isa<Argument>(v)) return true;
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst || depth > 12) return false;
    if (auto it = invariant_.find(inst); it != invariant_.end()) return it->second;
    bool ok = false;
    if (auto* call = dyn_cast<CallInst>(inst)) {
        ok = special_reg(call) != SpecialReg::None;
    } else if (isa<BinaryOperator>(inst) || isa<CastInst>(inst) || isa<CmpInst>(inst) || isa<SelectInst>(inst) ||
               isa<GetElementPtrInst>(inst) || isa<UnaryOperator>(inst)) {
        switch (inst->getOpcode()) {
            case Instruction::UDiv: case Instruction::SDiv: case Instruction::URem: case Instruction::SRem:
                ok = false;  // might trap where the original would not run
                break;
            default:
                ok = all_of(inst->operands(), [&](Use& u) { return is_invariant(u.get(), depth + 1); });
        }
    }
    invariant_[inst] = ok;
    return ok;
}

// Like is_invariant, but uniform values may be leaves too: a cheap varying
// value such as `t + threadIdx.y` (t a loop counter) is recomputed in the block
// that uses it from the scalar t, instead of keeping a whole vector in a slot.
bool WarpBuilder::remat_ok(Value* v, unsigned depth) {
    if (is_invariant(v, 0)) return true;
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst || depth > 8) return false;
    if (!uni_->is_varying(inst)) return depth > 0;
    // A GEP too: an address computed before a branch and used inside it is
    // recomputed there from its base and indices, so the access still sees
    // base + offsets (a slot would hold W opaque 64-bit pointers).
    if (!(isa<BinaryOperator>(inst) || isa<CastInst>(inst) || isa<GetElementPtrInst>(inst))) return false;
    switch (inst->getOpcode()) {
        case Instruction::UDiv: case Instruction::SDiv: case Instruction::URem: case Instruction::SRem:
        case Instruction::FDiv: case Instruction::FRem:
            return false;
        default:
            break;
    }
    if (inst->getType()->isFloatingPointTy()) return false;
    return all_of(inst->operands(), [&](Use& u) { return remat_ok(u.get(), depth + 1); });
}

void WarpBuilder::rematerialize_invariants() {
    // Clones go right before each block's original first instruction, in the
    // order they are created: operands first, so every clone follows its inputs.
    DenseMap<BasicBlock*, Instruction*> anchor;
    for (BasicBlock& bb : f_) anchor[&bb] = &*bb.getFirstInsertionPt();
    DenseMap<std::pair<BasicBlock*, Value*>, Value*> placed;
    std::function<Value*(Value*, BasicBlock*)> clone_into = [&](Value* v, BasicBlock* bb) -> Value* {
        auto* inst = dyn_cast<Instruction>(v);
        if (!inst || inst->getParent() == bb) return v;
        // A uniform leaf of a remat_ok value stays where it is (SSA: it
        // dominates the original, so it dominates the use too).
        if (!is_invariant(inst, 0) && !uni_->is_varying(inst)) return v;
        if (auto it = placed.find({bb, v}); it != placed.end()) return it->second;
        Instruction* c = inst->clone();
        for (Use& op : c->operands())
            if (!isa<Function>(op.get())) op.set(clone_into(op.get(), bb));
        c->setName(inst->getName() + ".remat");
        c->insertBefore(anchor[bb]->getIterator());
        if (uni_->is_varying(inst)) uni_->set_varying(c);
        invariant_[c] = is_invariant(inst, 0);
        placed[{bb, v}] = c;
        return c;
    };

    std::vector<Instruction*> candidates;
    for (BasicBlock& bb : f_)
        for (Instruction& inst : bb)
            if (is_invariant(&inst, 0) || (uni_->is_varying(&inst) && remat_ok(&inst, 0))) candidates.push_back(&inst);
    for (Instruction* inst : candidates) {
        SmallVector<Use*, 8> far;
        // Not into a phi: it reads the value at the end of a predecessor, not
        // in its own block (a slot carries it, as any value crossing blocks).
        for (Use& u : inst->uses())
            if (cast<Instruction>(u.getUser())->getParent() != inst->getParent() && !isa<PHINode>(u.getUser()))
                far.push_back(&u);
        for (Use* u : far) u->set(clone_into(inst, cast<Instruction>(u->getUser())->getParent()));
    }
}

// Pointers into a per-thread array are only usable as the GEP chain from the
// array (split_address turns it into lane-major addresses), never as a value
// kept in a slot: a chain used in another block is cloned there. Its indices
// stay ordinary values (slots if they cross blocks).
void WarpBuilder::rematerialize_private_geps() {
    auto root = [&](Value* v) {
        while (auto* g = dyn_cast<GetElementPtrInst>(v)) v = g->getPointerOperand();
        auto* a = dyn_cast<AllocaInst>(v);
        return a && private_elem_.count(a);
    };
    std::function<Value*(Value*, Instruction*)> clone_before = [&](Value* v, Instruction* at) -> Value* {
        auto* g = dyn_cast<GetElementPtrInst>(v);
        if (!g || g->getParent() == at->getParent()) return v;
        Instruction* c = g->clone();
        c->setOperand(0, clone_before(g->getPointerOperand(), at));
        c->insertBefore(at->getIterator());
        if (uni_->is_varying(g)) uni_->set_varying(c);
        return c;
    };
    std::vector<GetElementPtrInst*> geps;
    for (BasicBlock& bb : f_)
        for (Instruction& inst : bb)
            if (auto* g = dyn_cast<GetElementPtrInst>(&inst); g && root(g)) geps.push_back(g);
    for (GetElementPtrInst* g : geps) {
        SmallVector<Use*, 8> far;
        for (Use& u : g->uses())
            if (cast<Instruction>(u.getUser())->getParent() != g->getParent()) far.push_back(&u);
        for (Use* u : far) {
            auto* user = cast<Instruction>(u->getUser());
            u->set(clone_before(g, user));
        }
    }
}

void WarpBuilder::demote() {
    // The slot's uniformity is the value's; loads created for it inherit it.
    auto record = [&](AllocaInst* slot, bool varying) {
        demoted_.insert(slot);
        demoted_order_.push_back(slot);
        if (!varying) return;
        uni_->set_varying(slot);
        for (User* u : slot->users())
            if (auto* ld = dyn_cast<LoadInst>(u)) uni_->set_varying(ld);
    };

    // Single-input phis (LCSSA) are copies; dropping them saves a slot, as long
    // as they classify like their input (a uniform value leaving a loop with
    // divergent exits is varying outside, so that phi must stay).
    std::vector<PHINode*> copies;
    for (BasicBlock& bb : f_)
        for (PHINode& phi : bb.phis())
            if (phi.getNumIncomingValues() == 1 && uni_->is_varying(&phi) == uni_->is_varying(phi.getIncomingValue(0)))
                copies.push_back(&phi);
    for (PHINode* phi : copies) {
        phi->replaceAllUsesWith(phi->getIncomingValue(0));
        phi->eraseFromParent();
    }

    // Phis first: each becomes a slot written at the end of its predecessors
    // and one reload in its block...
    std::vector<PHINode*> phis;
    for (BasicBlock& bb : f_)
        for (PHINode& phi : bb.phis()) phis.push_back(&phi);
    std::vector<std::pair<AllocaInst*, bool>> phi_slots;
    for (PHINode* phi : phis) {
        bool varying = uni_->is_varying(phi);
        AllocaInst* slot = DemotePHIToStack(phi);
        if (!slot) continue;  // no uses: erased
        record(slot, varying);
        phi_slots.push_back({slot, varying});
    }

    // A varying slot's reload used in another block can simply be loaded again
    // there, if no per-lane path from the reload to the use writes the slot
    // (masked writes only change the lanes that take them). That avoids a copy
    // slot and its blend. Not for uniform slots: a scalar write changes the
    // value every lane sees.
    for (auto [slot, varying] : phi_slots) {
        if (!varying) continue;
        SmallVector<LoadInst*, 4> reloads;
        for (User* u : slot->users())
            if (auto* ld = dyn_cast<LoadInst>(u)) reloads.push_back(ld);
        for (LoadInst* ld : reloads) {
            SmallVector<Use*, 8> far;
            for (Use& u : ld->uses())
                if (cast<Instruction>(u.getUser())->getParent() != ld->getParent()) far.push_back(&u);
            for (Use* u : far) {
                auto* user = cast<Instruction>(u->getUser());
                if (!reload_is_safe(ld, slot, user)) continue;
                auto* again = new LoadInst(ld->getType(), slot, ld->getName() + ".again", user->getIterator());
                uni_->set_varying(again);
                u->set(again);
            }
        }
    }

    // Values that are pure functions of the thread/block indices, the kernel
    // parameters and constants are recomputed where they are used instead of
    // getting a slot: nothing to keep alive across blocks or barriers.
    rematerialize_invariants();
    rematerialize_private_geps();

    // ...then every value (reloads included) used outside its block gets a slot
    // written right after its definition. A copy, not a re-load at the use: the
    // original slot may be overwritten in between.
    std::vector<Instruction*> crossing;
    for (BasicBlock& bb : f_) {
        for (Instruction& inst : bb) {
            if (isa<AllocaInst>(inst)) continue;
            for (User* u : inst.users()) {
                if (cast<Instruction>(u)->getParent() != &bb) {
                    crossing.push_back(&inst);
                    break;
                }
            }
        }
    }
    for (Instruction* inst : crossing) {
        bool varying = uni_->is_varying(inst);
        record(DemoteRegToStack(*inst), varying);
    }
}

// ---------------------------------------------------------------- values

Value* WarpBuilder::scalar(Value* v) {
    if (isa<Constant>(v)) return v;
    if (auto it = scalar_map_.find(v); it != scalar_map_.end()) return it->second;
    fail("internal: no scalar value for", v);
    return PoisonValue::get(v->getType());
}

Value* WarpBuilder::vector(Value* v) {
    if (auto it = vector_map_.find(v); it != vector_map_.end()) return it->second;
    Value* s = scalar(v);
    if (auto* c = dyn_cast<Constant>(s)) return ConstantVector::getSplat(ElementCount::getFixed(W), c);
    return b_.CreateVectorSplat(W, s);
}


bool WarpBuilder::run(std::string* error) {
    if (!block_wide_) split_barriers();
    dt_ = std::make_unique<DominatorTree>(f_);
    pdt_ = std::make_unique<PostDominatorTree>(f_);
    li_ = std::make_unique<LoopInfo>(*dt_);
    if (!check_supported()) return *error = error_, false;

    uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_, fn_ != nullptr);
    if (!profile_ && insert_prefetches()) uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_, fn_ != nullptr);
    // The instrumented variant keeps the original branches so its per-thread
    // counts stay exact (if-conversion runs both sides' arithmetic).
    if (!profile_ && if_convert()) {
        dt_ = std::make_unique<DominatorTree>(f_);
        pdt_ = std::make_unique<PostDominatorTree>(f_);
        li_ = std::make_unique<LoopInfo>(*dt_);
        uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_, fn_ != nullptr);
    }
    demote();

    emit_prologue();
    if (error_.empty()) emit_level(nullptr);
    if (!error_.empty()) {
        *error = error_;
        if (w_ && !fn_) w_->eraseFromParent();
        return false;
    }
    sfence_if_streaming();
    if (!fn_) b_.CreateRet(b_.getInt32(CPUSIM_EXIT));
    else if (ret_slot_) b_.CreateRet(b_.CreateLoad(ret_slot_->getAllocatedType(), ret_slot_));
    else b_.CreateRetVoid();
    if (dispatch_) emit_dispatch();
    if (fn_) return true;
    if (profile_) k_.profiled = w_;
    else if (W <= k_.warp_size) k_.entry = w_;
    return true;
}


}  // namespace simt_internal

using namespace simt_internal;

bool build_warp_entry(Module& m, KernelInfo& kernel, Function& scalar, bool profile, unsigned lanes, bool suspending,
                      std::string* error, bool refill) {
    // Barriers are no-ops in a wide variant: the runtime only uses those when
    // the whole block fits in one call. (Kernels without barriers are free to
    // run any number of warps per call.)
    WarpBuilder builder(m, kernel, scalar, profile, lanes, lanes > kernel.warp_size && !suspending);
    if (refill) builder.set_refill();
    if (!builder.run(error)) return false;
    // The functions it calls without inlining, and the ones those call.
    auto& pending = pending_functions();
    while (!pending.empty()) {
        PendingFunction p = pending.back();
        pending.pop_back();
        ValueToValueMapTy vmap;
        Function* copy = CloneFunction(p.scalar, vmap);
        WarpBuilder fb(m, kernel, *copy, false, p.lanes, true, p.decl);
        bool ok = fb.run(error);
        copy->eraseFromParent();
        if (!ok) {
            *error = "in " + p.scalar->getName().str() + ": " + *error;
            pending.clear();
            return false;
        }
    }
    return true;
}

}  // namespace cpusim

namespace cpusim {

namespace {

bool used_in(const Value* v, const Function* f, SmallPtrSetImpl<const Value*>& seen) {
    if (!seen.insert(v).second) return false;
    for (const User* u : v->users()) {
        if (auto* i = dyn_cast<Instruction>(u)) {
            if (i->getFunction() == f) return true;
        } else if (isa<Constant>(u) && used_in(u, f, seen)) {
            return true;
        }
    }
    return false;
}

}  // namespace

void lower_shared_memory(Module& m, KernelInfo& k) {
    Function& f = *k.scalar;
    const DataLayout& dl = m.getDataLayout();
    constexpr unsigned kSharedAS = 3;

    std::vector<GlobalVariable*> statics, dynamics;
    for (GlobalVariable& g : m.globals()) {
        if (g.getAddressSpace() != kSharedAS) continue;
        SmallPtrSet<const Value*, 16> seen;
        if (!used_in(&g, &f, seen)) continue;
        // extern __shared__ T x[]; has no size: it is the launch's dynamic shared memory.
        (dl.getTypeAllocSize(g.getValueType()) == 0 ? dynamics : statics).push_back(&g);
    }
    if (statics.empty() && dynamics.empty()) return;

    std::vector<Constant*> consts(statics.begin(), statics.end());
    consts.insert(consts.end(), dynamics.begin(), dynamics.end());
    convertUsersOfConstantsToInstructions(consts, &f);

    LLVMContext& ctx = m.getContext();
    auto* ptr = PointerType::get(ctx, 0);
    FunctionCallee base_fn = m.getOrInsertFunction(kSharedBaseFn, FunctionType::get(ptr, false));
    if (auto* fn = dyn_cast<Function>(base_fn.getCallee())) {
        fn->setDoesNotThrow();
        fn->setDoesNotAccessMemory();
        fn->setWillReturn();
    }
    IRBuilder<> b(&*f.getEntryBlock().getFirstInsertionPt());
    Value* base = b.CreateCall(base_fn, {}, "shared.base");

    uint64_t offset = 0;
    auto place = [&](GlobalVariable* g, uint64_t align) {
        offset = alignTo(offset, align);
        Value* p = b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), base, offset, g->getName());
        Value* as = b.CreateAddrSpaceCast(p, g->getType());
        SmallVector<Use*, 8> uses;
        for (Use& u : g->uses())
            if (auto* i = dyn_cast<Instruction>(u.getUser()); i && i->getFunction() == &f) uses.push_back(&u);
        for (Use* u : uses) u->set(as);
        offset += dl.getTypeAllocSize(g->getValueType());
    };
    for (GlobalVariable* g : statics) place(g, std::max<uint64_t>(g->getAlign().valueOrOne().value(), 16));
    k.shared_bytes = alignTo(offset, 16);
    // Every dynamic shared array starts at the same place, right after the static ones.
    for (GlobalVariable* g : dynamics) {
        offset = k.shared_bytes;
        place(g, 16);
    }
}

}  // namespace cpusim