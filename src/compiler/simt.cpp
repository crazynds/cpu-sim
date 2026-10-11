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
#include "compiler/simt.h"

#include <array>
#include <map>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/Utils/ScalarEvolutionExpander.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Local.h>

#include "compiler/uniformity.h"
#include "device/kernel_abi.h"

using namespace llvm;

namespace cpusim {

namespace {


std::string describe(const Value* v) {
    std::string s;
    raw_string_ostream os(s);
    v->print(os);
    return s;
}

// A node of a loop level: a block that belongs directly to it, or a whole subloop.
struct Node {
    BasicBlock* block = nullptr;
    Loop* loop = nullptr;
    bool operator==(const Node& o) const { return block == o.block && loop == o.loop; }
};

class WarpBuilder {
public:
    // `lanes`: threads per call (32: one warp). `block_wide`: the caller
    // guarantees a whole block fits in one call, so barriers are no-ops.
    // `cooperative`: half of a block per call, the other half runs on the SMT
    // sibling; barriers are an inline 2-party barrier on epochs (no suspending).
    WarpBuilder(Module& m, KernelInfo& k, Function& scalar, bool profile, unsigned lanes, bool block_wide,
                bool cooperative = false)
        : W(lanes), block_wide_(block_wide || cooperative), coop_(cooperative), m_(m), k_(k), f_(scalar),
          profile_(profile), ctx_(m.getContext()), b_(ctx_) {}

    bool run(std::string* error);

private:
    // ---- preparation and analysis
    bool check_supported();
    bool insert_prefetches();
    void demote();

    // ---- emission
    void emit_prologue();
    void emit_level(Loop* level);
    void emit_node(const Node& n);
    void emit_block(BasicBlock* bb);
    void emit_loop(Loop* l);
    bool emit_instruction(Instruction& inst, Value* mask, bool divergent);
    void add_edge(BasicBlock* from, BasicBlock* to, Value* edge_mask);
    std::vector<Node> topo_order(Loop* level);
    Node node_of(BasicBlock* bb, Loop* level);

    // ---- values
    Type* vec_type(Type* t) { return FixedVectorType::get(t, W); }
    Value* scalar(Value* v);
    Value* vector(Value* v);
    Value* any_lane(Value* mask) {
        return b_.CreateICmpNE(b_.CreateBitCast(mask, b_.getIntNTy(W)), b_.getIntN(W, 0), "any");
    }
    Value* no_lanes() { return Constant::getNullValue(mask_type_); }
    struct SplitAddress {
        Value* base;     // uniform pointer (scalar)
        Value* offsets;  // varying byte offsets, <32 x i64>
        uint64_t scale = 1;  // a per-thread array: a thread's byte offsets count W times (lane-major layout)
    };
    // Per-thread arrays (`float v[8]` that did not become registers), stored
    // lane-major: element byte offset o of lane l at base + o * W + l * elem,
    // so one element of every lane is contiguous. `elem`: the size of every
    // access (arrays of one scalar type).
    DenseMap<const AllocaInst*, uint64_t> private_elem_;
    DenseMap<const AllocaInst*, uint64_t> private_frame_offset_;  // in the frame (variants that suspend)
    DenseMap<const AllocaInst*, Value*> private_stack_;           // or in the stack
    bool check_private_array(AllocaInst* a);
    Value* private_base(const AllocaInst* a);
    SplitAddress split_address(Value* ptr);
    // Varying byte offsets as a sum of terms coef * lanes[i] (coef: uniform
    // scalar, nullptr for 1; lanes: constants). Known when the thread index of
    // every lane is a compile-time constant (block shape fixed by the JIT).
    struct LinearForm {
        std::vector<std::pair<Value*, std::vector<int64_t>>> terms;
    };
    // ext: the value is seen through a zero (1) or sign (2) extension to i64.
    bool linear_form(Value* v, LinearForm& out, int ext = 0);
    // Per 16-lane chunk, from a linear form: 'c' contiguous, 'b' one address,
    // 0 unknown at compile time; and the chunk's first offset (scalar).
    std::vector<char> static_chunks(const LinearForm& f, uint64_t size, std::vector<Value*>& firsts);
    std::pair<Value*, Value*> split_index(Value* idx, bool is_signed);
    // `ptr` is the per-thread pointer; its offsets are recomputed in the rare
    // fallback paths so the common paths only need the cached pattern data.
    Value* gather(Type* elem, Value* ptr, Align align, Value* mask, const Twine& name);
    void scatter(Value* vals, Value* ptr, Align align, Value* mask);
    Value* half(Value* v, unsigned h);
    Value* concat_chunks(const std::vector<Value*>& parts);
    void fail(const std::string& what, const Value* v = nullptr) {
        if (error_.empty()) error_ = what + (v ? ": " + describe(v) : "");
    }

    void count(Instruction& inst, uint64_t* counts);
    void emit_counters(const uint64_t* counts, Value* mask);

    const unsigned W;
    const bool block_wide_;
    const bool coop_;
    Module& m_;
    KernelInfo& k_;
    Function& f_;
    bool profile_;
    Value* counters_ = nullptr;
    LLVMContext& ctx_;
    IRBuilder<> b_;
    std::string error_;

    std::unique_ptr<DominatorTree> dt_;
    std::unique_ptr<PostDominatorTree> pdt_;
    std::unique_ptr<LoopInfo> li_;
    std::unique_ptr<Uniformity> uni_;

    Function* w_ = nullptr;
    Type* mask_type_ = nullptr;
    Value* ctx_arg_ = nullptr;
    BasicBlock* alloca_block_ = nullptr;
    DenseMap<const Value*, Value*> scalar_map_, vector_map_;
    DenseMap<const BasicBlock*, AllocaInst*> block_mask_;
    DenseMap<const Loop*, AllocaInst*> next_mask_;
    DenseMap<const AllocaInst*, AllocaInst*> slots_;  // demoted scalar slot -> slot in the warp function
    DenseSet<const AllocaInst*> demoted_;
    std::vector<const AllocaInst*> demoted_order_;  // creation order, for a deterministic frame layout
    Value* special_[17] = {};
    Value* spec_lane_mask_ = nullptr;
    void emit_specialized_tid();
    void emit_coop_barrier(Value* mask);
    Value* emit_libmvec(CallInst& call, const std::string& name, unsigned arity);
    Value* emit_shfl(Value* v, Value* src);
    Value* emit_uniform_division(BinaryOperator* bin);
    Value* emit_ballot(Value* pred, Value* mask);
    bool emit_mma(CallInst* call, Value* mask);
    Value* emit_mmav(CallInst* call);
    std::map<std::vector<Value*>, std::array<Value*, 8>> mmav_done_;  // one inline mma per operand set

    // Barriers (DESIGN.md 4.4): each __syncthreads is a block of its own; the
    // warp saves every slot and mask to its frame there and returns.
    bool split_barriers();
    bool reload_is_safe(LoadInst* ld, AllocaInst* slot, Instruction* use);
    bool if_convert();
    bool convertible(BasicBlock* bb);
    void hoist_guarded(BasicBlock* from, BasicBlock* into, Value* guard);
    Value* emit_cond_load(CallInst* call, Value* mask);

    // Warp-invariant cache (kernels with barriers): values computable from the
    // context alone are computed once when the warp starts, kept in its frame,
    // and loaded where needed, instead of being recomputed after every resume.
    Value* cached(Value* v);
    // Records this variant's frame size (the plain and instrumented 32-lane
    // entries share one: the larger).
    void note_frame_size() {
        uint64_t bytes = alignTo(cache_end_, 64);
        uint64_t& slot = W == CPUSIM_WARP_SIZE ? k_.frame_bytes : k_.frame_bytes_wide[W == 64 ? 0 : 1];
        slot = std::max(slot, bytes);
    }
    Value* clone_invariant(Value* v, DenseMap<Value*, Value*>& memo);
    BasicBlock* entry_ = nullptr;
    BasicBlock* cache_block_ = nullptr;
    uint64_t cache_end_ = 0;
    bool is_uniform_loop(Loop* l);
    void emit_uniform_loop(Loop* l);
    bool is_invariant(Value* v, unsigned depth);
    void rematerialize_invariants();
    void rematerialize_private_geps();
    bool remat_ok(Value* v, unsigned depth);
    DenseMap<const Instruction*, bool> invariant_;
    void save_state(int resume_id);
    // A barrier in block `bb`: go through if the runtime allows it, else save
    // state and return. Leaves the builder at the code after the barrier.
    void emit_barrier(BasicBlock* bb);
    AllocaInst* pass_used_ = nullptr;  // this call already went through one barrier
    void emit_dispatch();
    DenseSet<const BasicBlock*> barrier_blocks_;
    std::vector<std::pair<AllocaInst*, uint64_t>> state_;  // slot -> frame offset
    BasicBlock* dispatch_ = nullptr;
    Value* frame_ = nullptr;
    Value* state_ptr_ = nullptr;
    std::vector<BasicBlock*> resumes_;
    // Per resume point: the frame stores emitted at its barrier.
    struct Save {
        AllocaInst* slot;
        uint64_t offset;
        LoadInst* load;
        StoreInst* store;
    };
    std::vector<std::vector<Save>> saves_;
    std::vector<BasicBlock*> save_blocks_;  // block that saves and returns at barrier k
    std::vector<BitVector> live_at_resume_;
    void compute_resume_liveness();
    // Per-thread liveness of the demoted slots after each barrier (in barrier order).
    void compute_thread_liveness();
    std::vector<const BasicBlock*> barrier_order_;
    DenseMap<const BasicBlock*, DenseSet<const AllocaInst*>> live_after_barrier_;
};

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
            if (auto* call = dyn_cast<CallInst>(&inst)) {
                Function* callee = call->getCalledFunction();
                if (!callee) fail("indirect calls are not supported yet", call);
                else if (callee->getName() == kSharedBaseFn || is_cond_load(call)) continue;
                else if (callee->getName() == kShflFn || callee->getName() == kBallotFn || is_mma(call) || is_mmav(call))
                    continue;
                else if (unsigned n; !libmvec_name(*call, &n).empty()) continue;
                else if (!callee->isIntrinsic()) fail("call to a function that could not be inlined", call);
                else if (callee->getIntrinsicID() == Intrinsic::nvvm_barrier0_popc ||
                         callee->getIntrinsicID() == Intrinsic::nvvm_barrier0_and ||
                         callee->getIntrinsicID() == Intrinsic::nvvm_barrier0_or)
                    fail("__syncthreads_count/_and/_or are not supported yet");
            } else if (auto* a = dyn_cast<AllocaInst>(&inst)) {
                if (!check_private_array(a)) return false;
            } else if (inst.isAtomic() || isa<FenceInst>(inst)) {
                fail("atomics are not supported yet", &inst);
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
            if (is_barrier(inst)) barriers.push_back(&inst);
    for (Instruction* bar : barriers) {
        BasicBlock* bb = bar->getParent();
        if (&*bb->getFirstNonPHIIt() != bar) bb = bb->splitBasicBlock(bar, "barrier");
        bb->splitBasicBlock(std::next(bar->getIterator()), "after.barrier");
        barrier_blocks_.insert(bb);
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
    if (!(isa<BinaryOperator>(inst) || isa<CastInst>(inst))) return false;
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
        for (Use& u : inst->uses())
            if (cast<Instruction>(u.getUser())->getParent() != inst->getParent()) far.push_back(&u);
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
WarpBuilder::SplitAddress WarpBuilder::split_address(Value* ptr) {
    auto* i64 = b_.getInt64Ty();
    auto* v64 = FixedVectorType::get(i64, W);
    if (auto* a = dyn_cast<AllocaInst>(ptr); a && private_elem_.count(a)) {
        // Lane l's copy of the array starts at base + l * elem.
        SmallVector<Constant*, 64> lanes;
        for (unsigned l = 0; l < W; ++l) lanes.push_back(ConstantInt::get(i64, l * private_elem_[a]));
        return {private_base(a), ConstantVector::get(lanes), W};
    }
    if (!uni_->is_varying(ptr)) return {scalar(ptr), Constant::getNullValue(v64)};
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
        return {b_.CreateGEP(b_.getInt8Ty(), inner.base, uniform), varying, inner.scale};
    }
    // Opaque varying pointer: offsets from address zero.
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
    const uint64_t size = elem(a->getAllocatedType());
    if (!size) return fail("per-thread array of mixed element sizes is not supported", a), false;
    SmallVector<Value*, 8> work{a};
    while (!work.empty()) {
        Value* v = work.pop_back_val();
        for (User* u : v->users()) {
            if (auto* gep = dyn_cast<GetElementPtrInst>(u); gep && gep->getPointerOperand() == v) {
                work.push_back(gep);
            } else if (auto* ld = dyn_cast<LoadInst>(u)) {
                if (dl.getTypeStoreSize(ld->getType()) != size)
                    return fail("per-thread array accessed with another element size", ld), false;
            } else if (auto* st = dyn_cast<StoreInst>(u); st && st->getPointerOperand() == v) {
                if (dl.getTypeStoreSize(st->getValueOperand()->getType()) != size)
                    return fail("per-thread array accessed with another element size", st), false;
            } else if (auto* ii = dyn_cast<IntrinsicInst>(u); ii && ii->isLifetimeStartOrEnd()) {
            } else if (is_mma(u) || is_mmav(u)) {  // a tensor core fragment, handed over whole (emit_mma)
            } else {
                return fail("per-thread array used other than by element loads and stores", u), false;
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

// Loads through varying addresses. The access pattern is found at run time
// from the offsets alone (usually invariant, so the tests leave loops or are
// cached per warp), first for the whole access, then per 16 lanes:
//   - every 16-lane chunk contiguous and starting at the same address (e.g.
//     Bs[k][threadIdx.x] in a 16x16 block): ONE vector load, replicated;
//   - every chunk contiguous: one masked load per chunk;
//   - every chunk on a single address: one broadcast per chunk;
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
                Value* row = b_.CreateMaskedLoad(v16, b_.CreateGEP(b_.getInt8Ty(), addr.base, firsts[0]), align, any_mask,
                                                 PoisonValue::get(v16));
                SmallVector<int, 64> rep;
                for (int i = 0; i < int(W); ++i) rep.push_back(i % 16);
                return b_.CreateShuffleVector(row, rep, name);
            }
            std::vector<Value*> parts(chunks);
            for (unsigned h = 0; h < chunks; ++h) {
                Value* m = half(mask, h);
                Value* base = b_.CreateGEP(b_.getInt8Ty(), addr.base, firsts[h]);
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
        bases[h] = b_.CreateGEP(b_.getInt8Ty(), addr.base, cached(firsts[h]));
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
            addr.base = again.base;
        }
        std::vector<Value*> parts(chunks);
        for (unsigned h = 0; h < chunks; ++h) {
            // Wider than a warp: one gather per chunk. Testing each chunk (3
            // paths each) multiplies the code by the chunks: the generic
            // fdtd3d module took 24 s to compile instead of 7. Only the
            // module without a known block shape gets here.
            if (chunks > 2) {
                parts[h] = b_.CreateMaskedGather(v16, b_.CreateGEP(b_.getInt8Ty(), addr.base, offs[h]), align, masks[h],
                                                 PoisonValue::get(v16));
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
            Value* ptrs = b_.CreateGEP(b_.getInt8Ty(), addr.base, offs[h]);
            Value* v_gather = b_.CreateMaskedGather(v16, ptrs, align, masks[h], PoisonValue::get(v16));
            b_.CreateBr(bb_done);
            b_.SetInsertPoint(bb_done);
            PHINode* phi = b_.CreatePHI(v16, 3);
            phi->addIncoming(v_cont, bb_cont);
            phi->addIncoming(v_same, bb_same);
            phi->addIncoming(v_gather, bb_gather);
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

// Same idea for stores: when every chunk is contiguous, one masked store per
// chunk; otherwise each chunk tests by itself (and scatters if needed, which
// also keeps "the last lane wins" when lanes share an address).
void WarpBuilder::scatter(Value* vals, Value* ptr, Align align, Value* mask) {
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
                b_.CreateMaskedStore(half(vals, h), b_.CreateGEP(b_.getInt8Ty(), addr.base, firsts[h]), align,
                                     half(mask, h));
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
        bases[h] = b_.CreateGEP(b_.getInt8Ty(), addr.base, cached(first));
    }
    all_contig = cached(all_contig);
    for (unsigned h = 0; h < chunks; ++h) cbits[h] = cached(cbits[h]);

    BasicBlock* contig = BasicBlock::Create(ctx_, "st.contig", w_);
    BasicBlock* mixed = BasicBlock::Create(ctx_, "st.mixed", w_);
    BasicBlock* done = BasicBlock::Create(ctx_, "st.done", w_);
    b_.CreateCondBr(all_contig, contig, mixed);

    b_.SetInsertPoint(contig);
    for (unsigned h = 0; h < chunks; ++h) b_.CreateMaskedStore(half(vals, h), bases[h], align, half(mask, h));
    b_.CreateBr(done);

    b_.SetInsertPoint(mixed);
    {
        SplitAddress again = split_address(ptr);
        for (unsigned h = 0; h < chunks; ++h) offs[h] = half(again.offsets, h);
        addr.base = again.base;
    }
    for (unsigned h = 0; h < chunks; ++h) {
        Value* m = half(mask, h);
        Value* v = half(vals, h);
        if (chunks > 2) {  // as for loads: one scatter per chunk (lanes in order, the last one wins)
            b_.CreateMaskedScatter(v, b_.CreateGEP(b_.getInt8Ty(), addr.base, offs[h]), align, m);
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
        b_.CreateMaskedScatter(v, b_.CreateGEP(b_.getInt8Ty(), addr.base, offs[h]), align, m);
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
    // base = warp_index * 32, a multiple of W here (groups are W lanes).
    Value* warp = b_.CreateAlignedLoad(i32, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_WARP_INDEX),
                                       Align(4), "warp");
    cast<Instruction>(warp)->setMetadata(
        LLVMContext::MD_range, MDNode::get(ctx_, {ConstantAsMetadata::get(b_.getInt32(0)),
                                                  ConstantAsMetadata::get(b_.getInt32((threads + 31) / 32))}));
    Value* base = b_.CreateMul(warp, b_.getInt32(CPUSIM_WARP_SIZE), "base", true, true);
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
    Function* perm = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_vpermi2var_d_512);
    auto slice = [&](Value* x, unsigned first) {
        SmallVector<int, 16> idx;
        for (unsigned i = 0; i < 16; ++i) idx.push_back(int(first + i));
        return b_.CreateShuffleVector(x, idx);
    };
    Value* five_bits = ConstantVector::getSplat(ElementCount::getFixed(16), b_.getInt32(31));
    std::vector<Value*> parts;
    for (unsigned w = 0; w < W; w += CPUSIM_WARP_SIZE) {
        Value* lo = slice(v, w);
        Value* hi = slice(v, w + 16);
        for (unsigned h = 0; h < 2; ++h)
            parts.push_back(b_.CreateCall(perm, {lo, b_.CreateAnd(slice(src, w + 16 * h), five_bits), hi}));
    }
    return concat_chunks(parts);
}

// __cpusim_ballot: every lane of a warp gets the 32-bit mask of its warp's
// lanes that are active here with pred != 0.
Value* WarpBuilder::emit_ballot(Value* pred, Value* mask) {
    Value* bits = b_.CreateAnd(b_.CreateICmpNE(pred, Constant::getNullValue(pred->getType())), mask);
    Value* words = b_.CreateBitCast(bits, FixedVectorType::get(b_.getInt32Ty(), W / CPUSIM_WARP_SIZE));
    SmallVector<int, 64> idx;
    for (unsigned l = 0; l < W; ++l) idx.push_back(int(l / CPUSIM_WARP_SIZE));
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

// __cpusim_mmav_bf16_f32(j, a, b0..b7, c0..c7): element j of one warp's
// D = A * B + C, with B and C as values and A by pointer (its fragment, a
// per-thread array). Element e of a fragment, over a warp's 32 lanes, is rows
// 2e and 2e + 1 of the tile (mma.h), so C's values are already vdpbf16ps
// accumulators, a row per 16 lanes, and B's are one register of two rows
// each, interleaved into k pairs with vpermw. A's pairs are broadcast from
// its fragment in memory (lane-major: row r at (r / 2) * W + (r % 2) * 16
// elements), which folds into the load units; a broadcast out of a register
// would cost a shuffle per vdpbf16ps on the same pipes. Loading A one step
// ahead (mma.h's note) keeps those loads off stores still in flight. The
// first of the 8 calls computes all of D; the others take their element.
Value* WarpBuilder::emit_mmav(CallInst* call) {
    std::vector<Value*> key{b_.GetInsertBlock()};
    for (unsigned i = 1; i < call->arg_size(); ++i) key.push_back(call->getArgOperand(i));
    std::array<Value*, 8>& out = mmav_done_[key];
    const unsigned j = unsigned(cast<ConstantInt>(call->getArgOperand(0))->getZExtValue());
    if (out[0]) return out[j];

    auto* i8 = b_.getInt8Ty();
    auto* v32bf = FixedVectorType::get(b_.getBFloatTy(), 32);
    auto slice = [&](Value* v, unsigned first, unsigned n) {
        SmallVector<int, 32> idx;
        for (unsigned i = 0; i < n; ++i) idx.push_back(int(first + i));
        return b_.CreateShuffleVector(v, idx);
    };
    Value* pa = call->getArgOperand(1);
    auto* fa = dyn_cast<AllocaInst>(getUnderlyingObject(pa, 12));
    if (!fa || !private_elem_.count(fa)) return fail("tensor core fragment not in a per-thread array", call), nullptr;
    SplitAddress sa = split_address(pa);
    if (auto* lanes = dyn_cast<Constant>(sa.offsets)) {
        for (unsigned l = 0; l < W; ++l) {
            auto* e = dyn_cast_or_null<ConstantInt>(lanes->getAggregateElement(l));
            if (!e || e->getZExtValue() != l * 2) return fail("tensor core fragment chosen per lane", call), nullptr;
        }
    } else {
        return fail("tensor core fragment chosen per lane", call), nullptr;
    }
    Value *bv[8], *c[8];
    for (unsigned e = 0; e < 8; ++e) {
        bv[e] = vector(call->getArgOperand(2 + e));
        c[e] = vector(call->getArgOperand(10 + e));
    }
    SmallVector<uint16_t, 32> pairs;
    for (unsigned n = 0; n < 16; ++n) pairs.push_back(uint16_t(n)), pairs.push_back(uint16_t(16 + n));
    Constant* pair_idx = ConstantDataVector::get(ctx_, ArrayRef<uint16_t>(pairs));
    Function* dp = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512bf16_dpbf16ps_512);
    Function* perm = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_permvar_hi_512);

    std::array<std::vector<Value*>, 8> parts;  // per element: 16-lane rows, in lane order
    for (unsigned w = 0; w < W / CPUSIM_WARP_SIZE; ++w) {
        const unsigned l0 = w * CPUSIM_WARP_SIZE;
        Value* abase = b_.CreateConstInBoundsGEP1_64(i8, sa.base, l0 * 2);
        Value* acc[16];
        for (unsigned r = 0; r < 16; ++r) acc[r] = slice(c[r / 2], l0 + (r % 2) * 16, 16);
        // k pair outermost: 16 independent accumulator chains, one B live.
        for (unsigned p = 0; p < 8; ++p) {
            Value* bp = b_.CreateBitCast(b_.CreateCall(perm, {slice(bv[p], l0, 32), pair_idx}), v32bf);
            for (unsigned r = 0; r < 16; ++r) {
                Value* at = b_.CreateConstInBoundsGEP1_64(i8, abase, (r / 2) * W * 2 + (r % 2) * 32 + 4 * p);
                Value* pr = b_.CreateAlignedLoad(b_.getInt32Ty(), at, Align(4));
                // A last: only the third operand has a {1to16} memory form
                // (each pair product is symmetric, so the order is free).
                acc[r] = b_.CreateCall(dp, {acc[r], bp, b_.CreateBitCast(b_.CreateVectorSplat(16, pr), v32bf)});
            }
        }
        for (unsigned e = 0; e < 8; ++e) parts[e].push_back(acc[2 * e]), parts[e].push_back(acc[2 * e + 1]);
    }
    for (unsigned e = 0; e < 8; ++e) out[e] = concat_chunks(parts[e]);
    return out[j];
}

// Cooperative variant: wait for the other half of the block, inline.
//     e = *epoch + 1; atomic store-release *epoch = e;
//     while (atomic load-acquire *other < e) pause;
// Only when some lane of this half is here, so the two halves count the same
// barriers (a half that already returned stops counting, see kernel_abi.h).
void WarpBuilder::emit_coop_barrier(Value* mask) {
    auto* ptr = PointerType::get(ctx_, 0);
    auto* i64 = b_.getInt64Ty();
    BasicBlock* arrive = BasicBlock::Create(ctx_, "coop.arrive", w_);
    BasicBlock* wait = BasicBlock::Create(ctx_, "coop.wait", w_);
    BasicBlock* spin = BasicBlock::Create(ctx_, "coop.spin", w_);
    BasicBlock* after = BasicBlock::Create(ctx_, "coop.after", w_);
    b_.CreateCondBr(any_lane(mask), arrive, after);

    b_.SetInsertPoint(arrive);
    auto field = [&](unsigned offset, const char* name) {
        return b_.CreateAlignedLoad(ptr, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, offset), Align(8), name);
    };
    Value* epoch = field(CPUSIM_CTX_COOP_EPOCH, "coop.epoch");
    Value* other = field(CPUSIM_CTX_COOP_OTHER, "coop.other");
    LoadInst* cur = b_.CreateAlignedLoad(i64, epoch, Align(8), "coop.cur");
    cur->setAtomic(AtomicOrdering::Monotonic);
    Value* next = b_.CreateAdd(cur, ConstantInt::get(i64, 1), "coop.next");
    b_.CreateAlignedStore(next, epoch, Align(8))->setAtomic(AtomicOrdering::Release);
    b_.CreateBr(wait);

    b_.SetInsertPoint(wait);
    LoadInst* seen = b_.CreateAlignedLoad(i64, other, Align(8), "coop.seen");
    seen->setAtomic(AtomicOrdering::Acquire);
    b_.CreateCondBr(b_.CreateICmpUGE(seen, next), after, spin);

    b_.SetInsertPoint(spin);
    b_.CreateCall(Intrinsic::getDeclaration(&m_, Intrinsic::x86_sse2_pause));
    b_.CreateBr(wait);

    b_.SetInsertPoint(after);
}

// ---------------------------------------------------------------- structure

Node WarpBuilder::node_of(BasicBlock* bb, Loop* level) {
    Loop* l = li_->getLoopFor(bb);
    if (l == level) return {bb, nullptr};
    while (l && l->getParentLoop() != level) l = l->getParentLoop();
    return {nullptr, l};
}

// Reverse post order of a loop level, with subloops collapsed and back edges to
// the level's header ignored: every node comes after all of its predecessors.
std::vector<Node> WarpBuilder::topo_order(Loop* level) {
    BasicBlock* start = level ? level->getHeader() : &f_.getEntryBlock();
    std::vector<Node> post;
    DenseSet<const void*> seen;
    auto key = [](const Node& n) -> const void* { return n.block ? (const void*)n.block : (const void*)n.loop; };

    std::function<void(const Node&)> dfs = [&](const Node& n) {
        if (!seen.insert(key(n)).second) return;
        SmallVector<BasicBlock*, 8> exits_from;
        if (n.block) {
            exits_from.push_back(n.block);
        } else {
            for (BasicBlock* bb : n.loop->blocks()) exits_from.push_back(bb);
        }
        for (BasicBlock* bb : exits_from) {
            for (BasicBlock* s : successors(bb)) {
                if (level && (!level->contains(s) || s == level->getHeader())) continue;
                if (n.loop && n.loop->contains(s)) continue;
                dfs(node_of(s, level));
            }
        }
        post.push_back(n);
    };
    dfs(node_of(start, level));
    return {post.rbegin(), post.rend()};
}

// ---------------------------------------------------------------- emission

void WarpBuilder::emit_prologue() {
    auto* i32 = b_.getInt32Ty();
    auto* ptr = PointerType::get(ctx_, 0);
    mask_type_ = FixedVectorType::get(b_.getInt1Ty(), W);

    w_ = Function::Create(FunctionType::get(i32, {ptr}, false), GlobalValue::ExternalLinkage,
                          (profile_ ? std::string("__cpusim_prof_")
                           : W == CPUSIM_WARP_SIZE ? std::string("__cpusim_entry_")
                                                   : "__cpusim_entry" + std::to_string(W) +
                                                         (coop_ ? "c_" : block_wide_ ? "_" : "s_")) +
                              k_.name,
                          m_);
    w_->addFnAttr(Attribute::NoUnwind);
    w_->addFnAttr("prefer-vector-width", "512");
    w_->addFnAttr("min-legal-vector-width", "512");
    ctx_arg_ = w_->getArg(0);
    ctx_arg_->setName("ctx");
    w_->addParamAttr(0, Attribute::NoAlias);
    w_->addParamAttr(0, Attribute::getWithAlignment(ctx_, Align(64)));
    w_->addParamAttr(0, Attribute::getWithDereferenceableBytes(ctx_, cpusim_ctx_size(W)));

    alloca_block_ = BasicBlock::Create(ctx_, "allocas", w_);
    BasicBlock* entry = BasicBlock::Create(ctx_, "entry", w_);
    entry_ = entry;
    b_.SetInsertPoint(entry);

    auto field = [&](unsigned offset) { return b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, offset); };
    auto load_u32 = [&](unsigned offset, const char* name) { return b_.CreateAlignedLoad(i32, field(offset), Align(4), name); };

    // Kernel parameters: uniform, read from the launch's parameter buffer.
    const DataLayout& dl = m_.getDataLayout();
    Value* params = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_PARAMS), Align(8), "params");
    uint64_t offset = 0;
    for (Argument& arg : f_.args()) {
        Type* t = arg.getType();
        uint64_t size, align;
        Type* byval = arg.getParamByValType();
        if (byval) {
            size = dl.getTypeAllocSize(byval);
            align = std::max<uint64_t>(dl.getABITypeAlign(byval).value(), arg.getParamAlign().valueOrOne().value());
        } else {
            size = dl.getTypeAllocSize(t);
            align = dl.getABITypeAlign(t).value();
        }
        offset = alignTo(offset, align);
        Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), params, offset);
        scalar_map_[&arg] = byval ? p : b_.CreateAlignedLoad(t, p, Align(align), arg.getName());
        if (!profile_ && W == CPUSIM_WARP_SIZE) k_.params.push_back({offset, size});
        offset += size;
    }
    k_.param_bytes = offset;
    if (profile_) counters_ = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_COUNTERS), Align(8), "counters");

    // Special registers.
    auto* v32 = FixedVectorType::get(i32, W);
    if (k_.spec_block[0]) {
        emit_specialized_tid();
    } else {
        special_[(int)SpecialReg::TidX] = b_.CreateAlignedLoad(v32, field(CPUSIM_CTX_TID(W, 0)), Align(64), "tid.x");
        special_[(int)SpecialReg::TidY] = b_.CreateAlignedLoad(v32, field(CPUSIM_CTX_TID(W, 1)), Align(64), "tid.y");
        special_[(int)SpecialReg::TidZ] = b_.CreateAlignedLoad(v32, field(CPUSIM_CTX_TID(W, 2)), Align(64), "tid.z");
    }
    // blockIdx within CUDA's grid limits: lets the optimizer (and
    // linear_form) prove that index arithmetic built on it cannot wrap.
    auto ranged = [&](Value* v, uint32_t limit) {
        cast<Instruction>(v)->setMetadata(
            LLVMContext::MD_range,
            MDNode::get(ctx_, {ConstantAsMetadata::get(b_.getInt32(0)), ConstantAsMetadata::get(b_.getInt32(limit))}));
        return v;
    };
    special_[(int)SpecialReg::CtaidX] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX, "ctaid.x"), 0x7fffffff);
    special_[(int)SpecialReg::CtaidY] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX + 4, "ctaid.y"), 65535);
    special_[(int)SpecialReg::CtaidZ] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX + 8, "ctaid.z"), 65535);
    if (k_.spec_block[0]) {
        special_[(int)SpecialReg::NtidX] = b_.getInt32(k_.spec_block[0]);
        special_[(int)SpecialReg::NtidY] = b_.getInt32(k_.spec_block[1]);
        special_[(int)SpecialReg::NtidZ] = b_.getInt32(k_.spec_block[2]);
    } else {
        special_[(int)SpecialReg::NtidX] = load_u32(CPUSIM_CTX_BLOCK_DIM, "ntid.x");
        special_[(int)SpecialReg::NtidY] = load_u32(CPUSIM_CTX_BLOCK_DIM + 4, "ntid.y");
        special_[(int)SpecialReg::NtidZ] = load_u32(CPUSIM_CTX_BLOCK_DIM + 8, "ntid.z");
    }
    special_[(int)SpecialReg::NctaidX] = load_u32(CPUSIM_CTX_GRID_DIM, "nctaid.x");
    special_[(int)SpecialReg::NctaidY] = load_u32(CPUSIM_CTX_GRID_DIM + 4, "nctaid.y");
    special_[(int)SpecialReg::NctaidZ] = load_u32(CPUSIM_CTX_GRID_DIM + 8, "nctaid.z");
    SmallVector<Constant*, 64> lanes;  // CUDA lane id: position inside the 32-thread warp
    for (unsigned i = 0; i < W; ++i) lanes.push_back(b_.getInt32(i % CPUSIM_WARP_SIZE));
    special_[(int)SpecialReg::LaneId] = ConstantVector::get(lanes);
    special_[(int)SpecialReg::WarpSize] = b_.getInt32(CPUSIM_WARP_SIZE);
    special_[(int)SpecialReg::SharedBase] = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_SHARED), Align(8), "shared");
    Value* lane_mask = spec_lane_mask_
                           ? spec_lane_mask_
                           : b_.CreateBitCast(b_.CreateAlignedLoad(b_.getIntNTy(W), field(CPUSIM_CTX_LANE_MASK(W)),
                                                                   Align(64), "lane_mask"),
                                              mask_type_, "initial_mask");

    // Everything read so far (context fields, kernel parameters) is constant
    // for the whole launch. Saying so lets the register allocator reload such
    // values where needed instead of spilling them at every entry.
    MDNode* empty = MDNode::get(ctx_, {});
    for (Instruction& inst : *entry) {
        auto* ld = dyn_cast<LoadInst>(&inst);
        if (!ld) continue;
        // Only the context and the parameters; never the frame (it changes
        // between calls, and within a block function between rounds).
        Value* base = getUnderlyingObject(ld->getPointerOperand());
        if (base != ctx_arg_ && base != params) continue;
        ld->setMetadata(LLVMContext::MD_invariant_load, empty);
        if (ld->getType()->isPointerTy() && ld == params)
            ld->setMetadata(LLVMContext::MD_dereferenceable,
                            MDNode::get(ctx_, ConstantAsMetadata::get(b_.getInt64(std::max<uint64_t>(k_.param_bytes, 1)))));
    }

    // With barriers, a warp either starts (state 0) or resumes after one.
    if (!barrier_blocks_.empty()) {
        frame_ = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_FRAME), Align(8), "frame");
        state_ptr_ = frame_;
        Value* state = b_.CreateAlignedLoad(i32, state_ptr_, Align(64), "state");
        BasicBlock* init = BasicBlock::Create(ctx_, "start", w_);
        dispatch_ = BasicBlock::Create(ctx_, "resume", w_);
        cache_block_ = BasicBlock::Create(ctx_, "warp.invariants", w_);
        b_.CreateCondBr(b_.CreateICmpEQ(state, b_.getInt32(0)), cache_block_, dispatch_);
        BranchInst::Create(init, cache_block_);
        b_.SetInsertPoint(init);
    }

    // Slots: one mask per block, one "next iteration" mask per loop, and the
    // demoted values (scalar for uniform values, <32 x T> for varying ones).
    IRBuilder<> ab(alloca_block_);
    for (BasicBlock& bb : f_) {
        AllocaInst* a = ab.CreateAlloca(mask_type_, nullptr, "mask." + bb.getName());
        block_mask_[&bb] = a;
        b_.CreateStore(&bb == &f_.getEntryBlock() ? lane_mask : no_lanes(), a);
    }
    for (Loop* l : li_->getLoopsInPreorder()) {
        AllocaInst* a = ab.CreateAlloca(mask_type_, nullptr, "next." + l->getHeader()->getName());
        next_mask_[l] = a;
        b_.CreateStore(no_lanes(), a);
    }
    for (const AllocaInst* slot : demoted_order_) {
        Type* t = slot->getAllocatedType();
        bool varying = uni_->is_varying(slot);
        if (varying && !(t->isIntegerTy() || t->isFloatingPointTy() || t->isPointerTy())) {
            fail("varying value of unsupported type", slot);
            continue;
        }
        slots_[slot] = ab.CreateAlloca(varying ? vec_type(t) : t, nullptr, slot->getName());
    }
    ab.CreateBr(entry);

    // Frame layout: the resume state at offset 0, then every slot.
    if (!barrier_blocks_.empty()) {
        const DataLayout& dl = m_.getDataLayout();
        uint64_t offset = 64;
        for (Instruction& inst : *alloca_block_) {
            auto* a = dyn_cast<AllocaInst>(&inst);
            if (!a) continue;
            Type* t = a->getAllocatedType();
            uint64_t align = std::min<uint64_t>(64, dl.getPrefTypeAlign(t).value());
            offset = alignTo(offset, align);
            state_.push_back({a, offset});
            offset += dl.getTypeAllocSize(t);
        }
        cache_end_ = alignTo(offset, 64);
        note_frame_size();

        // Per call, never saved: created after the frame layout.
        pass_used_ = new AllocaInst(Type::getInt1Ty(ctx_), 0, "barrier.pass.used", alloca_block_->getTerminator()->getIterator());
        new StoreInst(ConstantInt::getFalse(ctx_), pass_used_, entry_->getTerminator()->getIterator());
    }
}

// Stores every slot for now; compute_resume_liveness() later removes the
// stores of slots the code after the barrier never reads before writing.
void WarpBuilder::emit_barrier(BasicBlock* bb) {
    BasicBlock* resume = BasicBlock::Create(ctx_, "after." + bb->getName(), w_);
    BasicBlock* through = BasicBlock::Create(ctx_, "barrier.through", w_);
    BasicBlock* suspend = BasicBlock::Create(ctx_, "barrier.suspend", w_);
    resumes_.push_back(resume);
    barrier_order_.push_back(bb);
    Value* pass = b_.CreateAlignedLoad(b_.getInt32Ty(),
                                       b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_BARRIER_PASS),
                                       Align(4), "barrier.pass");
    Value* go = b_.CreateAnd(b_.CreateICmpNE(pass, b_.getInt32(0)),
                             b_.CreateNot(b_.CreateLoad(b_.getInt1Ty(), pass_used_)));
    b_.CreateCondBr(go, through, suspend);

    b_.SetInsertPoint(through);
    b_.CreateStore(b_.getTrue(), pass_used_);
    b_.CreateBr(resume);

    b_.SetInsertPoint(suspend);
    save_state(int(resumes_.size()));
    b_.CreateRet(b_.getInt32(CPUSIM_BARRIER));
    b_.SetInsertPoint(resume);
}

void WarpBuilder::save_state(int resume_id) {
    std::vector<Save> saves;
    for (auto [a, offset] : state_) {
        Type* t = a->getAllocatedType();
        Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset);
        LoadInst* ld = b_.CreateLoad(t, a);
        StoreInst* st = b_.CreateAlignedStore(ld, p, Align(std::min<uint64_t>(64, MinAlign(offset, 64))));
        saves.push_back({a, offset, ld, st});
    }
    b_.CreateAlignedStore(b_.getInt32(resume_id), state_ptr_, Align(64));
    saves_.push_back(std::move(saves));
    save_blocks_.push_back(b_.GetInsertBlock());
}

// Liveness of the demoted slots on the per-thread code. Each lane follows its
// own path through it, the linearization keeps every lane's order, and a
// masked write fully defines the slot for the lanes that execute it; so this
// is exact per lane, unlike the same analysis on the linearized code, which
// cannot see that a block skipped for lack of lanes also skips its readers.
void WarpBuilder::compute_thread_liveness() {
    DenseMap<const BasicBlock*, DenseSet<const AllocaInst*>> use, def, in;
    for (BasicBlock& bb : f_) {
        auto& u = use[&bb];
        auto& d = def[&bb];
        for (Instruction& inst : bb) {
            if (auto* ld = dyn_cast<LoadInst>(&inst)) {
                if (auto* a = dyn_cast<AllocaInst>(ld->getPointerOperand()); a && demoted_.contains(a) && !d.contains(a))
                    u.insert(a);
            } else if (auto* st = dyn_cast<StoreInst>(&inst)) {
                if (auto* a = dyn_cast<AllocaInst>(st->getPointerOperand()); a && demoted_.contains(a)) d.insert(a);
            }
        }
        in[&bb] = u;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : post_order(&f_.getEntryBlock())) {
            DenseSet<const AllocaInst*> out;
            for (BasicBlock* s : successors(bb))
                for (const AllocaInst* a : in[s]) out.insert(a);
            for (const AllocaInst* a : out) {
                if (def[bb].contains(a)) continue;
                if (in[bb].insert(a).second) changed = true;
            }
        }
    }
    for (const BasicBlock* bb : barrier_blocks_) {
        auto& live = live_after_barrier_[bb];
        for (const BasicBlock* s : successors(bb))
            for (const AllocaInst* a : in[s]) live.insert(a);
    }
}

// Liveness of the slots over the generated (already linearized) code: what
// runs after a resume point is exactly this code, so a slot must survive the
// barrier only if some path from the resume block reads it before writing it.
// A blend under a partial mask is a load followed by a store, so it counts as
// a read, which keeps the inactive lanes' values alive as they must be.
void WarpBuilder::compute_resume_liveness() {
    DenseMap<const AllocaInst*, unsigned> index;
    for (auto& [a, offset] : state_) index[a] = unsigned(index.size());
    const unsigned n = unsigned(index.size());
    DenseSet<const Instruction*> save_loads;
    for (auto& v : saves_)
        for (auto& s : v) save_loads.insert(s.load);

    DenseMap<const BasicBlock*, BitVector> use, def, in;
    for (BasicBlock& bb : *w_) {
        BitVector u(n), d(n);
        for (Instruction& inst : bb) {
            if (auto* ld = dyn_cast<LoadInst>(&inst); ld && !save_loads.contains(ld)) {
                if (auto* a = dyn_cast<AllocaInst>(ld->getPointerOperand()); a && index.count(a)) {
                    unsigned i = index[a];
                    if (!d[i]) u.set(i);
                }
            } else if (auto* st = dyn_cast<StoreInst>(&inst)) {
                if (auto* a = dyn_cast<AllocaInst>(st->getPointerOperand()); a && index.count(a)) d.set(index[a]);
            }
        }
        use[&bb] = u;
        def[&bb] = d;
        in[&bb] = u;
    }
    std::vector<BasicBlock*> order;
    for (BasicBlock* bb : post_order(&w_->getEntryBlock())) order.push_back(bb);
    for (BasicBlock* r : resumes_)  // not reachable from the entry until the dispatch exists
        for (BasicBlock* bb : post_order(r)) order.push_back(bb);
    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : order) {
            BitVector out(n);
            for (BasicBlock* s : successors(bb)) out |= in[s];
            // A barrier's return continues, later, at its resume point.
            for (size_t k = 0; k < save_blocks_.size(); ++k)
                if (save_blocks_[k] == bb) out |= in[resumes_[k]];
            BitVector cur = def[bb];
            cur.flip();
            cur &= out;
            cur |= use[bb];
            if (cur != in[bb]) {
                in[bb] = cur;
                changed = true;
            }
        }
    }

    // Demoted slots: the per-thread answer is exact; masks only exist here.
    compute_thread_liveness();
    DenseMap<const AllocaInst*, const AllocaInst*> original;  // warp slot -> per-thread slot
    for (auto& [orig, slot] : slots_) original[slot] = orig;
    for (size_t k = 0; k < resumes_.size(); ++k) {
        BitVector live = in[resumes_[k]];
        const auto& thread_live = live_after_barrier_[barrier_order_[k]];
        for (auto& [a, offset] : state_) {
            auto it = original.find(a);
            if (it != original.end() && !thread_live.contains(it->second)) live.reset(index[a]);
        }
        if (std::getenv("CPUSIM_DEVC_DEBUG")) {
            errs() << "resume " << (k + 1) << " (" << barrier_order_[k]->getName() << "):";
            for (auto& [a, offset] : state_) {
                auto it = original.find(a);
                bool wl = in[resumes_[k]][index[a]];
                bool tl = it != original.end() && thread_live.contains(it->second);
                if (wl || tl) errs() << " " << a->getName() << (wl ? "[w]" : "") << (tl ? "[t]" : "");
            }
            errs() << "\n";
        }
        live_at_resume_.push_back(live);
        for (Save& s : saves_[k]) {
            if (live[index[s.slot]]) continue;
            Value* addr = s.store->getPointerOperand();
            s.store->eraseFromParent();
            s.load->eraseFromParent();
            if (auto* gep = dyn_cast<Instruction>(addr); gep && gep->use_empty()) gep->eraseFromParent();
        }
    }
}

// The resume path: reload every slot and mask from the frame, then jump to
// the code right after the barrier the warp stopped at.
void WarpBuilder::emit_dispatch() {
    compute_resume_liveness();
    b_.SetInsertPoint(dispatch_);
    Value* state = b_.CreateAlignedLoad(b_.getInt32Ty(), state_ptr_, Align(64));
    BasicBlock* bad = BasicBlock::Create(ctx_, "resume.bad", w_);
    SwitchInst* sw = b_.CreateSwitch(state, bad, unsigned(resumes_.size()));
    for (size_t k = 0; k < resumes_.size(); ++k) {
        BasicBlock* restore = BasicBlock::Create(ctx_, "restore." + Twine(k + 1), w_);
        sw->addCase(b_.getInt32(uint32_t(k + 1)), restore);
        b_.SetInsertPoint(restore);
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!live_at_resume_[k][i]) continue;
            auto [a, offset] = state_[i];
            Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset);
            b_.CreateStore(b_.CreateAlignedLoad(a->getAllocatedType(), p, Align(std::min<uint64_t>(64, MinAlign(offset, 64)))), a);
        }
        b_.CreateBr(resumes_[k]);
    }
    b_.SetInsertPoint(bad);
    b_.CreateUnreachable();
}

void WarpBuilder::add_edge(BasicBlock* from, BasicBlock* to, Value* edge_mask) {
    AllocaInst* target = block_mask_[to];
    if (Loop* l = li_->getLoopFor(to); l && l->getHeader() == to && l->contains(from)) target = next_mask_[l];
    Value* old = b_.CreateLoad(mask_type_, target);
    b_.CreateStore(b_.CreateOr(old, edge_mask), target);
}

void WarpBuilder::emit_level(Loop* level) {
    for (const Node& n : topo_order(level)) emit_node(n);
}

void WarpBuilder::emit_node(const Node& n) {
    if (n.block) emit_block(n.block);
    else emit_loop(n.loop);
}

void WarpBuilder::emit_block(BasicBlock* bb) {
    Value* mask = b_.CreateLoad(mask_type_, block_mask_[bb], "m." + bb->getName());
    BasicBlock* body = BasicBlock::Create(ctx_, bb->getName(), w_);
    BasicBlock* next = BasicBlock::Create(ctx_, bb->getName() + ".next", w_);
    b_.CreateCondBr(any_lane(mask), body, next);
    b_.SetInsertPoint(body);

    bool divergent = uni_->is_divergent_block(bb);
    uint64_t counts[CPUSIM_COUNTERS] = {};
    for (Instruction& inst : *bb) {
        if (profile_) count(inst, counts);
        if (inst.isTerminator()) break;
        if (!emit_instruction(inst, mask, divergent)) return;
    }
    if (profile_) emit_counters(counts, mask);

    if (barrier_blocks_.contains(bb)) {
        // Stop here; the runtime calls the entry again once every warp of the
        // block got to this barrier.
        emit_barrier(bb);
        mask = b_.CreateLoad(mask_type_, block_mask_[bb], "m." + bb->getName());
    }

    if (auto* br = dyn_cast<BranchInst>(bb->getTerminator())) {
        if (br->isUnconditional()) {
            add_edge(bb, br->getSuccessor(0), mask);
        } else {
            Value* c = br->getCondition();
            Value *taken, *not_taken;
            if (uni_->is_varying(c)) {
                Value* cv = vector(c);
                taken = b_.CreateAnd(mask, cv);
                not_taken = b_.CreateAnd(mask, b_.CreateNot(cv));
            } else {
                Value* cs = scalar(c);
                taken = b_.CreateSelect(cs, mask, no_lanes());
                not_taken = b_.CreateSelect(cs, no_lanes(), mask);
            }
            add_edge(bb, br->getSuccessor(0), taken);
            add_edge(bb, br->getSuccessor(1), not_taken);
        }
    }
    b_.CreateBr(next);
    b_.SetInsertPoint(next);
}

// A loop where every branch (exits, back edges, and branches inside, nested
// loops included) is uniform: all active lanes walk the same path, so the
// mask is the one the loop was entered with, all the way through.
bool WarpBuilder::is_uniform_loop(Loop* l) {
    for (BasicBlock* bb : l->blocks()) {
        if (auto* br = dyn_cast<BranchInst>(bb->getTerminator()); br && br->isConditional() &&
                                                                    uni_->is_varying(br->getCondition()))
            return false;
    }
    return true;
}

// Emits a uniform loop with its own control flow (real scalar branches) and
// the entry mask for every block: no per-block masks, no per-iteration
// "any lane left" tests. Lanes leave together and join the exit's mask.
void WarpBuilder::emit_uniform_loop(Loop* l) {
    BasicBlock* header = l->getHeader();
    std::string name = header->getName().str();
    bool barriers = any_of(l->blocks(), [&](BasicBlock* bb) { return barrier_blocks_.contains(bb); });
    Value* entry_mask = b_.CreateLoad(mask_type_, block_mask_[header], "m.uloop." + name);
    BasicBlock* exit = BasicBlock::Create(ctx_, "uloop." + name + ".exit", w_);
    DenseMap<BasicBlock*, BasicBlock*> emitted;
    for (BasicBlock* bb : l->blocks()) emitted[bb] = BasicBlock::Create(ctx_, "u." + bb->getName(), w_);
    b_.CreateCondBr(any_lane(entry_mask), emitted[header], exit);

    for (BasicBlock* bb : l->blocks()) {
        b_.SetInsertPoint(emitted[bb]);
        // With barriers inside, a resume enters the loop in the middle: the
        // mask then comes back from the header's slot (restored with the rest).
        Value* mask = barriers ? b_.CreateLoad(mask_type_, block_mask_[header], "m.u") : entry_mask;
        bool divergent = uni_->is_divergent_block(bb);
        uint64_t counts[CPUSIM_COUNTERS] = {};
        for (Instruction& inst : *bb) {
            if (profile_) count(inst, counts);
            if (inst.isTerminator()) break;
            if (!emit_instruction(inst, mask, divergent)) return;
        }
        if (profile_) emit_counters(counts, mask);

        if (barrier_blocks_.contains(bb)) {
            emit_barrier(bb);
            mask = b_.CreateLoad(mask_type_, block_mask_[header], "m.u");
        }

        // Leaving the loop: the lanes join the target's mask, then continue
        // with the code after the loop.
        auto target = [&](BasicBlock* s) -> BasicBlock* {
            if (l->contains(s)) return emitted[s];
            BasicBlock* edge = BasicBlock::Create(ctx_, "u.exit." + s->getName(), w_);
            IRBuilder<>::InsertPointGuard guard(b_);
            b_.SetInsertPoint(edge);
            add_edge(bb, s, mask);
            b_.CreateBr(exit);
            return edge;
        };
        Instruction* term = bb->getTerminator();
        if (auto* br = dyn_cast<BranchInst>(term)) {
            if (br->isUnconditional())
                b_.CreateBr(target(br->getSuccessor(0)));
            else
                b_.CreateCondBr(scalar(br->getCondition()), target(br->getSuccessor(0)), target(br->getSuccessor(1)));
        } else {
            b_.CreateBr(exit);  // return / unreachable: these lanes are done
        }
    }
    b_.SetInsertPoint(exit);
}

void WarpBuilder::emit_loop(Loop* l) {
    if (is_uniform_loop(l)) return emit_uniform_loop(l);
    BasicBlock* header = l->getHeader();
    std::string name = header->getName().str();
    BasicBlock* top = BasicBlock::Create(ctx_, "loop." + name, w_);
    BasicBlock* exit = BasicBlock::Create(ctx_, "loop." + name + ".exit", w_);
    Value* entry_mask = b_.CreateLoad(mask_type_, block_mask_[header]);
    b_.CreateCondBr(any_lane(entry_mask), top, exit);

    b_.SetInsertPoint(top);
    for (BasicBlock* bb : l->blocks())
        if (bb != header) b_.CreateStore(no_lanes(), block_mask_[bb]);
    for (Loop* sub : l->getLoopsInPreorder()) b_.CreateStore(no_lanes(), next_mask_[sub]);

    emit_level(l);

    Value* next = b_.CreateLoad(mask_type_, next_mask_[l], "next." + name);
    b_.CreateStore(next, block_mask_[header]);
    b_.CreateCondBr(any_lane(next), top, exit);
    b_.SetInsertPoint(exit);
}

bool WarpBuilder::emit_instruction(Instruction& inst, Value* mask, bool divergent) {
    const bool varying = uni_->is_varying(&inst);
    auto set = [&](Value* v) {
        (varying ? vector_map_ : scalar_map_)[&inst] = v;
        return true;
    };

    // Per-thread arrays: only addressed through split_address; the alloca and
    // the GEPs on it produce nothing of their own (all their uses are loads
    // and stores, checked in check_private_array).
    if (auto* a = dyn_cast<AllocaInst>(&inst); a && private_elem_.count(a)) return true;
    if (auto* g = dyn_cast<GetElementPtrInst>(&inst)) {
        Value* v = g;
        while (auto* p = dyn_cast<GetElementPtrInst>(v)) v = p->getPointerOperand();
        if (auto* a = dyn_cast<AllocaInst>(v); a && private_elem_.count(a)) return true;
    }

    // Demoted slots.
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
        if (auto* a = dyn_cast<AllocaInst>(ld->getPointerOperand()); a && demoted_.contains(a)) {
            AllocaInst* slot = slots_[a];
            return set(b_.CreateLoad(slot->getAllocatedType(), slot, ld->getName()));
        }
    }
    if (auto* st = dyn_cast<StoreInst>(&inst)) {
        if (auto* a = dyn_cast<AllocaInst>(st->getPointerOperand()); a && demoted_.contains(a)) {
            AllocaInst* slot = slots_[a];
            if (!uni_->is_varying(a)) {
                b_.CreateStore(scalar(st->getValueOperand()), slot);
                return true;
            }
            Value* v = vector(st->getValueOperand());
            if (divergent) v = b_.CreateSelect(mask, v, b_.CreateLoad(slot->getAllocatedType(), slot));
            b_.CreateStore(v, slot);
            return true;
        }
    }
    if (isa<AllocaInst>(inst)) return true;  // demoted slots were recreated in the prologue

    if (auto* call = dyn_cast<CallInst>(&inst); call && is_cond_load(call)) return set(emit_cond_load(call, mask));

    // Software prefetch: one per 16-lane chunk, at the chunk's first address
    // (a chunk usually spans one or two cache lines).
    if (auto* call = dyn_cast<IntrinsicInst>(&inst); call && call->getIntrinsicID() == Intrinsic::prefetch) {
        Value* ptr = call->getArgOperand(0);
        Function* pf = Intrinsic::getDeclaration(&m_, Intrinsic::prefetch, {ptr->getType()});
        if (!uni_->is_varying(ptr)) {
            b_.CreateCall(pf, {scalar(ptr), call->getArgOperand(1), call->getArgOperand(2), call->getArgOperand(3)});
            return true;
        }
        SplitAddress addr = split_address(ptr);
        // Lanes of a chunk next to each other (known at compile time): one
        // prefetch for the chunk. Spread out (strided per lane, unknown): one
        // per lane, or 15 of the 16 lines would be missed.
        std::vector<char> kind;
        bool known = false;
        if (LinearForm form; linear_form(addr.offsets, form)) {
            std::vector<Value*> firsts;
            kind = static_chunks(form, 4, firsts);
            known = true;
        }
        // Prefetches the compiler added (insert_prefetches) are only there
        // for lanes a cache line or more apart: one per lane.
        if (call->getMetadata("cpusim.auto")) kind.assign(W / 16, 0);
        (void)known;
        for (unsigned h = 0; h < W / 16; ++h) {
            const bool together = h < kind.size() && kind[h] != 0;
            Value* m_bits = b_.CreateBitCast(half(mask, h), b_.getInt16Ty());
            for (unsigned l = 0; l < (together ? 1u : 16u); ++l) {
                Value* off = b_.CreateExtractElement(addr.offsets, uint64_t(16 * h + l));
                if (together) off = cached(off);
                Value* active = together ? b_.CreateICmpNE(m_bits, b_.getInt16(0))
                                         : b_.CreateICmpNE(b_.CreateAnd(m_bits, b_.getInt16(uint16_t(1u << l))),
                                                           b_.getInt16(0));
                Value* p = b_.CreateSelect(active, b_.CreateGEP(b_.getInt8Ty(), addr.base, off),
                                           b_.CreatePointerCast(ctx_arg_, addr.base->getType()));
                b_.CreateCall(pf, {p, call->getArgOperand(1), call->getArgOperand(2), call->getArgOperand(3)});
            }
        }
        return true;
    }

    if (is_mma(&inst)) return emit_mma(cast<CallInst>(&inst), mask);
    if (is_mmav(&inst)) return set(emit_mmav(cast<CallInst>(&inst)));

    // Intrinsics with special meaning.
    if (auto* call = dyn_cast<CallInst>(&inst)) {
        SpecialReg r = special_reg(call);
        if (r != SpecialReg::None) return set(special_[(int)r]);
        if (is_barrier(*call)) {
            if (coop_) emit_coop_barrier(mask);
            return true;
        }
        Intrinsic::ID id = call->getIntrinsicID();
        switch (id) {
            case Intrinsic::nvvm_bar_warp_sync:  // the warp runs in lockstep
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
            case Intrinsic::assume:
            case Intrinsic::experimental_noalias_scope_decl:
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            case Intrinsic::dbg_label:
                return true;
            default:
                break;
        }
    }

    // Memory through real pointers.
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
        if (ld->isVolatile()) return fail("volatile loads are not supported yet", ld), false;
        Value* ptr = ld->getPointerOperand();
        if (!uni_->is_varying(ptr)) {
            Value* v = b_.CreateAlignedLoad(ld->getType(), scalar(ptr), ld->getAlign(), ld->getName());
            if (!varying) return set(v);
            return set(b_.CreateVectorSplat(W, v));
        }
        return set(gather(ld->getType(), ptr, ld->getAlign(), mask, ld->getName()));
    }
    if (auto* st = dyn_cast<StoreInst>(&inst)) {
        if (st->isVolatile()) return fail("volatile stores are not supported yet", st), false;
        Value* ptr = st->getPointerOperand();
        Value* val = st->getValueOperand();
        if (!uni_->is_varying(ptr) && !uni_->is_varying(val)) {
            b_.CreateAlignedStore(scalar(val), scalar(ptr), st->getAlign());
            return true;
        }
        if (!uni_->is_varying(ptr)) {
            // Every lane writes the same address: the last active lane wins.
            b_.CreateMaskedScatter(vector(val), vector(ptr), st->getAlign(), mask);
            return true;
        }
        scatter(vector(val), ptr, st->getAlign(), mask);
        return true;
    }

    // Everything else is a pure computation.
    if (!varying) {
        Instruction* c = inst.clone();
        for (Use& u : c->operands())
            if (!isa<BasicBlock>(u.get()) && !isa<Function>(u.get()) && !isa<MetadataAsValue>(u.get()))
                u.set(scalar(u.get()));
        b_.Insert(c, inst.getName());
        return set(c);
    }

    if (auto* bin = dyn_cast<BinaryOperator>(&inst)) {
        if (Value* v = emit_uniform_division(bin)) return set(v);
        Value* lhs = vector(bin->getOperand(0));
        Value* rhs = vector(bin->getOperand(1));
        switch (bin->getOpcode()) {
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::URem:
            case Instruction::SRem:
                // Inactive lanes may hold a zero divisor; x86 would trap on it.
                // Not a constant one: no lane has zero, and keeping it constant
                // lets LLVM divide with shifts and multiplies instead of a
                // scalar idiv per lane.
                if (auto* c = dyn_cast<ConstantInt>(bin->getOperand(1)); c && !c->isZero()) break;
                rhs = b_.CreateSelect(mask, rhs, ConstantInt::get(rhs->getType(), 1));
                break;
            default:
                break;
        }
        Value* v = b_.CreateBinOp(bin->getOpcode(), lhs, rhs, bin->getName());
        if (auto* vi = dyn_cast<Instruction>(v)) vi->copyIRFlags(bin);
        return set(v);
    }
    if (auto* un = dyn_cast<UnaryOperator>(&inst)) {
        Value* v = b_.CreateUnOp(un->getOpcode(), vector(un->getOperand(0)), un->getName());
        if (auto* vi = dyn_cast<Instruction>(v)) vi->copyIRFlags(un);
        return set(v);
    }
    if (auto* cast = dyn_cast<CastInst>(&inst))
        return set(b_.CreateCast(cast->getOpcode(), vector(cast->getOperand(0)), vec_type(cast->getDestTy()),
                                 cast->getName()));
    if (auto* cmp = dyn_cast<CmpInst>(&inst)) {
        Value* v = b_.CreateCmp(cmp->getPredicate(), vector(cmp->getOperand(0)), vector(cmp->getOperand(1)),
                                cmp->getName());
        if (auto* vi = dyn_cast<Instruction>(v); vi && isa<FPMathOperator>(vi)) vi->copyFastMathFlags(cmp);
        return set(v);
    }
    if (auto* sel = dyn_cast<SelectInst>(&inst)) {
        Value* c = uni_->is_varying(sel->getCondition()) ? vector(sel->getCondition()) : scalar(sel->getCondition());
        return set(b_.CreateSelect(c, vector(sel->getTrueValue()), vector(sel->getFalseValue()), sel->getName()));
    }
    if (auto* fr = dyn_cast<FreezeInst>(&inst)) return set(b_.CreateFreeze(vector(fr->getOperand(0)), fr->getName()));
    if (auto* gep = dyn_cast<GetElementPtrInst>(&inst)) {
        Value* base = uni_->is_varying(gep->getPointerOperand()) ? vector(gep->getPointerOperand())
                                                                  : scalar(gep->getPointerOperand());
        SmallVector<Value*, 4> idx;
        for (Use& u : gep->indices()) idx.push_back(uni_->is_varying(u.get()) ? vector(u.get()) : scalar(u.get()));
        Value* v = b_.CreateGEP(gep->getSourceElementType(), base, idx, gep->getName(), gep->isInBounds());
        if (!v->getType()->isVectorTy()) v = b_.CreateVectorSplat(W, v);
        return set(v);
    }
    if (auto* call = dyn_cast<CallInst>(&inst); call && call->getCalledFunction()) {
        StringRef name = call->getCalledFunction()->getName();
        if (name == kShflFn) return set(emit_shfl(vector(call->getArgOperand(0)), vector(call->getArgOperand(1))));
        if (name == kBallotFn) return set(emit_ballot(vector(call->getArgOperand(0)), mask));
    }
    if (auto* call = dyn_cast<CallInst>(&inst)) {
        if (unsigned arity; true) {
            std::string name = libmvec_name(*call, &arity);
            if (!name.empty()) return set(emit_libmvec(*call, name, arity));
        }
        Intrinsic::ID id = call->getIntrinsicID();
        if (id != Intrinsic::not_intrinsic && isTriviallyVectorizable(id)) {
            SmallVector<Value*, 4> args;
            SmallVector<Type*, 2> tys;
            if (isVectorIntrinsicWithOverloadTypeAtArg(id, -1)) tys.push_back(vec_type(call->getType()));
            for (unsigned i = 0; i < call->arg_size(); ++i) {
                Value* a = call->getArgOperand(i);
                bool keep_scalar = isVectorIntrinsicWithScalarOpAtArg(id, i);
                args.push_back(keep_scalar ? scalar(a) : vector(a));
                if (isVectorIntrinsicWithOverloadTypeAtArg(id, i)) tys.push_back(args.back()->getType());
            }
            Function* decl = Intrinsic::getDeclaration(&m_, id, tys);
            CallInst* v = b_.CreateCall(decl, args, call->getName());
            if (isa<FPMathOperator>(v)) v->copyFastMathFlags(call);
            return set(v);
        }
    }
    fail("unsupported varying instruction", &inst);
    return false;
}

// ---------------------------------------------------------------- profiling

// Static per-thread cost of one instruction of the original kernel. Slots
// introduced by demotion are bookkeeping, not program memory traffic.
void WarpBuilder::count(Instruction& inst, uint64_t* c) {
    const DataLayout& dl = m_.getDataLayout();
    auto fp = [&](Type* t, uint64_t n) {
        t = t->getScalarType();
        if (t->isDoubleTy()) c[CPUSIM_CNT_FP64_FLOPS] += n;
        else if (t->isFloatingPointTy()) c[CPUSIM_CNT_FP32_FLOPS] += n;
    };
    auto is_slot = [&](Value* ptr) {
        auto* a = dyn_cast<AllocaInst>(ptr);
        return a && demoted_.contains(a);
    };
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
        if (is_slot(ld->getPointerOperand())) return;
        c[CPUSIM_CNT_LOADS] += 1;
        c[CPUSIM_CNT_LOAD_BYTES] += dl.getTypeStoreSize(ld->getType());
    } else if (auto* st = dyn_cast<StoreInst>(&inst)) {
        if (is_slot(st->getPointerOperand())) return;
        c[CPUSIM_CNT_STORES] += 1;
        c[CPUSIM_CNT_STORE_BYTES] += dl.getTypeStoreSize(st->getValueOperand()->getType());
    } else if (auto* br = dyn_cast<BranchInst>(&inst)) {
        if (br->isConditional()) c[CPUSIM_CNT_BRANCHES] += 1;
    } else if (auto* bin = dyn_cast<BinaryOperator>(&inst)) {
        if (bin->getType()->isFPOrFPVectorTy()) fp(bin->getType(), 1);
        else c[CPUSIM_CNT_INT_OPS] += 1;
    } else if (isa<ICmpInst>(inst)) {
        c[CPUSIM_CNT_INT_OPS] += 1;
    } else if (isa<FCmpInst>(inst)) {
        fp(inst.getOperand(0)->getType(), 1);
    } else if (unsigned n; isa<CallInst>(inst) && !isa<IntrinsicInst>(inst) &&
                           !libmvec_name(cast<CallInst>(inst), &n).empty()) {
        c[CPUSIM_CNT_FP_SPECIAL] += 1;  // erf, tanh, atan2...: library calls, not intrinsics
    } else if (auto* call = dyn_cast<IntrinsicInst>(&inst)) {
        switch (call->getIntrinsicID()) {
            case Intrinsic::fma:
            case Intrinsic::fmuladd:
                fp(call->getType(), 2);
                break;
            case Intrinsic::sqrt: case Intrinsic::exp: case Intrinsic::exp2: case Intrinsic::exp10:
            case Intrinsic::log: case Intrinsic::log2: case Intrinsic::log10: case Intrinsic::sin:
            case Intrinsic::cos: case Intrinsic::pow: case Intrinsic::powi:
                c[CPUSIM_CNT_FP_SPECIAL] += 1;
                break;
            case Intrinsic::minnum: case Intrinsic::maxnum: case Intrinsic::minimum: case Intrinsic::maximum:
            case Intrinsic::fabs: case Intrinsic::copysign: case Intrinsic::floor: case Intrinsic::ceil:
            case Intrinsic::trunc: case Intrinsic::round: case Intrinsic::rint: case Intrinsic::nearbyint:
                fp(call->getType(), 1);
                break;
            case Intrinsic::smin: case Intrinsic::smax: case Intrinsic::umin: case Intrinsic::umax:
            case Intrinsic::abs: case Intrinsic::ctpop: case Intrinsic::ctlz: case Intrinsic::cttz:
                c[CPUSIM_CNT_INT_OPS] += 1;
                break;
            default:
                break;
        }
    }
}

// counters[i] += active lanes * per-thread count, once per block execution.
void WarpBuilder::emit_counters(const uint64_t* counts, Value* mask) {
    auto* i64 = b_.getInt64Ty();
    Value* lanes = b_.CreateZExtOrTrunc(
        b_.CreateUnaryIntrinsic(Intrinsic::ctpop, b_.CreateBitCast(mask, b_.getIntNTy(W))), i64, "lanes");
    auto add = [&](int idx, Value* v) {
        Value* slot = b_.CreateConstInBoundsGEP1_64(i64, counters_, idx);
        b_.CreateStore(b_.CreateAdd(b_.CreateLoad(i64, slot), v), slot);
    };
    for (int i = 0; i < CPUSIM_CNT_BLOCK_RUNS; ++i)
        if (counts[i]) add(i, b_.CreateMul(lanes, ConstantInt::get(i64, counts[i])));
    add(CPUSIM_CNT_BLOCK_RUNS, ConstantInt::get(i64, 1));
    add(CPUSIM_CNT_ACTIVE_LANES, lanes);
}

bool WarpBuilder::run(std::string* error) {
    if (!block_wide_) split_barriers();
    dt_ = std::make_unique<DominatorTree>(f_);
    pdt_ = std::make_unique<PostDominatorTree>(f_);
    li_ = std::make_unique<LoopInfo>(*dt_);
    if (!check_supported()) return *error = error_, false;

    uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_);
    if (!profile_ && insert_prefetches()) uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_);
    // The instrumented variant keeps the original branches so its per-thread
    // counts stay exact (if-conversion runs both sides' arithmetic).
    if (!profile_ && if_convert()) {
        dt_ = std::make_unique<DominatorTree>(f_);
        pdt_ = std::make_unique<PostDominatorTree>(f_);
        li_ = std::make_unique<LoopInfo>(*dt_);
        uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_);
    }
    demote();

    emit_prologue();
    if (error_.empty()) emit_level(nullptr);
    if (!error_.empty()) {
        *error = error_;
        if (w_) w_->eraseFromParent();
        return false;
    }
    b_.CreateRet(b_.getInt32(CPUSIM_EXIT));
    if (dispatch_) emit_dispatch();
    if (profile_) k_.profiled = w_;
    else if (W == CPUSIM_WARP_SIZE) k_.entry = w_;
    return true;
}

}  // namespace

bool build_warp_entry(Module& m, KernelInfo& kernel, Function& scalar, bool profile, unsigned lanes, bool suspending,
                      std::string* error, bool cooperative) {
    // Barriers are no-ops in a wide variant: the runtime only uses those when
    // the whole block fits in one call. (Kernels without barriers are free to
    // run any number of warps per call.)
    WarpBuilder builder(m, kernel, scalar, profile, lanes, lanes > CPUSIM_WARP_SIZE && !suspending && !cooperative,
                        cooperative);
    return builder.run(error);
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
