#pragma once

// Internal to the SIMT pass (simt*.cpp): the WarpBuilder that turns one
// per-thread function into W-lane code, and the helpers its parts share.
// simt.cpp: preparation, if-conversion, values and run(); simt_memory.cpp:
// loads, stores, gathers and scatters; simt_ops.cpp: calls, atomics, warp
// operations, math and tensor cores; simt_emit.cpp: the linearized control
// flow, barriers and profiling counters.

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
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>

#include "compiler/uniformity.h"
#include "device/kernel_abi.h"

using namespace llvm;

// (Only the SIMT pass includes this header.)
using namespace llvm;

namespace cpusim {
namespace simt_internal {

std::string describe(const Value* v);

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
    // `target`: build the body of this declaration (from vector_fn) instead
    // of a kernel entry: `scalar` is a device function called per lane.
    WarpBuilder(Module& m, KernelInfo& k, Function& scalar, bool profile, unsigned lanes, bool block_wide,
                Function* target = nullptr)
        : W(lanes), block_wide_(block_wide), m_(m), k_(k), f_(scalar),
          profile_(profile), ctx_(m.getContext()), b_(ctx_), fn_(target) {}

    bool run(std::string* error);
    void set_refill() { refill_ = true; }

private:
    bool refill_ = false;  // __cpusim_entryr_: lanes refill (make_refill_kernel)
    // ---- preparation and analysis
    bool check_supported();
    bool insert_prefetches();
    void demote();

    // ---- device functions that are not inlined (recursive, or called
    // through a pointer): each has a W-lane version taking the context, the
    // lane mask and every argument per lane, built from the same per-thread
    // code by a WarpBuilder of its own (build_pending_functions).
    Function* vector_fn(Function* f);
    bool emit_call(CallInst* call, Value* mask);

    // ---- atomics: one locked instruction per active lane, or one for the
    // warp when every lane updates the same address.
    bool emit_atomic(Instruction& inst, Value* mask);
    // `ptr`: the address of every lane in `mask` (default: the instruction's
    // own, uniform). Null if the operation has no reduction.
    Value* emit_uniform_rmw(AtomicRMWInst* rmw, Value* mask, Value* ptr = nullptr);
    Value* emit_refill(bool first, Value* mask);
    // Operations emit_uniform_rmw reduces across lanes.
    static bool groupable(const AtomicRMWInst* rmw) {
        using Op = AtomicRMWInst::BinOp;
        if (rmw->getType()->isIntegerTy()) switch (rmw->getOperation()) {
                case Op::Add: case Op::Sub: case Op::Or: case Op::Xor: case Op::And:
                case Op::Max: case Op::Min: case Op::UMax: case Op::UMin:
                    return true;
                default:
                    return false;
            }
        return rmw->getOperation() == Op::FAdd;
    }
    DenseMap<const Value*, std::pair<Value*, Value*>> cmpxchg_parts_;  // cmpxchg -> (old values, success)
    // An empty asm with a memory clobber: volatile accesses stay in order
    // and are never merged, hoisted or removed.
    // Divisors x86's idiv traps on and a GPU does not (the result is then
    // unspecified): zero, and -1 under the most negative dividend. Replaced
    // by 1, which gives the wrapped result for the latter.
    Value* trapping_divisor(Instruction::BinaryOps op, Value* n, Value* d) {
        Value* bad = b_.CreateICmpEQ(d, Constant::getNullValue(d->getType()));
        if (op == Instruction::SDiv || op == Instruction::SRem) {
            const unsigned bits = d->getType()->getScalarSizeInBits();
            Value* min = ConstantInt::get(d->getType(), APInt::getSignedMinValue(bits));
            bad = b_.CreateOr(bad, b_.CreateAnd(b_.CreateICmpEQ(d, Constant::getAllOnesValue(d->getType())),
                                                b_.CreateICmpEQ(n, min)));
        }
        return bad;
    }
    void compiler_barrier() {
        b_.CreateCall(InlineAsm::get(FunctionType::get(b_.getVoidTy(), false), "", "~{memory}", true));
    }
    Value* call_vector_fn(Function* f, CallInst* call, Value* mask);

    // ---- emission
    void emit_prologue();
    void emit_level(Loop* level);
    void emit_node(const Node& n);
    void emit_block(BasicBlock* bb);
    void emit_loop(Loop* l);
    bool emit_instruction(Instruction& inst, Value* mask, bool divergent);
    void add_edge(BasicBlock* from, BasicBlock* to, Value* edge_mask);
    std::vector<Node> topo_order(Loop* level);
    bool waits(const Node& n);
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
        // The pointer the address is computed from (a parameter, the shared
        // memory base, a loaded pointer...) and base's distance from it in
        // bytes: the address is root + uoff + offsets. An access stays inside
        // root's object, so uoff + offsets is smaller than that object.
        // root is null for an opaque varying pointer.
        Value* root = nullptr;
        Value* uoff = nullptr;  // scalar i64
        bool small = false;     // root's object is shared memory or a per-thread array: under 2 GiB
    };
    // Per-thread arrays (`float v[8]` that did not become registers), stored
    // lane-major: element byte offset o of lane l at base + o * W + l * elem,
    // so one element of every lane is contiguous. `elem`: the size of every
    // access (arrays of one scalar type).
    DenseMap<const AllocaInst*, uint64_t> private_elem_;
    DenseMap<const AllocaInst*, uint64_t> private_frame_offset_;  // in the frame (variants that suspend)
    DenseMap<const AllocaInst*, Value*> private_stack_;           // or in the stack
    bool check_private_array(AllocaInst* a);
    // Per-thread arrays whose address goes to a call (a function that is not
    // inlined): W copies one after the other, lane l's at base + l * size,
    // so each lane's pointer is a real one.
    DenseSet<const AllocaInst*> escaping_;
    static bool escapes(const AllocaInst* a) {
        SmallVector<const Value*, 8> work{a};
        while (!work.empty()) {
            const Value* v = work.pop_back_val();
            for (const User* u : v->users()) {
                if (isa<GetElementPtrInst>(u) || isa<BitCastInst>(u)) work.push_back(u);
                else if (auto* st = dyn_cast<StoreInst>(u); st && st->getValueOperand() == v) return true;
                else if (auto* call = dyn_cast<CallInst>(u)) {
                    if (auto* ii = dyn_cast<IntrinsicInst>(call); ii && ii->isLifetimeStartOrEnd()) continue;
                    if (is_mma(call) || is_mmav(call)) continue;
                    return true;
                } else if (isa<PtrToIntInst>(u) || isa<PHINode>(u) || isa<SelectInst>(u)) {
                    return true;
                }
            }
        }
        return false;
    }
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
    // Groups of g consecutive lanes (g = 8, 4 or 2, the largest that fits),
    // each contiguous, on uniform terms that are the same within a group but
    // not across groups: rows of a tile, g values per row, with a stride
    // known only at run time (wmma's sub-byte fragments: 4 words of 8 rows).
    // Returns g (0: no such pattern) and each group's first offset.
    unsigned static_groups(const LinearForm& f, uint64_t size, std::vector<Value*>& firsts);
    std::pair<Value*, Value*> split_index(Value* idx, bool is_signed);
    // `ptr` is the per-thread pointer; its offsets are recomputed in the rare
    // fallback paths so the common paths only need the cached pattern data.
    Value* gather(Type* elem, Value* ptr, Align align, Value* mask, const Twine& name);
    void scatter(Value* vals, Value* ptr, Align align, Value* mask, bool nontemporal = false);
    void store_chunk(Value* v, Value* base, Align align, Value* mask, bool nontemporal);
    // base + offset in bytes. When split_address found no base (null), the
    // address is the integer itself: a pointer made by a GEP on null has no
    // provenance, and LLVM treats a plain store through it as undefined.
    Value* byte_ptr(Value* base, Value* offset) {
        if (isa<ConstantPointerNull>(base)) return b_.CreateIntToPtr(offset, base->getType());
        return b_.CreateGEP(b_.getInt8Ty(), base, offset);
    }
    // Non-temporal stores are weakly ordered on x86: before the block ends (the
    // next kernel may read them from another core) or meets its SMT sibling at
    // a barrier, an sfence makes them visible.
    void sfence_if_streaming() {
        for (Instruction& i : instructions(f_))
            if (isa<StoreInst>(i) && i.hasMetadata(LLVMContext::MD_nontemporal)) {
                b_.CreateCall(Intrinsic::getDeclaration(&m_, Intrinsic::x86_sse_sfence));
                return;
            }
    }
    Value* half(Value* v, unsigned h);
    // A masked gather. AVX-512 gathers 32 and 64-bit elements only; LLVM
    // turns a gather of bytes into one load per lane. A byte (or an aligned
    // 16-bit value) is gathered instead as the aligned 32-bit word holding it,
    // then shifted out: that word never crosses a page, so it is readable
    // whenever the byte is.
    Value* masked_gather(Type* vty, Value* ptrs, Align align, Value* mask) {
        auto* vt = cast<FixedVectorType>(vty);
        Type* et = vt->getElementType();
        const unsigned bits = et->getPrimitiveSizeInBits();
        if (!(bits == 8 || (bits == 16 && align.value() >= 2)))
            return b_.CreateMaskedGather(vty, ptrs, align, mask, PoisonValue::get(vty));
        const unsigned n = vt->getNumElements();
        auto* i64v = FixedVectorType::get(b_.getInt64Ty(), n);
        auto* i32v = FixedVectorType::get(b_.getInt32Ty(), n);
        Value* p = b_.CreatePtrToInt(ptrs, i64v);
        Value* word = b_.CreateIntToPtr(b_.CreateAnd(p, ConstantInt::get(i64v, ~uint64_t(3))),
                                        FixedVectorType::get(PointerType::get(ctx_, 0), n));
        Value* shift = b_.CreateTrunc(b_.CreateShl(b_.CreateAnd(p, ConstantInt::get(i64v, 3)), ConstantInt::get(i64v, 3)), i32v);
        Value* w = b_.CreateMaskedGather(i32v, word, Align(4), mask, PoisonValue::get(i32v));
        Value* v = b_.CreateTrunc(b_.CreateLShr(w, shift), FixedVectorType::get(b_.getIntNTy(bits), n));
        return et->isIntegerTy() ? v : b_.CreateBitCast(v, vty);
    }
    // One 16-lane chunk of a gather or scatter at addr.base + offs (byte
    // offsets, <16 x i64>): with 32-bit indices from addr.root when they are
    // known to fit, else with the 64-bit offsets (see gather16 below).
    Value* gather16(Type* v16, const SplitAddress& addr, Value* offs, Align align, Value* mask);
    void scatter16(Value* v, const SplitAddress& addr, Value* offs, Align align, Value* mask);
    bool narrow_fits(Type* elem, Align align);
    Value* narrow_index(const SplitAddress& addr, Value* offs);
    Value* gather16_narrow(Type* v16, const SplitAddress& addr, Value* offs, Value* mask);
    void scatter16_narrow(Value* v, const SplitAddress& addr, Value* offs, Value* mask);
    Value* root_is_small(Value* root);
    Value* concat_chunks(const std::vector<Value*>& parts);
    void fail(const std::string& what, const Value* v = nullptr) {
        if (error_.empty()) error_ = what + (v ? ": " + describe(v) : "");
    }

    void count(Instruction& inst, uint64_t* counts);
    void emit_counters(const uint64_t* counts, Value* mask);

    const unsigned W;
    const bool block_wide_;
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

    Function* fn_ = nullptr;           // function mode: the declaration being defined
    AllocaInst* ret_slot_ = nullptr;   // function mode: the lanes' return values
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
    void emit_class_tid();
    Value* emit_libmvec(CallInst& call, const std::string& name, unsigned arity);
    Value* emit_shfl(Value* v, Value* src);
    Value* emit_uniform_division(BinaryOperator* bin);
    Value* emit_ballot(Value* pred, Value* mask);
    bool emit_mma(CallInst* call, Value* mask);
    Value* emit_mmav(CallInst* call);
    std::map<std::vector<Value*>, std::array<Value*, 8>> mmav_done_;  // one inline mma per operand set
    AllocaInst* mma_scratch_ = nullptr;
    // AMX for bf16 mma (decided when compiling, from the function's target
    // features: nothing is checked at run time). Opt-in (CPUSIM_AMX=1 for
    // cucc): with the accumulator moved between the tile and zmm registers
    // around every mma, a 4096^3 GEMM on a Xeon 6548Y+ took 4.6 TFLOPS
    // against 5.4 with vdpbf16ps; it pays only with the accumulator kept in
    // a tile through the loop over k (docs/optimization/iteration-3/34).
    bool amx_bf16() const {
        static const bool wanted = std::getenv("CPUSIM_AMX") && std::atoi(std::getenv("CPUSIM_AMX")) != 0;
        return wanted && f_.getFnAttribute("target-features").getValueAsString().contains("+amx-bf16");
    }
    // p through an empty asm that also clobbers memory: LLVM no longer knows
    // what it points to and cannot move loads above it, so it neither
    // forwards earlier stores to loads through it nor promotes the memory to
    // registers; the loads themselves stay ordinary.
    Value* launder(Value* p) {
        return b_.CreateCall(
            InlineAsm::get(FunctionType::get(p->getType(), {p->getType()}, false), "", "=r,0,~{memory}", true), {p});
    }
    // 2 KiB on the stack for an inline mma's converted A (fp16: 32 x 16
    // floats at most) or row sums (int8).
    Value* mma_scratch() {
        if (!mma_scratch_) {
            IRBuilder<> eb(&w_->getEntryBlock(), w_->getEntryBlock().begin());
            mma_scratch_ = eb.CreateAlloca(ArrayType::get(b_.getInt8Ty(), 2048), nullptr, "mma.scratch");
            mma_scratch_->setAlignment(Align(64));
        }
        return mma_scratch_;
    }

    // Barriers (DESIGN.md 4.5): each __syncthreads is a block of its own; the
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
        uint64_t& slot = W <= k_.warp_size ? k_.frame_bytes : k_.frame_bytes_wide[W == 2 * k_.warp_size ? 0 : 1];
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
    DenseSet<const BasicBlock*> yield_blocks_;  // barrier blocks that are yield points
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


// A call to a device function that was not inlined (recursive), or through
// a pointer.
bool is_device_call(const CallInst* call);
// A function of the process the kernel runs in (malloc, free, the device
// printf calls): called once per active lane with that lane's arguments.
bool is_external_call(const CallInst* call);
// Versions of device functions declared by vector_fn and not built yet.
struct PendingFunction {
    Function* decl;
    Function* scalar;
    unsigned lanes;
};
std::vector<PendingFunction>& pending_functions();
bool is_barrier(const Instruction& inst);
// The libmvec name of a math call ("expf"), and its number of arguments;
// empty when there is none (simt.cpp).
std::string libmvec_name(const CallInst& call, unsigned* arity);

}  // namespace simt_internal
}  // namespace cpusim
