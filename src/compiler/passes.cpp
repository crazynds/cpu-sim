// The IR passes of cpusim-devc around the SIMT pass: lowering of what the
// per-thread code calls (printf, malloc, atomics, device globals, barrier
// votes), clean-ups of the warp code, and what the variant choice reads
// (kernel_features). The order they run in is in main.cpp.
#include "compiler/passes.h"

#include <cstdio>
#include <map>
#include <algorithm>
#include <string>
#include <vector>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>

#include "compiler/simt.h"
#include "compiler/uniformity.h"
#include "device/kernel_abi.h"

using namespace llvm;

namespace cpusim {

// Works around an LLVM 19 code generator hang (an endless DAG combine):
//     %m = select i1 %c, <128 x i1> %a, <128 x i1> %b
// followed by bitcasts of %m, as the masks of 128-lane and wider variants get
// after O3. The same select on the integer the mask is made of is fine. Runs
// after the optimizer, which would turn it back.
void widen_mask_selects(Module& m) {
    for (Function& f : m)
        for (BasicBlock& bb : f)
            for (Instruction& inst : make_early_inc_range(bb)) {
                auto* sel = dyn_cast<SelectInst>(&inst);
                auto* vt = sel ? dyn_cast<FixedVectorType>(sel->getType()) : nullptr;
                if (!vt || !vt->getElementType()->isIntegerTy(1) || vt->getNumElements() <= 64 ||
                    sel->getCondition()->getType()->isVectorTy())
                    continue;
                IRBuilder<> b(sel);
                Type* it = b.getIntNTy(vt->getNumElements());
                Value* r = b.CreateSelect(sel->getCondition(), b.CreateBitCast(sel->getTrueValue(), it),
                                          b.CreateBitCast(sel->getFalseValue(), it));
                sel->replaceAllUsesWith(b.CreateBitCast(r, vt));
                sel->eraseFromParent();
            }
}

// Masks under a uniform guard. The warp code tests a uniform condition c by
// giving the taken side the mask select(c, m, 0) and running a block only if
// any lane of its mask is set. Inside, c is then known to hold, so the mask
// is m (often a constant: all lanes), but LLVM does not see it: the select
// sits before the branch. Here, for each edge of a conditional branch, the
// scalar conditions known to hold along it (through `and`/`or`, `not`, and
// "some lane of select(c, x, 0) is set") turn select(c, x, y) into x in the
// code that edge dominates. Masked loads and stores with a constant mask then
// become plain ones. Run on the warp code after SROA and InstCombine.
namespace {
// The value a mask M has whenever some lane of it is set, if there is one:
// x for select(c, x, 0) (c then holds, recorded in `holds`), and for a phi
// whose incoming values are each zero (or poison) or reduce to the same x.
Value* when_nonzero(Value* m, SmallPtrSetImpl<Value*>& holds, unsigned depth = 0) {
    using namespace PatternMatch;
    Value *c, *x;
    if (depth > 6) return nullptr;
    if (match(m, m_Select(m_Value(c), m_Value(x), m_Zero())) && !c->getType()->isVectorTy()) {
        holds.insert(c);
        Value* inner = when_nonzero(x, holds, depth + 1);
        return inner ? inner : x;
    }
    if (auto* phi = dyn_cast<PHINode>(m)) {
        Value* common = nullptr;
        for (Value* in : phi->incoming_values()) {
            if (auto* k = dyn_cast<Constant>(in); k && (k->isNullValue() || isa<UndefValue>(k))) continue;
            Value* r = when_nonzero(in, holds, depth + 1);
            if (!r) r = in;
            if (common && common != r) return nullptr;
            common = r;
        }
        return common;
    }
    return nullptr;
}

struct Facts {
    SmallPtrSet<Value*, 8> holds;          // scalar conditions known true
    SmallVector<std::pair<Value*, Value*>, 4> masks;  // mask -> its value
};

void implied(Value* v, bool val, Facts& f, unsigned depth = 0) {
    using namespace PatternMatch;
    if (depth > 8 || !v->getType()->isIntegerTy(1)) return;
    if (val) f.holds.insert(v);
    Value *a, *b, *m;
    if (val && match(v, m_LogicalAnd(m_Value(a), m_Value(b)))) {
        implied(a, true, f, depth + 1), implied(b, true, f, depth + 1);
    } else if (!val && match(v, m_LogicalOr(m_Value(a), m_Value(b)))) {
        implied(a, false, f, depth + 1), implied(b, false, f, depth + 1);
    } else if (match(v, m_Not(m_Value(a)))) {
        implied(a, !val, f, depth + 1);
    } else if (auto* cmp = dyn_cast<ICmpInst>(v); cmp && match(cmp, m_ICmp(m_BitCast(m_Value(m)), m_Zero())) &&
                                                  m->getType()->isVectorTy()) {
        // Some lane of m is set.
        if ((cmp->getPredicate() == ICmpInst::ICMP_NE) == val)
            if (Value* x = when_nonzero(m, f.holds)) f.masks.push_back({m, x});
    }
}
}  // namespace

bool refine_uniform_masks(Function& f) {
    DominatorTree dt(f);
    bool changed = false;
    auto replace = [&](Value* from, Value* to, const BasicBlockEdge& edge) {
        auto* def = dyn_cast<Instruction>(to);
        for (Use& use : make_early_inc_range(from->uses()))
            if (dt.dominates(edge, use) && (!def || dt.dominates(def, use))) {
                use.set(to);
                changed = true;
            }
    };
    for (BasicBlock& bb : f) {
        auto* br = dyn_cast<BranchInst>(bb.getTerminator());
        if (!br || !br->isConditional() || br->getSuccessor(0) == br->getSuccessor(1)) continue;
        for (unsigned side = 0; side < 2; ++side) {
            Facts facts;
            implied(br->getCondition(), side == 0, facts);
            BasicBlockEdge edge(&bb, br->getSuccessor(side));
            for (auto& [m, x] : facts.masks) replace(m, x, edge);
            for (Value* c : facts.holds)
                for (User* u : make_early_inc_range(c->users()))
                    if (auto* sel = dyn_cast<SelectInst>(u);
                        sel && sel->getCondition() == c && sel->getType()->isVectorTy())
                        replace(sel, sel->getTrueValue(), edge);
        }
    }
    return changed;
}

// memcpy, memmove and memset as plain loads and stores, which the SIMT pass
// widens like any other access: struct copies (`a[i] = p`, `float4` copies
// through a reinterpret_cast), and the copies to or from a local that SROA
// then turns into registers. Up to 256 bytes, in 4-byte pieces when the size
// allows (then 2, then 1); longer or variable sizes become a loop.
bool expand_mem_intrinsics(Function& f, TargetMachine* tm) {
    bool changed = false;
    std::vector<MemIntrinsic*> calls;
    for (Instruction& inst : instructions(f))
        if (auto* mi = dyn_cast<MemIntrinsic>(&inst)) calls.push_back(mi);
    for (MemIntrinsic* mi : calls) {
        auto* len = dyn_cast<ConstantInt>(mi->getLength());
        const bool vol = mi->isVolatile();
        if (!len || len->getZExtValue() > 256) {
            TargetTransformInfo tti = tm->getTargetTransformInfo(f);
            if (auto* cpy = dyn_cast<MemCpyInst>(mi)) expandMemCpyAsLoop(cpy, tti);
            else if (auto* mv = dyn_cast<MemMoveInst>(mi)) {
                if (!expandMemMoveAsLoop(mv, tti)) continue;
            } else if (auto* ms = dyn_cast<MemSetInst>(mi)) expandMemSetAsLoop(ms);
            else continue;
            mi->eraseFromParent();
            changed = true;
            continue;
        }
        const uint64_t n = len->getZExtValue();
        const uint64_t unit = n % 4 == 0 ? 4 : n % 2 == 0 ? 2 : 1;
        IRBuilder<> b(mi);
        Type* t = b.getIntNTy(unsigned(unit * 8));
        const Align da = mi->getDestAlign().valueOrOne();
        auto dst = [&](uint64_t off) { return b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), mi->getDest(), off); };
        if (auto* ms = dyn_cast<MemSetInst>(mi)) {
            Value* byte = b.CreateZExt(ms->getValue(), t);
            Value* v = byte;
            for (uint64_t k = 1; k < unit; ++k) v = b.CreateOr(v, b.CreateShl(byte, k * 8));
            for (uint64_t off = 0; off < n; off += unit) b.CreateAlignedStore(v, dst(off), commonAlignment(da, off), vol);
        } else {
            auto* mt = cast<MemTransferInst>(mi);
            const Align sa = mt->getSourceAlign().valueOrOne();
            // All loads first: a memmove's ranges may overlap.
            std::vector<Value*> vals;
            for (uint64_t off = 0; off < n; off += unit)
                vals.push_back(b.CreateAlignedLoad(t, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), mt->getSource(), off),
                                                   commonAlignment(sa, off), vol));
            for (uint64_t off = 0, k = 0; off < n; off += unit, ++k)
                b.CreateAlignedStore(vals[k], dst(off), commonAlignment(da, off), vol);
        }
        mi->eraseFromParent();
        changed = true;
    }
    return changed;
}

// NVVM's atomic increment and decrement become LLVM's uinc_wrap and
// udec_wrap, and its memory barriers full fences, which x86 code generation
// knows.
void lower_nvvm_atomics(Module& m) {
    std::vector<IntrinsicInst*> calls;
    for (Function& f : m)
        for (Instruction& i : instructions(f))
            if (auto* ii = dyn_cast<IntrinsicInst>(&i)) switch (ii->getIntrinsicID()) {
                    case Intrinsic::nvvm_atomic_load_inc_32:
                    case Intrinsic::nvvm_atomic_load_dec_32:
                    case Intrinsic::nvvm_membar_cta:
                    case Intrinsic::nvvm_membar_gl:
                    case Intrinsic::nvvm_membar_sys:
                        calls.push_back(ii);
                        break;
                    default:
                        break;
                }
    for (IntrinsicInst* ii : calls) {
        IRBuilder<> b(ii);
        Intrinsic::ID id = ii->getIntrinsicID();
        if (id == Intrinsic::nvvm_atomic_load_inc_32 || id == Intrinsic::nvvm_atomic_load_dec_32) {
            auto op = id == Intrinsic::nvvm_atomic_load_inc_32 ? AtomicRMWInst::UIncWrap : AtomicRMWInst::UDecWrap;
            Value* r = b.CreateAtomicRMW(op, ii->getArgOperand(0), ii->getArgOperand(1), MaybeAlign(4),
                                         AtomicOrdering::Monotonic);
            ii->replaceAllUsesWith(r);
        } else {
            b.CreateFence(AtomicOrdering::SequentiallyConsistent);
        }
        ii->eraseFromParent();
    }
}

// __device__ and __constant__ variables (every global the device code can
// write, or the host can, through cudaMemcpyToSymbol) live in the VRAM, one
// copy per program, as on a GPU: every module of the program (the generic one
// and each block shape's) reaches them through a pointer of its own,
// `__cpusim_sym_<name>`, which the runtime sets when it loads the module.
// `__cpusim_init_<name>` keeps the initial value, copied to the VRAM when the
// generic module is loaded. Returns the manifest lines: "symbol <name> <size>
// <alignment> [managed [initial bytes in hex]]".
// The bytes of a constant without relocations (numbers, arrays and structs of
// them, zero), as laid out in memory; false if it has an address in it.
bool constant_bytes(const DataLayout& dl, const Constant* c, uint8_t* out) {
    if (isa<ConstantAggregateZero>(c) || isa<UndefValue>(c)) return true;  // the buffer starts zeroed
    if (auto* i = dyn_cast<ConstantInt>(c)) {
        const APInt& v = i->getValue();
        for (unsigned b = 0; b < dl.getTypeStoreSize(c->getType()); ++b) out[b] = uint8_t(v.extractBitsAsZExtValue(8, b * 8));
        return true;
    }
    if (auto* f = dyn_cast<ConstantFP>(c)) {
        APInt v = f->getValueAPF().bitcastToAPInt();
        for (unsigned b = 0; b < dl.getTypeStoreSize(c->getType()); ++b) out[b] = uint8_t(v.extractBitsAsZExtValue(8, b * 8));
        return true;
    }
    if (auto* d = dyn_cast<ConstantDataSequential>(c)) {
        StringRef raw = d->getRawDataValues();
        std::memcpy(out, raw.data(), raw.size());
        return true;
    }
    if (auto* st = dyn_cast<ConstantStruct>(c)) {
        const StructLayout* sl = dl.getStructLayout(st->getType());
        for (unsigned i = 0; i < st->getNumOperands(); ++i)
            if (!constant_bytes(dl, st->getOperand(i), out + sl->getElementOffset(i))) return false;
        return true;
    }
    if (auto* a = dyn_cast<ConstantArray>(c)) {
        const uint64_t step = dl.getTypeAllocSize(a->getType()->getElementType());
        for (unsigned i = 0; i < a->getNumOperands(); ++i)
            if (!constant_bytes(dl, a->getOperand(i), out + i * step)) return false;
        return true;
    }
    return false;
}

std::string lower_device_globals(Module& m) {
    LLVMContext& ctx = m.getContext();
    const DataLayout& dl = m.getDataLayout();
    std::vector<GlobalVariable*> syms;
    for (GlobalVariable& g : m.globals()) {
        if (g.isDeclaration() || g.isConstant() || g.getAddressSpace() == 3 || !g.hasName()) continue;
        if (g.getName().starts_with("llvm.") || g.getName().starts_with("__cpusim")) continue;
        syms.push_back(&g);
    }
    if (syms.empty()) return "";
    // Clang lists them in llvm.compiler.used to keep them; they go away here.
    for (const char* n : {"llvm.used", "llvm.compiler.used"})
        if (GlobalVariable* u = m.getNamedGlobal(n)) u->eraseFromParent();
    std::string lines;
    MDNode* empty = MDNode::get(ctx, {});
    for (GlobalVariable* g : syms) {
        const std::string name = g->getName().str();
        const bool managed = g->getSection() == "cpusim_managed";  // __managed__ (device header)
        Type* t = g->getValueType();
        const uint64_t size = dl.getTypeAllocSize(t);
        const uint64_t align = std::max<uint64_t>(g->getAlign().valueOrOne().value(), dl.getABITypeAlign(t).value());
        // Weak: the objects of one module compiled kernel by kernel (cucc)
        // each define them; the link keeps one.
        auto* init = new GlobalVariable(m, t, true, GlobalValue::WeakODRLinkage,
                                        g->hasInitializer() ? g->getInitializer() : Constant::getNullValue(t),
                                        "__cpusim_init_" + name);
        init->setVisibility(GlobalValue::ProtectedVisibility);
        init->setAlignment(Align(align));
        auto* pty = PointerType::get(ctx, g->getAddressSpace());
        auto* slot = new GlobalVariable(m, pty, false, GlobalValue::WeakAnyLinkage, ConstantPointerNull::get(pty),
                                        "__cpusim_sym_" + name);
        slot->setVisibility(GlobalValue::ProtectedVisibility);
        convertUsersOfConstantsToInstructions({g});
        DenseMap<Function*, Value*> loads;
        for (Use& u : make_early_inc_range(g->uses())) {
            auto* i = dyn_cast<Instruction>(u.getUser());
            if (!i) continue;
            Value*& l = loads[i->getFunction()];
            if (!l) {
                IRBuilder<> b(&*i->getFunction()->getEntryBlock().getFirstInsertionPt());
                LoadInst* ld = b.CreateAlignedLoad(pty, slot, Align(8), name);
                ld->setMetadata(LLVMContext::MD_invariant_load, empty);
                ld->setMetadata(LLVMContext::MD_nonnull, empty);
                l = ld;
            }
            u.set(l);
        }
        // Still referenced from another global's initializer: the initial copy.
        if (!g->use_empty()) g->replaceAllUsesWith(ConstantExpr::getPointerBitCastOrAddrSpaceCast(init, g->getType()));
        g->eraseFromParent();
        lines += "symbol " + name + " " + std::to_string(size) + " " + std::to_string(align);
        if (managed) {
            // The host copy starts from the same value (Clang leaves the host
            // side of a device variable without its initializer).
            lines += " managed";
            std::vector<uint8_t> bytes(size);
            if (init->hasInitializer() && !init->getInitializer()->isNullValue() &&
                constant_bytes(dl, init->getInitializer(), bytes.data())) {
                static const char* hex = "0123456789abcdef";
                lines += " ";
                for (uint8_t b : bytes) lines += hex[b >> 4], lines += hex[b & 15];
            }
        }
        lines += "\n";
    }
    return lines;
}

// __syncthreads_count / _and / _or as per-thread code around an ordinary
// barrier: each thread adds its vote to a counter in shared memory (one
// atomic per warp: the address is the same for every lane), the barrier, then
// every thread reads the total. Three counters in turn: the one the next call
// but one will use is cleared right after this call's barrier, when every
// thread has read it (before the previous barrier) and nobody adds to it yet
// (not before the next one). The kernel clears the first two at its start,
// followed by a barrier.
void lower_barrier_votes(Module& m, Function& kernel) {
    std::vector<IntrinsicInst*> votes;
    for (Instruction& i : instructions(kernel))
        if (auto* ii = dyn_cast<IntrinsicInst>(&i))
            if (ii->getIntrinsicID() == Intrinsic::nvvm_barrier0_popc ||
                ii->getIntrinsicID() == Intrinsic::nvvm_barrier0_and ||
                ii->getIntrinsicID() == Intrinsic::nvvm_barrier0_or)
                votes.push_back(ii);
    if (votes.empty()) return;
    LLVMContext& ctx = m.getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    auto* counters_ty = ArrayType::get(i32, 3);
    auto* counters = new GlobalVariable(m, counters_ty, false, GlobalValue::InternalLinkage, UndefValue::get(counters_ty),
                                        "__cpusim_barrier_votes." + kernel.getName(), nullptr,
                                        GlobalValue::NotThreadLocal, 3);
    counters->setAlignment(Align(16));
    Function* barrier = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_barrier0);
    auto counter = [&](IRBuilder<>& b, Value* index) {
        Value* p = b.CreateGEP(counters_ty, counters, {b.getInt32(0), index});
        return b.CreateAddrSpaceCast(p, PointerType::get(ctx, 0));
    };

    IRBuilder<> eb(&*kernel.getEntryBlock().getFirstInsertionPt());
    AllocaInst* generation = eb.CreateAlloca(i32, nullptr, "vote.generation");
    eb.CreateStore(eb.getInt32(0), generation);
    for (int k = 0; k < 2; ++k) eb.CreateStore(eb.getInt32(0), counter(eb, eb.getInt32(k)));
    eb.CreateCall(barrier);

    for (IntrinsicInst* v : votes) {
        IRBuilder<> b(v);
        const Intrinsic::ID id = v->getIntrinsicID();
        Value* gen = b.CreateLoad(i32, generation);
        b.CreateStore(b.CreateAdd(gen, b.getInt32(1)), generation);
        Value* slot = b.CreateURem(gen, b.getInt32(3));
        Value* nonzero = b.CreateICmpNE(v->getArgOperand(0), b.getInt32(0));
        // _and counts the threads whose vote fails.
        Value* one = b.CreateZExt(id == Intrinsic::nvvm_barrier0_and ? b.CreateNot(nonzero) : nonzero, i32);
        b.CreateAtomicRMW(AtomicRMWInst::Add, counter(b, slot), one, MaybeAlign(4), AtomicOrdering::Monotonic);
        b.CreateCall(barrier);
        Value* total = b.CreateLoad(i32, counter(b, slot));
        cast<LoadInst>(total)->setAtomic(AtomicOrdering::Monotonic);
        b.CreateStore(b.getInt32(0), counter(b, b.CreateURem(b.CreateAdd(gen, b.getInt32(2)), b.getInt32(3))));
        Value* r = id == Intrinsic::nvvm_barrier0_popc ? total
                   : id == Intrinsic::nvvm_barrier0_and ? b.CreateZExt(b.CreateICmpEQ(total, b.getInt32(0)), i32)
                                                        : b.CreateZExt(b.CreateICmpNE(total, b.getInt32(0)), i32);
        v->replaceAllUsesWith(r);
        v->eraseFromParent();
    }
}

// Device malloc refuses objects of 2 GiB or more (malloc(SIZE_MAX) returns
// null), as a GPU does with a heap below that size: kernels index every
// object outside the span of the huge allocations with 32-bit offsets
// (kernel_abi.h, cpusim_warp_ctx::huge_lo), and the runtime only knows the
// VRAM and shared pool ones.
void cap_device_malloc(Function& f) {
    for (Instruction& i : instructions(f)) {
        auto* call = dyn_cast<CallInst>(&i);
        Function* callee = call ? call->getCalledFunction() : nullptr;
        if (!callee || callee->getName() != "malloc" || call->arg_size() != 1) continue;
        IRBuilder<> b(call);
        Value* n = call->getArgOperand(0);
        auto* t = cast<IntegerType>(n->getType());
        Value* big = b.CreateICmpUGE(n, ConstantInt::get(t, uint64_t(1) << 31));
        call->setArgOperand(0, b.CreateSelect(big, ConstantInt::get(t, t->getMask()), n));
    }
}

// Half precision where the CPU has it (AVX512-FP16, decided when compiling):
// __hfma's emulation (the header's __cpusim_hfma_half, a sum rounded to odd
// in double) becomes the CPU's fma, one rounding as on the GPU.
void lower_half_fma(Module& m, bool fp16) {
    Function* f = m.getFunction("__cpusim_hfma_half");
    if (!f || !fp16) return;
    LLVMContext& ctx = m.getContext();
    Function* fma = Intrinsic::getDeclaration(&m, Intrinsic::fma, {Type::getHalfTy(ctx)});
    for (User* u : make_early_inc_range(f->users())) {
        auto* call = dyn_cast<CallInst>(u);
        if (!call || call->getCalledFunction() != f) continue;
        IRBuilder<> b(call);
        auto h = [&](Value* v) { return b.CreateBitCast(v, b.getHalfTy()); };
        Value* r = b.CreateCall(fma, {h(call->getArgOperand(0)), h(call->getArgOperand(1)), h(call->getArgOperand(2))});
        call->replaceAllUsesWith(b.CreateBitCast(r, b.getInt16Ty()));
        call->eraseFromParent();
    }
}

// fptrunc(sqrt(fpext h)) to half is the half-precision square root (float
// has more than twice half's precision, so the double rounding is exact):
// vsqrtph where the CPU has AVX512-FP16, the same promotion elsewhere.
void shrink_half_sqrt(Function& f) {
    for (Instruction& i : make_early_inc_range(instructions(f))) {
        auto* tr = dyn_cast<FPTruncInst>(&i);
        if (!tr || !tr->getType()->isHalfTy()) continue;
        auto* sq = dyn_cast<IntrinsicInst>(tr->getOperand(0));
        if (!sq || sq->getIntrinsicID() != Intrinsic::sqrt || !sq->getType()->isFloatTy()) continue;
        auto* ext = dyn_cast<FPExtInst>(sq->getArgOperand(0));
        if (!ext || !ext->getOperand(0)->getType()->isHalfTy()) continue;
        IRBuilder<> b(tr);
        Value* r = b.CreateUnaryIntrinsic(Intrinsic::sqrt, ext->getOperand(0));
        tr->replaceAllUsesWith(r);
        tr->eraseFromParent();
    }
}

// Device printf: Clang packs the arguments in a local struct and calls
// vprintf(format, &struct). Each call becomes
//   h = __cpusim_printf_begin(format); __cpusim_printf_<kind>(h, value)...;
//   __cpusim_printf_end(h)
// with every argument passed by value (integers widened to 64 bits with their
// size, doubles, pointers), calls the SIMT pass makes once per active lane.
void lower_printf(Function& f) {
    Module& m = *f.getParent();
    LLVMContext& ctx = m.getContext();
    std::vector<CallInst*> calls;
    for (Instruction& i : instructions(f))
        if (auto* c = dyn_cast<CallInst>(&i); c && c->getCalledFunction() && c->getCalledFunction()->getName() == "vprintf")
            calls.push_back(c);
    if (calls.empty()) return;
    Type* i64 = Type::getInt64Ty(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* ptr = PointerType::get(ctx, 0);
    FunctionCallee begin = m.getOrInsertFunction("__cpusim_printf_begin", i64, ptr);
    FunctionCallee arg_int = m.getOrInsertFunction("__cpusim_printf_int", Type::getVoidTy(ctx), i64, i64, i32);
    FunctionCallee arg_float =
        m.getOrInsertFunction("__cpusim_printf_float", Type::getVoidTy(ctx), i64, Type::getDoubleTy(ctx));
    FunctionCallee arg_ptr = m.getOrInsertFunction("__cpusim_printf_pointer", Type::getVoidTy(ctx), i64, ptr);
    FunctionCallee end = m.getOrInsertFunction("__cpusim_printf_end", i32, i64);
    const DataLayout& dl = m.getDataLayout();
    for (CallInst* call : calls) {
        // The stores into the argument struct, by offset.
        std::map<int64_t, StoreInst*> stores;
        std::vector<Instruction*> dead;
        auto* buf = dyn_cast<AllocaInst>(getUnderlyingObject(call->getArgOperand(1)));
        bool ok = isa<ConstantPointerNull>(call->getArgOperand(1)) || buf;
        if (buf) {
            SmallVector<std::pair<Value*, int64_t>, 8> work{{buf, 0}};
            while (ok && !work.empty()) {
                auto [v, off] = work.pop_back_val();
                for (User* u : v->users()) {
                    if (auto* g = dyn_cast<GetElementPtrInst>(u)) {
                        APInt o(64, 0);
                        if (!g->accumulateConstantOffset(dl, o)) ok = false;
                        else work.push_back({g, off + o.getSExtValue()}), dead.push_back(g);
                    } else if (auto* st = dyn_cast<StoreInst>(u); st && st->getPointerOperand() == v) {
                        stores[off] = st;
                    } else if (auto* ii = dyn_cast<IntrinsicInst>(u); ii && ii->isLifetimeStartOrEnd()) {
                        dead.push_back(ii);
                    } else if (u != call) {
                        ok = false;
                    }
                }
            }
        }
        if (!ok) continue;  // left alone: the SIMT pass reports it
        IRBuilder<> b(call);
        Value* h = b.CreateCall(begin, {call->getArgOperand(0)});
        for (auto& [off, st] : stores) {
            Value* v = st->getValueOperand();
            Type* t = v->getType();
            if (t->isPointerTy()) b.CreateCall(arg_ptr, {h, v});
            else if (t->isFloatingPointTy()) b.CreateCall(arg_float, {h, b.CreateFPExt(v, b.getDoubleTy())});
            else if (t->isIntegerTy())
                b.CreateCall(arg_int, {h, b.CreateSExtOrTrunc(v, i64), b.getInt32(int32_t(dl.getTypeStoreSize(t)))});
        }
        Value* r = b.CreateCall(end, {h});
        call->replaceAllUsesWith(r);
        call->eraseFromParent();
        for (auto& [off, st] : stores) st->eraseFromParent();
        for (auto it = dead.rbegin(); it != dead.rend(); ++it)
            if ((*it)->use_empty()) (*it)->eraseFromParent();
        if (buf && buf->use_empty()) buf->eraseFromParent();
    }
}

// Loops that wait for memory another thread writes: their exit depends on a
// volatile or atomic read made inside them (a flag, a lock). On a GPU the
// writer, in another warp or another branch of the same warp, runs meanwhile;
// here the warp's lanes run together and the warps of a block in turn. Each
// such loop gets a yield point at the end of its iterations (the warp
// suspends there and the others run), and the kernel keeps to its 32-lane
// variants (several warps in one vector could wait for each other).
bool mark_spin_loops(Function& f) {
    DominatorTree dt(f);
    LoopInfo li(dt);
    Module& m = *f.getParent();
    FunctionCallee yield = m.getOrInsertFunction(cpusim::kYieldFn, Type::getVoidTy(m.getContext()));
    if (auto* fn = dyn_cast<Function>(yield.getCallee())) {
        fn->addFnAttr(Attribute::Convergent);
        fn->addFnAttr(Attribute::NoUnwind);
    }
    bool any = false;
    for (Loop* l : li.getLoopsInPreorder()) {
        bool barrier = false;
        for (BasicBlock* bb : l->blocks())
            for (Instruction& i : *bb)
                if (auto* ii = dyn_cast<IntrinsicInst>(&i))
                    barrier |= ii->getIntrinsicID() == Intrinsic::nvvm_barrier0;
        if (barrier) continue;  // its barriers already let the other warps run
        // Does an exit condition depend, inside the loop, on such a read?
        SmallVector<BasicBlock*, 4> exiting;
        l->getExitingBlocks(exiting);
        SmallVector<Value*, 16> work;
        for (BasicBlock* bb : exiting)
            if (auto* br = dyn_cast<BranchInst>(bb->getTerminator()); br && br->isConditional())
                work.push_back(br->getCondition());
        SmallPtrSet<Value*, 32> seen;
        bool spins = false;
        while (!work.empty() && !spins) {
            auto* i = dyn_cast<Instruction>(work.pop_back_val());
            if (!i || !l->contains(i) || !seen.insert(i).second) continue;
            if (auto* ld = dyn_cast<LoadInst>(i); ld && (ld->isVolatile() || ld->isAtomic())) spins = true;
            else if (isa<AtomicRMWInst>(i) || isa<AtomicCmpXchgInst>(i)) spins = true;
            else
                for (Value* op : i->operands()) work.push_back(op);
        }
        if (!spins) continue;
        BasicBlock* latch = l->getLoopLatch();
        if (!latch) continue;
        CallInst::Create(yield, "", latch->getTerminator()->getIterator());
        any = true;
    }
    return any;
}

// Live values are counted with the liveness of the per-thread code; see
// Features in passes.h.
Features kernel_features(Function& f) {
    Features r;
    DominatorTree dt(f);
    PostDominatorTree pdt(f);
    LoopInfo li(dt);
    cpusim::Uniformity uni(f, dt, pdt, li);
    const DataLayout& dl = f.getParent()->getDataLayout();
    auto units = [&](const Value* v) -> unsigned {
        if (!uni.is_varying(v) || v->getType()->isVoidTy() || !v->getType()->isSized()) return 0;
        return unsigned((dl.getTypeSizeInBits(v->getType()).getFixedValue() + 31) / 32);
    };
    // Live-out sets by iteration over the blocks, then each block backwards.
    DenseMap<const BasicBlock*, SmallPtrSet<const Value*, 32>> live_in, live_out;
    for (bool changed = true; changed;) {
        changed = false;
        for (BasicBlock& bb : reverse(f)) {
            SmallPtrSet<const Value*, 32> out;
            for (BasicBlock* s : successors(&bb)) {
                for (const Value* v : live_in[s]) out.insert(v);
                for (PHINode& phi : s->phis())
                    if (auto* in = phi.getIncomingValueForBlock(&bb); in && (isa<Instruction>(in) || isa<Argument>(in)))
                        out.insert(in);
            }
            SmallPtrSet<const Value*, 32> in = out;
            for (Instruction& i : reverse(bb)) {
                in.erase(&i);
                if (isa<PHINode>(i)) continue;
                for (const Use& u : i.operands())
                    if (isa<Instruction>(u.get()) || isa<Argument>(u.get())) in.insert(u.get());
            }
            if (in.size() != live_in[&bb].size()) {
                live_in[&bb] = in;
                changed = true;
            }
            live_out[&bb] = std::move(out);
        }
    }
    for (BasicBlock& bb : f) {
        SmallPtrSet<const Value*, 32> live = live_out[&bb];
        unsigned now = 0;
        for (const Value* v : live) now += units(v);
        r.live_units = std::max(r.live_units, now);
        for (Instruction& i : reverse(bb)) {
            if (live.erase(&i)) now -= units(&i);
            if (!isa<PHINode>(i))
                for (const Use& u : i.operands())
                    if ((isa<Instruction>(u.get()) || isa<Argument>(u.get())) && live.insert(u.get()).second)
                        now += units(u.get());
            r.live_units = std::max(r.live_units, now);
            ++r.instructions;
            if (isa<LoadInst>(i)) ++r.loads;
            if (isa<StoreInst>(i)) ++r.stores;
            if (i.getOpcode() == Instruction::FDiv || i.getOpcode() == Instruction::SDiv ||
                i.getOpcode() == Instruction::UDiv || i.getOpcode() == Instruction::SRem || i.getOpcode() == Instruction::URem)
                ++r.slow_ops;
            if (auto* c = dyn_cast<CallInst>(&i); c && c->getCalledFunction() && !isa<IntrinsicInst>(c)) ++r.slow_ops;
            if (auto* ii = dyn_cast<IntrinsicInst>(&i); ii && (ii->getIntrinsicID() == Intrinsic::sqrt ||
                                                               ii->getIntrinsicID() == Intrinsic::exp ||
                                                               ii->getIntrinsicID() == Intrinsic::log ||
                                                               ii->getIntrinsicID() == Intrinsic::pow))
                ++r.slow_ops;
        }
    }
    r.loops = unsigned(li.getLoopsInPreorder().size());
    return r;
}

// Loops with a barrier stay loops. Unrolled, every copy of the barrier is
// one more place where a warp suspends and resumes, and the function grows
// with each: HeCBench sc (a loop of 32 prefix sums with barriers) took 8
// minutes to compile, nearly all of it in register allocation, and then
// hung; kept rolled it compiles in 3 s and runs. A barrier costs the same
// inside a loop as outside.
void keep_barrier_loops_rolled(Function& f) {
    DominatorTree dt(f);
    LoopInfo li(dt);
    for (Loop* l : li.getLoopsInPreorder()) {
        bool barrier = false;
        for (BasicBlock* bb : l->blocks())
            for (Instruction& i : *bb)
                if (auto* ii = dyn_cast<IntrinsicInst>(&i)) {
                    const Intrinsic::ID id = ii->getIntrinsicID();
                    barrier |= id == Intrinsic::nvvm_barrier0 || id == Intrinsic::nvvm_bar_sync ||
                               id == Intrinsic::nvvm_barrier || id == Intrinsic::nvvm_barrier_n;
                }
        if (barrier) l->setLoopAlreadyUnrolled();
    }
}

bool has_alloca(const Function& f) {
    for (const BasicBlock& bb : f)
        for (const Instruction& i : bb)
            if (isa<AllocaInst>(i)) return true;
    return false;
}

}  // namespace cpusim
