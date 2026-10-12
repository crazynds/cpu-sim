// cpusim-devc: device compiler back end. Takes the per-thread LLVM IR that
// Clang produces for the device side of a .cu file and emits a native x86
// object with one warp entry point per kernel, plus the module manifest.
//
//   cpusim-devc device.bc -o kernels.o --manifest kernels.manifest [--emit-ll out.ll] [--mcpu native|znver4|...]
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>

#include "compiler/simt.h"
#include "device/kernel_abi.h"

using namespace llvm;
using cpusim::KernelInfo;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "cpusim-devc: error: %s\n", msg.c_str());
    std::exit(1);
}

struct Options {
    std::string input, output, manifest, emit_ll, emit_scalar_ll, emit_raw_ll, emit_asm;
    std::string cpu = "native";  // the CPU this runs on (the JIT runs where the kernels run)
    uint32_t specialize[3] = {0, 0, 0};
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "-o") o.output = value();
        else if (a == "--manifest") o.manifest = value();
        else if (a == "--emit-ll") o.emit_ll = value();
        else if (a == "--emit-scalar-ll") o.emit_scalar_ll = value();
        else if (a == "--emit-raw-ll") o.emit_raw_ll = value();  // warp code before optimization
        else if (a == "--emit-asm") o.emit_asm = value();        // x86 assembly of the final code
        else if (a == "--mcpu") o.cpu = value();
        else if (a == "--specialize") {
            // Block shape known: blockDim becomes a constant, threadIdx mostly too.
            std::string v = value();
            if (std::sscanf(v.c_str(), "%u,%u,%u", &o.specialize[0], &o.specialize[1], &o.specialize[2]) != 3)
                die("--specialize expects X,Y,Z");
        }
        else if (!a.empty() && a[0] == '-') die("unknown option " + a);
        else o.input = a;
    }
    if (o.input.empty() || o.output.empty() || o.manifest.empty())
        die("usage: cpusim-devc device.bc -o kernels.o --manifest kernels.manifest");
    if (o.cpu == "native") o.cpu = sys::getHostCPUName().str();
    return o;
}

// Kernels are listed in !nvvm.annotations as {ptr @f, !"kernel", i32 1}.
std::vector<KernelInfo> find_kernels(Module& m) {
    std::vector<KernelInfo> kernels;
    if (NamedMDNode* md = m.getNamedMetadata("nvvm.annotations")) {
        for (MDNode* node : md->operands()) {
            if (node->getNumOperands() < 3) continue;
            auto* kind = dyn_cast<MDString>(node->getOperand(1));
            if (!kind || kind->getString() != "kernel") continue;
            auto* fn = mdconst::dyn_extract_or_null<Function>(node->getOperand(0));
            if (fn && !fn->isDeclaration()) {
                KernelInfo k;
                k.name = fn->getName().str();
                k.scalar = fn;
                kernels.push_back(k);
            }
        }
    }
    return kernels;
}

void write_ll(Module& m, const std::string& path) {
    std::error_code ec;
    raw_fd_ostream out(path, ec, sys::fs::OF_Text);
    if (ec) die(path + ": " + ec.message());
    m.print(out, nullptr);
}

void run_pipeline(Module& m, TargetMachine* tm, const std::string& pipeline) {
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PipelineTuningOptions pto;
    PassBuilder pb(tm, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    ModulePassManager mpm;
    if (pipeline == "O3") {
        mpm = pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    } else if (auto err = pb.parsePassPipeline(mpm, pipeline)) {
        die("bad pass pipeline: " + toString(std::move(err)));
    }
    mpm.run(m, mam);
}

// The same, for one function ("loop(...),sroa,..." without the function()).
void run_function_pipeline(Function& f, TargetMachine* tm, const std::string& pipeline) {
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassBuilder pb(tm, PipelineTuningOptions());
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    FunctionPassManager fpm;
    if (auto err = pb.parsePassPipeline(fpm, pipeline)) die("bad pass pipeline: " + toString(std::move(err)));
    fpm.run(f, fam);
}

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
// <alignment>".
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
        Type* t = g->getValueType();
        const uint64_t size = dl.getTypeAllocSize(t);
        const uint64_t align = std::max<uint64_t>(g->getAlign().valueOrOne().value(), dl.getABITypeAlign(t).value());
        auto* init = new GlobalVariable(m, t, true, GlobalValue::ExternalLinkage,
                                        g->hasInitializer() ? g->getInitializer() : Constant::getNullValue(t),
                                        "__cpusim_init_" + name);
        init->setVisibility(GlobalValue::ProtectedVisibility);
        init->setAlignment(Align(align));
        auto* pty = PointerType::get(ctx, g->getAddressSpace());
        auto* slot = new GlobalVariable(m, pty, false, GlobalValue::ExternalLinkage, ConstantPointerNull::get(pty),
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
        lines += "symbol " + name + " " + std::to_string(size) + " " + std::to_string(align) + "\n";
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

bool has_alloca(const Function& f) {
    for (const BasicBlock& bb : f)
        for (const Instruction& i : bb)
            if (isa<AllocaInst>(i)) return true;
    return false;
}

// The device IR targets NVPTX. Same pointer size and type layout as x86-64 for
// everything CUDA code uses, so switching the target is enough.
void retarget(Module& m, TargetMachine* tm, const std::vector<KernelInfo>& kernels) {
    m.setTargetTriple(tm->getTargetTriple().str());
    m.setDataLayout(tm->createDataLayout());
    if (NamedMDNode* md = m.getNamedMetadata("nvvm.annotations")) m.eraseNamedMetadata(md);
    if (NamedMDNode* md = m.getNamedMetadata("nvvmir.version")) m.eraseNamedMetadata(md);

    // The module becomes a shared object: a global another module could
    // preempt (vtables and other linkonce data, __device__ variables) cannot
    // be addressed PC-relative. Protected keeps them visible to dlsym
    // (cudaMemcpyToSymbol) without making them preemptible.
    for (GlobalVariable& g : m.globals())
        if (!g.isDeclaration() && !g.hasLocalLinkage()) g.setVisibility(GlobalValue::ProtectedVisibility);
    for (Function& f : m) {
        f.removeFnAttr("target-cpu");
        f.removeFnAttr("target-features");
        f.removeFnAttr("uniform-work-group-size");
        f.addFnAttr("target-cpu", tm->getTargetCPU());
        f.addFnAttr("target-features", tm->getTargetFeatureString());
        if (f.isDeclaration()) continue;
        bool is_kernel = false;
        for (const auto& k : kernels) is_kernel |= k.scalar == &f;
        // Every device function is inlined into the kernels (DESIGN.md 5.6).
        f.removeFnAttr(Attribute::NoInline);
        f.removeFnAttr(Attribute::OptimizeNone);
        if (!is_kernel) {
            f.addFnAttr(Attribute::AlwaysInline);
            f.setLinkage(GlobalValue::InternalLinkage);
        }
    }
}

// __cpusim_block_<name>: the barrier rounds of one block in generated code,
// with the warp entry inlined, so switching warps at a barrier costs a loop
// iteration instead of a return, a call and a function prologue.
void build_block_function(Module& m, KernelInfo& k) {
    LLVMContext& ctx = m.getContext();
    auto* i32 = Type::getInt32Ty(ctx);
    auto* i8 = Type::getInt8Ty(ctx);
    auto* ptr = PointerType::get(ctx, 0);
    Function* f = Function::Create(FunctionType::get(i32, {ptr, i32, ptr}, false), GlobalValue::ExternalLinkage,
                                   "__cpusim_block_" + k.name, m);
    f->addFnAttr(Attribute::NoUnwind);
    for (const char* a : {"target-cpu", "target-features", "prefer-vector-width", "min-legal-vector-width"})
        if (k.entry->hasFnAttribute(a)) f->addFnAttr(k.entry->getFnAttribute(a));
    Argument* ctxs = f->getArg(0);
    Argument* n = f->getArg(1);
    Argument* done = f->getArg(2);

    BasicBlock* entry = BasicBlock::Create(ctx, "entry", f);
    BasicBlock* round = BasicBlock::Create(ctx, "round", f);
    BasicBlock* warp = BasicBlock::Create(ctx, "warp", f);
    BasicBlock* run = BasicBlock::Create(ctx, "run", f);
    BasicBlock* next = BasicBlock::Create(ctx, "next", f);
    BasicBlock* end_round = BasicBlock::Create(ctx, "end.round", f);
    BasicBlock* exit = BasicBlock::Create(ctx, "exit", f);
    IRBuilder<> b(entry);
    b.CreateBr(round);

    b.SetInsertPoint(round);
    b.CreateBr(warp);

    b.SetInsertPoint(warp);
    PHINode* w = b.CreatePHI(i32, 2, "w");
    PHINode* left = b.CreatePHI(i32, 2, "left");  // warps still running after this round
    w->addIncoming(b.getInt32(0), round);
    left->addIncoming(b.getInt32(0), round);
    Value* done_w = b.CreateGEP(i8, done, w);
    b.CreateCondBr(b.CreateICmpNE(b.CreateLoad(i8, done_w), b.getInt8(0)), next, run);

    b.SetInsertPoint(run);
    Value* c = b.CreateGEP(i8, ctxs, b.CreateMul(b.CreateZExt(w, Type::getInt64Ty(ctx)),
                                                   b.getInt64(cpusim_stagger(cpusim_ctx_size(CPUSIM_WARP_SIZE)))));
    CallInst* call = b.CreateCall(k.entry, {c});
    Value* finished = b.CreateICmpEQ(call, b.getInt32(CPUSIM_EXIT));
    b.CreateStore(b.CreateZExt(finished, i8), done_w);
    Value* still = b.CreateAdd(left, b.CreateZExt(b.CreateNot(finished), i32));
    b.CreateBr(next);

    b.SetInsertPoint(next);
    PHINode* left2 = b.CreatePHI(i32, 2);
    left2->addIncoming(left, warp);
    left2->addIncoming(still, run);
    Value* w1 = b.CreateAdd(w, b.getInt32(1));
    w->addIncoming(w1, next);
    left->addIncoming(left2, next);
    b.CreateCondBr(b.CreateICmpULT(w1, n), warp, end_round);

    b.SetInsertPoint(end_round);
    b.CreateCondBr(b.CreateICmpNE(left2, b.getInt32(0)), round, exit);

    b.SetInsertPoint(exit);
    b.CreateRet(b.getInt32(CPUSIM_EXIT));

    InlineFunctionInfo ifi;
    InlineFunction(*call, ifi);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    // Tuning of LLVM's own passes for warp code. CPUSIM_LLVM_FLAGS adds more
    // (for experiments), e.g. "-unswitch-threshold=200".
    {
        // Loops in warp code test invariant access-pattern flags; letting LLVM
        // version them (unswitch) takes those tests out of the loops.
        std::vector<std::string> flags = {"cpusim-devc", "-unswitch-threshold=2000"};
        if (const char* extra = std::getenv("CPUSIM_LLVM_FLAGS")) {
            std::string all = extra;
            size_t pos = 0;
            while (pos < all.size()) {
                size_t end = all.find(' ', pos);
                if (end == std::string::npos) end = all.size();
                if (end > pos) flags.push_back(all.substr(pos, end - pos));
                pos = end + 1;
            }
        }
        std::vector<const char*> argv2;
        for (auto& f : flags) argv2.push_back(f.c_str());
        cl::ParseCommandLineOptions(int(argv2.size()), argv2.data());
    }

    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();

    LLVMContext ctx;
    SMDiagnostic diag;
    std::unique_ptr<Module> m = parseIRFile(opt.input, diag, ctx);
    if (!m) {
        diag.print("cpusim-devc", errs());
        return 1;
    }

    std::string triple = "x86_64-unknown-linux-gnu", err;
    const Target* target = TargetRegistry::lookupTarget(triple, err);
    if (!target) die(err);
    TargetOptions to;
    std::unique_ptr<TargetMachine> tm(target->createTargetMachine(triple, opt.cpu, "", to, Reloc::PIC_, std::nullopt,
                                                                  CodeGenOptLevel::Aggressive));
    if (!tm) die("cannot create a target machine for " + opt.cpu);

    std::vector<KernelInfo> kernels = find_kernels(*m);
    for (auto& k : kernels)
        for (int i = 0; i < 3; ++i) k.spec_block[i] = opt.specialize[i];
    retarget(*m, tm.get(), kernels);
    lower_nvvm_atomics(*m);
    const std::string symbol_lines = lower_device_globals(*m);

    // Per-thread clean up before vectorizing: inline everything, promote locals
    // to registers, simplify. No unrolling or vectorization here: those would
    // obscure the per-thread structure the SIMT pass works on.
    run_pipeline(*m, tm.get(), "always-inline,globaldce");
    for (auto& k : kernels) lower_barrier_votes(*m, *k.scalar);
    for (auto& k : kernels) cpusim::lower_shared_memory(*m, k);
    run_pipeline(*m, tm.get(),
                 "globaldce,"
                 "function(sroa<modify-cfg>,early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,"
                 "loop-mssa(licm<allowspeculation>),gvn,instcombine<no-verify-fixpoint>,simplifycfg,dse,adce,"
                 "lower-switch,lcssa)");
    // Per-thread arrays (float v[4]) usually live through loops of a fixed
    // trip count: unrolled, their indices are constants and SROA turns them
    // into registers. Only kernels that still have arrays, so the others keep
    // their loops as written.
    // (Device functions that were not inlined too: they become warp code as well.)
    std::vector<Function*> per_thread;
    for (auto& k : kernels) per_thread.push_back(k.scalar);
    for (Function& f : *m)
        if (!f.isDeclaration() && f.hasFnAttribute(Attribute::AlwaysInline)) per_thread.push_back(&f);
    for (Function* f : per_thread) lower_printf(*f);
    for (Function* f : per_thread)
        if (expand_mem_intrinsics(*f, tm.get()) && !has_alloca(*f))
            run_function_pipeline(*f, tm.get(),
                                  "early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,lower-switch,lcssa");
        else if (has_alloca(*f)) {
            run_function_pipeline(*f, tm.get(),
                                  "loop-simplify,lcssa,loop(loop-rotate,indvars,loop-unroll-full),sroa<modify-cfg>,"
                                  "early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,gvn,"
                                  "instcombine<no-verify-fixpoint>,simplifycfg,dse,adce,lower-switch,lcssa");
        }
    for (Function* f : per_thread) removeUnreachableBlocks(*f);
    if (!opt.emit_scalar_ll.empty()) write_ll(*m, opt.emit_scalar_ll);

    std::string manifest = "cpusim-module " + std::to_string(CPUSIM_KERNEL_ABI) + "\n" + symbol_lines;
    int failures = 0;
    for (auto& k : kernels) {
        std::string why;
        // The SIMT builder rewrites the scalar function, so every variant but
        // the last starts from a copy.
        // Without a block shape (the module compiled ahead of time), a wide
        // variant past this many instructions is dropped: code generation
        // grows faster than linearly with size (register allocation), and the
        // JIT builds the variant the launch's block shape needs anyway.
        constexpr size_t kGenericBudget = 12000;
        bool dropped = false;
        auto build = [&](bool profile, unsigned lanes, bool suspending = false, bool cooperative = false) {
            ValueToValueMapTy vmap;
            Function* copy = CloneFunction(k.scalar, vmap);
            bool ok = cpusim::build_warp_entry(*m, k, *copy, profile, lanes, suspending, &why, cooperative);
            copy->eraseFromParent();
            if (ok && k.spec_block[0] == 0 && lanes > 32) {
                std::string name = "__cpusim_entry" + std::to_string(lanes) +
                                   (cooperative ? "c_" : suspending ? "s_" : "_") + k.name;
                if (Function* f = m->getFunction(name); f && f->getInstructionCount() > kGenericBudget) {
                    if (std::getenv("CPUSIM_DEVC_DEBUG"))
                        std::fprintf(stderr, "cpusim-devc: %s: %u instructions, left to the JIT\n", name.c_str(),
                                     f->getInstructionCount());
                    f->eraseFromParent();
                    dropped = true;
                }
            }
            return ok;
        };
        // build() for a variant the runtime may choose: false if dropped.
        auto kept = [&]() {
            bool d = dropped;
            dropped = false;
            return !d;
        };
        // The compute capability the kernel's tensor core types need, as on
        // a GPU: fp16 since sm_70, int8 since sm_72, 4-bit and 1-bit since
        // sm_75, bf16, tf32 and double since sm_80.
        int min_sm = 0;
        for (BasicBlock& bb : *k.scalar)
            for (Instruction& inst : bb)
                if (cpusim::is_mma(&inst) || cpusim::is_mmav(&inst)) {
                    StringRef types = cast<CallInst>(inst).getCalledFunction()->getName();
                    const bool sm80 = types.contains("bf16") || types.contains("tf32") || types.contains("f64");
                    const bool sm75 = types.contains("s4_") || types.contains("u4_") || types.contains("b1");
                    min_sm = std::max(min_sm, sm80 ? 80 : sm75 ? 75 : types.contains("8_") ? 72 : 70);
                }
        for (BasicBlock& bb : *k.scalar)
            for (Instruction& inst : bb)
                if (auto* call = dyn_cast<IntrinsicInst>(&inst))
                    k.has_barriers |= call->getIntrinsicID() == Intrinsic::nvvm_barrier0 ||
                                      call->getIntrinsicID() == Intrinsic::nvvm_bar_sync ||
                                      call->getIntrinsicID() == Intrinsic::nvvm_barrier ||
                                      call->getIntrinsicID() == Intrinsic::nvvm_barrier_n;
        // With a known block shape (JIT at launch) only the variants the
        // runtime can pick are built, and no instrumented one (profiling runs
        // the generic module).
        const bool spec = k.spec_block[0] != 0;
        const uint64_t threads = uint64_t(k.spec_block[0]) * k.spec_block[1] * k.spec_block[2];
        auto wanted = [&](unsigned lanes, bool block_wide) {
            if (!spec) return true;
            if (block_wide) return k.has_barriers ? threads <= lanes && threads > lanes / 2 : lanes <= threads;
            return lanes < threads;  // suspending variants: several groups per block
        };
        // Tensor core kernels: an inline mma is a few hundred instructions per
        // warp, repeated for every warp of a wide variant, which costs
        // compile time and gains nothing: each warp's mma is enough
        // independent work, and the runtime's measurement picked the 32-lane
        // variant over the block-wide one (samples/tensor_gemm: 2x faster).
        const bool tensor = min_sm > 0;
        bool ok = build(false, 32) && (spec || build(true, 32));
        // Wider variants: with barriers, a whole block per call makes them free
        // (64..512 lanes); without, 2 or 4 warps per call give the core more
        // independent work. The runtime measures which one wins.
        for (unsigned i = 0; ok && i < (k.has_barriers ? 4u : 2u); ++i) {
            if (!wanted(64u << i, true) || tensor) continue;
            ok = build(false, 64u << i);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_WIDE_SHIFT + i);
        }
        // Groups of 2 and 4 warps that still suspend at barriers: fewer switches.
        for (unsigned i = 0; ok && k.has_barriers && !tensor && i < 2; ++i) {
            if (!wanted(64u << i, false)) continue;
            ok = build(false, 64u << i, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_SUSP_SHIFT + i);
        }
        // Half a block per call, the two SMT siblings of a core together: for
        // blocks of L+1 to 2L threads.
        for (unsigned i = 0; ok && k.has_barriers && !tensor && i < 4; ++i) {
            const unsigned lanes = 64u << i;
            if (spec && !(threads > lanes && threads <= 2 * lanes)) continue;
            ok = build(false, lanes, false, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_COOP_SHIFT + i);
        }
        if (!ok) {
            std::fprintf(stderr, "cpusim-devc: kernel %s: %s\n", k.name.c_str(), why.c_str());
            ++failures;
            continue;
        }
        if (k.has_barriers) {
            build_block_function(*m, k);
            k.wide_variants |= CPUSIM_META_BLOCK_FN;
        }
        // What the runtime needs to know about the kernel, readable with dlsym.
        auto* i32 = Type::getInt32Ty(ctx);
        auto* meta_ty = StructType::get(ctx, {i32, i32, i32, i32, i32, i32});
        new GlobalVariable(*m, meta_ty, true, GlobalValue::ExternalLinkage,
                           ConstantStruct::get(meta_ty, {ConstantInt::get(i32, CPUSIM_KERNEL_ABI),
                                                         ConstantInt::get(i32, k.frame_bytes),
                                                         ConstantInt::get(i32, k.shared_bytes),
                                                         ConstantInt::get(i32, (k.has_barriers ? CPUSIM_META_HAS_BARRIERS : 0) |
                                                                                   k.wide_variants),
                                                         ConstantInt::get(i32, k.frame_bytes_wide[0]),
                                                         ConstantInt::get(i32, k.frame_bytes_wide[1])}),
                           "__cpusim_meta_" + k.name);
        manifest += "kernel " + k.name + " " + std::to_string(k.param_bytes) + " " + std::to_string(k.params.size());
        for (const auto& p : k.params) manifest += " " + std::to_string(p.offset) + ":" + std::to_string(p.size);
        if (min_sm) manifest += " sm=" + std::to_string(min_sm);
        if (k.shared_bytes) manifest += " smem=" + std::to_string(k.shared_bytes);
        manifest += "\n";
    }
    if (failures) return 1;

    // Only the entries remain visible; the scalar kernels are dropped.
    for (auto& k : kernels) k.scalar->eraseFromParent();
    // So are the per-thread device functions left after inlining (recursive
    // ones, or those whose address is taken): their vector versions replaced
    // every call. An address still in use (a table of function pointers, a
    // vtable, the comparisons that pick the target of a call through a
    // pointer) becomes a number made from the function's name: the same in
    // every module of the program (the generic one and each block shape's),
    // so a pointer stored by one kernel is called right by any other. Its
    // upper bits make it a non-canonical address, never a real one.
    {
        std::vector<Function*> left;
        for (Function& f : *m)
            if (!f.isDeclaration() && f.hasFnAttribute(Attribute::AlwaysInline)) left.push_back(&f);
        for (Function* f : left) f->dropAllReferences();
        for (Function* f : left) {
            if (!f->use_empty()) {
                uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a
                for (char c : f->getName()) h = (h ^ uint8_t(c)) * 0x100000001b3ull;
                const uint64_t id = 0x8000000000000000ull | (h & 0x00007ffffffffff0ull);
                f->replaceAllUsesWith(
                    ConstantExpr::getIntToPtr(ConstantInt::get(Type::getInt64Ty(ctx), id), f->getType()));
            }
            f->eraseFromParent();
        }
    }
    new GlobalVariable(*m, Type::getInt32Ty(ctx), true, GlobalValue::ExternalLinkage,
                       ConstantInt::get(Type::getInt32Ty(ctx), CPUSIM_KERNEL_ABI), "__cpusim_abi_version");

    if (!opt.emit_raw_ll.empty()) write_ll(*m, opt.emit_raw_ll);
    if (verifyModule(*m, &errs())) die("internal error: generated IR does not verify");
    for (Function& f : *m)
        if (!f.isDeclaration()) {
            run_function_pipeline(f, tm.get(), "sroa,early-cse,instcombine");
            if (refine_uniform_masks(f) && verifyFunction(f, &errs()))
                die("internal error: IR does not verify after refining masks in " + f.getName().str());
        }
    run_pipeline(*m, tm.get(), "O3");
    widen_mask_selects(*m);
    if (!opt.emit_ll.empty()) write_ll(*m, opt.emit_ll);

    std::error_code ec;
    if (!opt.emit_asm.empty()) {
        // Code generation changes the module, so the listing comes from a copy.
        std::unique_ptr<Module> copy = CloneModule(*m);
        raw_fd_ostream asm_out(opt.emit_asm, ec, sys::fs::OF_Text);
        if (ec) die(opt.emit_asm + ": " + ec.message());
        legacy::PassManager apm;
        if (tm->addPassesToEmitFile(apm, asm_out, nullptr, CodeGenFileType::AssemblyFile))
            die("cannot emit assembly");
        apm.run(*copy);
    }
    raw_fd_ostream out(opt.output, ec, sys::fs::OF_None);
    if (ec) die(opt.output + ": " + ec.message());
    legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, out, nullptr, CodeGenFileType::ObjectFile)) die("cannot emit an object file");
    pm.run(*m);
    out.close();

    std::ofstream mf(opt.manifest);
    mf << manifest;
    return mf ? 0 : 1;
}
