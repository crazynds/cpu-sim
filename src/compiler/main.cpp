// cpusim-devc: device compiler back end. Takes the per-thread LLVM IR that
// Clang produces for the device side of a .cu file and emits a native x86
// object with one warp entry point per kernel, plus the module manifest.
//
//   cpusim-devc device.bc -o kernels.o --manifest kernels.manifest [--emit-ll out.ll] [--mcpu native|znver4|...]
#include <cstdio>
#include <fstream>
#include <map>
#include <algorithm>
#include <set>
#include <thread>
#include <string>
#include <vector>

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Program.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Transforms/Utils/SplitModule.h>
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

#include "common/cpu_id.h"
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
    // With "native": the extensions this CPU really has (not the defaults of
    // its model: a virtual machine may hide some), and its signature in the
    // manifest, so the runtime uses this code only on the same kind of CPU.
    std::string features;
    uint64_t signature = 0;
    uint32_t specialize[3] = {0, 0, 0};
    // With --specialize: only these kernels (the ones launched with that
    // block shape); empty: all of them.
    std::set<std::string> only;
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
        else if (a == "--kernels") {  // name,name,...
            std::string v = value();
            for (size_t b = 0, e; b <= v.size(); b = e + 1) {
                e = v.find(',', b);
                if (e == std::string::npos) e = v.size();
                if (e > b) o.only.insert(v.substr(b, e - b));
            }
        }
        else if (!a.empty() && a[0] == '-') die("unknown option " + a);
        else o.input = a;
    }
    if (o.input.empty() || o.output.empty() || o.manifest.empty())
        die("usage: cpusim-devc device.bc -o kernels.o --manifest kernels.manifest");
    if (o.cpu == "native") {
        o.cpu = sys::getHostCPUName().str();
        for (const auto& f : sys::getHostCPUFeatures()) o.features += (o.features.empty() ? "" : ",") + std::string(f.getValue() ? "+" : "-") + f.getKey().str();
        o.signature = cpusim::cpu_signature();
    }
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

// Loops with a barrier stay loops: unrolled, every copy of the barrier is
// one more place where a warp suspends and resumes, and a barrier costs the
// same inside a loop as outside. (HeCBench sc: 32 copies of a prefix sum
// with barriers in its loops, over ten minutes to compile.)
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
        // Every device function is inlined into the kernels (DESIGN.md 5.5).
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
                                                   b.getInt64(cpusim_stagger(cpusim_ctx_size(k.warp_lanes)))));
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
    std::unique_ptr<TargetMachine> tm(target->createTargetMachine(triple, opt.cpu, opt.features, to, Reloc::PIC_, std::nullopt,
                                                                  CodeGenOptLevel::Aggressive));
    if (!tm) die("cannot create a target machine for " + opt.cpu);

    // The warp size the code was compiled for: the device header leaves it in
    // __cpusim_warp_size (CPUSIM_WARP_SIZE at compile time; 32 by default).
    unsigned warp_size = 32;
    if (GlobalVariable* g = m->getGlobalVariable("__cpusim_warp_size")) {
        if (auto* c = dyn_cast_or_null<ConstantInt>(g->hasInitializer() ? g->getInitializer() : nullptr))
            warp_size = unsigned(c->getZExtValue());
        g->eraseFromParent();
    }
    if (warp_size != 16 && warp_size != 32 && warp_size != 64) die("warp size must be 16, 32 or 64");

    std::vector<KernelInfo> kernels = find_kernels(*m);
    // A specialized module holds only the kernels launched with its shape:
    // building every kernel of the file once per shape made files with many
    // kernels take kernels x shapes x widths builds (segsort, hadamard, sc:
    // over ten minutes). The others run the generic module's code.
    // Not with device functions whose address is taken (vtables, tables of
    // function pointers): a call through a pointer reaches the vector
    // versions built for the kernels that set the pointers up, so every
    // kernel stays.
    bool addresses_taken = false;
    {
        std::set<const Function*> entry_points;
        for (const KernelInfo& k : kernels) entry_points.insert(k.scalar);
        for (Function& f : *m)
            if (!f.isDeclaration() && !entry_points.count(&f) && f.hasAddressTaken(nullptr, true, true, true, true))
                addresses_taken = true;
    }
    if (!opt.only.empty() && !addresses_taken) {
        for (KernelInfo& k : kernels)
            if (!opt.only.count(k.name) && k.scalar->use_empty()) k.scalar->deleteBody();
        kernels.erase(std::remove_if(kernels.begin(), kernels.end(),
                                     [&](const KernelInfo& k) { return !opt.only.count(k.name); }),
                      kernels.end());
    }
    for (auto& k : kernels) {
        for (int i = 0; i < 3; ++i) k.spec_block[i] = opt.specialize[i];
        k.warp_size = k.warp_lanes = warp_size;
    }
    retarget(*m, tm.get(), kernels);
    // GEMMs written with wmma: a trace function per candidate kernel, for
    // the runtime to recognize the launch and run it on AMX tiles.
    if (!std::getenv("CPUSIM_NO_WMMA_TRACE"))
        for (auto& k : kernels) {
            std::string why;
            k.traced = cpusim::build_wmma_trace(*m, k, &why);
            if (std::getenv("CPUSIM_DEVC_DEBUG") && why != "no mma_sync")
                std::fprintf(stderr, "cpusim-devc: %s: wmma GEMM trace %s%s\n", k.name.c_str(),
                             k.traced ? "built" : "not built: ", why.c_str());
        }
    lower_half_fma(*m, StringRef(tm->getTargetFeatureString()).contains("+avx512fp16") ||
                           tm->getMCSubtargetInfo()->checkFeatures("+avx512fp16"));
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
    for (Function* f : per_thread) cap_device_malloc(*f);
    for (Function* f : per_thread)
        if (expand_mem_intrinsics(*f, tm.get()) && !has_alloca(*f))
            run_function_pipeline(*f, tm.get(),
                                  "early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,lower-switch,lcssa");
        else if (has_alloca(*f)) {
            keep_barrier_loops_rolled(*f);
            run_function_pipeline(*f, tm.get(),
                                  "loop-simplify,lcssa,loop(loop-rotate,indvars,loop-unroll-full),sroa<modify-cfg>,"
                                  "early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,gvn,"
                                  "instcombine<no-verify-fixpoint>,simplifycfg,dse,adce,lower-switch,lcssa");
        }
    for (Function* f : per_thread) removeUnreachableBlocks(*f);
    for (Function* f : per_thread) shrink_half_sqrt(*f);
    for (auto& k : kernels) k.spins = mark_spin_loops(*k.scalar);
    if (!opt.emit_scalar_ll.empty()) write_ll(*m, opt.emit_scalar_ll);

    char cpu_tag[32] = "";
    if (opt.signature) std::snprintf(cpu_tag, sizeof(cpu_tag), " cpu=%016llx", (unsigned long long)opt.signature);
    std::string manifest = "cpusim-module " + std::to_string(CPUSIM_KERNEL_ABI) + " warp=" + std::to_string(warp_size) +
                           cpu_tag + "\n" + symbol_lines;
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
        // Any module: a wide variant whose size, estimated from the warp
        // entry's (it grows about with the lanes), passes this is not built
        // at all. A kernel already large at 32 lanes (HeCBench hadamard: a
        // fully unrolled template) took minutes per wide variant, for code
        // with no registers left to gain from width.
        constexpr size_t kWideBudget = 24000;
        size_t warp_entry_size = 0;
        bool dropped = false;
        auto build = [&](bool profile, unsigned lanes, bool suspending = false, bool cooperative = false) {
            if (!profile && lanes > k.warp_lanes && warp_entry_size &&
                warp_entry_size * lanes / k.warp_lanes > (k.spec_block[0] ? kWideBudget : kGenericBudget)) {
                if (std::getenv("CPUSIM_DEVC_DEBUG"))
                    std::fprintf(stderr, "cpusim-devc: %s: %u lanes not built (warp entry: %zu instructions)\n",
                                 k.name.c_str(), lanes, warp_entry_size);
                dropped = true;
                return true;
            }
            ValueToValueMapTy vmap;
            Function* copy = CloneFunction(k.scalar, vmap);
            bool ok = cpusim::build_warp_entry(*m, k, *copy, profile, lanes, suspending, &why, cooperative);
            copy->eraseFromParent();
            if (ok && k.spec_block[0] == 0 && lanes > k.warp_size) {
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
                    k.has_barriers |= k.spins || call->getIntrinsicID() == Intrinsic::nvvm_barrier0 ||
                                      call->getIntrinsicID() == Intrinsic::nvvm_bar_sync ||
                                      call->getIntrinsicID() == Intrinsic::nvvm_barrier ||
                                      call->getIntrinsicID() == Intrinsic::nvvm_barrier_n;
        // Lanes that never look at each other: nothing shared, nothing
        // collective, in the kernel or what it calls. They can run apart
        // (CPUSIM_META_LANE_INDEPENDENT) or take other threads (refill).
        bool collective_free = !k.has_barriers && !k.spins && k.shared_bytes == 0 && min_sm == 0;
        {
            bool& lane_independent = collective_free;
            std::vector<Function*> todo{k.scalar};
            std::set<Function*> seen{k.scalar};
            while (lane_independent && !todo.empty()) {
                Function* f = todo.back();
                todo.pop_back();
                for (BasicBlock& bb : *f)
                    for (Instruction& inst : bb) {
                        auto* call = dyn_cast<CallBase>(&inst);
                        if (!call) continue;
                        Function* callee = call->getCalledFunction();
                        if (!callee) {
                            if (!call->isInlineAsm()) lane_independent = false;  // unknown target
                            continue;
                        }
                        const StringRef n = callee->getName();
                        if (n == cpusim::kShflFn || n == cpusim::kBallotFn || n == cpusim::kSharedBaseFn ||
                            n == cpusim::kYieldFn || n.starts_with(cpusim::kMmaPrefix) ||
                            n.starts_with(cpusim::kMmavPrefix) || n.starts_with(cpusim::kTensorPrefix) ||
                            n.contains("grid_barrier") ||
                            (n.starts_with("llvm.nvvm.") &&
                             (n.contains("shfl") || n.contains("vote") || n.contains("match") || n.contains("redux") ||
                              n.contains("bar") || n.contains("wmma") || n.contains("mma")))) {
                            lane_independent = false;
                            break;
                        }
                        if (!callee->isDeclaration() && seen.insert(callee).second) todo.push_back(callee);
                    }
            }
        }
        // Spreading lanes over workers: only for the generic module (a module
        // specialized for a block shape has its entry masks built in).
        bool lane_independent = collective_free && k.spec_block[0] == 0;
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
        k.has_barriers |= k.spins;
        // Spinning kernels: one warp per call, as on a GPU (wide variants put
        // the waiting warp and the one it waits for in the same vector).
        const bool narrow = tensor || k.spins;
        const unsigned L = k.warp_size;
        if (tensor && L != 32) {
            std::fprintf(stderr, "cpusim-devc: kernel %s: tensor cores (wmma) need warps of 32 threads, not %u\n",
                         k.name.c_str(), L);
            ++failures;
            continue;
        }
        // A block smaller than a warp: the warp entry has only the lanes of
        // threads that exist (16, or 32 of a warp of 64), with the warp's
        // semantics (lane ids, ballots, shuffles) unchanged.
        if (spec && !tensor)
            while (k.warp_lanes > 16 && threads <= k.warp_lanes / 2) k.warp_lanes /= 2;
        bool ok = build(false, k.warp_lanes) && (spec || build(true, k.warp_lanes));
        if (Function* f = m->getFunction("__cpusim_entry_" + k.name)) warp_entry_size = f->getInstructionCount();
        // Wider variants (lanes 2L << i, L the warp size): with barriers, a
        // whole block per call makes them free (up to 512 lanes); without, 2
        // or 4 warps per call give the core more independent work. The
        // runtime measures which one wins.
        for (unsigned i = 0; ok && (2 * L << i) <= CPUSIM_MAX_LANES && (k.has_barriers || i < 2); ++i) {
            if (!wanted(2 * L << i, true) || narrow) continue;
            ok = build(false, 2 * L << i);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_WIDE_SHIFT + i);
        }
        // Groups of 2 and 4 warps that still suspend at barriers: fewer switches.
        for (unsigned i = 0; ok && k.has_barriers && !narrow && i < 2; ++i) {
            if (!wanted(2 * L << i, false)) continue;
            ok = build(false, 2 * L << i, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_SUSP_SHIFT + i);
        }
        // Half a block per call, the two SMT siblings of a core together: for
        // blocks of L+1 to 2L threads.
        for (unsigned i = 0; ok && k.has_barriers && !narrow && (2 * L << i) <= CPUSIM_MAX_LANES; ++i) {
            const unsigned lanes = 2 * L << i;
            if (spec && !(threads > lanes && threads <= 2 * lanes)) continue;
            ok = build(false, lanes, false, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_COOP_SHIFT + i);
        }
        if (!ok) {
            std::fprintf(stderr, "cpusim-devc: kernel %s: %s\n", k.name.c_str(), why.c_str());
            ++failures;
            continue;
        }
        // Opt-in at run time (CPUSIM_BLOCK_FN), slower where measured: built
        // only when asked for at compile time too (one more large function
        // per kernel with barriers otherwise).
        static const bool block_fn = std::getenv("CPUSIM_BLOCK_FN") != nullptr;
        if (block_fn && k.has_barriers && L == 32 && k.warp_lanes == 32) {
            build_block_function(*m, k);
            k.wide_variants |= CPUSIM_META_BLOCK_FN;
        }
        // Lanes that refill (make_refill_kernel): one more candidate for the
        // runtime's measurement, for kernels with a loop.
        static const bool no_refill = std::getenv("CPUSIM_NO_REFILL") != nullptr;
        if (collective_free && !no_refill) {
            ValueToValueMapTy vmap;
            Function* copy = CloneFunction(k.scalar, vmap);
            std::string rwhy;
            bool built = cpusim::make_refill_kernel(*copy, k, &rwhy);
            if (built && std::getenv("CPUSIM_DEVC_REFILL_LL")) copy->print(errs());  // the rewritten per-thread code
            built = built && cpusim::build_warp_entry(*m, k, *copy, false, k.warp_lanes, false, &rwhy, false, true);
            copy->eraseFromParent();
            if (built) {
                k.wide_variants |= CPUSIM_META_REFILL;
            } else {
                if (Function* f = m->getFunction("__cpusim_entryr_" + k.name)) f->eraseFromParent();
                if (std::getenv("CPUSIM_DEVC_DEBUG"))
                    std::fprintf(stderr, "cpusim-devc: %s: no refill entry: %s\n", k.name.c_str(), rwhy.c_str());
            }
        }
        // Spreading lanes pays only when a lane's work is serial in the warp
        // code: atomics (one lane at a time) and gathers or scatters (one
        // memory access per lane, latency bound), in a loop. Plain vector
        // code costs the same with one lane active as with all of them.
        if (lane_independent) {
            bool serial = false;
            if (Function* e = m->getFunction("__cpusim_entry_" + k.name); e && !e->isDeclaration()) {
                DominatorTree dt(*e);
                LoopInfo li(dt);
                for (BasicBlock& bb : *e)
                    for (Instruction& inst : bb) {
                        if (!li.getLoopFor(&bb)) continue;
                        if (isa<AtomicRMWInst>(inst) || isa<AtomicCmpXchgInst>(inst)) serial = true;
                        if (auto* ii = dyn_cast<IntrinsicInst>(&inst))
                            serial |= ii->getIntrinsicID() == Intrinsic::masked_gather ||
                                      ii->getIntrinsicID() == Intrinsic::masked_scatter;
                    }
            }
            lane_independent = serial;
        }
        // What the runtime needs to know about the kernel, readable with dlsym.
        auto* i32 = Type::getInt32Ty(ctx);
        auto* meta_ty = StructType::get(ctx, {i32, i32, i32, i32, i32, i32, i32, i32});
        new GlobalVariable(*m, meta_ty, true, GlobalValue::ExternalLinkage,
                           ConstantStruct::get(meta_ty, {ConstantInt::get(i32, CPUSIM_KERNEL_ABI),
                                                         ConstantInt::get(i32, k.frame_bytes),
                                                         ConstantInt::get(i32, k.shared_bytes),
                                                         ConstantInt::get(i32, (k.has_barriers ? CPUSIM_META_HAS_BARRIERS : 0) |
                                                                                   (k.spins ? CPUSIM_META_YIELDS : 0) |
                                                                                   (lane_independent ? CPUSIM_META_LANE_INDEPENDENT : 0) |
                                                                                   k.wide_variants),
                                                         ConstantInt::get(i32, k.frame_bytes_wide[0]),
                                                         ConstantInt::get(i32, k.frame_bytes_wide[1]),
                                                         ConstantInt::get(i32, k.warp_size),
                                                         ConstantInt::get(i32, k.warp_lanes)}),
                           "__cpusim_meta_" + k.name);
        manifest += "kernel " + k.name + " " + std::to_string(k.param_bytes) + " " + std::to_string(k.params.size());
        for (const auto& p : k.params) manifest += " " + std::to_string(p.offset) + ":" + std::to_string(p.size);
        if (min_sm) manifest += " sm=" + std::to_string(min_sm);
        if (k.shared_bytes) manifest += " smem=" + std::to_string(k.shared_bytes);
        if (k.traced) manifest += " trace";
        if (lane_independent) manifest += " indep";
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
    bool function_pointers = false;  // some address became a number (see below)
    {
        std::vector<Function*> left;
        for (Function& f : *m)
            if (!f.isDeclaration() && f.hasFnAttribute(Attribute::AlwaysInline)) left.push_back(&f);
        for (Function* f : left) f->dropAllReferences();
        for (Function* f : left) {
            if (!f->use_empty()) {
                function_pointers = true;
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
            run_function_pipeline(f, tm.get(), "sroa,early-cse,instcombine<no-verify-fixpoint>");
            if (refine_uniform_masks(f) && verifyFunction(f, &errs()))
                die("internal error: IR does not verify after refining masks in " + f.getName().str());
        }
    std::error_code ec;
    // Optimization and code generation in parallel: the module split into
    // parts (kernels that share nothing go apart), each optimized and turned
    // into an object in a thread of its own with its own context, the
    // objects joined into one. One file's kernels used to go one after the
    // other (HeCBench hadamard: 18 kernels, 90 s on one core).
    const unsigned parts = std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), 16);
    size_t kernel_functions = 0;
    for (Function& f : *m)
        if (!f.isDeclaration() && f.getName().starts_with("__cpusim_")) ++kernel_functions;
    // (Not with function pointers: a vtable of numbers is read where it is
    // defined; split apart, the calls through it went wrong.)
    if (parts > 1 && kernel_functions > 1 && !function_pointers && opt.emit_ll.empty() && opt.emit_asm.empty()) {
        // A definition another part may use must survive its own part's
        // clean-up: linkonce (vtables, inline functions) would be dropped
        // where unused and missing where used.
        for (GlobalValue& g : m->global_values())
            if (!g.isDeclaration() && (g.hasLinkOnceLinkage() || g.hasAvailableExternallyLinkage()))
                g.setLinkage(GlobalValue::WeakODRLinkage);
        std::vector<SmallString<0>> bitcode;
        SplitModule(
            *m, std::min<unsigned>(parts, unsigned(kernel_functions)),
            [&](std::unique_ptr<Module> part) {
                SmallString<0> buf;
                raw_svector_ostream os(buf);
                WriteBitcodeToFile(*part, os);
                bitcode.push_back(std::move(buf));
            },
            false);
        std::vector<std::string> objects(bitcode.size());
        std::vector<std::string> errors(bitcode.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < bitcode.size(); ++i)
            threads.emplace_back([&, i] {
                LLVMContext c;
                Expected<std::unique_ptr<Module>> part =
                    parseBitcodeFile(MemoryBufferRef(StringRef(bitcode[i].data(), bitcode[i].size()), "part"), c);
                if (!part) {
                    errors[i] = toString(part.takeError());
                    return;
                }
                std::unique_ptr<TargetMachine> t(target->createTargetMachine(
                    triple, opt.cpu, opt.features, to, Reloc::PIC_, std::nullopt, CodeGenOptLevel::Aggressive));
                run_pipeline(**part, t.get(), "O3");
                widen_mask_selects(**part);
                objects[i] = opt.output + ".part" + std::to_string(i) + ".o";
                std::error_code e;
                raw_fd_ostream out(objects[i], e, sys::fs::OF_None);
                legacy::PassManager pm;
                if (e || t->addPassesToEmitFile(pm, out, nullptr, CodeGenFileType::ObjectFile)) {
                    errors[i] = "cannot emit " + objects[i];
                    return;
                }
                pm.run(**part);
            });
        for (auto& th : threads) th.join();
        for (const auto& e : errors)
            if (!e.empty()) die(e);
        std::vector<std::string> cmd{CPUSIM_CLANG_PATH, "-r", "-nostdlib", "-o", opt.output};
        cmd.insert(cmd.end(), objects.begin(), objects.end());
        std::vector<StringRef> args(cmd.begin(), cmd.end());
        std::string msg;
        const int rc = sys::ExecuteAndWait(cmd[0], args, std::nullopt, {}, 0, 0, &msg);
        for (const auto& o : objects) sys::fs::remove(o);
        if (rc != 0) die("joining the objects failed: " + msg);
        std::ofstream mf(opt.manifest);
        mf << manifest;
        return mf ? 0 : 1;
    }

    run_pipeline(*m, tm.get(), "O3");
    widen_mask_selects(*m);
    if (!opt.emit_ll.empty()) write_ll(*m, opt.emit_ll);

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
