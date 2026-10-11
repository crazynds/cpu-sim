// cpusim-devc: device compiler back end. Takes the per-thread LLVM IR that
// Clang produces for the device side of a .cu file and emits a native x86
// object with one warp entry point per kernel, plus the module manifest.
//
//   cpusim-devc device.bc -o kernels.o --manifest kernels.manifest [--emit-ll out.ll] [--mcpu native|znver4|...]
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/IRBuilder.h>
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
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>

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

// Small fixed-size copies to or from a local (struct copies: `auto p = a[i]`,
// `a[i] = p`) as 4-byte loads and stores, which SROA then turns into
// registers; it leaves the local in memory when it is copied with memcpy.
bool expand_local_memcpys(Function& f) {
    bool changed = false;
    for (BasicBlock& bb : f)
        for (Instruction& inst : make_early_inc_range(bb)) {
            auto* cpy = dyn_cast<MemCpyInst>(&inst);
            auto* len = cpy ? dyn_cast<ConstantInt>(cpy->getLength()) : nullptr;
            if (!len || len->getZExtValue() > 256 || len->getZExtValue() % 4 != 0 || cpy->isVolatile()) continue;
            if (!isa<AllocaInst>(getUnderlyingObject(cpy->getDest())) &&
                !isa<AllocaInst>(getUnderlyingObject(cpy->getSource())))
                continue;
            IRBuilder<> b(cpy);
            Type* i32 = b.getInt32Ty();
            const Align da = cpy->getDestAlign().valueOrOne(), sa = cpy->getSourceAlign().valueOrOne();
            for (uint64_t off = 0; off < len->getZExtValue(); off += 4) {
                Value* v = b.CreateAlignedLoad(i32, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), cpy->getSource(), off),
                                               commonAlignment(sa, off));
                b.CreateAlignedStore(v, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), cpy->getDest(), off),
                                     commonAlignment(da, off));
            }
            cpy->eraseFromParent();
            changed = true;
        }
    return changed;
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

    // Per-thread clean up before vectorizing: inline everything, promote locals
    // to registers, simplify. No unrolling or vectorization here: those would
    // obscure the per-thread structure the SIMT pass works on.
    run_pipeline(*m, tm.get(), "always-inline,globaldce");
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
    for (auto& k : kernels)
        if (has_alloca(*k.scalar)) {
            expand_local_memcpys(*k.scalar);
            run_function_pipeline(*k.scalar, tm.get(),
                                  "loop-simplify,lcssa,loop(loop-rotate,indvars,loop-unroll-full),sroa<modify-cfg>,"
                                  "early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,gvn,"
                                  "instcombine<no-verify-fixpoint>,simplifycfg,dse,adce,lower-switch,lcssa");
        }
    for (auto& k : kernels) removeUnreachableBlocks(*k.scalar);
    if (!opt.emit_scalar_ll.empty()) write_ll(*m, opt.emit_scalar_ll);

    std::string manifest = "cpusim-module " + std::to_string(CPUSIM_KERNEL_ABI) + "\n";
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
        // a GPU: fp16 since sm_70, int8 since sm_72, bf16 since sm_80.
        int min_sm = 0;
        for (BasicBlock& bb : *k.scalar)
            for (Instruction& inst : bb)
                if (cpusim::is_mma(&inst) || cpusim::is_mmav(&inst)) {
                    StringRef types = cast<CallInst>(inst).getCalledFunction()->getName();
                    min_sm = std::max(min_sm, types.contains("bf16") ? 80 : types.contains("8_") ? 72 : 70);
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
        bool ok = build(false, 32) && (spec || build(true, 32));
        // Wider variants: with barriers, a whole block per call makes them free
        // (64..512 lanes); without, 2 or 4 warps per call give the core more
        // independent work. The runtime measures which one wins.
        for (unsigned i = 0; ok && i < (k.has_barriers ? 4u : 2u); ++i) {
            if (!wanted(64u << i, true)) continue;
            ok = build(false, 64u << i);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_WIDE_SHIFT + i);
        }
        // Groups of 2 and 4 warps that still suspend at barriers: fewer switches.
        for (unsigned i = 0; ok && k.has_barriers && i < 2; ++i) {
            if (!wanted(64u << i, false)) continue;
            ok = build(false, 64u << i, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_SUSP_SHIFT + i);
        }
        // Half a block per call, the two SMT siblings of a core together: for
        // blocks of L+1 to 2L threads.
        for (unsigned i = 0; ok && k.has_barriers && i < 4; ++i) {
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
        manifest += "\n";
    }
    if (failures) return 1;

    // Only the entries remain visible; the scalar kernels are dropped.
    for (auto& k : kernels) k.scalar->eraseFromParent();
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
