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
#include <llvm/Analysis/PostDominators.h>
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
#include "compiler/passes.h"
#include "compiler/simt.h"
#include "compiler/uniformity.h"
#include "device/kernel_abi.h"

using namespace llvm;
using cpusim::KernelInfo;
using cpusim::cap_device_malloc;
using cpusim::expand_mem_intrinsics;
using cpusim::Features;
using cpusim::has_alloca;
using cpusim::keep_barrier_loops_rolled;
using cpusim::kernel_features;
using cpusim::lower_barrier_votes;
using cpusim::lower_device_globals;
using cpusim::lower_half_fma;
using cpusim::lower_nvvm_atomics;
using cpusim::lower_printf;
using cpusim::mark_spin_loops;
using cpusim::refine_uniform_masks;
using cpusim::shrink_half_sqrt;
using cpusim::widen_mask_selects;

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
    // Only these kernels (the JIT: the one being launched); empty: all.
    std::set<std::string> only;
    bool list = false;  // --list: print the kernels (and whether they can be compiled apart), nothing else
    uint32_t shape_class = 0;  // --specialize-class: a class of shapes (cpusim_shape_class) instead of one
    int barriers = 0;          // --barriers-only (1) / --no-barriers (-1): only those kernels
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
        else if (a == "--list") o.list = true;
        else if (a == "--specialize-class") o.shape_class = uint32_t(std::stoul(value()));
        else if (a == "--barriers-only") o.barriers = 1;
        else if (a == "--no-barriers") o.barriers = -1;
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
    if (o.list && !o.input.empty()) return o;
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
    // --kernels: the run-time JIT compiles the kernel being launched, not
    // every kernel of the file (the code of that kernel is the same).
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
    // --list: one line per kernel, and "together" when they must be
    // compiled in one module (cucc otherwise runs one cpusim-devc each).
    if (opt.list) {
        for (const KernelInfo& k : kernels) std::printf("kernel %s\n", k.name.c_str());
        if (addresses_taken) std::printf("together\n");
        return 0;
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
        // (Classes assume groups of at least 32 lanes.)
        if (warp_size >= 32) k.spec_class = opt.shape_class;
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
        bool dropped = false;
        auto build = [&](bool profile, unsigned lanes, bool suspending = false) {
            ValueToValueMapTy vmap;
            Function* copy = CloneFunction(k.scalar, vmap);
            bool ok = cpusim::build_warp_entry(*m, k, *copy, profile, lanes, suspending, &why);
            copy->eraseFromParent();
            if (ok && k.spec_block[0] == 0 && k.spec_class == 0 && lanes > k.warp_size) {
                std::string name = "__cpusim_entry" + std::to_string(lanes) +
                                   (suspending ? "s_" : "_") + k.name;
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
        bool lane_independent = collective_free && k.spec_block[0] == 0 && k.spec_class == 0;
        // With a known block shape (JIT at launch) only the variants the
        // runtime can pick are built, and no instrumented one (profiling runs
        // the generic module).
        const bool spec = k.spec_block[0] != 0 || k.spec_class != 0;
        // (A class: the fewest threads of its shapes.)
        const uint64_t threads = k.spec_class ? ((k.spec_class & 0xff) == 2 ? 32 : 128)
                                              : uint64_t(k.spec_block[0]) * k.spec_block[1] * k.spec_block[2];
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
        // Variants of 128 lanes and more for a thread with many live values:
        // every value is lanes / 16 vector registers, so they spill, and
        // register allocation over them took minutes (HeCBench hadamard, 128+
        // live values: 215 s for one kernel). They never won: past 47 live
        // values the runtime picked 32 or 64 lanes in every measured kernel
        // (the widest winner had 29), and hadamard's 256 and 128s ran 2-5x
        // slower than 32.
        constexpr unsigned kWideLiveLimit = 64;
        const bool heavy = kernel_features(*k.scalar).live_units > kWideLiveLimit;
        k.has_barriers |= k.spins;
        static const bool print_features = std::getenv("CPUSIM_DEVC_FEATURES") != nullptr;
        if (print_features && k.spec_block[0] == 0 && k.spec_class == 0) {
            const Features ft = kernel_features(*k.scalar);
            std::fprintf(stderr, "cpusim-features %s live=%u insts=%u loads=%u stores=%u slow=%u loops=%u barriers=%d\n",
                         k.name.c_str(), ft.live_units, ft.instructions, ft.loads, ft.stores, ft.slow_ops, ft.loops,
                         int(k.has_barriers));
        }
        // A module for a class of shapes holds the kernels without barriers,
        // the one for an exact shape next to it those with (cucc).
        if ((opt.barriers > 0 && !k.has_barriers) || (opt.barriers < 0 && k.has_barriers)) continue;
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
        // Wider variants (lanes 2L << i, L the warp size): with barriers, a
        // whole block per call makes them free (up to 512 lanes); without, 2
        // or 4 warps per call give the core more independent work. The
        // runtime measures which one wins.
        for (unsigned i = 0; ok && (2 * L << i) <= CPUSIM_MAX_LANES && (k.has_barriers || i < 2); ++i) {
            if (!wanted(2 * L << i, true) || narrow || (heavy && (2 * L << i) >= 128)) continue;
            ok = build(false, 2 * L << i);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_WIDE_SHIFT + i);
        }
        // Groups of 2 and 4 warps that still suspend at barriers: fewer switches.
        for (unsigned i = 0; ok && k.has_barriers && !narrow && i < 2; ++i) {
            if (!wanted(2 * L << i, false) || (heavy && (2 * L << i) >= 128)) continue;
            ok = build(false, 2 * L << i, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_SUSP_SHIFT + i);
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
            built = built && cpusim::build_warp_entry(*m, k, *copy, false, k.warp_lanes, false, &rwhy, true);
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
        if (k.has_barriers) manifest += " barriers";
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
    new GlobalVariable(*m, Type::getInt32Ty(ctx), true, GlobalValue::WeakODRLinkage,
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
    // (CPUSIM_DEVC_THREADS: cucc's share of the cores for this process.)
    const char* threads_env = std::getenv("CPUSIM_DEVC_THREADS");
    const unsigned parts = threads_env ? unsigned(std::max(1, std::atoi(threads_env)))
                                       : std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), 16);
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
