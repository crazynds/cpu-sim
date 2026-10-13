// Trace functions for kernels that may be a matrix multiplication written
// with wmma (docs/optimization/iteration-3/35).
//
// For a candidate kernel, __cpusim_trace_<kernel>(params, sregs, rec) is its
// scalar copy run for one thread: the address arithmetic and the control
// flow are the kernel's, and every wmma call becomes a call that records it
// (__cpusim_rec_*, in the runtime). The runtime runs it for every warp of a
// launch and checks, on the real values, that the launch is C = A * B: then
// the GEMM runs on AMX tiles instead of the kernel.
//
// A candidate writes no memory except through wmma (stores only to private
// arrays), uses no shared memory, barriers or atomics, and calls nothing but
// wmma and intrinsics without side effects (other device functions are
// inlined into the copy first). Anything else: no trace function.

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <map>
#include <string>

#include "compiler/simt.h"

using namespace llvm;

namespace cpusim {

namespace {

enum WmmaOp { NotWmma, Load, Store, Mma, Fill, OtherWmma };

// Element type codes shared with the runtime (device/wmma_trace.cpp).
enum TypeCode { TOther = 0, TBf16 = 1, TS8 = 2, TU8 = 3, TF16 = 4, TF32 = 5, TS32 = 6 };

struct WmmaCall {
    WmmaOp op = NotWmma;
    int kind = -1;     // load: 0 matrix_a, 1 matrix_b, 2 accumulator
    int type = TOther;
    int layout = 0;    // a/b load: 0 row-major, 1 col-major (from the fragment type)
};

// `name` as a whole template argument: ", float>" or ", float, void>".
bool token(const std::string& s, const std::string& name) {
    for (size_t at = s.find(", " + name); at != std::string::npos; at = s.find(", " + name, at + 1)) {
        const char next = at + 2 + name.size() < s.size() ? s[at + 2 + name.size()] : '\0';
        if (next == ',' || next == '>') return true;
    }
    return false;
}

WmmaCall classify(const CallInst* call) {
    WmmaCall w;
    const Function* f = call->getCalledFunction();
    if (!f) return w;
    const std::string d = demangle(f->getName().str());
    if (d.find("nvcuda::wmma::") == std::string::npos) return w;
    auto has = [&](const char* s) { return d.find(s) != std::string::npos; };
    if (has("wmma::load_matrix_sync<")) w.op = Load;
    else if (has("wmma::store_matrix_sync<")) w.op = Store;
    else if (has("wmma::mma_sync<")) w.op = Mma;
    else if (has("wmma::fill_fragment<")) w.op = Fill;
    else return w.op = OtherWmma, w;
    // The template arguments of the function itself (before the parameter list).
    const std::string head = d.substr(0, d.find('('));
    w.kind = head.find("matrix_a") != std::string::npos   ? 0
             : head.find("matrix_b") != std::string::npos ? 1
                                                          : 2;
    if (has("__nv_bfloat16")) w.type = TBf16;
    else if (has("__half")) w.type = TF16;
    else if (has("signed char")) w.type = TS8;
    else if (has("unsigned char")) w.type = TU8;
    else if (token(head, "float")) w.type = TF32;
    else if (token(head, "int")) w.type = TS32;
    w.layout = has("col_major") ? 1 : 0;
    return w;
}

int sreg_index(Intrinsic::ID id) {
    switch (id) {
        case Intrinsic::nvvm_read_ptx_sreg_tid_x: return 0;
        case Intrinsic::nvvm_read_ptx_sreg_tid_y: return 1;
        case Intrinsic::nvvm_read_ptx_sreg_tid_z: return 2;
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_x: return 3;
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_y: return 4;
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_z: return 5;
        case Intrinsic::nvvm_read_ptx_sreg_ntid_x: return 6;
        case Intrinsic::nvvm_read_ptx_sreg_ntid_y: return 7;
        case Intrinsic::nvvm_read_ptx_sreg_ntid_z: return 8;
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_x: return 9;
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_y: return 10;
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_z: return 11;
        case Intrinsic::nvvm_read_ptx_sreg_laneid: return 12;
        case Intrinsic::nvvm_read_ptx_sreg_warpsize: return 13;
        default: return -1;
    }
}

bool fail(std::string* why, const std::string& w) {
    if (why) *why = w;
    return false;
}

// The checks on the copy, once the other device functions are inlined.
bool candidate(Function& t, std::string* why) {
    int mmas = 0;
    for (Instruction& i : instructions(t)) {
        for (const Use& op : i.operands())
            if (auto* pt = dyn_cast<PointerType>(op->getType()); pt && pt->getAddressSpace() == 3)
                return fail(why, "shared memory");
        if (auto* call = dyn_cast<CallInst>(&i)) {
            WmmaCall w = classify(call);
            if (w.op == Mma) ++mmas;
            if (w.op == OtherWmma) return fail(why, "wmma operation other than load/store/mma/fill");
            if (w.op != NotWmma) continue;
            if (auto* ii = dyn_cast<IntrinsicInst>(call)) {
                if (sreg_index(ii->getIntrinsicID()) >= 0 || ii->isAssumeLikeIntrinsic() || !ii->mayWriteToMemory())
                    continue;
                return fail(why, "intrinsic with side effects: " + ii->getCalledFunction()->getName().str());
            }
            if (call->getCalledFunction() && call->getCalledFunction()->getName().starts_with("__cpusim_shared_base"))
                return fail(why, "shared memory");
            return fail(why, "call to " + (call->getCalledFunction() ? call->getCalledFunction()->getName().str()
                                                                     : std::string("a pointer")));
        }
        if (isa<AtomicRMWInst>(i) || isa<AtomicCmpXchgInst>(i) || isa<FenceInst>(i)) return fail(why, "atomics");
        if (auto* st = dyn_cast<StoreInst>(&i))
            if (!isa<AllocaInst>(getUnderlyingObject(st->getPointerOperand(), 16)))
                return fail(why, "store to memory other than private arrays");
    }
    if (!mmas) return fail(why, "no mma_sync");
    // Fragments only through wmma: an element changed by the kernel itself
    // (acc.x[i] *= alpha) would not be in the records.
    std::vector<const Value*> frags;
    for (Instruction& i : instructions(t))
        if (auto* call = dyn_cast<CallInst>(&i)) {
            // The fragment arguments: load (frag, p, ldm), store (p, frag, ...),
            // mma (d, a, b, c), fill (frag, value by reference).
            const WmmaOp op = classify(call).op;
            std::vector<unsigned> args = op == Load || op == Fill ? std::vector<unsigned>{0}
                                         : op == Store            ? std::vector<unsigned>{1}
                                         : op == Mma              ? std::vector<unsigned>{0, 1, 2, 3}
                                                                  : std::vector<unsigned>{};
            for (unsigned a : args)
                if (auto* o = dyn_cast<AllocaInst>(getUnderlyingObject(call->getArgOperand(a), 16))) frags.push_back(o);
        }
    for (const Value* o : frags) {
        SmallVector<const Value*, 16> work{o};
        while (!work.empty()) {
            const Value* v = work.pop_back_val();
            for (const User* u : v->users()) {
                if (isa<GetElementPtrInst>(u) || isa<BitCastInst>(u)) {
                    work.push_back(u);
                } else if (auto* call = dyn_cast<CallInst>(u)) {
                    if (classify(call).op != NotWmma) continue;
                    if (auto* ii = dyn_cast<IntrinsicInst>(call); ii && ii->isLifetimeStartOrEnd()) continue;
                    if (isa<DbgInfoIntrinsic>(call)) continue;
                    return fail(why, "fragment passed to something other than wmma");
                } else {
                    std::string text;
                    raw_string_ostream os(text);
                    u->print(os);
                    return fail(why, "fragment elements used directly:" + text);
                }
            }
        }
    }
    return true;
}

}  // namespace

bool build_wmma_trace(Module& m, KernelInfo& k, std::string* why) {
    Function& f = *k.scalar;
    LLVMContext& ctx = m.getContext();
    const DataLayout& dl = m.getDataLayout();
    // Cheap pre-check: some mma_sync in the kernel or what it calls directly.
    bool any_mma = false;
    for (Instruction& i : instructions(f))
        if (auto* call = dyn_cast<CallInst>(&i)) any_mma |= classify(call).op == Mma;
    if (!any_mma) return fail(why, "no mma_sync");

    auto* ptr = PointerType::get(ctx, 0);
    auto* fty = FunctionType::get(Type::getVoidTy(ctx), {ptr, ptr, ptr}, false);
    Function* t = Function::Create(fty, GlobalValue::ExternalLinkage, "__cpusim_trace_" + k.name, m);
    t->setVisibility(GlobalValue::ProtectedVisibility);
    for (const char* a : {"target-cpu", "target-features"})
        if (f.hasFnAttribute(a)) t->addFnAttr(f.getFnAttribute(a));
    Argument* params = t->getArg(0);
    Argument* sregs = t->getArg(1);
    Argument* rec = t->getArg(2);

    // The kernel's arguments, read from the launch's parameter buffer (the
    // same layout as the warp code's, simt.cpp).
    BasicBlock* prologue = BasicBlock::Create(ctx, "trace.params", t);
    IRBuilder<> b(prologue);
    ValueToValueMapTy vmap;
    uint64_t offset = 0;
    for (Argument& arg : f.args()) {
        Type* ty = arg.getType();
        uint64_t size, align;
        Type* byval = arg.getParamByValType();
        if (byval) {
            size = dl.getTypeAllocSize(byval);
            align = std::max<uint64_t>(dl.getABITypeAlign(byval).value(), arg.getParamAlign().valueOrOne().value());
        } else {
            size = dl.getTypeAllocSize(ty);
            align = dl.getABITypeAlign(ty).value();
        }
        offset = alignTo(offset, align);
        Value* p = b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), params, offset);
        vmap[&arg] = byval ? p : b.CreateAlignedLoad(ty, p, Align(align));
        offset += size;
    }
    SmallVector<ReturnInst*, 4> returns;
    CloneFunctionInto(t, &f, vmap, CloneFunctionChangeType::LocalChangesOnly, returns);
    b.CreateBr(cast<BasicBlock>(vmap[&f.getEntryBlock()]));

    // Inline everything but wmma.
    for (bool changed = true; changed;) {
        changed = false;
        for (Instruction& i : instructions(*t))
            if (auto* call = dyn_cast<CallInst>(&i)) {
                Function* callee = call->getCalledFunction();
                if (!callee || callee->isDeclaration() || callee->isIntrinsic() || classify(call).op != NotWmma) continue;
                InlineFunctionInfo info;
                if (InlineFunction(*call, info).isSuccess()) {
                    changed = true;
                    break;
                }
            }
    }
    if (!candidate(*t, why)) {
        t->eraseFromParent();
        return false;
    }

    // Special registers from the sregs array; wmma calls become records.
    auto* i32 = Type::getInt32Ty(ctx);
    auto* f64 = Type::getDoubleTy(ctx);
    // Fragments are told apart by their address in the running trace (arrays
    // of fragments indexed by loop counters included).
    FunctionCallee rec_load = m.getOrInsertFunction(
        "__cpusim_rec_load", FunctionType::get(Type::getVoidTy(ctx), {ptr, ptr, i32, i32, i32, ptr, i32}, false));
    FunctionCallee rec_store = m.getOrInsertFunction(
        "__cpusim_rec_store", FunctionType::get(Type::getVoidTy(ctx), {ptr, ptr, i32, ptr, i32, i32}, false));
    FunctionCallee rec_mma =
        m.getOrInsertFunction("__cpusim_rec_mma", FunctionType::get(Type::getVoidTy(ctx), {ptr, ptr, ptr, ptr, ptr}, false));
    FunctionCallee rec_fill =
        m.getOrInsertFunction("__cpusim_rec_fill", FunctionType::get(Type::getVoidTy(ctx), {ptr, ptr, f64}, false));
    std::vector<Instruction*> dead;
    for (Instruction& i : instructions(*t)) {
        auto* call = dyn_cast<CallInst>(&i);
        if (!call) continue;
        IRBuilder<> rb(call);
        if (auto* ii = dyn_cast<IntrinsicInst>(call)) {
            const int s = sreg_index(ii->getIntrinsicID());
            if (s < 0) continue;
            Value* v = rb.CreateAlignedLoad(i32, rb.CreateConstInBoundsGEP1_64(i32, sregs, s), Align(4));
            ii->replaceAllUsesWith(v);
            dead.push_back(ii);
            continue;
        }
        const WmmaCall w = classify(call);
        auto c32 = [&](int v) { return ConstantInt::get(i32, v); };
        switch (w.op) {
            case Load: {  // (fragment&, const T* p, unsigned ldm[, layout_t])
                Value* layout = call->arg_size() > 3 ? rb.CreateZExtOrTrunc(call->getArgOperand(3), i32) : c32(w.layout);
                rb.CreateCall(rec_load, {rec, call->getArgOperand(0), c32(w.kind), c32(w.type), layout,
                                         call->getArgOperand(1), rb.CreateZExtOrTrunc(call->getArgOperand(2), i32)});
                break;
            }
            case Store: {  // (T* p, const fragment&, unsigned ldm, layout_t)
                rb.CreateCall(rec_store, {rec, call->getArgOperand(1), c32(w.type), call->getArgOperand(0),
                                          rb.CreateZExtOrTrunc(call->getArgOperand(2), i32),
                                          rb.CreateZExtOrTrunc(call->getArgOperand(3), i32)});
                break;
            }
            case Mma:  // (d&, a, b, c)
                rb.CreateCall(rec_mma, {rec, call->getArgOperand(0), call->getArgOperand(1),
                                        call->getArgOperand(2), call->getArgOperand(3)});
                break;
            case Fill: {  // (fragment&, const T& value)
                Value* v = call->getArgOperand(1);
                Type* et = w.type == TS32 ? Type::getInt32Ty(ctx) : w.type == TF32 ? Type::getFloatTy(ctx) : nullptr;
                Value* d = ConstantFP::get(f64, 1e300);  // unknown element type: never zero, never a GEMM
                if (et) {
                    Value* x = rb.CreateLoad(et, v);
                    d = et->isIntegerTy() ? rb.CreateSIToFP(x, f64) : rb.CreateFPExt(x, f64);
                }
                rb.CreateCall(rec_fill, {rec, call->getArgOperand(0), d});
                break;
            }
            default:
                continue;
        }
        dead.push_back(call);
    }
    for (Instruction* i : dead) i->eraseFromParent();
    return true;
}

}  // namespace cpusim
