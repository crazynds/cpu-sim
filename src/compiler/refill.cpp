// Lanes that refill (make_refill_kernel in simt.h).
//
// A warp runs its lanes' threads in step: a loop goes on while any lane is
// in it, so when the threads of a warp loop for different counts (a search
// whose depth depends on the data, like HeCBench's nqueen) most lanes sit
// masked off until the longest one ends. For a kernel whose lanes never look
// at each other, a lane that is done can take another thread instead. The
// kernel is rewritten so that its main loop L (the largest top-level loop)
// becomes the loop of the whole call:
//
//     entry:      lin = first()                       (lane's first thread)
//     find0:      if lin == ~0: exit;  thread state from lin
//                 P (code before L) ... -> head,  or a return -> lin = next(), find0
//     head:       if refill: find1;  else L's header
//     L:          the loop, as written; back edges -> head; exits go to
//     B:          a copy of the code after L; its return -> refill = true, head
//     find1:      lin = next(); if lin == ~0: exit;  thread state from lin, refill = false
//                 A (a copy of P) ... -> head,  or a return -> refill = true, head
//
// `head` is the only way in from outside, so everything from head on is one
// loop: a lane that leaves L runs B, find1 and A while the other lanes keep
// iterating L, and the warp leaves when every lane is out of threads. Every
// exit leaves one loop level only (find1 is no loop of its own: lanes that
// left two loops at once stayed in the outer loop's mask, and it spun).
// threadIdx, blockIdx and the lane id are the lane's current thread's, kept
// in variables that find0 and find1 set from the linear index lin = block *
// threads per block + thread. Values that cross blocks are demoted to
// memory first, so the copies need no SSA repair; the variables are
// promoted back at the end.
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "compiler/simt.h"
#include "compiler/uniformity.h"

using namespace llvm;

namespace cpusim {

namespace {

// As LLVM's reg2mem: a value used in another block, or by a phi.
bool escapes(const Instruction& inst) {
    if (!inst.getType()->isSized()) return false;
    for (const User* u : inst.users()) {
        const auto* ui = cast<Instruction>(u);
        if (ui->getParent() != inst.getParent() || isa<PHINode>(ui)) return true;
    }
    return false;
}

// Blocks reachable from `starts` without entering `stop`.
std::vector<BasicBlock*> reachable(ArrayRef<BasicBlock*> starts, const SmallPtrSetImpl<BasicBlock*>& stop) {
    SmallPtrSet<BasicBlock*, 32> seen;
    std::vector<BasicBlock*> order, todo(starts.begin(), starts.end());
    while (!todo.empty()) {
        BasicBlock* bb = todo.back();
        todo.pop_back();
        if (stop.contains(bb) || !seen.insert(bb).second) continue;
        order.push_back(bb);
        for (BasicBlock* s : successors(bb)) todo.push_back(s);
    }
    return order;
}

}  // namespace

bool make_refill_kernel(Function& f, const KernelInfo& k, std::string* why) {
    auto no = [&](const char* w) {
        if (why) *why = w;
        return false;
    };
    LLVMContext& ctx = f.getContext();
    Module& m = *f.getParent();
    auto* i32 = Type::getInt32Ty(ctx);
    auto* i64 = Type::getInt64Ty(ctx);
    if (!f.getReturnType()->isVoidTy()) return no("returns a value");

    // Special registers: the ones a thread's identity gives are replaced;
    // ntid, nctaid and the warp size are the same for every thread; any
    // other (a warp or SM id) would tell where the thread runs.
    for (BasicBlock& bb : f)
        for (Instruction& inst : bb)
            if (auto* call = dyn_cast<CallInst>(&inst); call && call->getCalledFunction() &&
                                                         call->getCalledFunction()->getName().starts_with("llvm.nvvm.read.ptx.sreg.") &&
                                                         special_reg(call) == SpecialReg::None)
                return no("a special register other than the thread's indices and the launch's dimensions");

    // The main loop (the top-level loop with the most instructions) must be
    // one whose threads leave it in different iterations: with the same trip
    // count everywhere the lanes stay together and refilling only costs
    // (nbody, md: 30% slower with the refill entry offered).
    BasicBlock* main_header = nullptr;
    {
        DominatorTree dt(f);
        PostDominatorTree pdt(f);
        LoopInfo li(dt);
        size_t best = 0;
        Loop* l0 = nullptr;
        for (Loop* l : li) {
            size_t n = 0;
            for (BasicBlock* bb : l->blocks()) n += bb->size();
            if (n > best) best = n, l0 = l;
        }
        if (!l0) return no("no loop");
        Uniformity uni(f, dt, pdt, li);
        SmallVector<BasicBlock*, 4> exiting;
        l0->getExitingBlocks(exiting);
        bool divergent = false;
        for (BasicBlock* bb : exiting) divergent |= uni.is_divergent_branch(bb);
        if (!divergent) return no("every thread runs the main loop the same number of times");
        main_header = l0->getHeader();
    }

    // One return block: the end of a thread.
    SmallVector<ReturnInst*, 4> rets;
    for (BasicBlock& bb : f)
        if (auto* r = dyn_cast<ReturnInst>(bb.getTerminator())) rets.push_back(r);
    if (rets.empty()) return no("no return");
    BasicBlock* done = BasicBlock::Create(ctx, "thread.done", &f);
    ReturnInst::Create(ctx, done);
    for (ReturnInst* r : rets) ReplaceInstWithInst(r, BranchInst::Create(done));

    // A new entry block holding the static allocas.
    BasicBlock* old_entry = &f.getEntryBlock();
    SmallVector<AllocaInst*, 16> allocas;  // (static only while in the entry block)
    for (Instruction& inst : *old_entry)
        if (auto* a = dyn_cast<AllocaInst>(&inst); a && a->isStaticAlloca()) allocas.push_back(a);
    BasicBlock* entry = BasicBlock::Create(ctx, "refill.entry", &f, old_entry);
    BranchInst::Create(old_entry, entry);
    for (AllocaInst* a : allocas) a->moveBefore(entry->getTerminator());
    const auto at = [&]() { return std::optional<BasicBlock::iterator>(entry->getTerminator()->getIterator()); };

    // Every value that crosses blocks into memory.
    std::vector<AllocaInst*> promote;
    {
        SmallVector<Instruction*, 64> values;
        SmallVector<PHINode*, 16> phis;
        for (BasicBlock& bb : f) {
            if (&bb == entry) continue;
            for (Instruction& inst : bb) {
                // As reg2mem: escaping values first (phis included: their
                // uses elsewhere), then every phi.
                if (escapes(inst)) values.push_back(&inst);
                if (auto* phi = dyn_cast<PHINode>(&inst)) phis.push_back(phi);
            }
        }
        for (Instruction* inst : values) promote.push_back(DemoteRegToStack(*inst, false, at()));
        for (PHINode* phi : phis) promote.push_back(DemotePHIToStack(phi, at()));
    }

    // The thread's identity: variables, set from lin.
    IRBuilder<> eb(entry->getTerminator());
    enum { TidX, TidY, TidZ, CtaX, CtaY, CtaZ, Lane, NState };
    AllocaInst* state[NState];
    for (auto& a : state) promote.push_back(a = eb.CreateAlloca(i32));
    AllocaInst* lin_slot = eb.CreateAlloca(i64, nullptr, "lin");
    promote.push_back(lin_slot);
    AllocaInst* refill_slot = eb.CreateAlloca(eb.getInt1Ty(), nullptr, "refill");
    promote.push_back(refill_slot);
    eb.CreateStore(eb.getFalse(), refill_slot);
    {
        SmallVector<CallInst*, 16> reads;
        for (BasicBlock& bb : f)
            for (Instruction& inst : bb)
                if (auto* call = dyn_cast<CallInst>(&inst)) {
                    SpecialReg r = special_reg(call);
                    if (r == SpecialReg::TidX || r == SpecialReg::TidY || r == SpecialReg::TidZ ||
                        r == SpecialReg::CtaidX || r == SpecialReg::CtaidY || r == SpecialReg::CtaidZ ||
                        r == SpecialReg::LaneId)
                        reads.push_back(call);
                }
        for (CallInst* call : reads) {
            int s = 0;
            switch (special_reg(call)) {
                case SpecialReg::TidX: s = TidX; break;
                case SpecialReg::TidY: s = TidY; break;
                case SpecialReg::TidZ: s = TidZ; break;
                case SpecialReg::CtaidX: s = CtaX; break;
                case SpecialReg::CtaidY: s = CtaY; break;
                case SpecialReg::CtaidZ: s = CtaZ; break;
                default: s = Lane; break;
            }
            IRBuilder<> b(call);
            Value* v = b.CreateLoad(i32, state[s]);
            if (v->getType() != call->getType()) v = b.CreateZExtOrTrunc(v, call->getType());
            call->replaceAllUsesWith(v);
            call->eraseFromParent();
        }
    }

    // The main loop, chosen above (demotion leaves the blocks as they were).
    DominatorTree dt(f);
    LoopInfo li(dt);
    Loop* main = li.getLoopFor(main_header);
    if (!main || main->getHeader() != main_header) return no("the main loop changed");
    if (!main->getLoopPreheader() && !InsertPreheaderForLoop(main, &dt, &li, nullptr, false))
        return no("the main loop has no preheader");
    BasicBlock* header = main->getHeader();
    BasicBlock* preheader = main->getLoopPreheader();  // (copying P gives the header a second way in)
    SmallPtrSet<BasicBlock*, 32> in_loop(main->block_begin(), main->block_end());
    // P starts at the old entry (or at the preheader, if the loop was the entry).
    BasicBlock* p_start = in_loop.contains(old_entry) ? preheader : old_entry;
    SmallVector<BasicBlock*, 8> exits;
    for (BasicBlock* bb : main->blocks())
        for (BasicBlock* s : successors(bb))
            if (!in_loop.contains(s) && !is_contained(exits, s)) exits.push_back(s);
    SmallPtrSet<BasicBlock*, 1> none;
    const std::vector<BasicBlock*> pre = reachable({p_start}, in_loop);
    const std::vector<BasicBlock*> post = reachable(exits, none);
    for (BasicBlock* bb : post)
        if (in_loop.contains(bb)) return no("the code after the main loop goes back to it");

    // Copies of P (A) and of the code after the loop (B).
    auto copy = [&](const std::vector<BasicBlock*>& blocks, const char* suffix, ValueToValueMapTy& vm) {
        std::vector<BasicBlock*> out;
        for (BasicBlock* bb : blocks) {
            BasicBlock* c = CloneBasicBlock(bb, vm, suffix, &f);
            vm[bb] = c;
            out.push_back(c);
        }
        for (BasicBlock* c : out)
            for (Instruction& inst : *c) RemapInstruction(&inst, vm, RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    };
    ValueToValueMapTy va, vb;
    copy(pre, ".refill", va);
    copy(post, ".after", vb);

    FunctionCallee first_fn = m.getOrInsertFunction(kRefillFirstFn, FunctionType::get(i64, false));
    FunctionCallee next_fn = m.getOrInsertFunction(kRefillNextFn, FunctionType::get(i64, false));
    BasicBlock* exit = BasicBlock::Create(ctx, "refill.exit", &f);
    ReturnInst::Create(ctx, exit);
    // The thread state from lin, then on to `to`.
    auto set_thread = [&](IRBuilder<>& b, Value* lin, BasicBlock* to) {
        auto sreg = [&](Intrinsic::ID id) { return b.CreateZExt(b.CreateIntrinsic(id, {}, {}), i64); };
        Value* bx = sreg(Intrinsic::nvvm_read_ptx_sreg_ntid_x);
        Value* by = sreg(Intrinsic::nvvm_read_ptx_sreg_ntid_y);
        Value* bz = sreg(Intrinsic::nvvm_read_ptx_sreg_ntid_z);
        Value* gx = sreg(Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
        Value* gy = sreg(Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
        Value* plane = b.CreateMul(bx, by);
        Value* threads = b.CreateMul(plane, bz);
        Value* block = b.CreateUDiv(lin, threads);
        Value* t = b.CreateURem(lin, threads);
        Value* v[NState];
        v[TidX] = b.CreateURem(t, bx);
        v[TidY] = b.CreateURem(b.CreateUDiv(t, bx), by);
        v[TidZ] = b.CreateUDiv(t, plane);
        v[CtaX] = b.CreateURem(block, gx);
        v[CtaY] = b.CreateURem(b.CreateUDiv(block, gx), gy);
        v[CtaZ] = b.CreateUDiv(block, b.CreateMul(gx, gy));
        v[Lane] = b.CreateURem(t, b.getInt64(k.warp_size));
        for (int s = 0; s < NState; ++s) b.CreateStore(b.CreateTrunc(v[s], i32), state[s]);
        b.CreateBr(to);
    };
    // find0: a loop of its own, before the main one.
    BasicBlock* find0 = BasicBlock::Create(ctx, "find0.head", &f);
    BasicBlock* find0_thread = BasicBlock::Create(ctx, "find0.thread", &f);
    BasicBlock* next0 = BasicBlock::Create(ctx, "find0.next", &f);
    {
        IRBuilder<> b(find0);
        Value* lin = b.CreateLoad(i64, lin_slot);
        b.CreateCondBr(b.CreateICmpEQ(lin, b.getInt64(~0ull)), exit, find0_thread);
        b.SetInsertPoint(find0_thread);
        set_thread(b, lin, p_start);
        b.SetInsertPoint(next0);
        b.CreateStore(b.CreateCall(next_fn), lin_slot);
        b.CreateBr(find0);
    }
    // head and find1, inside the main loop.
    BasicBlock* head = BasicBlock::Create(ctx, "refill.head", &f);
    BasicBlock* find1 = BasicBlock::Create(ctx, "find1", &f);
    BasicBlock* find1_thread = BasicBlock::Create(ctx, "find1.thread", &f);
    BasicBlock* done1 = BasicBlock::Create(ctx, "thread.done.refill", &f);
    {
        IRBuilder<> b(head);
        b.CreateCondBr(b.CreateLoad(b.getInt1Ty(), refill_slot), find1, header);
    }
    {
        IRBuilder<> b(find1);
        Value* lin = b.CreateCall(next_fn);
        b.CreateCondBr(b.CreateICmpEQ(lin, b.getInt64(~0ull)), exit, find1_thread);
        b.SetInsertPoint(find1_thread);
        b.CreateStore(b.getFalse(), refill_slot);
        set_thread(b, lin, cast<BasicBlock>(va[p_start]));
        b.SetInsertPoint(done1);
        b.CreateStore(b.getTrue(), refill_slot);
        b.CreateBr(head);
    }
    {
        Instruction* term = entry->getTerminator();
        IRBuilder<> b(term);
        b.CreateStore(b.CreateCall(first_fn), lin_slot);
        b.CreateBr(find0);
        term->eraseFromParent();
    }
    // The end of a thread: in P, the next one through find0; in B and A, through head.
    auto retarget = [&](BasicBlock* bb, BasicBlock* to) {
        if (bb) ReplaceInstWithInst(bb->getTerminator(), BranchInst::Create(to));
    };
    if (is_contained(pre, done)) retarget(done, next0);
    if (Value* c = va.lookup(done)) retarget(cast<BasicBlock>(c), done1);
    if (Value* c = vb.lookup(done)) retarget(cast<BasicBlock>(c), done1);
    // Into the main loop only through head: from the preheader (P and A) and
    // L's back edges. L's exits go into B.
    SmallVector<BasicBlock*, 32> to_head(main->block_begin(), main->block_end());
    to_head.push_back(preheader);
    to_head.push_back(cast<BasicBlock>(va[preheader]));
    for (BasicBlock* bb : to_head) {
        Instruction* term = bb->getTerminator();
        for (unsigned i = 0; i < term->getNumSuccessors(); ++i) {
            BasicBlock* s = term->getSuccessor(i);
            if (s == header) term->setSuccessor(i, head);
            else if (in_loop.contains(bb) && !in_loop.contains(s)) term->setSuccessor(i, cast<BasicBlock>(vb[s]));
        }
    }

    removeUnreachableBlocks(f);
    std::vector<AllocaInst*> live;
    for (AllocaInst* a : promote)
        if (a->getParent() && isAllocaPromotable(a)) live.push_back(a);
    DominatorTree dt2(f);
    PromoteMemToReg(live, dt2);
    // Promotion leaves phis nobody reads (values the copies no longer use).
    for (bool changed = true; changed;) {
        changed = false;
        for (BasicBlock& bb : f)
            for (Instruction& inst : make_early_inc_range(bb))
                if (isInstructionTriviallyDead(&inst)) {
                    inst.eraseFromParent();
                    changed = true;
                }
    }
    // The loops back in LCSSA form, as the warp builder expects: a value
    // leaving a loop through a phi of its exit block is how
    // the uniformity analysis learns that lanes leaving in different
    // iterations carry different values (without it a uniform step counter
    // read after a divergent loop was taken as one value for all lanes).
    DominatorTree dt3(f);
    LoopInfo li3(dt3);
    // (Not loop-simplify form too: its extra preheaders and exit blocks made
    // nqueen's refill entry 50% slower.)
    for (Loop* l : li3) formLCSSARecursively(*l, dt3, &li3, nullptr);
    return true;
}

}  // namespace cpusim
