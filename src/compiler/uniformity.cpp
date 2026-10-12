#include "compiler/uniformity.h"
#include "compiler/simt.h"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IntrinsicsNVPTX.h>

using namespace llvm;

namespace cpusim {

SpecialReg special_reg(const Value* v) {
    if (auto* c = dyn_cast<CallInst>(v); c && c->getCalledFunction() && c->getCalledFunction()->getName() == kSharedBaseFn)
        return SpecialReg::SharedBase;
    auto* call = dyn_cast<IntrinsicInst>(v);
    if (!call) return SpecialReg::None;
    switch (call->getIntrinsicID()) {
        case Intrinsic::nvvm_read_ptx_sreg_tid_x: return SpecialReg::TidX;
        case Intrinsic::nvvm_read_ptx_sreg_tid_y: return SpecialReg::TidY;
        case Intrinsic::nvvm_read_ptx_sreg_tid_z: return SpecialReg::TidZ;
        case Intrinsic::nvvm_read_ptx_sreg_ntid_x: return SpecialReg::NtidX;
        case Intrinsic::nvvm_read_ptx_sreg_ntid_y: return SpecialReg::NtidY;
        case Intrinsic::nvvm_read_ptx_sreg_ntid_z: return SpecialReg::NtidZ;
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_x: return SpecialReg::CtaidX;
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_y: return SpecialReg::CtaidY;
        case Intrinsic::nvvm_read_ptx_sreg_ctaid_z: return SpecialReg::CtaidZ;
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_x: return SpecialReg::NctaidX;
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_y: return SpecialReg::NctaidY;
        case Intrinsic::nvvm_read_ptx_sreg_nctaid_z: return SpecialReg::NctaidZ;
        case Intrinsic::nvvm_read_ptx_sreg_laneid: return SpecialReg::LaneId;
        case Intrinsic::nvvm_read_ptx_sreg_warpsize: return SpecialReg::WarpSize;
        default: return SpecialReg::None;
    }
}

namespace {

bool is_back_edge(const BasicBlock* from, const BasicBlock* to, const DominatorTree& dt) {
    return dt.dominates(to, from);
}

}  // namespace

Uniformity::Uniformity(Function& f, DominatorTree& dt, PostDominatorTree& pdt, LoopInfo& li, bool varying_args)
    : f_(f), dt_(dt), pdt_(pdt), li_(li) {
    if (varying_args)
        for (Argument& a : f.args()) varying_.insert(&a);
    ReversePostOrderTraversal<Function*> rpot(&f);
    rpo_.assign(rpot.begin(), rpot.end());

    bool changed = true;
    while (changed) {
        changed = propagate_values();
        for (BasicBlock* bb : rpo_) {
            auto* br = dyn_cast<BranchInst>(bb->getTerminator());
            if (!br || !br->isConditional() || divergent_branches_.contains(bb)) continue;
            if (!is_varying(br->getCondition())) continue;
            divergent_branches_.insert(bb);
            propagate_branch(bb);
            changed = true;
        }
    }
    compute_divergent_blocks();
}

bool Uniformity::mark(const Value* v) { return varying_.insert(v).second; }

bool Uniformity::propagate_values() {
    bool any = false, changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : rpo_) {
            for (Instruction& inst : *bb) {
                if (is_varying(&inst) || inst.getType()->isVoidTy()) continue;
                bool v = false;
                if (isa<AllocaInst>(inst)) {
                    v = true;  // private memory: every lane has its own copy
                } else if (auto* ld = dyn_cast<LoadInst>(&inst)) {
                    v = is_varying(ld->getPointerOperand());
                } else if (inst.isAtomic()) {
                    v = true;
                } else if (auto* call = dyn_cast<CallInst>(&inst)) {
                    SpecialReg r = special_reg(call);
                    if (r == SpecialReg::TidX || r == SpecialReg::TidY || r == SpecialReg::TidZ || r == SpecialReg::LaneId)
                        v = true;
                    else if (r != SpecialReg::None)
                        v = false;
                    else if (is_cond_load(call))
                        v = any_of(call->args(), [&](const Use& u) { return is_varying(u.get()); });
                    else if (call->getCalledFunction() && call->getCalledFunction()->isIntrinsic() &&
                             isTriviallyVectorizable(call->getIntrinsicID()))
                        v = any_of(call->args(), [&](const Use& u) { return is_varying(u.get()); });
                    else
                        v = true;
                } else {
                    v = any_of(inst.operands(), [&](const Use& u) { return is_varying(u.get()); });
                }
                if (v && mark(&inst)) changed = any = true;
            }
        }
    }
    return any;
}

// Finds the blocks where paths starting at different successors of the
// divergent branch in `a` meet again (label propagation over forward edges, in
// reverse post order), and makes their phis varying. Loops the branch can
// leave get their exit values (LCSSA phis) marked varying too.
bool Uniformity::propagate_branch(BasicBlock* a) {
    DenseMap<const BasicBlock*, const BasicBlock*> label;
    for (BasicBlock* s : successors(a)) label[s] = s;

    DenseSet<const BasicBlock*> joins;
    bool after = false;
    for (BasicBlock* b : rpo_) {
        if (b == a) {
            after = true;
            continue;
        }
        if (!after) continue;
        SmallPtrSet<const BasicBlock*, 4> incoming;
        for (BasicBlock* p : predecessors(b)) {
            if (is_back_edge(p, b, dt_)) continue;
            if (p == a) incoming.insert(b);
            else if (auto it = label.find(p); it != label.end()) incoming.insert(it->second);
        }
        if (incoming.size() >= 2) {
            joins.insert(b);
            label[b] = b;
        } else if (incoming.size() == 1 && !label.count(b)) {
            label[b] = *incoming.begin();
        }
    }

    // Two back edges carrying different labels make the loop header a join.
    for (Loop* l = li_.getLoopFor(a); l; l = l->getParentLoop()) {
        SmallPtrSet<const BasicBlock*, 4> latch_labels;
        SmallVector<BasicBlock*, 4> latches;
        l->getLoopLatches(latches);
        for (BasicBlock* latch : latches)
            if (auto it = label.find(latch); it != label.end()) latch_labels.insert(it->second);
        if (latch_labels.size() >= 2) joins.insert(l->getHeader());

        // Lanes may leave this loop in different iterations.
        SmallVector<BasicBlock*, 4> exits;
        l->getExitBlocks(exits);
        for (BasicBlock* e : exits)
            for (PHINode& phi : e->phis()) mark(&phi);
    }

    bool changed = false;
    for (const BasicBlock* j : joins)
        for (const PHINode& phi : j->phis()) changed |= mark(&phi);
    return changed;
}

// A block is divergent when it is control dependent on a divergent branch, or
// on any branch of a divergent block.
void Uniformity::compute_divergent_blocks() {
    SmallVector<BasicBlock*, 16> work;
    for (const BasicBlock* bb : divergent_branches_) work.push_back(const_cast<BasicBlock*>(bb));
    DenseSet<const BasicBlock*> processed;
    while (!work.empty()) {
        BasicBlock* a = work.pop_back_val();
        if (!processed.insert(a).second) continue;
        DomTreeNode* ipdom_node = pdt_.getNode(a) ? pdt_.getNode(a)->getIDom() : nullptr;
        BasicBlock* ipdom = ipdom_node ? ipdom_node->getBlock() : nullptr;
        for (BasicBlock* s : successors(a)) {
            for (DomTreeNode* n = pdt_.getNode(s); n && n->getBlock() && n->getBlock() != ipdom; n = n->getIDom()) {
                BasicBlock* b = n->getBlock();
                if (divergent_blocks_.insert(b).second && b->getTerminator()->getNumSuccessors() > 1)
                    work.push_back(b);
            }
        }
    }
}

}  // namespace cpusim
