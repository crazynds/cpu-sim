#pragma once

#include <llvm/ADT/DenseSet.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>

namespace cpusim {

// Which values differ between the lanes of a warp (varying) and which are the
// same for all of them (uniform), plus which blocks may run with only part of
// the warp active. Works on the per-thread SSA form of a kernel.
//
// Sources of divergence: threadIdx, the lane id, per-thread memory (allocas)
// and anything loaded through a varying pointer. Divergent branches add
// "sync dependence": a phi at a point where paths from different sides of a
// divergent branch meet is varying, and so is every value leaving a loop whose
// lanes may exit in different iterations.
class Uniformity {
public:
    Uniformity(llvm::Function& f, llvm::DominatorTree& dt, llvm::PostDominatorTree& pdt, llvm::LoopInfo& li);

    bool is_varying(const llvm::Value* v) const { return varying_.contains(v); }
    void set_varying(const llvm::Value* v) { varying_.insert(v); }

    // True if the block may execute with fewer lanes than the warp started with.
    bool is_divergent_block(const llvm::BasicBlock* bb) const { return divergent_blocks_.contains(bb); }
    bool is_divergent_branch(const llvm::BasicBlock* bb) const { return divergent_branches_.contains(bb); }

private:
    bool propagate_values();
    bool propagate_branch(llvm::BasicBlock* a);
    void compute_divergent_blocks();
    bool mark(const llvm::Value* v);

    llvm::Function& f_;
    llvm::DominatorTree& dt_;
    llvm::PostDominatorTree& pdt_;
    llvm::LoopInfo& li_;
    std::vector<llvm::BasicBlock*> rpo_;
    llvm::DenseSet<const llvm::Value*> varying_;
    llvm::DenseSet<const llvm::BasicBlock*> divergent_branches_;
    llvm::DenseSet<const llvm::BasicBlock*> divergent_blocks_;
};

// The special registers behind threadIdx and friends.
enum class SpecialReg { None, TidX, TidY, TidZ, NtidX, NtidY, NtidZ, CtaidX, CtaidY, CtaidZ, NctaidX, NctaidY, NctaidZ,
                        LaneId, WarpSize, SharedBase };
SpecialReg special_reg(const llvm::Value* v);

}  // namespace cpusim
