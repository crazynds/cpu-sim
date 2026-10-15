// The SIMT pass: the linearized control flow (loop levels, masks, uniform
// loops), barriers and their saved state, and the profiling counters.
#include "compiler/simt_internal.h"

namespace cpusim {
namespace simt_internal {

// ---------------------------------------------------------------- structure

Node WarpBuilder::node_of(BasicBlock* bb, Loop* level) {
    Loop* l = li_->getLoopFor(bb);
    if (l == level) return {bb, nullptr};
    while (l && l->getParentLoop() != level) l = l->getParentLoop();
    return {nullptr, l};
}

// Reverse post order of a loop level, with subloops collapsed and back edges to
// the level's header ignored: every node comes after all of its predecessors.
// The node holds a yield point (a loop waiting for another thread's write).
bool WarpBuilder::waits(const Node& n) {
    if (yield_blocks_.empty()) return false;
    if (n.block) return yield_blocks_.contains(n.block);
    return any_of(n.loop->blocks(), [&](BasicBlock* bb) { return yield_blocks_.contains(bb); });
}

std::vector<Node> WarpBuilder::topo_order(Loop* level) {
    BasicBlock* start = level ? level->getHeader() : &f_.getEntryBlock();
    std::vector<Node> post;
    DenseSet<const void*> seen;
    auto key = [](const Node& n) -> const void* { return n.block ? (const void*)n.block : (const void*)n.loop; };

    std::function<void(const Node&)> dfs = [&](const Node& n) {
        if (!seen.insert(key(n)).second) return;
        SmallVector<BasicBlock*, 8> exits_from;
        if (n.block) {
            exits_from.push_back(n.block);
        } else {
            for (BasicBlock* bb : n.loop->blocks()) exits_from.push_back(bb);
        }
        // Nodes that wait for other threads (yield points) are visited first,
        // so they come after their siblings in the order: a branch that
        // writes a flag runs before the branch that waits for it.
        SmallVector<Node, 8> next;
        for (BasicBlock* bb : exits_from) {
            for (BasicBlock* s : successors(bb)) {
                if (level && (!level->contains(s) || s == level->getHeader())) continue;
                if (n.loop && n.loop->contains(s)) continue;
                next.push_back(node_of(s, level));
            }
        }
        std::stable_partition(next.begin(), next.end(), [&](const Node& x) { return waits(x); });
        for (const Node& x : next) dfs(x);
        post.push_back(n);
    };
    dfs(node_of(start, level));
    return {post.rbegin(), post.rend()};
}

// ---------------------------------------------------------------- emission

void WarpBuilder::emit_prologue() {
    auto* i32 = b_.getInt32Ty();
    auto* ptr = PointerType::get(ctx_, 0);
    mask_type_ = FixedVectorType::get(b_.getInt1Ty(), W);

    w_ = fn_ ? fn_
             : Function::Create(FunctionType::get(i32, {ptr}, false), GlobalValue::ExternalLinkage,
                                (refill_ ? std::string("__cpusim_entryr_")
                                 : profile_ ? std::string("__cpusim_prof_")
                                 : W <= k_.warp_size ? std::string("__cpusim_entry_")
                                                         : "__cpusim_entry" + std::to_string(W) +
                                                               (block_wide_ ? "_" : "s_")) +
                                    k_.name,
                                m_);
    w_->addFnAttr(Attribute::NoUnwind);
    w_->addFnAttr("prefer-vector-width", "512");
    // Frames holding every lane's per-thread arrays can be megabytes: touch
    // each page on the way down, so an overflow hits the guard page and
    // faults instead of writing past it.
    w_->addFnAttr("probe-stack", "inline-asm");
    w_->addFnAttr("min-legal-vector-width", "512");
    ctx_arg_ = w_->getArg(0);
    ctx_arg_->setName("ctx");
    w_->addParamAttr(0, Attribute::NoAlias);
    w_->addParamAttr(0, Attribute::getWithAlignment(ctx_, Align(64)));
    w_->addParamAttr(0, Attribute::getWithDereferenceableBytes(ctx_, cpusim_ctx_size(W)));

    alloca_block_ = BasicBlock::Create(ctx_, "allocas", w_);
    BasicBlock* entry = BasicBlock::Create(ctx_, "entry", w_);
    entry_ = entry;
    b_.SetInsertPoint(entry);

    auto field = [&](unsigned offset) { return b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, offset); };
    auto load_u32 = [&](unsigned offset, const char* name) { return b_.CreateAlignedLoad(i32, field(offset), Align(4), name); };

    // Kernel parameters: uniform, read from the launch's parameter buffer.
    // A device function's arguments: one per lane, after the context and the mask.
    const DataLayout& dl = m_.getDataLayout();
    Value* params = fn_ ? nullptr : b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_PARAMS), Align(8), "params");
    uint64_t offset = 0;
    if (fn_)
        for (Argument& arg : f_.args()) vector_map_[&arg] = w_->getArg(2 + arg.getArgNo());
    for (Argument& arg : fn_ ? make_range(f_.arg_end(), f_.arg_end()) : f_.args()) {
        Type* t = arg.getType();
        uint64_t size, align;
        Type* byval = arg.getParamByValType();
        if (byval) {
            size = dl.getTypeAllocSize(byval);
            align = std::max<uint64_t>(dl.getABITypeAlign(byval).value(), arg.getParamAlign().valueOrOne().value());
        } else {
            size = dl.getTypeAllocSize(t);
            align = dl.getABITypeAlign(t).value();
        }
        offset = alignTo(offset, align);
        Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), params, offset);
        scalar_map_[&arg] = byval ? p : b_.CreateAlignedLoad(t, p, Align(align), arg.getName());
        if (!profile_ && !refill_ && W <= k_.warp_size) k_.params.push_back({offset, size});
        offset += size;
    }
    if (!fn_) k_.param_bytes = offset;
    if (profile_) counters_ = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_COUNTERS), Align(8), "counters");

    // Special registers.
    auto* v32 = FixedVectorType::get(i32, W);
    if (k_.spec_block[0]) {
        emit_specialized_tid();
    } else if (k_.spec_class) {
        emit_class_tid();
    } else {
        special_[(int)SpecialReg::TidX] = b_.CreateAlignedLoad(v32, field(CPUSIM_CTX_TID(W, 0)), Align(64), "tid.x");
        special_[(int)SpecialReg::TidY] = b_.CreateAlignedLoad(v32, field(CPUSIM_CTX_TID(W, 1)), Align(64), "tid.y");
        special_[(int)SpecialReg::TidZ] = b_.CreateAlignedLoad(v32, field(CPUSIM_CTX_TID(W, 2)), Align(64), "tid.z");
    }
    // blockIdx within CUDA's grid limits: lets the optimizer (and
    // linear_form) prove that index arithmetic built on it cannot wrap.
    auto ranged = [&](Value* v, uint32_t limit) {
        cast<Instruction>(v)->setMetadata(
            LLVMContext::MD_range,
            MDNode::get(ctx_, {ConstantAsMetadata::get(b_.getInt32(0)), ConstantAsMetadata::get(b_.getInt32(limit))}));
        return v;
    };
    special_[(int)SpecialReg::CtaidX] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX, "ctaid.x"), 0x7fffffff);
    special_[(int)SpecialReg::CtaidY] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX + 4, "ctaid.y"), 65535);
    special_[(int)SpecialReg::CtaidZ] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX + 8, "ctaid.z"), 65535);
    if (k_.spec_block[0]) {
        special_[(int)SpecialReg::NtidX] = b_.getInt32(k_.spec_block[0]);
        special_[(int)SpecialReg::NtidY] = b_.getInt32(k_.spec_block[1]);
        special_[(int)SpecialReg::NtidZ] = b_.getInt32(k_.spec_block[2]);
    } else if (k_.spec_class) {  // 1D, blockDim.x a uniform
        special_[(int)SpecialReg::NtidX] = load_u32(CPUSIM_CTX_BLOCK_DIM, "ntid.x");
        special_[(int)SpecialReg::NtidY] = b_.getInt32(1);
        special_[(int)SpecialReg::NtidZ] = b_.getInt32(1);
    } else {
        special_[(int)SpecialReg::NtidX] = load_u32(CPUSIM_CTX_BLOCK_DIM, "ntid.x");
        special_[(int)SpecialReg::NtidY] = load_u32(CPUSIM_CTX_BLOCK_DIM + 4, "ntid.y");
        special_[(int)SpecialReg::NtidZ] = load_u32(CPUSIM_CTX_BLOCK_DIM + 8, "ntid.z");
    }
    special_[(int)SpecialReg::NctaidX] = load_u32(CPUSIM_CTX_GRID_DIM, "nctaid.x");
    special_[(int)SpecialReg::NctaidY] = load_u32(CPUSIM_CTX_GRID_DIM + 4, "nctaid.y");
    special_[(int)SpecialReg::NctaidZ] = load_u32(CPUSIM_CTX_GRID_DIM + 8, "nctaid.z");
    SmallVector<Constant*, 64> lanes;  // CUDA lane id: position inside the warp
    for (unsigned i = 0; i < W; ++i) lanes.push_back(b_.getInt32(i % k_.warp_size));
    special_[(int)SpecialReg::LaneId] = ConstantVector::get(lanes);
    special_[(int)SpecialReg::WarpSize] = b_.getInt32(k_.warp_size);
    special_[(int)SpecialReg::SharedBase] = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_SHARED), Align(8), "shared");
    Value* lane_mask = fn_ ? w_->getArg(1)
                       : refill_ ? Constant::getAllOnesValue(mask_type_)  // lanes without a thread leave at once
                       : spec_lane_mask_
                           ? spec_lane_mask_
                           : b_.CreateBitCast(b_.CreateAlignedLoad(b_.getIntNTy(W), field(CPUSIM_CTX_LANE_MASK(W)),
                                                                   Align(64), "lane_mask"),
                                              mask_type_, "initial_mask");

    // Everything read so far (context fields, kernel parameters) is constant
    // for the whole launch. Saying so lets the register allocator reload such
    // values where needed instead of spilling them at every entry.
    MDNode* empty = MDNode::get(ctx_, {});
    for (Instruction& inst : *entry) {
        auto* ld = dyn_cast<LoadInst>(&inst);
        if (!ld) continue;
        // Only the context and the parameters; never the frame (it changes
        // between calls, and within a block function between rounds).
        Value* base = getUnderlyingObject(ld->getPointerOperand());
        if (base != ctx_arg_ && base != params) continue;
        ld->setMetadata(LLVMContext::MD_invariant_load, empty);
        if (ld->getType()->isPointerTy() && params && ld == params)
            ld->setMetadata(LLVMContext::MD_dereferenceable,
                            MDNode::get(ctx_, ConstantAsMetadata::get(b_.getInt64(std::max<uint64_t>(k_.param_bytes, 1)))));
    }

    // With barriers, a warp either starts (state 0) or resumes after one.
    if (!barrier_blocks_.empty()) {
        frame_ = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_FRAME), Align(8), "frame");
        state_ptr_ = frame_;
        Value* state = b_.CreateAlignedLoad(i32, state_ptr_, Align(64), "state");
        BasicBlock* init = BasicBlock::Create(ctx_, "start", w_);
        dispatch_ = BasicBlock::Create(ctx_, "resume", w_);
        cache_block_ = BasicBlock::Create(ctx_, "warp.invariants", w_);
        b_.CreateCondBr(b_.CreateICmpEQ(state, b_.getInt32(0)), cache_block_, dispatch_);
        BranchInst::Create(init, cache_block_);
        b_.SetInsertPoint(init);
    }

    // Slots: one mask per block, one "next iteration" mask per loop, and the
    // demoted values (scalar for uniform values, <32 x T> for varying ones).
    IRBuilder<> ab(alloca_block_);
    for (BasicBlock& bb : f_) {
        AllocaInst* a = ab.CreateAlloca(mask_type_, nullptr, "mask." + bb.getName());
        block_mask_[&bb] = a;
        b_.CreateStore(&bb == &f_.getEntryBlock() ? lane_mask : no_lanes(), a);
    }
    for (Loop* l : li_->getLoopsInPreorder()) {
        AllocaInst* a = ab.CreateAlloca(mask_type_, nullptr, "next." + l->getHeader()->getName());
        next_mask_[l] = a;
        b_.CreateStore(no_lanes(), a);
    }
    for (const AllocaInst* slot : demoted_order_) {
        Type* t = slot->getAllocatedType();
        bool varying = uni_->is_varying(slot);
        if (varying && !(t->isIntegerTy() || t->isFloatingPointTy() || t->isPointerTy())) {
            fail("varying value of unsupported type", slot);
            continue;
        }
        slots_[slot] = ab.CreateAlloca(varying ? vec_type(t) : t, nullptr, slot->getName());
    }
    if (fn_ && !f_.getReturnType()->isVoidTy()) {
        ret_slot_ = ab.CreateAlloca(vec_type(f_.getReturnType()), nullptr, "ret");
        ab.CreateStore(PoisonValue::get(ret_slot_->getAllocatedType()), ret_slot_);
    }
    ab.CreateBr(entry);

    // Frame layout: the resume state at offset 0, then every slot.
    if (!barrier_blocks_.empty()) {
        const DataLayout& dl = m_.getDataLayout();
        uint64_t offset = 64;
        for (Instruction& inst : *alloca_block_) {
            auto* a = dyn_cast<AllocaInst>(&inst);
            if (!a) continue;
            Type* t = a->getAllocatedType();
            uint64_t align = std::min<uint64_t>(64, dl.getPrefTypeAlign(t).value());
            offset = alignTo(offset, align);
            state_.push_back({a, offset});
            offset += dl.getTypeAllocSize(t);
        }
        cache_end_ = alignTo(offset, 64);
        note_frame_size();

        // Per call, never saved: created after the frame layout.
        pass_used_ = new AllocaInst(Type::getInt1Ty(ctx_), 0, "barrier.pass.used", alloca_block_->getTerminator()->getIterator());
        new StoreInst(ConstantInt::getFalse(ctx_), pass_used_, entry_->getTerminator()->getIterator());
    }
}

// Stores every slot for now; compute_resume_liveness() later removes the
// stores of slots the code after the barrier never reads before writing.
void WarpBuilder::emit_barrier(BasicBlock* bb) {
    if (yield_blocks_.contains(bb)) {
        // A yield point: always suspend (the other warps must run), and
        // resume right here.
        BasicBlock* resume = BasicBlock::Create(ctx_, "after.yield", w_);
        resumes_.push_back(resume);
        barrier_order_.push_back(bb);
        save_state(int(resumes_.size()));
        b_.CreateRet(b_.getInt32(CPUSIM_YIELD));
        b_.SetInsertPoint(resume);
        return;
    }
    BasicBlock* resume = BasicBlock::Create(ctx_, "after." + bb->getName(), w_);
    BasicBlock* through = BasicBlock::Create(ctx_, "barrier.through", w_);
    BasicBlock* suspend = BasicBlock::Create(ctx_, "barrier.suspend", w_);
    resumes_.push_back(resume);
    barrier_order_.push_back(bb);
    Value* pass = b_.CreateAlignedLoad(b_.getInt32Ty(),
                                       b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_BARRIER_PASS),
                                       Align(4), "barrier.pass");
    Value* go = b_.CreateAnd(b_.CreateICmpNE(pass, b_.getInt32(0)),
                             b_.CreateNot(b_.CreateLoad(b_.getInt1Ty(), pass_used_)));
    b_.CreateCondBr(go, through, suspend);

    b_.SetInsertPoint(through);
    b_.CreateStore(b_.getTrue(), pass_used_);
    b_.CreateBr(resume);

    b_.SetInsertPoint(suspend);
    save_state(int(resumes_.size()));
    b_.CreateRet(b_.getInt32(CPUSIM_BARRIER));
    b_.SetInsertPoint(resume);
}

void WarpBuilder::save_state(int resume_id) {
    std::vector<Save> saves;
    for (auto [a, offset] : state_) {
        Type* t = a->getAllocatedType();
        Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset);
        LoadInst* ld = b_.CreateLoad(t, a);
        StoreInst* st = b_.CreateAlignedStore(ld, p, Align(std::min<uint64_t>(64, MinAlign(offset, 64))));
        saves.push_back({a, offset, ld, st});
    }
    b_.CreateAlignedStore(b_.getInt32(resume_id), state_ptr_, Align(64));
    saves_.push_back(std::move(saves));
    save_blocks_.push_back(b_.GetInsertBlock());
}

// Liveness of the demoted slots on the per-thread code. Each lane follows its
// own path through it, the linearization keeps every lane's order, and a
// masked write fully defines the slot for the lanes that execute it; so this
// is exact per lane, unlike the same analysis on the linearized code, which
// cannot see that a block skipped for lack of lanes also skips its readers.
void WarpBuilder::compute_thread_liveness() {
    DenseMap<const BasicBlock*, DenseSet<const AllocaInst*>> use, def, in;
    for (BasicBlock& bb : f_) {
        auto& u = use[&bb];
        auto& d = def[&bb];
        for (Instruction& inst : bb) {
            if (auto* ld = dyn_cast<LoadInst>(&inst)) {
                if (auto* a = dyn_cast<AllocaInst>(ld->getPointerOperand()); a && demoted_.contains(a) && !d.contains(a))
                    u.insert(a);
            } else if (auto* st = dyn_cast<StoreInst>(&inst)) {
                if (auto* a = dyn_cast<AllocaInst>(st->getPointerOperand()); a && demoted_.contains(a)) d.insert(a);
            }
        }
        in[&bb] = u;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : post_order(&f_.getEntryBlock())) {
            DenseSet<const AllocaInst*> out;
            for (BasicBlock* s : successors(bb))
                for (const AllocaInst* a : in[s]) out.insert(a);
            for (const AllocaInst* a : out) {
                if (def[bb].contains(a)) continue;
                if (in[bb].insert(a).second) changed = true;
            }
        }
    }
    for (const BasicBlock* bb : barrier_blocks_) {
        auto& live = live_after_barrier_[bb];
        for (const BasicBlock* s : successors(bb))
            for (const AllocaInst* a : in[s]) live.insert(a);
    }
}

// Liveness of the slots over the generated (already linearized) code: what
// runs after a resume point is exactly this code, so a slot must survive the
// barrier only if some path from the resume block reads it before writing it.
// A blend under a partial mask is a load followed by a store, so it counts as
// a read, which keeps the inactive lanes' values alive as they must be.
void WarpBuilder::compute_resume_liveness() {
    DenseMap<const AllocaInst*, unsigned> index;
    for (auto& [a, offset] : state_) index[a] = unsigned(index.size());
    const unsigned n = unsigned(index.size());
    DenseSet<const Instruction*> save_loads;
    for (auto& v : saves_)
        for (auto& s : v) save_loads.insert(s.load);

    DenseMap<const BasicBlock*, BitVector> use, def, in;
    for (BasicBlock& bb : *w_) {
        BitVector u(n), d(n);
        for (Instruction& inst : bb) {
            if (auto* ld = dyn_cast<LoadInst>(&inst); ld && !save_loads.contains(ld)) {
                if (auto* a = dyn_cast<AllocaInst>(ld->getPointerOperand()); a && index.count(a)) {
                    unsigned i = index[a];
                    if (!d[i]) u.set(i);
                }
            } else if (auto* st = dyn_cast<StoreInst>(&inst)) {
                if (auto* a = dyn_cast<AllocaInst>(st->getPointerOperand()); a && index.count(a)) d.set(index[a]);
            }
        }
        use[&bb] = u;
        def[&bb] = d;
        in[&bb] = u;
    }
    std::vector<BasicBlock*> order;
    for (BasicBlock* bb : post_order(&w_->getEntryBlock())) order.push_back(bb);
    for (BasicBlock* r : resumes_)  // not reachable from the entry until the dispatch exists
        for (BasicBlock* bb : post_order(r)) order.push_back(bb);
    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : order) {
            BitVector out(n);
            for (BasicBlock* s : successors(bb)) out |= in[s];
            // A barrier's return continues, later, at its resume point.
            for (size_t k = 0; k < save_blocks_.size(); ++k)
                if (save_blocks_[k] == bb) out |= in[resumes_[k]];
            BitVector cur = def[bb];
            cur.flip();
            cur &= out;
            cur |= use[bb];
            if (cur != in[bb]) {
                in[bb] = cur;
                changed = true;
            }
        }
    }

    // Demoted slots: the per-thread answer is exact; masks only exist here.
    compute_thread_liveness();
    DenseMap<const AllocaInst*, const AllocaInst*> original;  // warp slot -> per-thread slot
    for (auto& [orig, slot] : slots_) original[slot] = orig;
    for (size_t k = 0; k < resumes_.size(); ++k) {
        BitVector live = in[resumes_[k]];
        const auto& thread_live = live_after_barrier_[barrier_order_[k]];
        for (auto& [a, offset] : state_) {
            auto it = original.find(a);
            if (it != original.end() && !thread_live.contains(it->second)) live.reset(index[a]);
        }
        if (std::getenv("CPUSIM_DEVC_DEBUG")) {
            errs() << "resume " << (k + 1) << " (" << barrier_order_[k]->getName() << "):";
            for (auto& [a, offset] : state_) {
                auto it = original.find(a);
                bool wl = in[resumes_[k]][index[a]];
                bool tl = it != original.end() && thread_live.contains(it->second);
                if (wl || tl) errs() << " " << a->getName() << (wl ? "[w]" : "") << (tl ? "[t]" : "");
            }
            errs() << "\n";
        }
        live_at_resume_.push_back(live);
        for (Save& s : saves_[k]) {
            if (live[index[s.slot]]) continue;
            Value* addr = s.store->getPointerOperand();
            s.store->eraseFromParent();
            s.load->eraseFromParent();
            if (auto* gep = dyn_cast<Instruction>(addr); gep && gep->use_empty()) gep->eraseFromParent();
        }
    }
}

// The resume path: reload every slot and mask from the frame, then jump to
// the code right after the barrier the warp stopped at.
void WarpBuilder::emit_dispatch() {
    compute_resume_liveness();
    b_.SetInsertPoint(dispatch_);
    Value* state = b_.CreateAlignedLoad(b_.getInt32Ty(), state_ptr_, Align(64));
    BasicBlock* bad = BasicBlock::Create(ctx_, "resume.bad", w_);
    SwitchInst* sw = b_.CreateSwitch(state, bad, unsigned(resumes_.size()));
    for (size_t k = 0; k < resumes_.size(); ++k) {
        BasicBlock* restore = BasicBlock::Create(ctx_, "restore." + Twine(k + 1), w_);
        sw->addCase(b_.getInt32(uint32_t(k + 1)), restore);
        b_.SetInsertPoint(restore);
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!live_at_resume_[k][i]) continue;
            auto [a, offset] = state_[i];
            Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset);
            b_.CreateStore(b_.CreateAlignedLoad(a->getAllocatedType(), p, Align(std::min<uint64_t>(64, MinAlign(offset, 64)))), a);
        }
        b_.CreateBr(resumes_[k]);
    }
    b_.SetInsertPoint(bad);
    b_.CreateUnreachable();
}

void WarpBuilder::add_edge(BasicBlock* from, BasicBlock* to, Value* edge_mask) {
    AllocaInst* target = block_mask_[to];
    if (Loop* l = li_->getLoopFor(to); l && l->getHeader() == to && l->contains(from)) target = next_mask_[l];
    Value* old = b_.CreateLoad(mask_type_, target);
    b_.CreateStore(b_.CreateOr(old, edge_mask), target);
}

void WarpBuilder::emit_level(Loop* level) {
    for (const Node& n : topo_order(level)) emit_node(n);
}

void WarpBuilder::emit_node(const Node& n) {
    if (n.block) emit_block(n.block);
    else emit_loop(n.loop);
}

void WarpBuilder::emit_block(BasicBlock* bb) {
    Value* mask = b_.CreateLoad(mask_type_, block_mask_[bb], "m." + bb->getName());
    BasicBlock* body = BasicBlock::Create(ctx_, bb->getName(), w_);
    BasicBlock* next = BasicBlock::Create(ctx_, bb->getName() + ".next", w_);
    b_.CreateCondBr(any_lane(mask), body, next);
    b_.SetInsertPoint(body);

    bool divergent = uni_->is_divergent_block(bb);
    uint64_t counts[CPUSIM_COUNTERS] = {};
    for (Instruction& inst : *bb) {
        if (profile_) count(inst, counts);
        if (inst.isTerminator()) break;
        if (!emit_instruction(inst, mask, divergent)) return;
    }
    if (profile_) emit_counters(counts, mask);

    if (barrier_blocks_.contains(bb)) {
        // Stop here; the runtime calls the entry again once every warp of the
        // block got to this barrier.
        emit_barrier(bb);
        mask = b_.CreateLoad(mask_type_, block_mask_[bb], "m." + bb->getName());
    }

    if (auto* ret = dyn_cast<ReturnInst>(bb->getTerminator()); ret && ret_slot_) {
        Value* v = vector(ret->getReturnValue());
        if (divergent) v = b_.CreateSelect(mask, v, b_.CreateLoad(ret_slot_->getAllocatedType(), ret_slot_));
        b_.CreateStore(v, ret_slot_);
    }
    if (auto* br = dyn_cast<BranchInst>(bb->getTerminator())) {
        if (br->isUnconditional()) {
            add_edge(bb, br->getSuccessor(0), mask);
        } else {
            Value* c = br->getCondition();
            Value *taken, *not_taken;
            if (uni_->is_varying(c)) {
                Value* cv = vector(c);
                taken = b_.CreateAnd(mask, cv);
                not_taken = b_.CreateAnd(mask, b_.CreateNot(cv));
            } else {
                Value* cs = scalar(c);
                taken = b_.CreateSelect(cs, mask, no_lanes());
                not_taken = b_.CreateSelect(cs, no_lanes(), mask);
            }
            add_edge(bb, br->getSuccessor(0), taken);
            add_edge(bb, br->getSuccessor(1), not_taken);
        }
    }
    b_.CreateBr(next);
    b_.SetInsertPoint(next);
}

// A loop where every branch (exits, back edges, and branches inside, nested
// loops included) is uniform: all active lanes walk the same path, so the
// mask is the one the loop was entered with, all the way through.
bool WarpBuilder::is_uniform_loop(Loop* l) {
    for (BasicBlock* bb : l->blocks()) {
        if (auto* br = dyn_cast<BranchInst>(bb->getTerminator()); br && br->isConditional() &&
                                                                    uni_->is_varying(br->getCondition()))
            return false;
    }
    return true;
}

// Emits a uniform loop with its own control flow (real scalar branches) and
// the entry mask for every block: no per-block masks, no per-iteration
// "any lane left" tests. Lanes leave together and join the exit's mask.
void WarpBuilder::emit_uniform_loop(Loop* l) {
    BasicBlock* header = l->getHeader();
    std::string name = header->getName().str();
    bool barriers = any_of(l->blocks(), [&](BasicBlock* bb) { return barrier_blocks_.contains(bb); });
    Value* entry_mask = b_.CreateLoad(mask_type_, block_mask_[header], "m.uloop." + name);
    BasicBlock* exit = BasicBlock::Create(ctx_, "uloop." + name + ".exit", w_);
    DenseMap<BasicBlock*, BasicBlock*> emitted;
    for (BasicBlock* bb : l->blocks()) emitted[bb] = BasicBlock::Create(ctx_, "u." + bb->getName(), w_);
    b_.CreateCondBr(any_lane(entry_mask), emitted[header], exit);

    for (BasicBlock* bb : l->blocks()) {
        b_.SetInsertPoint(emitted[bb]);
        // With barriers inside, a resume enters the loop in the middle: the
        // mask then comes back from the header's slot (restored with the rest).
        Value* mask = barriers ? b_.CreateLoad(mask_type_, block_mask_[header], "m.u") : entry_mask;
        bool divergent = uni_->is_divergent_block(bb);
        uint64_t counts[CPUSIM_COUNTERS] = {};
        for (Instruction& inst : *bb) {
            if (profile_) count(inst, counts);
            if (inst.isTerminator()) break;
            if (!emit_instruction(inst, mask, divergent)) return;
        }
        if (profile_) emit_counters(counts, mask);

        if (barrier_blocks_.contains(bb)) {
            emit_barrier(bb);
            mask = b_.CreateLoad(mask_type_, block_mask_[header], "m.u");
        }

        // Leaving the loop: the lanes join the target's mask, then continue
        // with the code after the loop.
        auto target = [&](BasicBlock* s) -> BasicBlock* {
            if (l->contains(s)) return emitted[s];
            BasicBlock* edge = BasicBlock::Create(ctx_, "u.exit." + s->getName(), w_);
            IRBuilder<>::InsertPointGuard guard(b_);
            b_.SetInsertPoint(edge);
            add_edge(bb, s, mask);
            b_.CreateBr(exit);
            return edge;
        };
        Instruction* term = bb->getTerminator();
        if (auto* br = dyn_cast<BranchInst>(term)) {
            if (br->isUnconditional())
                b_.CreateBr(target(br->getSuccessor(0)));
            else
                b_.CreateCondBr(scalar(br->getCondition()), target(br->getSuccessor(0)), target(br->getSuccessor(1)));
        } else {
            b_.CreateBr(exit);  // return / unreachable: these lanes are done
        }
    }
    b_.SetInsertPoint(exit);
}

void WarpBuilder::emit_loop(Loop* l) {
    if (is_uniform_loop(l)) return emit_uniform_loop(l);
    BasicBlock* header = l->getHeader();
    std::string name = header->getName().str();
    BasicBlock* top = BasicBlock::Create(ctx_, "loop." + name, w_);
    BasicBlock* exit = BasicBlock::Create(ctx_, "loop." + name + ".exit", w_);
    Value* entry_mask = b_.CreateLoad(mask_type_, block_mask_[header]);
    b_.CreateCondBr(any_lane(entry_mask), top, exit);

    b_.SetInsertPoint(top);
    for (BasicBlock* bb : l->blocks())
        if (bb != header) b_.CreateStore(no_lanes(), block_mask_[bb]);
    for (Loop* sub : l->getLoopsInPreorder()) b_.CreateStore(no_lanes(), next_mask_[sub]);

    emit_level(l);

    Value* next = b_.CreateLoad(mask_type_, next_mask_[l], "next." + name);
    b_.CreateStore(next, block_mask_[header]);
    b_.CreateCondBr(any_lane(next), top, exit);
    b_.SetInsertPoint(exit);
}

bool WarpBuilder::emit_instruction(Instruction& inst, Value* mask, bool divergent) {
    const bool varying = uni_->is_varying(&inst);
    auto set = [&](Value* v) {
        (varying ? vector_map_ : scalar_map_)[&inst] = v;
        return true;
    };

    // Per-thread arrays: only addressed through split_address; the alloca and
    // the GEPs on it produce nothing of their own (all their uses are loads
    // and stores, checked in check_private_array).
    if (auto* a = dyn_cast<AllocaInst>(&inst); a && private_elem_.count(a)) return true;
    if (auto* g = dyn_cast<GetElementPtrInst>(&inst)) {
        Value* v = g;
        while (auto* p = dyn_cast<GetElementPtrInst>(v)) v = p->getPointerOperand();
        if (auto* a = dyn_cast<AllocaInst>(v); a && private_elem_.count(a)) return true;
    }

    // Demoted slots.
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
        if (auto* a = dyn_cast<AllocaInst>(ld->getPointerOperand()); a && demoted_.contains(a)) {
            AllocaInst* slot = slots_[a];
            return set(b_.CreateLoad(slot->getAllocatedType(), slot, ld->getName()));
        }
    }
    if (auto* st = dyn_cast<StoreInst>(&inst)) {
        if (auto* a = dyn_cast<AllocaInst>(st->getPointerOperand()); a && demoted_.contains(a)) {
            AllocaInst* slot = slots_[a];
            if (!uni_->is_varying(a)) {
                b_.CreateStore(scalar(st->getValueOperand()), slot);
                return true;
            }
            Value* v = vector(st->getValueOperand());
            if (divergent) v = b_.CreateSelect(mask, v, b_.CreateLoad(slot->getAllocatedType(), slot));
            b_.CreateStore(v, slot);
            return true;
        }
    }
    if (auto* a = dyn_cast<AllocaInst>(&inst); a && escaping_.contains(a)) {
        auto* copies = ArrayType::get(a->getAllocatedType(), W);
        SmallVector<Constant*, 64> lane_idx;
        for (unsigned l = 0; l < W; ++l) lane_idx.push_back(b_.getInt32(l));
        if (frame_) {
            // A variant that suspends at barriers: the copies live in the
            // warp's frame (the stack is gone on resume), at a fixed offset;
            // the pointers are made where the frame is loaded, every call.
            auto [it, fresh] = private_frame_offset_.try_emplace(a, 0);
            if (fresh) {
                it->second = alignTo(cache_end_, 64);
                cache_end_ = it->second + m_.getDataLayout().getTypeAllocSize(copies);
                note_frame_size();
            }
            IRBuilder<> fb(cast<Instruction>(frame_)->getNextNode());
            Value* base = fb.CreateConstInBoundsGEP1_64(fb.getInt8Ty(), frame_, it->second);
            return set(fb.CreateGEP(copies, base, {fb.getInt32(0), ConstantVector::get(lane_idx)}, a->getName()));
        }
        IRBuilder<> ab(alloca_block_->getTerminator());
        AllocaInst* base = ab.CreateAlloca(copies, nullptr, a->getName() + ".lanes");
        base->setAlignment(std::max(a->getAlign(), Align(16)));
        SmallVector<Constant*, 64> lanes;
        for (unsigned l = 0; l < W; ++l) lanes.push_back(b_.getInt32(l));
        // Next to the storage: the lanes' pointers are the same all through
        // the call, and its uses may be in any block.
        return set(ab.CreateGEP(copies, base, {ab.getInt32(0), ConstantVector::get(lanes)}, a->getName()));
    }
    if (isa<AllocaInst>(inst)) return true;  // demoted slots were recreated in the prologue

    if (auto* call = dyn_cast<CallInst>(&inst); call && is_cond_load(call)) return set(emit_cond_load(call, mask));

    // Software prefetch: one per 16-lane chunk, at the chunk's first address
    // (a chunk usually spans one or two cache lines).
    if (auto* call = dyn_cast<IntrinsicInst>(&inst); call && call->getIntrinsicID() == Intrinsic::prefetch) {
        Value* ptr = call->getArgOperand(0);
        Function* pf = Intrinsic::getDeclaration(&m_, Intrinsic::prefetch, {ptr->getType()});
        if (!uni_->is_varying(ptr)) {
            b_.CreateCall(pf, {scalar(ptr), call->getArgOperand(1), call->getArgOperand(2), call->getArgOperand(3)});
            return true;
        }
        SplitAddress addr = split_address(ptr);
        // Lanes of a chunk next to each other (known at compile time): one
        // prefetch for the chunk. Spread out (strided per lane, unknown): one
        // per lane, or 15 of the 16 lines would be missed.
        std::vector<char> kind;
        bool known = false;
        if (LinearForm form; linear_form(addr.offsets, form)) {
            std::vector<Value*> firsts;
            kind = static_chunks(form, 4, firsts);
            known = true;
        }
        // Prefetches the compiler added (insert_prefetches) are only there
        // for lanes a cache line or more apart: one per lane.
        if (call->getMetadata("cpusim.auto")) kind.assign(W / 16, 0);
        (void)known;
        for (unsigned h = 0; h < W / 16; ++h) {
            const bool together = h < kind.size() && kind[h] != 0;
            Value* m_bits = b_.CreateBitCast(half(mask, h), b_.getInt16Ty());
            for (unsigned l = 0; l < (together ? 1u : 16u); ++l) {
                Value* off = b_.CreateExtractElement(addr.offsets, uint64_t(16 * h + l));
                if (together) off = cached(off);
                Value* active = together ? b_.CreateICmpNE(m_bits, b_.getInt16(0))
                                         : b_.CreateICmpNE(b_.CreateAnd(m_bits, b_.getInt16(uint16_t(1u << l))),
                                                           b_.getInt16(0));
                Value* p = b_.CreateSelect(active, byte_ptr(addr.base, off),
                                           b_.CreatePointerCast(ctx_arg_, addr.base->getType()));
                b_.CreateCall(pf, {p, call->getArgOperand(1), call->getArgOperand(2), call->getArgOperand(3)});
            }
        }
        return true;
    }

    if (is_mma(&inst)) return emit_mma(cast<CallInst>(&inst), mask);
    if (is_mmav(&inst)) return set(emit_mmav(cast<CallInst>(&inst)));
    if (auto* call = dyn_cast<CallInst>(&inst); call && is_device_call(call)) return emit_call(call, mask);
    if (auto* call = dyn_cast<CallInst>(&inst); call && is_external_call(call)) {
        unsigned arity;
        if (libmvec_name(*call, &arity).empty()) return emit_atomic(inst, mask);
    }

    // Intrinsics with special meaning.
    if (auto* call = dyn_cast<CallInst>(&inst)) {
        SpecialReg r = special_reg(call);
        if (r != SpecialReg::None) return set(special_[(int)r]);
        if (is_barrier(*call)) return true;  // suspending variants: the block's end does it (emit_barrier)
        if (is_yield(*call)) return true;  // the block's end does it (emit_barrier)
        Intrinsic::ID id = call->getIntrinsicID();
        switch (id) {
            case Intrinsic::nvvm_bar_warp_sync:  // the warp runs in lockstep
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
            case Intrinsic::assume:
            case Intrinsic::experimental_noalias_scope_decl:
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            case Intrinsic::dbg_label:
                return true;
            default:
                break;
        }
    }

    if (isa<AtomicRMWInst>(inst) || isa<AtomicCmpXchgInst>(inst) || (isa<LoadInst>(inst) && inst.isAtomic()) ||
        (isa<StoreInst>(inst) && inst.isAtomic()))
        return emit_atomic(inst, mask);
    if (isa<FenceInst>(inst)) {
        b_.Insert(inst.clone());
        return true;
    }

    // Memory through real pointers.
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
        if (ld->isVolatile()) compiler_barrier();
        Value* ptr = ld->getPointerOperand();
        if (!uni_->is_varying(ptr)) {
            LoadInst* v = b_.CreateAlignedLoad(ld->getType(), scalar(ptr), ld->getAlign(), ld->getName());
            v->setVolatile(ld->isVolatile());
            if (!varying) return set(v);
            return set(b_.CreateVectorSplat(W, v));
        }
        return set(gather(ld->getType(), ptr, ld->getAlign(), mask, ld->getName()));
    }
    if (auto* st = dyn_cast<StoreInst>(&inst)) {
        if (st->isVolatile()) compiler_barrier();
        Value* ptr = st->getPointerOperand();
        Value* val = st->getValueOperand();
        if (!uni_->is_varying(ptr) && !uni_->is_varying(val)) {
            b_.CreateAlignedStore(scalar(val), scalar(ptr), st->getAlign())->setVolatile(st->isVolatile());
            return true;
        }
        if (!uni_->is_varying(ptr)) {
            // Every lane writes the same address: the last active lane wins.
            b_.CreateMaskedScatter(vector(val), vector(ptr), st->getAlign(), mask);
            return true;
        }
        scatter(vector(val), ptr, st->getAlign(), mask, st->hasMetadata(LLVMContext::MD_nontemporal));
        return true;
    }

    // Everything else is a pure computation.
    if (!varying) {
        Instruction* c = inst.clone();
        for (Use& u : c->operands())
            if (!isa<BasicBlock>(u.get()) && !isa<Function>(u.get()) && !isa<MetadataAsValue>(u.get()))
                u.set(scalar(u.get()));
        if (auto* bin = dyn_cast<BinaryOperator>(c); bin && bin->isIntDivRem())
            if (auto* k = dyn_cast<ConstantInt>(bin->getOperand(1)); !k || k->isZero() || k->isMinusOne()) {
                Value* d = bin->getOperand(1);
                bin->setOperand(1, b_.CreateSelect(trapping_divisor(bin->getOpcode(), bin->getOperand(0), d),
                                                   ConstantInt::get(d->getType(), 1), d));
            }
        b_.Insert(c, inst.getName());
        return set(c);
    }

    if (auto* ev = dyn_cast<ExtractValueInst>(&inst)) {
        if (auto it = cmpxchg_parts_.find(ev->getAggregateOperand()); it != cmpxchg_parts_.end())
            return set(ev->getIndices()[0] == 0 ? it->second.first : it->second.second);
    }
    if (auto* bin = dyn_cast<BinaryOperator>(&inst)) {
        if (Value* v = emit_uniform_division(bin)) return set(v);
        Value* lhs = vector(bin->getOperand(0));
        Value* rhs = vector(bin->getOperand(1));
        switch (bin->getOpcode()) {
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::URem:
            case Instruction::SRem:
                // Inactive lanes may hold a zero divisor; x86 would trap on it.
                // Not a constant one: no lane has zero, and keeping it constant
                // lets LLVM divide with shifts and multiplies instead of a
                // scalar idiv per lane.
                if (auto* c = dyn_cast<ConstantInt>(bin->getOperand(1)); c && !c->isZero() && !c->isMinusOne()) break;
                rhs = b_.CreateSelect(b_.CreateAnd(mask, b_.CreateNot(trapping_divisor(bin->getOpcode(), lhs, rhs))), rhs,
                                      ConstantInt::get(rhs->getType(), 1));
                break;
            default:
                break;
        }
        Value* v = b_.CreateBinOp(bin->getOpcode(), lhs, rhs, bin->getName());
        if (auto* vi = dyn_cast<Instruction>(v)) vi->copyIRFlags(bin);
        return set(v);
    }
    if (auto* un = dyn_cast<UnaryOperator>(&inst)) {
        Value* v = b_.CreateUnOp(un->getOpcode(), vector(un->getOperand(0)), un->getName());
        if (auto* vi = dyn_cast<Instruction>(v)) vi->copyIRFlags(un);
        return set(v);
    }
    if (auto* cast = dyn_cast<CastInst>(&inst))
        return set(b_.CreateCast(cast->getOpcode(), vector(cast->getOperand(0)), vec_type(cast->getDestTy()),
                                 cast->getName()));
    if (auto* cmp = dyn_cast<CmpInst>(&inst)) {
        Value* v = b_.CreateCmp(cmp->getPredicate(), vector(cmp->getOperand(0)), vector(cmp->getOperand(1)),
                                cmp->getName());
        if (auto* vi = dyn_cast<Instruction>(v); vi && isa<FPMathOperator>(vi)) vi->copyFastMathFlags(cmp);
        return set(v);
    }
    if (auto* sel = dyn_cast<SelectInst>(&inst)) {
        Value* c = uni_->is_varying(sel->getCondition()) ? vector(sel->getCondition()) : scalar(sel->getCondition());
        return set(b_.CreateSelect(c, vector(sel->getTrueValue()), vector(sel->getFalseValue()), sel->getName()));
    }
    if (auto* fr = dyn_cast<FreezeInst>(&inst)) return set(b_.CreateFreeze(vector(fr->getOperand(0)), fr->getName()));
    if (auto* gep = dyn_cast<GetElementPtrInst>(&inst)) {
        Value* base = uni_->is_varying(gep->getPointerOperand()) ? vector(gep->getPointerOperand())
                                                                  : scalar(gep->getPointerOperand());
        SmallVector<Value*, 4> idx;
        for (Use& u : gep->indices()) idx.push_back(uni_->is_varying(u.get()) ? vector(u.get()) : scalar(u.get()));
        Value* v = b_.CreateGEP(gep->getSourceElementType(), base, idx, gep->getName(), gep->isInBounds());
        if (!v->getType()->isVectorTy()) v = b_.CreateVectorSplat(W, v);
        return set(v);
    }
    if (auto* call = dyn_cast<CallInst>(&inst); call && call->getCalledFunction()) {
        StringRef name = call->getCalledFunction()->getName();
        if (name == kShflFn) return set(emit_shfl(vector(call->getArgOperand(0)), vector(call->getArgOperand(1))));
        if (name == kBallotFn) return set(emit_ballot(vector(call->getArgOperand(0)), mask));
        if (name == kRefillFirstFn || name == kRefillNextFn) return set(emit_refill(name == kRefillFirstFn, mask));
    }
    if (auto* call = dyn_cast<CallInst>(&inst)) {
        if (unsigned arity; true) {
            std::string name = libmvec_name(*call, &arity);
            if (!name.empty()) return set(emit_libmvec(*call, name, arity));
        }
        Intrinsic::ID id = call->getIntrinsicID();
        if (id != Intrinsic::not_intrinsic && isTriviallyVectorizable(id)) {
            SmallVector<Value*, 4> args;
            SmallVector<Type*, 2> tys;
            if (isVectorIntrinsicWithOverloadTypeAtArg(id, -1)) tys.push_back(vec_type(call->getType()));
            for (unsigned i = 0; i < call->arg_size(); ++i) {
                Value* a = call->getArgOperand(i);
                bool keep_scalar = isVectorIntrinsicWithScalarOpAtArg(id, i);
                args.push_back(keep_scalar ? scalar(a) : vector(a));
                if (isVectorIntrinsicWithOverloadTypeAtArg(id, i)) tys.push_back(args.back()->getType());
            }
            Function* decl = Intrinsic::getDeclaration(&m_, id, tys);
            CallInst* v = b_.CreateCall(decl, args, call->getName());
            if (isa<FPMathOperator>(v)) v->copyFastMathFlags(call);
            return set(v);
        }
    }
    // An intrinsic with no vector form (lround, ldexp...): once per lane.
    if (auto* call = dyn_cast<IntrinsicInst>(&inst)) {
        auto lane_type = [](Type* t) { return t->isIntegerTy() || t->isFloatingPointTy() || t->isPointerTy(); };
        if (lane_type(call->getType()) && all_of(call->args(), [&](Use& u) { return lane_type(u->getType()); }))
            return emit_atomic(inst, mask);
    }
    fail("unsupported varying instruction", &inst);
    return false;
}

// ---------------------------------------------------------------- profiling

// Static per-thread cost of one instruction of the original kernel. Slots
// introduced by demotion are bookkeeping, not program memory traffic.
void WarpBuilder::count(Instruction& inst, uint64_t* c) {
    const DataLayout& dl = m_.getDataLayout();
    auto fp = [&](Type* t, uint64_t n) {
        t = t->getScalarType();
        if (t->isDoubleTy()) c[CPUSIM_CNT_FP64_FLOPS] += n;
        else if (t->isFloatingPointTy()) c[CPUSIM_CNT_FP32_FLOPS] += n;
    };
    auto is_slot = [&](Value* ptr) {
        auto* a = dyn_cast<AllocaInst>(ptr);
        return a && demoted_.contains(a);
    };
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
        if (is_slot(ld->getPointerOperand())) return;
        c[CPUSIM_CNT_LOADS] += 1;
        c[CPUSIM_CNT_LOAD_BYTES] += dl.getTypeStoreSize(ld->getType());
    } else if (auto* st = dyn_cast<StoreInst>(&inst)) {
        if (is_slot(st->getPointerOperand())) return;
        c[CPUSIM_CNT_STORES] += 1;
        c[CPUSIM_CNT_STORE_BYTES] += dl.getTypeStoreSize(st->getValueOperand()->getType());
    } else if (auto* br = dyn_cast<BranchInst>(&inst)) {
        if (br->isConditional()) c[CPUSIM_CNT_BRANCHES] += 1;
    } else if (auto* bin = dyn_cast<BinaryOperator>(&inst)) {
        if (bin->getType()->isFPOrFPVectorTy()) fp(bin->getType(), 1);
        else c[CPUSIM_CNT_INT_OPS] += 1;
    } else if (isa<ICmpInst>(inst)) {
        c[CPUSIM_CNT_INT_OPS] += 1;
    } else if (isa<FCmpInst>(inst)) {
        fp(inst.getOperand(0)->getType(), 1);
    } else if (unsigned n; isa<CallInst>(inst) && !isa<IntrinsicInst>(inst) &&
                           !libmvec_name(cast<CallInst>(inst), &n).empty()) {
        c[CPUSIM_CNT_FP_SPECIAL] += 1;  // erf, tanh, atan2...: library calls, not intrinsics
    } else if (auto* call = dyn_cast<IntrinsicInst>(&inst)) {
        switch (call->getIntrinsicID()) {
            case Intrinsic::fma:
            case Intrinsic::fmuladd:
                fp(call->getType(), 2);
                break;
            case Intrinsic::sqrt: case Intrinsic::exp: case Intrinsic::exp2: case Intrinsic::exp10:
            case Intrinsic::log: case Intrinsic::log2: case Intrinsic::log10: case Intrinsic::sin:
            case Intrinsic::cos: case Intrinsic::pow: case Intrinsic::powi:
                c[CPUSIM_CNT_FP_SPECIAL] += 1;
                break;
            case Intrinsic::minnum: case Intrinsic::maxnum: case Intrinsic::minimum: case Intrinsic::maximum:
            case Intrinsic::fabs: case Intrinsic::copysign: case Intrinsic::floor: case Intrinsic::ceil:
            case Intrinsic::trunc: case Intrinsic::round: case Intrinsic::rint: case Intrinsic::nearbyint:
                fp(call->getType(), 1);
                break;
            case Intrinsic::smin: case Intrinsic::smax: case Intrinsic::umin: case Intrinsic::umax:
            case Intrinsic::abs: case Intrinsic::ctpop: case Intrinsic::ctlz: case Intrinsic::cttz:
                c[CPUSIM_CNT_INT_OPS] += 1;
                break;
            default:
                break;
        }
    }
}

// counters[i] += active lanes * per-thread count, once per block execution.
void WarpBuilder::emit_counters(const uint64_t* counts, Value* mask) {
    auto* i64 = b_.getInt64Ty();
    Value* lanes = b_.CreateZExtOrTrunc(
        b_.CreateUnaryIntrinsic(Intrinsic::ctpop, b_.CreateBitCast(mask, b_.getIntNTy(W))), i64, "lanes");
    auto add = [&](int idx, Value* v) {
        Value* slot = b_.CreateConstInBoundsGEP1_64(i64, counters_, idx);
        b_.CreateStore(b_.CreateAdd(b_.CreateLoad(i64, slot), v), slot);
    };
    for (int i = 0; i < CPUSIM_CNT_BLOCK_RUNS; ++i)
        if (counts[i]) add(i, b_.CreateMul(lanes, ConstantInt::get(i64, counts[i])));
    add(CPUSIM_CNT_BLOCK_RUNS, ConstantInt::get(i64, 1));
    add(CPUSIM_CNT_ACTIVE_LANES, lanes);
}


}  // namespace simt_internal
}  // namespace cpusim
