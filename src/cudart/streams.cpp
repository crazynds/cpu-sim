#include "cudart/streams.h"

#include "common/spin.h"

#include <algorithm>
#include <atomic>
#include <map>

namespace cpusim::cudart {

struct Op {
    std::function<cudaError_t()> work;
    std::vector<CompletionPtr> deps;
    CompletionPtr done;
    bool sync = false;
};

// One stream: its operations, run in order by its thread.
class Queue {
public:
    Queue(Scheduler& s, unsigned flags, bool legacy) : sched_(s), flags_(flags), legacy_(legacy) {}
    ~Queue() { stop(); }

    // Under the scheduler's mutex.
    CompletionPtr last() const { return last_; }
    bool blocking() const { return !legacy_ && !(flags_ & cudaStreamNonBlocking); }

    void push(Op op) {
        last_ = op.done;
        {
            std::lock_guard lock(mutex_);
            ops_.push_back(std::move(op));
            pending_.fetch_add(1, std::memory_order_release);
            if (!thread_.joinable()) thread_ = std::thread([this] { run(); });
        }
        cv_.notify_one();
    }
    void set_last(CompletionPtr c) { last_ = std::move(c); }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run();

    Scheduler& sched_;
    const unsigned flags_;
    const bool legacy_;
    CompletionPtr last_;  // the last operation issued (scheduler's mutex)
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Op> ops_;
    std::atomic<int> pending_{0};  // ops_.size(), readable without the lock
    bool stopping_ = false;
    std::thread thread_;
    friend class Scheduler;
};

void Queue::run() {
    while (true) {
        Op op;
        {
            std::unique_lock lock(mutex_);
            if (ops_.empty() && !stopping_) {  // the next operation usually follows soon
                lock.unlock();
                spin_until([&] { return pending_.load(std::memory_order_acquire) != 0; });
                lock.lock();
            }
            cv_.wait(lock, [&] { return stopping_ || !ops_.empty(); });
            if (ops_.empty()) return;  // stopping, and everything ran
            op = std::move(ops_.front());
            ops_.pop_front();
            pending_.fetch_sub(1, std::memory_order_relaxed);
        }
        for (auto& d : op.deps) d->wait();
        cudaError_t e = op.work ? op.work() : cudaSuccess;
        if (e != cudaSuccess && !op.sync) sched_.record_error(e);
        op.done->finish(e);
    }
}

Scheduler::Scheduler() = default;

Scheduler::~Scheduler() {
    for (Queue* q : streams_) q->stop();
    if (legacy_) legacy_->stop();
}

Queue* Scheduler::resolve(cudaStream_t stream) {
    if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        if (!legacy_) legacy_ = std::make_unique<Queue>(*this, 0, true);
        return legacy_.get();
    }
    return stream->queue;
}

CompletionPtr Scheduler::enqueue(cudaStream_t stream, std::function<cudaError_t()> work, Options opt) {
    Op op;
    op.work = std::move(work);
    op.deps = std::move(opt.after);
    op.done = std::make_shared<Completion>();
    op.sync = opt.sync;
    std::unique_lock lock(mutex_);
    Queue* q = resolve(stream);
    auto add = [&](const CompletionPtr& c) {
        if (c && !c->ready()) op.deps.push_back(c);
    };
    add(q->last());
    if (q == legacy_.get()) {
        // The legacy default stream waits for every blocking stream...
        for (Queue* s : streams_)
            if (s->blocking()) add(s->last());
    } else if (q->blocking() && legacy_) {
        // ...and blocking streams wait for it.
        add(legacy_->last());
    }
    op.deps.erase(std::remove_if(op.deps.begin(), op.deps.end(), [](const CompletionPtr& c) { return !c || c->ready(); }),
                  op.deps.end());
    if (opt.inline_if_idle && op.deps.empty()) {
        // Nothing pending: run it now on this thread, still in issue order
        // (later operations depend on it through last()).
        CompletionPtr done = op.done;
        q->set_last(done);
        lock.unlock();
        cudaError_t e = op.work ? op.work() : cudaSuccess;
        if (e != cudaSuccess && !op.sync) record_error(e);
        done->finish(e);
        return done;
    }
    if (opt.on_queue) opt.on_queue();
    CompletionPtr done = op.done;
    q->push(std::move(op));
    return done;
}

bool Scheduler::idle(cudaStream_t stream) {
    std::lock_guard lock(mutex_);
    CompletionPtr c = resolve(stream)->last();
    return !c || c->ready();
}

void Scheduler::wait_stream(cudaStream_t stream) {
    CompletionPtr c;
    {
        std::lock_guard lock(mutex_);
        c = resolve(stream)->last();
    }
    if (c) c->wait();
}

void Scheduler::wait_all() {
    std::vector<CompletionPtr> all;
    {
        std::lock_guard lock(mutex_);
        if (legacy_) all.push_back(legacy_->last());
        for (Queue* s : streams_) all.push_back(s->last());
    }
    for (auto& c : all)
        if (c) c->wait();
}

Queue* Scheduler::create(unsigned flags) {
    std::lock_guard lock(mutex_);
    auto* q = new Queue(*this, flags, false);
    streams_.push_back(q);
    return q;
}

void Scheduler::destroy(Queue* q) {
    {
        std::lock_guard lock(mutex_);
        streams_.erase(std::remove(streams_.begin(), streams_.end(), q), streams_.end());
    }
    delete q;  // runs what it still has, then joins its thread
}

void Scheduler::record_error(cudaError_t e) {
    std::lock_guard lock(error_mutex_);
    if (error_ == cudaSuccess) error_ = e;
}

cudaError_t Scheduler::peek_error() {
    std::lock_guard lock(error_mutex_);
    return error_;
}

// A kernel that faulted leaves the context unusable, as on a GPU: its error
// is reported by every later call that reports asynchronous errors, until
// cudaDeviceReset.
static bool sticky(cudaError_t e) {
    return e == cudaErrorIllegalAddress || e == cudaErrorLaunchFailure || e == cudaErrorAssert;
}

cudaError_t Scheduler::take_error() {
    std::lock_guard lock(error_mutex_);
    cudaError_t e = error_;
    if (!sticky(e)) error_ = cudaSuccess;
    return e;
}

void Scheduler::clear_error() {
    std::lock_guard lock(error_mutex_);
    error_ = cudaSuccess;
}

namespace {
std::mutex pinned_mutex;
struct PinnedRange {
    size_t size;
    bool owned;  // allocated by cudaMallocHost (not registered)
};
std::map<const char*, PinnedRange> pinned;  // start -> range
}  // namespace

void pinned_add(const void* p, size_t size, bool owned) {
    std::lock_guard lock(pinned_mutex);
    pinned[static_cast<const char*>(p)] = {size, owned};
}

bool pinned_remove(const void* p) {
    std::lock_guard lock(pinned_mutex);
    return pinned.erase(static_cast<const char*>(p)) != 0;
}

bool pinned_owned(const void* p) {
    std::lock_guard lock(pinned_mutex);
    auto it = pinned.find(static_cast<const char*>(p));
    return it != pinned.end() && it->second.owned;
}

bool pinned_contains(const void* p, size_t size) {
    std::lock_guard lock(pinned_mutex);
    auto c = static_cast<const char*>(p);
    auto it = pinned.upper_bound(c);
    if (it == pinned.begin()) return false;
    --it;
    return c >= it->first && c + size <= it->first + it->second.size;
}

}  // namespace cpusim::cudart
