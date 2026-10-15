// The CUDA runtime: CUDA graphs, built explicitly or captured from a stream,
// and their launch.
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

extern char** environ;

#include "common/cpu_id.h"
#include "cudart/streams.h"
#include "cuda_runtime_api.h"
#include "device/device.h"
#include "device/kernel_abi.h"
#include "cudart/runtime_internal.h"

using namespace cpusim::rt;

#pragma GCC visibility push(hidden)
namespace cpusim::rt {
std::mutex capture_mutex;
std::map<cudaStream_t, CUgraph_st*> capturing_streams;

// The graph `stream` is being captured into, or null.
CUgraph_st* capturing(cudaStream_t stream) {
    std::lock_guard lock(capture_mutex);
    auto it = capturing_streams.find(stream);
    return it == capturing_streams.end() ? nullptr : it->second;
}

CUgraphNode_st* add_node(CUgraph_st* g, CUgraphNode_st::Kind kind) {
    g->nodes.push_back(std::make_unique<CUgraphNode_st>());
    g->nodes.back()->kind = kind;
    return g->nodes.back().get();
}

// A kernel node's parameters copied from `args` (the sizes come from the
// kernel's manifest entry).
cudaError_t set_kernel(CUgraphNode_st* n, const void* func, dim3 grid, dim3 block, size_t shared, void** args) {
    std::vector<std::pair<uint64_t, uint64_t>> params;
    {
        Registry& r = registry();
        std::lock_guard lock(r.mutex);
        auto it = r.functions.find(func);
        if (it == r.functions.end() || !it->second.kernel) return record(Error::InvalidDeviceFunction);
        params = it->second.kernel->params;
    }
    n->func = func, n->grid = grid, n->block = block, n->shared = shared;
    n->args.clear();
    n->arg_slots.clear();
    for (size_t i = 0; i < params.size(); ++i) {
        const size_t at = (n->args.size() + 15) & ~size_t(15);
        n->args.resize(at + params[i].second);
        std::memcpy(n->args.data() + at, args[i], params[i].second);
        n->arg_slots.push_back({at, params[i].second});
    }
    return cudaSuccess;
}

void set_memset(CUgraphNode_st* n, const cudaMemsetParams& p) {
    n->memset = p;
    n->pattern.clear();
    if (p.elementSize > 1) {
        n->pattern.resize(p.width * p.elementSize);
        for (size_t i = 0; i < p.width; ++i) std::memcpy(n->pattern.data() + i * p.elementSize, &p.value, p.elementSize);
    }
}
}  // namespace cpusim::rt
#pragma GCC visibility pop

extern "C" {

// ---- CUDA Graphs (the nodes and capture state are above)

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode) {
    if (!is_object(stream)) return record(Error::NotSupported);  // not the legacy default stream
    std::lock_guard lock(capture_mutex);
    if (capturing_streams.count(stream)) return record(Error::InvalidValue);
    capturing_streams[stream] = new CUgraph_st;
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    if (!graph) return record(Error::InvalidValue);
    std::lock_guard lock(capture_mutex);
    auto it = capturing_streams.find(stream);
    if (it == capturing_streams.end()) return record(Error::InvalidValue);
    CUgraph_st* g = it->second;
    // Every stream that joined this capture leaves it.
    for (auto i = capturing_streams.begin(); i != capturing_streams.end();)
        i = i->second == g ? capturing_streams.erase(i) : std::next(i);
    *graph = g;
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* status) {
    if (!status) return record(Error::InvalidValue);
    *status = capturing(stream) ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int) {
    if (!graph) return record(Error::InvalidValue);
    *graph = new CUgraph_st;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    delete graph;
    return cudaSuccess;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* count) {
    if (!graph || !count) return record(Error::InvalidValue);
    if (!nodes) {
        *count = graph->nodes.size();
        return cudaSuccess;
    }
    const size_t n = std::min(*count, graph->nodes.size());
    for (size_t i = 0; i < n; ++i) nodes[i] = graph->nodes[i].get();
    *count = n;
    return cudaSuccess;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t*, size_t,
                                   const struct cudaKernelNodeParams* p) {
    if (!node || !graph || !p) return record(Error::InvalidValue);
    CUgraphNode_st* n = add_node(graph, CUgraphNode_st::Kernel);
    *node = n;
    return set_kernel(n, p->func, p->gridDim, p->blockDim, p->sharedMemBytes, p->kernelParams);
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t*, size_t,
                                   const struct cudaMemsetParams* p) {
    if (!node || !graph || !p || (p->elementSize != 1 && p->elementSize != 2 && p->elementSize != 4))
        return record(Error::InvalidValue);
    CUgraphNode_st* n = add_node(graph, CUgraphNode_st::Memset);
    set_memset(n, *p);
    *node = n;
    return cudaSuccess;
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t*, size_t,
                                     void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    if (!node || !graph) return record(Error::InvalidValue);
    CUgraphNode_st* n = add_node(graph, CUgraphNode_st::Memcpy);
    n->dst = dst, n->src = src, n->count = count, n->copy_kind = kind;
    *node = n;
    return cudaSuccess;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t*, size_t,
                                 const struct cudaHostNodeParams* p) {
    if (!node || !graph || !p || !p->fn) return record(Error::InvalidValue);
    CUgraphNode_st* n = add_node(graph, CUgraphNode_st::Host);
    n->fn = p->fn, n->data = p->userData;
    *node = n;
    return cudaSuccess;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t*, size_t) {
    if (!node || !graph) return record(Error::InvalidValue);
    *node = add_node(graph, CUgraphNode_st::Empty);
    return cudaSuccess;
}

namespace {
// A kernel node's parameters as cudaLaunchKernel takes them.
std::vector<void*> kernel_args(CUgraphNode_st& n) {
    std::vector<void*> a;
    for (auto [at, size] : n.arg_slots) a.push_back(n.args.data() + at);
    return a;
}
}  // namespace

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* p) {
    if (!node || !p || node->kind != CUgraphNode_st::Kernel) return record(Error::InvalidValue);
    p->func = const_cast<void*>(node->func);
    p->gridDim = node->grid, p->blockDim = node->block;
    p->sharedMemBytes = unsigned(node->shared);
    p->kernelParams = nullptr, p->extra = nullptr;
    return cudaSuccess;
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams* p) {
    if (!node || !p || node->kind != CUgraphNode_st::Kernel) return record(Error::InvalidValue);
    return set_kernel(node, p->func, p->gridDim, p->blockDim, p->sharedMemBytes, p->kernelParams);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long) {
    if (!exec || !graph) return record(Error::InvalidValue);
    auto* e = new CUgraphExec_st;
    for (auto& n : graph->nodes) {
        e->index[n.get()] = e->nodes.size();
        e->nodes.push_back(*n);
    }
    *exec = e;
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags) {
    return cudaGraphInstantiate(exec, graph, flags);
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec, cudaGraphNode_t node,
                                             const struct cudaKernelNodeParams* p) {
    if (!exec || !node || !p) return record(Error::InvalidValue);
    auto it = exec->index.find(node);
    if (it == exec->index.end() || exec->nodes[it->second].kind != CUgraphNode_st::Kernel)
        return record(Error::InvalidValue);
    return set_kernel(&exec->nodes[it->second], p->func, p->gridDim, p->blockDim, p->sharedMemBytes, p->kernelParams);
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    if (!exec) return record(Error::InvalidResourceHandle);
    for (CUgraphNode_st& n : exec->nodes) {
        cudaError_t e = cudaSuccess;
        switch (n.kind) {
            case CUgraphNode_st::Kernel: {
                std::vector<void*> a = kernel_args(n);
                e = cudaLaunchKernel(n.func, n.grid, n.block, a.data(), n.shared, stream);
                break;
            }
            case CUgraphNode_st::Memcpy:
                e = cudaMemcpyAsync(n.dst, n.src, n.count, n.copy_kind, stream);
                break;
            case CUgraphNode_st::Memset: {
                const cudaMemsetParams& m = n.memset;
                const size_t pitch = m.pitch ? m.pitch : m.width * m.elementSize;
                if (m.elementSize == 1) {
                    e = cudaMemset2DAsync(m.dst, pitch, int(m.value), m.width, m.height, stream);
                } else {
                    for (size_t y = 0; y < m.height && e == cudaSuccess; ++y)
                        e = cudaMemcpyAsync(static_cast<char*>(m.dst) + y * pitch, n.pattern.data(), n.pattern.size(),
                                            cudaMemcpyHostToDevice, stream);
                }
                break;
            }
            case CUgraphNode_st::Host:
                e = cudaLaunchHostFunc(stream, n.fn, n.data);
                break;
            case CUgraphNode_st::Empty:
                break;
        }
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t) {
    return exec ? cudaSuccess : record(Error::InvalidResourceHandle);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    delete exec;
    return cudaSuccess;
}

}  // extern "C"
