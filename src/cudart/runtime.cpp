// The CUDA runtime API on top of the device interface. One device (0); the
// backend is chosen on first use (see open_device).
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

#include "cudart/streams.h"
#include "cuda_runtime_api.h"
#include "device/device.h"
#include "device/kernel_abi.h"

using cpusim::Error;

namespace {

struct Runtime {
    std::unique_ptr<cpusim::Device> device;
    Error init_error = Error::Success;
    // After `device`: destroyed first, so the queued work still has a device.
    cpusim::cudart::Scheduler sched;
};

Runtime& runtime() {
    static Runtime rt;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string message;
        rt.device = cpusim::open_device(&rt.init_error, &message);
        if (!rt.device && std::getenv("CPUSIM_QUIET") == nullptr)
            std::fprintf(stderr, "cpusim: no device: %s\n", message.c_str());
    });
    return rt;
}

thread_local cudaError_t last_error = cudaSuccess;

cudaError_t record(Error e) {
    auto err = static_cast<cudaError_t>(e);
    if (err != cudaSuccess) last_error = err;
    return err;
}
cudaError_t record(cudaError_t err) {
    if (err != cudaSuccess) last_error = err;
    return err;
}

// At a synchronization point: the first error of asynchronous work, if any.
cudaError_t async_error() { return record(runtime().sched.take_error()); }

// Returns the device or records why there is none.
cpusim::Device* device(cudaError_t* err) {
    Runtime& rt = runtime();
    if (!rt.device) *err = record(rt.init_error == Error::Success ? Error::NoDevice : rt.init_error);
    return rt.device.get();
}

// ---- modules registered by the compiler-generated constructors

struct KernelDesc {
    std::string name;
    uint64_t param_bytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> params;  // offset, size
    int min_sm = 0;  // compute capability (major * 10 + minor) it needs, 0: any
    uint64_t shared_bytes = 0;  // static __shared__
    int max_dynamic_shared = -1;  // cudaFuncAttributeMaxDynamicSharedMemorySize (-1: the default)
};

// A module recompiled for one block shape (JIT at launch).
struct JitModule {
    bool tried = false, ok = false;
    uint64_t handle = 0;
    std::map<std::string, uint64_t> functions;
};

// A __device__ or __constant__ variable: one copy in the VRAM for the whole
// program, allocated when the generic module is loaded.
struct SymbolDesc {
    std::string name;
    uint64_t size = 0, align = 1;
    uint64_t address = 0;
    bool managed = false;  // __managed__: kept equal to `host` around launches
    void* host = nullptr;  // its host copy (the registered shadow)
    std::vector<uint8_t> initial;  // managed: its initial value, for the host copy
};

struct Module {
    const cpusim_fatbin_header* header = nullptr;
    std::map<std::string, KernelDesc> kernels;
    std::vector<SymbolDesc> symbols;
    // For Device::bind_symbols: "<address> <size> <name>" per line.
    std::string symbol_table() const {
        std::string t;
        for (const auto& s : symbols) t += std::to_string(s.address) + " " + std::to_string(s.size) + " " + s.name + "\n";
        return t;
    }
    bool loaded = false;
    uint64_t handle = 0;
    std::map<std::array<uint32_t, 3>, JitModule> jit;  // by block shape
    std::map<std::array<uint32_t, 3>, bool> dumped;    // CPUSIM_DUMP_ASM, by block shape ({0,0,0}: generic)
};

struct Function {
    Module* module = nullptr;
    const KernelDesc* kernel = nullptr;
    uint64_t handle = 0;  // device function, resolved on first launch
};

struct Variable {
    Module* module = nullptr;
    std::string name;  // device name
    size_t size = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<Module>> modules;
    std::map<const void*, Function> functions;  // host stub -> kernel
    std::map<const void*, Variable> variables;  // host shadow -> __device__ / __constant__ variable
};

Registry& registry() {
    static Registry r;
    return r;
}

// ---- profiling (CPUSIM_PROFILE=1)
//
// The first launch of each (kernel, grid, block) runs the instrumented entry,
// which counts operations; later launches of the same shape run the fast one
// and are timed. Rates combine both: counts from the instrumented run, time
// from the uninstrumented ones. Each launch still runs exactly once, so kernels
// that accumulate into their outputs keep working. Counts of data-dependent
// kernels are those of the first launch.

struct ProfileEntry {
    std::string kernel;
    uint32_t grid[3], block[3];
    uint64_t counters[CPUSIM_COUNTERS] = {};
    uint64_t profiled_ns = 0;
    uint64_t timed_runs = 0, timed_ns = 0;
};

struct Profiler {
    bool enabled = false;
    std::mutex mutex;
    std::vector<ProfileEntry> entries;

    Profiler() {
        const char* v = std::getenv("CPUSIM_PROFILE");
        enabled = v && *v && std::strcmp(v, "0") != 0;
        if (enabled) std::atexit([] { report(); });
    }

    ProfileEntry* find(const std::string& kernel, const cpusim::LaunchConfig& cfg) {
        for (auto& e : entries)
            if (e.kernel == kernel && !std::memcmp(e.grid, cfg.grid, sizeof(e.grid)) &&
                !std::memcmp(e.block, cfg.block, sizeof(e.block)))
                return &e;
        return nullptr;
    }

    static void report();
};

// Never destroyed: the report runs from atexit, after static destructors may have run.
Profiler& profiler() {
    static Profiler* p = new Profiler;
    return *p;
}

std::string human(double v, const char* unit) {
    const char* prefixes[] = {"", "K", "M", "G", "T", "P"};
    int i = 0;
    while (v >= 1000 && i < 5) v /= 1000, ++i;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3g %s%s", v, prefixes[i], unit);
    return buf;
}

void Profiler::report() {
    Profiler& p = profiler();
    std::lock_guard lock(p.mutex);
    std::fflush(stdout);
    std::fprintf(stderr, "\n==== cpusim profile (%zu kernel shapes) ====\n", p.entries.size());
    for (const auto& e : p.entries) {
        const uint64_t* c = e.counters;
        double secs = e.timed_runs ? e.timed_ns / 1e9 / e.timed_runs : 0;
        double flops = double(c[CPUSIM_CNT_FP32_FLOPS] + c[CPUSIM_CNT_FP64_FLOPS]);
        double bytes = double(c[CPUSIM_CNT_LOAD_BYTES] + c[CPUSIM_CNT_STORE_BYTES]);
        double lanes = c[CPUSIM_CNT_BLOCK_RUNS] ? double(c[CPUSIM_CNT_ACTIVE_LANES]) / c[CPUSIM_CNT_BLOCK_RUNS] : 0;
        std::fprintf(stderr, "%s  grid (%u,%u,%u) block (%u,%u,%u)\n", e.kernel.c_str(), e.grid[0], e.grid[1],
                     e.grid[2], e.block[0], e.block[1], e.block[2]);
        if (e.timed_runs)
            std::fprintf(stderr, "  time            %.3f ms (mean of %llu uninstrumented launches)\n", secs * 1e3,
                         (unsigned long long)e.timed_runs);
        else
            std::fprintf(stderr, "  time            only the instrumented launch ran (%.3f ms, not representative)\n",
                         e.profiled_ns / 1e6);
        std::fprintf(stderr, "  instrumented    %.3f ms (first launch, counting)\n", e.profiled_ns / 1e6);
        std::fprintf(stderr, "  fp32 flops      %s (%llu)\n", human(double(c[CPUSIM_CNT_FP32_FLOPS]), "").c_str(),
                     (unsigned long long)c[CPUSIM_CNT_FP32_FLOPS]);
        std::fprintf(stderr, "  fp64 flops      %s (%llu)\n", human(double(c[CPUSIM_CNT_FP64_FLOPS]), "").c_str(),
                     (unsigned long long)c[CPUSIM_CNT_FP64_FLOPS]);
        std::fprintf(stderr, "  special fp ops  %s\n", human(double(c[CPUSIM_CNT_FP_SPECIAL]), "").c_str());
        std::fprintf(stderr, "  integer ops     %s (%llu)\n", human(double(c[CPUSIM_CNT_INT_OPS]), "").c_str(),
                     (unsigned long long)c[CPUSIM_CNT_INT_OPS]);
        std::fprintf(stderr, "  loads / stores  %s / %s (%s / %s)\n", human(double(c[CPUSIM_CNT_LOADS]), "").c_str(),
                     human(double(c[CPUSIM_CNT_STORES]), "").c_str(), human(double(c[CPUSIM_CNT_LOAD_BYTES]), "B").c_str(),
                     human(double(c[CPUSIM_CNT_STORE_BYTES]), "B").c_str());
        std::fprintf(stderr, "  branches        %s\n", human(double(c[CPUSIM_CNT_BRANCHES]), "").c_str());
        std::fprintf(stderr, "  SIMD efficiency %.1f%% (%.1f of 32 lanes active per block executed)\n",
                     100.0 * lanes / CPUSIM_WARP_SIZE, lanes);
        if (secs > 0) {
            std::fprintf(stderr, "  throughput      %s, %s, %s\n", human(flops / secs, "FLOP/s").c_str(),
                         human(double(c[CPUSIM_CNT_INT_OPS]) / secs, "IOP/s").c_str(),
                         human(bytes / secs, "B/s").c_str());
            std::fprintf(stderr, "  intensity       %.2f FLOP/B\n", bytes ? flops / bytes : 0.0);
        }
    }
}

// ---- Specialized code: the device code compiled for the launch's block shape
//
// With blockDim known, threadIdx becomes a constant (or constant + uniform)
// vector and the compiler resolves address patterns statically. cucc already
// did it for the block shapes constant at the program's launches (the objects
// are in the fatbin); for any other shape the runtime recompiles the bitcode
// (JIT), cached on disk by content, so only the first run of a program pays
// for it. CPUSIM_JIT=0 runs the generic code instead; CPUSIM_VERBOSE=1
// reports which one each shape gets.

bool specialization_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CPUSIM_JIT");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return on;
}

bool run_tool(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid;
    if (posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), environ) != 0) return false;
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::string jit_cache_dir() {
    std::string dir;
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) dir = x;
    else if (const char* h = std::getenv("HOME"); h && *h) dir = std::string(h) + "/.cache";
    else dir = "/tmp";
    dir += "/cpusim";
    mkdir(dir.c_str(), 0700);
    dir += "/jit";
    mkdir(dir.c_str(), 0700);
    return dir;
}

uint64_t bitcode_hash(const Module& m) {
    const auto* bc = reinterpret_cast<const uint8_t*>(m.header) + m.header->bitcode_offset;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < m.header->bitcode_size; ++i) h = (h ^ bc[i]) * 1099511628211ull;
    for (const char* v = CPUSIM_VERSION; *v; ++v) h = (h ^ uint8_t(*v)) * 1099511628211ull;
#ifdef CPUSIM_DEVC_PATH
    // The compiler that would build it: a rebuilt cpusim-devc (same version
    // string during development) must not reuse modules of the old one.
    static const uint64_t devc_id = []() -> uint64_t {
        struct stat st {};
        if (stat(CPUSIM_DEVC_PATH, &st) != 0) return uint64_t(0);
        return uint64_t(st.st_mtim.tv_sec) * 1000000007ull + uint64_t(st.st_mtim.tv_nsec) + uint64_t(st.st_size) * 31;
    }();
    h = (h ^ devc_id) * 1099511628211ull;
#endif
    return h ^ CPUSIM_KERNEL_ABI;
}

// Path of the specialized shared object, compiling it if it is not cached.
bool jit_object(const Module& m, const std::array<uint32_t, 3>& block, std::string* path) {
#if defined(CPUSIM_DEVC_PATH) && defined(CPUSIM_CLANG_PATH)
    const auto* base = reinterpret_cast<const uint8_t*>(m.header);
    const uint8_t* bc = base + m.header->bitcode_offset;
    const uint64_t n = m.header->bitcode_size;
    const uint64_t h = bitcode_hash(m);

    char name[96];
    std::snprintf(name, sizeof(name), "/%016llx-%ux%ux%u", (unsigned long long)h, block[0], block[1], block[2]);
    std::string stem = jit_cache_dir() + name;
    *path = stem + ".so";
    if (access(path->c_str(), R_OK) == 0) return true;

    std::string tmp = stem + "." + std::to_string(getpid());
    {
        std::ofstream f(tmp + ".bc", std::ios::binary);
        f.write(reinterpret_cast<const char*>(bc), std::streamsize(n));
        if (!f) return false;
    }
    char spec[48];
    std::snprintf(spec, sizeof(spec), "%u,%u,%u", block[0], block[1], block[2]);
    auto start = std::chrono::steady_clock::now();
    bool ok = run_tool({CPUSIM_DEVC_PATH, tmp + ".bc", "-o", tmp + ".o", "--manifest", tmp + ".manifest",
                        "--specialize", spec}) &&
              run_tool({CPUSIM_CLANG_PATH, "-shared", "-nostdlib", "-o", tmp + ".so", tmp + ".o", "-Wl,--as-needed",
                        "-lmvec", "-lm", "-L" CPUSIM_LIB_DIR_PATH, "-lcpusim_tensor",
                        "-Wl,-rpath," CPUSIM_LIB_DIR_PATH}) &&
              std::rename((tmp + ".so").c_str(), path->c_str()) == 0;
    for (const char* ext : {".bc", ".o", ".manifest", ".so"}) std::remove((tmp + ext).c_str());
    if (std::getenv("CPUSIM_VERBOSE"))
        std::fprintf(stderr, "cpusim: JIT for block (%u,%u,%u): %s in %.1f s\n", block[0], block[1], block[2],
                     ok ? "compiled" : "FAILED",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return ok;
#else
    (void)m, (void)block, (void)path;
    return false;
#endif
}

// ---- CPUSIM_DUMP_ASM=1: x86 assembly of the kernels a program launches
//
// For each module and block shape launched, writes
//     ./cpusim-asm-<program>-<module hash>-<X>x<Y>x<Z>.txt
// (or -generic.txt when the generic code runs: JIT off or profiling), compiled
// from the embedded bitcode exactly as the launch's code is. The listing keeps
// LLVM's comments (loop depth, block names), which objdump would lose.

bool dump_asm_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CPUSIM_DUMP_ASM");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return on;
}

void dump_asm(Module& m, const std::array<uint32_t, 3>& block) {
#if defined(CPUSIM_DEVC_PATH)
    if (!dump_asm_enabled() || m.dumped[block]) return;
    m.dumped[block] = true;
    const bool generic = block[0] == 0;
    char shape[48];
    if (generic) std::snprintf(shape, sizeof(shape), "generic");
    else std::snprintf(shape, sizeof(shape), "%ux%ux%u", block[0], block[1], block[2]);
    char hash[24];
    std::snprintf(hash, sizeof(hash), "%08llx", (unsigned long long)(bitcode_hash(m) >> 32));
    const std::string out = std::string("cpusim-asm-") + program_invocation_short_name + "-" + hash + "-" + shape + ".txt";
    if (!m.header->bitcode_size) {
        std::fprintf(stderr, "cpusim: CPUSIM_DUMP_ASM: module without bitcode (old cucc?), nothing written\n");
        return;
    }

    std::string tmp = jit_cache_dir() + "/asm." + std::to_string(getpid()) + "." + hash + "." + shape;
    {
        std::ofstream f(tmp + ".bc", std::ios::binary);
        f.write(reinterpret_cast<const char*>(m.header) + m.header->bitcode_offset,
                std::streamsize(m.header->bitcode_size));
    }
    std::vector<std::string> args = {CPUSIM_DEVC_PATH, tmp + ".bc", "-o", tmp + ".o", "--manifest",
                                     tmp + ".manifest", "--emit-asm", tmp + ".s"};
    if (!generic) {
        char spec[48];
        std::snprintf(spec, sizeof(spec), "%u,%u,%u", block[0], block[1], block[2]);
        args.push_back("--specialize");
        args.push_back(spec);
    }
    bool ok = run_tool(args);
    if (ok) {
        // Unwind directives are noise here; the rest is kept as LLVM wrote it.
        std::vector<std::string> lines;
        {
            std::ifstream s(tmp + ".s");
            for (std::string l; std::getline(s, l);)
                if (l.rfind("\t.cfi_", 0) != 0) lines.push_back(std::move(l));
        }
        // Index: each function, the line where it starts and its instruction count.
        struct Fn {
            std::string name;
            size_t line = 0, insns = 0;
        };
        std::vector<Fn> fns;
        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& l = lines[i];
            if (l.rfind("__cpusim_", 0) == 0 && l.back() == ':') fns.push_back({l.substr(0, l.size() - 1), i});
            else if (!fns.empty() && l.size() > 1 && l[0] == '\t' && l[1] != '.' && l[1] != '#') ++fns.back().insns;
        }

        std::ostringstream h;
        h << "# cpusim: x86-64 assembly of the device code (for the host CPU)\n"
          << "# program: " << program_invocation_short_name << ", module " << hash << ", block shape: " << shape
          << (generic ? " (generic code, not specialized)\n" : " (JIT specialized)\n")
          << "# kernels:";
        for (auto& [name, k] : m.kernels) h << " " << name;
        h << "\n#\n"
             "# Per kernel <K>:\n"
             "#   __cpusim_entry_<K>        one warp (32 lanes) up to the next barrier or the end\n"
             "#   __cpusim_entry<L>_<K>     L lanes in lockstep (64..512), whole block, barriers free\n"
             "#   __cpusim_entry<L>s_<K>    L = 64 or 128 lanes that suspend at barriers like a warp\n"
             "#   __cpusim_prof_<K>         instrumented twin (CPUSIM_PROFILE)\n"
             "#   __cpusim_block_<K>        whole block as rounds of warps (CPUSIM_BLOCK_FN=1)\n"
             "# The runtime picks the width by measurement (CPUSIM_VERBOSE=1 shows it).\n#\n"
             "# functions (line in this file, instructions):\n";
        std::string header = h.str();
        const size_t first = size_t(std::count(header.begin(), header.end(), '\n')) + 1 +  // + blank
                             size_t(std::count_if(fns.begin(), fns.end(), [](const Fn& x) { return x.insns; }));
        std::ofstream f(out);
        f << header;
        for (auto& fn : fns) if (fn.insns) f << "#   " << first + fn.line + 1 << "\t" << fn.insns << "\t" << fn.name << "\n";
        f << "\n";
        for (auto& l : lines) f << l << "\n";
        ok = bool(f);
    }
    for (const char* ext : {".bc", ".o", ".manifest", ".s"}) std::remove((tmp + ext).c_str());
    std::fprintf(stderr, "cpusim: %s %s\n", ok ? "assembly written to" : "FAILED to write assembly", out.c_str());
#else
    (void)m, (void)block;
#endif
}

// Loads the generic module of `m` (once): its variables get their VRAM and
// initial values.
Error load_generic(Module& m, cpusim::Device* d) {
    if (m.loaded) return Error::Success;
    const auto* base = reinterpret_cast<const uint8_t*>(m.header);
    Error e = d->load_module(base + m.header->object_offset, m.header->object_size, &m.handle);
    if (e != Error::Success) return e;
    for (auto& sym : m.symbols) {
        // VRAM allocations are aligned to 256 bytes, more than any variable needs.
        e = d->malloc(std::max<uint64_t>(sym.size, 1), &sym.address);
        if (e != Error::Success) return e;
    }
    if (!m.symbols.empty()) {
        e = d->bind_symbols(m.handle, m.symbol_table(), true);
        if (e != Error::Success) return e;
    }
    // __managed__ variables: their host copies (registered by the program).
    for (auto& sym : m.symbols)
        if (sym.managed)
            for (auto& [host, var] : registry().variables)
                if (var.module == &m && var.name == sym.name) sym.host = const_cast<void*>(host);
    m.loaded = true;
    return Error::Success;
}

// The device function to launch for this block shape, or 0 to use the generic one.
// The module specialized for `block` by the compiler, if it is in the fatbin.
const cpusim_fatbin_specialized* specialized_object(const Module& m, const std::array<uint32_t, 3>& block) {
    const auto* table = reinterpret_cast<const cpusim_fatbin_specialized*>(reinterpret_cast<const uint8_t*>(m.header) +
                                                                           m.header->specialized_offset);
    for (uint64_t i = 0; i < m.header->specialized_count; ++i)
        if (table[i].block[0] == block[0] && table[i].block[1] == block[1] && table[i].block[2] == block[2])
            return &table[i];
    return nullptr;
}

uint64_t jit_function(Module& m, const KernelDesc& k, const std::array<uint32_t, 3>& block, cpusim::Device* d) {
    if (!specialization_enabled() || (!m.header->bitcode_size && !m.header->specialized_count)) return 0;
    JitModule& j = m.jit[block];
    if (!j.tried) {
        j.tried = true;
        std::string path;
        if (const cpusim_fatbin_specialized* s = specialized_object(m, block)) {
            // Compiled ahead of time: nothing to wait for.
            j.ok = d->load_module(reinterpret_cast<const uint8_t*>(m.header) + s->offset, s->size, &j.handle) ==
                   Error::Success;
            if (std::getenv("CPUSIM_VERBOSE"))
                std::fprintf(stderr, "cpusim: block (%u,%u,%u): specialized ahead of time%s\n", block[0], block[1],
                             block[2], j.ok ? "" : ", but it does not load");
        } else if (m.header->bitcode_size && jit_object(m, block, &path)) {
            std::ifstream f(path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            j.ok = !bytes.empty() && d->load_module(bytes.data(), bytes.size(), &j.handle) == Error::Success;
        }
        // The program's variables, already in the VRAM (the generic module
        // is loaded first).
        if (j.ok && !m.symbols.empty()) j.ok = d->bind_symbols(j.handle, m.symbol_table(), false) == Error::Success;
        if (!j.ok && std::getenv("CPUSIM_VERBOSE"))
            std::fprintf(stderr, "cpusim: JIT unavailable, using the generic kernels\n");
    }
    if (!j.ok) return 0;
    auto it = j.functions.find(k.name);
    if (it != j.functions.end()) return it->second;
    uint64_t fn = 0;
    if (d->get_function(j.handle, k.name, &fn) != Error::Success) fn = 0;
    j.functions[k.name] = fn;
    return fn;
}

struct LaunchConfiguration {
    dim3 grid, block;
    size_t shared = 0;
    void* stream = nullptr;
};
thread_local std::vector<LaunchConfiguration> call_configs;

bool parse_manifest(const char* text, size_t size, Module* m) {
    std::istringstream in(std::string(text, size));
    std::string word;
    int version = 0;
    if (!(in >> word >> version) || word != "cpusim-module" || version != CPUSIM_KERNEL_ABI) return false;
    // One kernel per line: name, parameter bytes and slots, then options
    // (sm=NN: the compute capability its tensor core types need).
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        if (!(ls >> word)) continue;
        if (word == "symbol") {
            SymbolDesc sym;
            if (!(ls >> sym.name >> sym.size >> sym.align)) return false;
            std::string flag, hex;
            sym.managed = (ls >> flag) && flag == "managed";
            if (sym.managed && (ls >> hex) && hex.size() == 2 * sym.size)
                for (size_t i = 0; i < sym.size; ++i) sym.initial.push_back(uint8_t(std::stoul(hex.substr(2 * i, 2), nullptr, 16)));
            m->symbols.push_back(sym);
            continue;
        }
        if (word != "kernel") return false;
        KernelDesc k;
        size_t count = 0;
        if (!(ls >> k.name >> k.param_bytes >> count)) return false;
        for (size_t i = 0; i < count; ++i) {
            std::string slot;
            ls >> slot;
            auto colon = slot.find(':');
            if (colon == std::string::npos) return false;
            k.params.emplace_back(std::stoull(slot.substr(0, colon)), std::stoull(slot.substr(colon + 1)));
        }
        for (std::string opt; ls >> opt;)
            if (opt.rfind("sm=", 0) == 0) k.min_sm = std::atoi(opt.c_str() + 3);
            else if (opt.rfind("smem=", 0) == 0) k.shared_bytes = std::stoull(opt.substr(5));
        m->kernels[k.name] = k;
    }
    return true;
}

#define REQUIRE_DEVICE(dev)                    \
    cudaError_t init_err_ = cudaSuccess;       \
    cpusim::Device* dev = device(&init_err_);  \
    if (!dev) return init_err_

}  // namespace

extern "C" {

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return record(Error::InvalidValue);
    cudaError_t err = cudaSuccess;
    *count = device(&err) ? 1 : 0;
    return err;
}

cudaError_t cudaGetDevice(int* dev) {
    if (!dev) return record(Error::InvalidValue);
    *dev = 0;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int dev) {
    REQUIRE_DEVICE(d);
    (void)d;
    return dev == 0 ? cudaSuccess : record(Error::InvalidDevice);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int dev) {
    if (!prop) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    if (dev != 0) return record(Error::InvalidDevice);
    const auto& p = d->properties();
    *prop = {};
    std::memcpy(prop->name, p.name, sizeof(prop->name));
    std::memcpy(prop->uuid.bytes, p.uuid, sizeof(prop->uuid.bytes));
#define COPY(f) prop->f = p.f
#define COPY3(f) COPY(f[0]), COPY(f[1]), COPY(f[2])
    COPY(totalGlobalMem); COPY(sharedMemPerBlock); COPY(regsPerBlock); COPY(warpSize); COPY(memPitch);
    COPY(maxThreadsPerBlock); COPY3(maxThreadsDim); COPY3(maxGridSize); COPY(totalConstMem); COPY(major);
    COPY(minor); COPY(textureAlignment); COPY(texturePitchAlignment); COPY(multiProcessorCount); COPY(integrated);
    COPY(canMapHostMemory); COPY(concurrentKernels); COPY(ECCEnabled); COPY(pciBusID); COPY(pciDeviceID);
    COPY(pciDomainID); COPY(asyncEngineCount); COPY(unifiedAddressing); COPY(memoryBusWidth); COPY(l2CacheSize);
    COPY(persistingL2CacheMaxSize); COPY(maxThreadsPerMultiProcessor); COPY(streamPrioritiesSupported);
    COPY(globalL1CacheSupported); COPY(localL1CacheSupported); COPY(sharedMemPerMultiprocessor);
    COPY(regsPerMultiprocessor); COPY(managedMemory); COPY(isMultiGpuBoard); COPY(pageableMemoryAccess);
    COPY(concurrentManagedAccess); COPY(computePreemptionSupported); COPY(cooperativeLaunch);
    COPY(sharedMemPerBlockOptin); COPY(maxBlocksPerMultiProcessor); COPY(reservedSharedMemPerBlock);
    COPY(hostRegisterSupported); COPY(memoryPoolsSupported); COPY(clockRate); COPY(memoryClockRate);
#undef COPY3
#undef COPY
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int* version) {
    if (!version) return record(Error::InvalidValue);
    *version = runtime().device ? CUDART_VERSION : 0;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    if (!version) return record(Error::InvalidValue);
    *version = CUDART_VERSION;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device) {
    if (!value) return record(Error::InvalidValue);
    cudaDeviceProp p;
    if (cudaError_t e = cudaGetDeviceProperties(&p, device); e != cudaSuccess) return e;
    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock: *value = int(p.maxThreadsPerBlock); break;
        case cudaDevAttrMaxBlockDimX: *value = int(p.maxThreadsDim[0]); break;
        case cudaDevAttrMaxBlockDimY: *value = int(p.maxThreadsDim[1]); break;
        case cudaDevAttrMaxBlockDimZ: *value = int(p.maxThreadsDim[2]); break;
        case cudaDevAttrMaxGridDimX: *value = int(p.maxGridSize[0]); break;
        case cudaDevAttrMaxGridDimY: *value = int(p.maxGridSize[1]); break;
        case cudaDevAttrMaxGridDimZ: *value = int(p.maxGridSize[2]); break;
        case cudaDevAttrMaxSharedMemoryPerBlock: *value = int(p.sharedMemPerBlock); break;
        case cudaDevAttrTotalConstantMemory: *value = int(p.totalConstMem); break;
        case cudaDevAttrWarpSize: *value = int(p.warpSize); break;
        case cudaDevAttrMaxPitch: *value = int(p.memPitch); break;
        case cudaDevAttrMaxRegistersPerBlock: *value = int(p.regsPerBlock); break;
        case cudaDevAttrClockRate: *value = int(p.clockRate); break;
        case cudaDevAttrTextureAlignment: *value = int(p.textureAlignment); break;
        case cudaDevAttrGpuOverlap: *value = int(p.asyncEngineCount > 0); break;
        case cudaDevAttrMultiProcessorCount: *value = int(p.multiProcessorCount); break;
        case cudaDevAttrKernelExecTimeout: *value = 0; break;
        case cudaDevAttrIntegrated: *value = int(p.integrated); break;
        case cudaDevAttrCanMapHostMemory: *value = int(p.canMapHostMemory); break;
        case cudaDevAttrComputeMode: *value = 0; break;
        case cudaDevAttrConcurrentKernels: *value = int(p.concurrentKernels); break;
        case cudaDevAttrEccEnabled: *value = int(p.ECCEnabled); break;
        case cudaDevAttrPciBusId: *value = int(p.pciBusID); break;
        case cudaDevAttrPciDeviceId: *value = int(p.pciDeviceID); break;
        case cudaDevAttrTccDriver: *value = 0; break;
        case cudaDevAttrMemoryClockRate: *value = int(p.memoryClockRate); break;
        case cudaDevAttrGlobalMemoryBusWidth: *value = int(p.memoryBusWidth); break;
        case cudaDevAttrL2CacheSize: *value = int(p.l2CacheSize); break;
        case cudaDevAttrMaxThreadsPerMultiProcessor: *value = int(p.maxThreadsPerMultiProcessor); break;
        case cudaDevAttrAsyncEngineCount: *value = int(p.asyncEngineCount); break;
        case cudaDevAttrUnifiedAddressing: *value = int(p.unifiedAddressing); break;
        case cudaDevAttrPciDomainId: *value = int(p.pciDomainID); break;
        case cudaDevAttrTexturePitchAlignment: *value = int(p.texturePitchAlignment); break;
        case cudaDevAttrComputeCapabilityMajor: *value = int(p.major); break;
        case cudaDevAttrComputeCapabilityMinor: *value = int(p.minor); break;
        case cudaDevAttrStreamPrioritiesSupported: *value = int(p.streamPrioritiesSupported); break;
        case cudaDevAttrGlobalL1CacheSupported: *value = int(p.globalL1CacheSupported); break;
        case cudaDevAttrLocalL1CacheSupported: *value = int(p.localL1CacheSupported); break;
        case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = int(p.sharedMemPerMultiprocessor); break;
        case cudaDevAttrMaxRegistersPerMultiprocessor: *value = int(p.regsPerMultiprocessor); break;
        case cudaDevAttrManagedMemory: *value = int(p.managedMemory); break;
        case cudaDevAttrIsMultiGpuBoard: *value = int(p.isMultiGpuBoard); break;
        case cudaDevAttrMultiGpuBoardGroupID: *value = 0; break;
        case cudaDevAttrPageableMemoryAccess: *value = int(p.pageableMemoryAccess); break;
        case cudaDevAttrConcurrentManagedAccess: *value = int(p.concurrentManagedAccess); break;
        case cudaDevAttrComputePreemptionSupported: *value = int(p.computePreemptionSupported); break;
        case cudaDevAttrCanUseHostPointerForRegisteredMem: *value = 1; break;
        case cudaDevAttrCooperativeLaunch: *value = int(p.cooperativeLaunch); break;
        case cudaDevAttrCooperativeMultiDeviceLaunch: *value = 0; break;
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = int(p.sharedMemPerBlockOptin); break;
        case cudaDevAttrCanFlushRemoteWrites: *value = 0; break;
        case cudaDevAttrHostRegisterSupported: *value = int(p.hostRegisterSupported); break;
        case cudaDevAttrPageableMemoryAccessUsesHostPageTables: *value = int(p.pageableMemoryAccess); break;
        case cudaDevAttrDirectManagedMemAccessFromHost: *value = 1; break;
        case cudaDevAttrMaxBlocksPerMultiprocessor: *value = int(p.maxBlocksPerMultiProcessor); break;
        case cudaDevAttrMaxPersistingL2CacheSize: *value = int(p.persistingL2CacheMaxSize); break;
        case cudaDevAttrMaxAccessPolicyWindowSize: *value = 0; break;
        case cudaDevAttrReservedSharedMemoryPerBlock: *value = int(p.reservedSharedMemPerBlock); break;
        case cudaDevAttrMemoryPoolsSupported: *value = int(p.memoryPoolsSupported); break;
        default: return record(Error::InvalidValue);
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    REQUIRE_DEVICE(d);
    runtime().sched.wait_all();
    if (Error e = d->synchronize(); e != Error::Success) return record(e);
    return async_error();
}

// Limits are recorded and reported back; none constrains the device. Device
// code runs on the workers' own stacks (megabytes, not CUDA's kilobyte), and
// there is no device printf FIFO or heap to size.
namespace {
std::mutex limits_mutex;
size_t limits[7] = {1024, 1 << 20, 8 << 20, 2, 2048, 32, 0};
}  // namespace

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    if (unsigned(limit) >= 7) return record(Error::UnsupportedLimit);
    std::lock_guard lock(limits_mutex);
    limits[limit] = value;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t* value, cudaLimit limit) {
    if (!value) return record(Error::InvalidValue);
    if (unsigned(limit) >= 7) return record(Error::UnsupportedLimit);
    std::lock_guard lock(limits_mutex);
    *value = limits[limit];
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    cudaDeviceSynchronize();
    runtime().sched.clear_error();
    last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** ptr, size_t size) {
    if (!ptr) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    uint64_t dptr = 0;
    Error e = d->malloc(size, &dptr);
    *ptr = e == Error::Success ? reinterpret_cast<void*>(dptr) : nullptr;
    return record(e);
}

cudaError_t cudaFree(void* ptr) {
    REQUIRE_DEVICE(d);
    runtime().sched.wait_all();  // like CUDA: no work still using it
    return record(d->free(reinterpret_cast<uint64_t>(ptr)));
}

cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned int) {
    if (!ptr) return record(Error::InvalidValue);
    *ptr = nullptr;
    REQUIRE_DEVICE(d);
    uint64_t p = 0;
    Error e = d->malloc_managed(size, &p);
    *ptr = reinterpret_cast<void*>(p);
    return record(e);
}

// Nothing moves: managed memory is host RAM seen at one address by both sides.
cudaError_t cudaMemPrefetchAsync(const void*, size_t, int, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaMemAdvise(const void*, size_t, cudaMemoryAdvise, int) { return cudaSuccess; }

// Page-locked host memory comes from the shared pool: the same address in the
// application and in kernels, as page-locked memory is under unified
// addressing (a kernel may use it directly, mapped or not). Without a pool,
// ordinary memory.
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) { return cudaMallocHost(ptr, size); }

namespace {
struct MappedRange {
    char* host;
    char* shadow;  // == host in-process
    size_t size;
};
std::mutex mapped_mutex;
std::vector<MappedRange> mapped_ranges;
}  // namespace

cudaError_t cudaHostGetDevicePointer(void** dev, void* host, unsigned int) {
    if (!dev) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    {
        std::lock_guard lock(mapped_mutex);
        for (const auto& r : mapped_ranges)
            if (static_cast<char*>(host) >= r.host && static_cast<char*>(host) < r.host + r.size) {
                *dev = r.shadow + (static_cast<char*>(host) - r.host);
                return cudaSuccess;
            }
    }
    if (!d->is_managed_pointer(host)) return record(Error::InvalidValue);  // not from cudaHostAlloc
    *dev = host;
    return cudaSuccess;
}

// Page locking means nothing here. Mapped registration (kernels use the
// memory in place): in-process, the program's memory is the kernels'; with
// the driver, kernels run in its process and see a copy in the shared pool,
// made equal to the program's memory around every launch (copied in before,
// back after), so a kernel reads what the program wrote before launching it
// and the program reads what the kernel wrote once it is done.
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    if (!ptr || !size) return record(Error::InvalidValue);
    if (flags & cudaHostRegisterMapped) {
        REQUIRE_DEVICE(d);
        char* shadow = static_cast<char*>(ptr);
        if (std::strcmp(d->backend_name(), "inprocess") != 0) {
            uint64_t p = 0;
            if (Error e = d->malloc_managed(size, &p); e != Error::Success) return record(e);
            shadow = reinterpret_cast<char*>(p);
            std::memcpy(shadow, ptr, size);
        }
        std::lock_guard lock(mapped_mutex);
        mapped_ranges.push_back({static_cast<char*>(ptr), shadow, size});
    }
    cpusim::cudart::pinned_add(ptr, size, false);
    return cudaSuccess;
}
cudaError_t cudaHostUnregister(void* ptr) {
    {
        std::lock_guard lock(mapped_mutex);
        for (auto it = mapped_ranges.begin(); it != mapped_ranges.end(); ++it)
            if (it->host == ptr) {
                if (it->shadow != it->host) {
                    runtime().sched.wait_all();
                    std::memcpy(it->host, it->shadow, it->size);
                    cudaFree(it->shadow);
                }
                mapped_ranges.erase(it);
                break;
            }
    }
    return cpusim::cudart::pinned_remove(ptr) ? cudaSuccess : record(Error::InvalidValue);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (!ptr) return record(Error::InvalidValue);
    *ptr = nullptr;
    if (!size) return cudaSuccess;
    REQUIRE_DEVICE(d);
    uint64_t p = 0;
    if (Error e = d->malloc_managed(size, &p); e == Error::Success) {
        *ptr = reinterpret_cast<void*>(p);
        cpusim::cudart::pinned_add(*ptr, size, false);  // page-locked host memory, for the queries
    } else if (e == Error::NotSupported) {
        // Recorded so asynchronous copies from it are not staged.
        *ptr = std::malloc(size);
        if (!*ptr) return record(Error::MemoryAllocation);
        cpusim::cudart::pinned_add(*ptr, size, true);
    } else {
        return record(e);
    }
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    if (!ptr) return cudaSuccess;
    cudaError_t init_err = cudaSuccess;
    if (cpusim::Device* d = device(&init_err); d && d->is_managed_pointer(ptr)) {
        cpusim::cudart::pinned_remove(ptr);
        return cudaFree(ptr);
    }
    // Only what cudaMallocHost allocated itself (not a registered buffer).
    if (!cpusim::cudart::pinned_owned(ptr)) return record(Error::InvalidValue);
    runtime().sched.wait_all();  // no copy still reading it
    cpusim::cudart::pinned_remove(ptr);
    std::free(ptr);
    return cudaSuccess;
}

}  // extern "C"

namespace {

// Which way a copy goes, with managed memory counted as host memory (it is,
// on this side).
cudaMemcpyKind resolve_kind(cpusim::Device* d, void* dst, const void* src, cudaMemcpyKind kind) {
    bool dm = d->is_managed_pointer(dst), sm = d->is_managed_pointer(src);
    bool dd = d->is_device_pointer(dst), sd = d->is_device_pointer(src);
    if (dm || sm || kind == cudaMemcpyDefault)
        return dd ? (sd ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                  : (sd ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    return kind;
}

cudaError_t copy_now(cpusim::Device* d, void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    auto D = [](const void* p) { return reinterpret_cast<uint64_t>(p); };
    switch (kind) {
        case cudaMemcpyHostToHost: std::memmove(dst, src, count); return cudaSuccess;
        case cudaMemcpyHostToDevice: return static_cast<cudaError_t>(d->memcpy_htod(D(dst), src, count));
        case cudaMemcpyDeviceToHost: return static_cast<cudaError_t>(d->memcpy_dtoh(dst, D(src), count));
        case cudaMemcpyDeviceToDevice: return static_cast<cudaError_t>(d->memcpy_dtod(D(dst), D(src), count));
        default: return cudaErrorInvalidMemcpyDirection;
    }
}

// Queues a copy on `stream`. Host data an asynchronous copy reads is staged
// when the copy cannot run right away (CUDA's rule for pageable memory: the
// buffer may be reused as soon as the call returns), unless it is pinned or
// managed memory.
cpusim::cudart::CompletionPtr queue_copy(cpusim::Device* d, void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream, bool sync) {
    kind = resolve_kind(d, dst, src, kind);
    auto staged = std::make_shared<std::vector<uint8_t>>();
    const bool host_src = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyHostToHost;
    const bool stage = host_src && !sync && !d->is_managed_pointer(src) && !cpusim::cudart::pinned_contains(src, count);
    cpusim::cudart::EnqueueOptions opt;
    opt.inline_if_idle = true;
    opt.sync = sync;
    if (stage) opt.on_queue = [staged, src, count] { staged->assign(static_cast<const uint8_t*>(src), static_cast<const uint8_t*>(src) + count); };
    return runtime().sched.enqueue(
        stream,
        [d, dst, src, count, kind, staged] { return copy_now(d, dst, staged->empty() ? src : staged->data(), count, kind); },
        std::move(opt));
}

cpusim::cudart::CompletionPtr queue_memset(cpusim::Device* d, void* ptr, int value, size_t count, cudaStream_t stream) {
    return runtime().sched.enqueue(stream, [d, ptr, value, count]() -> cudaError_t {
        if (d->is_managed_pointer(ptr)) {
            std::memset(ptr, value, count);
            return cudaSuccess;
        }
        return static_cast<cudaError_t>(d->memset(reinterpret_cast<uint64_t>(ptr), value, count));
    });
}

}  // namespace

extern "C" {

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    // Synchronous: in order on the default stream, and back when done.
    auto done = queue_copy(d, dst, src, count, kind, nullptr, true);
    done->wait();
    if (done->error != cudaSuccess) return record(done->error);
    return async_error();
}

cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    if (count == 0) return cudaSuccess;
    REQUIRE_DEVICE(d);
    if (!d->is_device_pointer(ptr) && !d->is_managed_pointer(ptr)) return record(Error::InvalidValue);
    auto done = queue_memset(d, ptr, value, count, nullptr);  // asynchronous on device memory, like CUDA's
    // On managed memory the host may look right after (the GPU's memset is
    // done by then too): wait.
    if (d->is_managed_pointer(ptr)) {
        done->wait();
        return async_error();
    }
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    if (!free || !total) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    uint64_t f = 0, t = 0;
    Error e = d->mem_get_info(&f, &t);
    *free = f;
    *total = t;
    return record(e);
}

// ---- streams and events (see cudart/streams.h)

}  // extern "C"

// An event's last record: the operation that stamps it.
struct CUevent_st {
    unsigned flags = 0;
    std::mutex mutex;
    cpusim::cudart::CompletionPtr pending;  // null: never recorded
    std::shared_ptr<std::chrono::steady_clock::time_point> when;
    CUgraph_st* captured = nullptr;  // recorded in a stream being captured into this graph
};

// ---- CUDA Graphs
//
// A node is an operation with its arguments copied (a kernel's parameter
// values, as cudaLaunchKernel copies them). Nodes are kept in the order they
// were added: each one after the nodes it depends on (in a capture, the
// program issued them in that order; explicit nodes name earlier ones). A
// launch enqueues them in that order on its stream.
struct CUgraphNode_st {
    enum Kind { Kernel, Memcpy, Memset, Host, Empty } kind = Empty;
    const void* func = nullptr;
    dim3 grid, block;
    size_t shared = 0;
    std::vector<uint8_t> args;                         // parameter values, packed
    std::vector<std::pair<size_t, size_t>> arg_slots;  // offset, size in `args`
    void* dst = nullptr;
    const void* src = nullptr;
    size_t count = 0;
    cudaMemcpyKind copy_kind = cudaMemcpyDefault;
    cudaMemsetParams memset{};
    std::vector<uint8_t> pattern;  // one row of a memset with elements wider than a byte
    cudaHostFn_t fn = nullptr;
    void* data = nullptr;
};

struct CUgraph_st {
    std::vector<std::unique_ptr<CUgraphNode_st>> nodes;
};

struct CUgraphExec_st {
    std::vector<CUgraphNode_st> nodes;
    std::map<const CUgraphNode_st*, size_t> index;  // the graph's node -> ours
};

namespace {
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
}  // namespace

extern "C" {

cudaError_t cudaStreamCreate(cudaStream_t* stream) { return cudaStreamCreateWithFlags(stream, cudaStreamDefault); }

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    return cudaStreamCreateWithPriority(stream, flags, 0);
}

// Priorities are recorded and reported; streams are not scheduled by them.
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority) {
    if (!stream) return record(Error::InvalidValue);
    *stream = new CUstream_st{runtime().sched.create(flags), flags, std::clamp(priority, -5, 0)};
    return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* least, int* greatest) {
    if (least) *least = 0;
    if (greatest) *greatest = -5;
    return cudaSuccess;
}

namespace {
// The legacy and per-thread default streams are not objects.
bool is_object(cudaStream_t s) { return s && s != cudaStreamLegacy && s != cudaStreamPerThread; }
}  // namespace

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!is_object(stream)) return record(Error::InvalidResourceHandle);
    runtime().sched.destroy(stream->queue);  // its work still runs first
    delete stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    runtime().sched.wait_stream(stream);
    return async_error();
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags) {
    if (!flags) return record(Error::InvalidValue);
    *flags = is_object(stream) ? stream->flags : cudaStreamDefault;
    return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    if (!priority) return record(Error::InvalidValue);
    *priority = is_object(stream) ? stream->priority : 0;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return runtime().sched.idle(stream) ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int) {
    if (!event) return record(Error::InvalidResourceHandle);
    {
        // An event recorded during a capture: this stream joins the capture
        // (a fork); its operations go into the same graph.
        std::lock_guard lock(event->mutex);
        if (event->captured) {
            std::lock_guard clock(capture_mutex);
            capturing_streams[stream] = event->captured;
            return cudaSuccess;
        }
    }
    if (capturing(stream)) return cudaSuccess;  // ordering inside a graph is its node order
    cpusim::cudart::EnqueueOptions opt;
    {
        std::lock_guard lock(event->mutex);
        if (event->pending) opt.after.push_back(event->pending);
    }
    runtime().sched.enqueue(stream, nullptr, std::move(opt));
    return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* data) {
    if (!fn) return record(Error::InvalidValue);
    if (CUgraph_st* g = capturing(stream)) {
        CUgraphNode_st* n = add_node(g, CUgraphNode_st::Host);
        n->fn = fn, n->data = data;
        return cudaSuccess;
    }
    runtime().sched.enqueue(stream, [fn, data] {
        fn(data);
        return cudaSuccess;
    });
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t cb, void* data, unsigned int) {
    if (!cb) return record(Error::InvalidValue);
    runtime().sched.enqueue(stream, [stream, cb, data] {
        cb(stream, cudaSuccess, data);
        return cudaSuccess;
    });
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return record(Error::InvalidValue);
    if (CUgraph_st* g = capturing(stream)) {
        CUgraphNode_st* n = add_node(g, CUgraphNode_st::Memcpy);
        n->dst = dst, n->src = src, n->count = count, n->copy_kind = kind;
        return cudaSuccess;
    }
    REQUIRE_DEVICE(d);
    queue_copy(d, dst, src, count, kind, stream, false);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (CUgraph_st* g = capturing(stream)) {
        set_memset(add_node(g, CUgraphNode_st::Memset), {ptr, count, unsigned(value & 0xff), 1, count, 1});
        return cudaSuccess;
    }
    REQUIRE_DEVICE(d);
    if (!d->is_device_pointer(ptr) && !d->is_managed_pointer(ptr)) return record(Error::InvalidValue);
    queue_memset(d, ptr, value, count, stream);
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) { return cudaEventCreateWithFlags(event, cudaEventDefault); }

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (!event) return record(Error::InvalidValue);
    *event = new CUevent_st;
    (*event)->flags = flags;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;  // a record still queued keeps its own time stamp alive
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return record(Error::InvalidResourceHandle);
    if (CUgraph_st* g = capturing(stream)) {
        std::lock_guard lock(event->mutex);
        event->captured = g;
        return cudaSuccess;
    }
    auto when = std::make_shared<std::chrono::steady_clock::time_point>();
    auto done = runtime().sched.enqueue(stream, [when] {
        *when = std::chrono::steady_clock::now();  // everything before it in the stream ran
        return cudaSuccess;
    });
    std::lock_guard lock(event->mutex);
    event->pending = done;
    event->when = when;
    event->captured = nullptr;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) return record(Error::InvalidResourceHandle);
    cpusim::cudart::CompletionPtr c;
    {
        std::lock_guard lock(event->mutex);
        c = event->pending;
    }
    if (c) c->wait();
    return async_error();
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!event) return record(Error::InvalidResourceHandle);
    std::lock_guard lock(event->mutex);
    return !event->pending || event->pending->ready() ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return record(Error::InvalidValue);
    if (!start || !end) return record(Error::InvalidResourceHandle);
    std::scoped_lock lock(start->mutex, end->mutex);
    if (!start->pending || !end->pending || (start->flags | end->flags) & cudaEventDisableTiming)
        return record(Error::InvalidResourceHandle);
    if (!start->pending->ready() || !end->pending->ready()) return cudaErrorNotReady;
    *ms = std::chrono::duration<float, std::milli>(*end->when - *start->when).count();
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) {
    cudaError_t e = last_error;
    last_error = cudaSuccess;
    if (e == cudaSuccess) e = runtime().sched.take_error();  // from asynchronous work
    return e;
}

cudaError_t cudaPeekAtLastError(void) { return last_error != cudaSuccess ? last_error : runtime().sched.peek_error(); }

// ---- kernel registration (called from the constructors Clang emits)

struct FatbinWrapper {
    int magic;
    int version;
    const void* data;
    void* unused;
};

void** __cudaRegisterFatBinary(void* fatbin_wrapper) {
    auto* w = static_cast<const FatbinWrapper*>(fatbin_wrapper);
    auto m = std::make_unique<Module>();
    auto* h = w ? static_cast<const cpusim_fatbin_header*>(w->data) : nullptr;
    if (!h || std::memcmp(h->magic, CPUSIM_FATBIN_MAGIC, 8) != 0 || h->version != CPUSIM_KERNEL_ABI) {
        std::fprintf(stderr, "cpusim: this program's device code was not built by the CPU SIMT compiler\n");
    } else if (!parse_manifest(reinterpret_cast<const char*>(h) + h->manifest_offset, h->manifest_size, m.get())) {
        std::fprintf(stderr, "cpusim: malformed kernel manifest\n");
    } else {
        m->header = h;
    }
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    r.modules.push_back(std::move(m));
    return reinterpret_cast<void**>(r.modules.back().get());
}

void __cudaRegisterFatBinaryEnd(void**) {}
void __cudaUnregisterFatBinary(void**) {}

int __cudaRegisterFunction(void** handle, const char* host_fun, char*, const char* device_name, int, uint3*, uint3*,
                           dim3*, dim3*, int*) {
    auto* m = reinterpret_cast<Module*>(handle);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    Function f;
    f.module = m;
    if (auto it = m->kernels.find(device_name); it != m->kernels.end()) f.kernel = &it->second;
    r.functions[host_fun] = f;
    return 0;
}

void __cudaRegisterVar(void** handle, char* host_var, char*, const char* device_name, int, size_t size, int, int) {
    auto* m = reinterpret_cast<Module*>(handle);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    r.variables[host_var] = {m, device_name, size};
    // A __managed__ variable's host copy starts from its initial value.
    for (const auto& sym : m->symbols)
        if (sym.managed && sym.name == device_name && sym.initial.size() == size)
            std::memcpy(host_var, sym.initial.data(), size);
}

unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shared, void* stream) {
    call_configs.push_back({grid, block, shared, stream});
    return 0;
}

cudaError_t __cudaPopCallConfiguration(dim3* grid, dim3* block, size_t* shared, void* stream) {
    if (call_configs.empty()) return record(Error::MissingConfiguration);
    LaunchConfiguration c = call_configs.back();
    call_configs.pop_back();
    *grid = c.grid;
    *block = c.block;
    *shared = c.shared;
    *static_cast<void**>(stream) = c.stream;
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shared,
                             cudaStream_t stream) {
    REQUIRE_DEVICE(d);
    if (CUgraph_st* g = capturing(stream)) return set_kernel(add_node(g, CUgraphNode_st::Kernel), func, grid, block, shared, args);
    {
        // A bad configuration is reported by the launch itself, as in CUDA.
        const auto& p = d->properties();
        const uint32_t g[3] = {grid.x, grid.y, grid.z}, b[3] = {block.x, block.y, block.z};
        uint64_t threads = 1;
        for (int i = 0; i < 3; ++i) {
            if (!g[i] || !b[i] || b[i] > uint32_t(p.maxThreadsDim[i]) || g[i] > uint32_t(p.maxGridSize[i]))
                return record(Error::InvalidConfiguration);
            threads *= b[i];
        }
        if (threads > uint64_t(p.maxThreadsPerBlock)) return record(Error::InvalidConfiguration);
        if (shared > p.sharedMemPerBlockOptin) return record(Error::InvalidValue);
    }
    Registry& r = registry();
    std::unique_lock lock(r.mutex);
    auto it = r.functions.find(func);
    if (it == r.functions.end() || !it->second.module->header) return record(Error::InvalidDeviceFunction);
    Function& f = it->second;
    if (!f.kernel) {
        // Registered, but the compiler did not produce code for it.
        return record(Error::NoKernelImageForDevice);
    }
    // Tensor core types the CPU cannot run natively (no AVX512-BF16 for bf16,
    // no AVX512-VNNI for int8): like a binary built for a newer GPU.
    if (f.kernel->min_sm > d->properties().major * 10 + d->properties().minor)
        return record(Error::NoKernelImageForDevice);
    Module& m = *f.module;
    if (Error e = load_generic(m, d); e != Error::Success) return record(e);
    if (!f.handle) {
        Error e = d->get_function(m.handle, f.kernel->name, &f.handle);
        if (e != Error::Success) return record(e);
    }
    const KernelDesc& k = *f.kernel;
    uint64_t function = f.handle;
    // Instrumented launches run the generic module (the JIT one has no
    // instrumented entries).
    bool specialized = false;
    if (!profiler().enabled)
        if (uint64_t jf = jit_function(m, k, {block.x, block.y, block.z}, d)) function = jf, specialized = true;
    if (dump_asm_enabled()) dump_asm(m, specialized ? std::array<uint32_t, 3>{block.x, block.y, block.z}
                                                    : std::array<uint32_t, 3>{0, 0, 0});
    lock.unlock();

    std::vector<uint8_t> params(k.param_bytes);
    for (size_t i = 0; i < k.params.size(); ++i) std::memcpy(params.data() + k.params[i].first, args[i], k.params[i].second);

    cpusim::LaunchConfig cfg;
    cfg.grid[0] = grid.x, cfg.grid[1] = grid.y, cfg.grid[2] = grid.z;
    cfg.block[0] = block.x, cfg.block[1] = block.y, cfg.block[2] = block.z;
    cfg.shared_bytes = static_cast<uint32_t>(shared);

    // CPUSIM_DEBUG=1: the driver runs the launch step by step with guard pages
    // and reports the first invalid access; =2 also traces every step.
    static const int debug = [] {
        const char* v = std::getenv("CPUSIM_DEBUG");
        return v ? std::atoi(v) : 0;
    }();
    cfg.debug = debug;

    Profiler& prof = profiler();
    // __managed__ variables of the kernel's module: the host's values in,
    // the kernel's out (see the device header).
    struct Managed {
        uint64_t address;
        void* host;
        uint64_t size;
    };
    std::vector<Managed> managed;
    for (const auto& sym : m.symbols)
        if (sym.managed && sym.host) managed.push_back({sym.address, sym.host, sym.size});
    if (!prof.enabled && !debug) {
        // Queued on its stream; returns right away.
        auto buf = std::make_shared<std::vector<uint8_t>>(std::move(params));
        runtime().sched.enqueue(stream, [d, function, cfg, buf, name = k.name, managed]() -> cudaError_t {
            for (const Managed& v : managed) d->memcpy_htod(v.address, v.host, v.size);
            std::vector<MappedRange> copies;
            {
                std::lock_guard lock(mapped_mutex);
                for (const auto& r : mapped_ranges)
                    if (r.shadow != r.host) copies.push_back(r);
            }
            for (const auto& r : copies) std::memcpy(r.shadow, r.host, r.size);
            cpusim::LaunchResult result;
            Error e = d->launch(function, cfg, buf->data(), buf->size(), &result);
            for (const auto& r : copies) std::memcpy(r.host, r.shadow, r.size);
            for (const Managed& v : managed) d->memcpy_dtoh(v.host, v.address, v.size);
            if (e != Error::Success && result.report[0])
                std::fprintf(stderr, "cpusim: kernel %s failed:\n  %s\n", name.c_str(), result.report);
            return static_cast<cudaError_t>(e);
        });
        return cudaSuccess;
    }
    // Profiling and debug runs report on the result: synchronous, after
    // everything issued before.
    runtime().sched.wait_all();
    if (!prof.enabled) {
        cpusim::LaunchResult result;
        Error e = d->launch(function, cfg, params.data(), params.size(), &result);
        if (e != Error::Success && result.report[0])
            std::fprintf(stderr, "cpusim: kernel %s failed:\n  %s\n", k.name.c_str(), result.report);
        return record(e);
    }

    ProfileEntry* entry;
    {
        std::lock_guard plock(prof.mutex);
        entry = prof.find(k.name, cfg);
        cfg.profile = entry == nullptr;
    }
    cpusim::LaunchResult result;
    Error e = d->launch(function, cfg, params.data(), params.size(), &result);
    if (e != Error::Success) return record(e);
    std::lock_guard plock(prof.mutex);
    if (cfg.profile) {
        ProfileEntry ne;
        ne.kernel = k.name;
        std::memcpy(ne.grid, cfg.grid, sizeof(ne.grid));
        std::memcpy(ne.block, cfg.block, sizeof(ne.block));
        std::memcpy(ne.counters, result.counters, sizeof(ne.counters));
        ne.profiled_ns = result.kernel_ns;
        prof.entries.push_back(ne);
    } else {
        entry->timed_runs += 1;
        entry->timed_ns += result.kernel_ns;
    }
    return cudaSuccess;
}

// ---- kernel attributes and occupancy

namespace {
constexpr int kResidentPerSM = 2;  // workers (logical CPUs) per multiprocessor
// The registered kernel behind a host stub, or null.
KernelDesc* kernel_of(const void* func) {
    Registry& r = registry();
    auto it = r.functions.find(func);
    return it == r.functions.end() ? nullptr : const_cast<KernelDesc*>(it->second.kernel);
}
}  // namespace

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func) {
    if (!attr) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    KernelDesc* k = kernel_of(func);
    if (!k) return record(Error::InvalidDeviceFunction);
    const auto& p = d->properties();
    *attr = {};
    attr->sharedSizeBytes = k->shared_bytes;
    attr->maxThreadsPerBlock = p.maxThreadsPerBlock;
    attr->numRegs = 32;  // no registers to count: a plausible figure for occupancy arithmetic
    attr->ptxVersion = attr->binaryVersion = p.major * 10 + p.minor;
    attr->maxDynamicSharedSizeBytes =
        k->max_dynamic_shared >= 0 ? k->max_dynamic_shared : int(p.sharedMemPerBlock - k->shared_bytes);
    attr->preferredShmemCarveout = -1;
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value) {
    REQUIRE_DEVICE(d);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    KernelDesc* k = kernel_of(func);
    if (!k) return record(Error::InvalidDeviceFunction);
    if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) {
        if (value < 0 || uint64_t(value) + k->shared_bytes > d->properties().sharedMemPerBlockOptin)
            return record(Error::InvalidValue);
        k->max_dynamic_shared = value;
    } else if (attr != cudaFuncAttributePreferredSharedMemoryCarveout) {
        return record(Error::InvalidValue);
    }
    return cudaSuccess;
}

cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache) {
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    return kernel_of(func) ? cudaSuccess : record(Error::InvalidDeviceFunction);
}

// As a GPU counts: blocks that fit at once in one multiprocessor by threads
// (in whole warps), by shared memory and by the per-multiprocessor limit.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* blocks, const void* func, int block_size,
                                                                   size_t dynamic_shared, unsigned int) {
    if (!blocks || block_size <= 0) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    uint64_t shared;
    {
        Registry& r = registry();
        std::lock_guard lock(r.mutex);
        KernelDesc* k = kernel_of(func);
        if (!k) return record(Error::InvalidDeviceFunction);
        shared = k->shared_bytes + dynamic_shared;
    }
    const auto& p = d->properties();
    if (block_size > p.maxThreadsPerBlock) {
        *blocks = 0;
        return cudaSuccess;
    }
    const int warps = (block_size + 31) / 32;
    // Blocks run whole, one per worker (logical CPU): two per multiprocessor
    // at a time, whatever the size. A program that sizes a persistent or
    // cooperative grid from this gets every block resident.
    int n = std::min({p.maxBlocksPerMultiProcessor, p.maxThreadsPerMultiProcessor / (warps * 32), kResidentPerSM});
    if (shared) n = std::min<int>(n, int(p.sharedMemPerMultiprocessor / shared));
    *blocks = std::max(n, 0);
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* blocks, const void* func, int block_size,
                                                          size_t dynamic_shared) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(blocks, func, block_size, dynamic_shared, 0);
}

// Every block of the grid must be resident at once (this_grid().sync()
// waits for all of them): at most the occupancy times the multiprocessors.
cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shared,
                                        cudaStream_t stream) {
    REQUIRE_DEVICE(d);
    int per_sm = 0;
    if (cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, func, int(block.x * block.y * block.z),
                                                                      shared);
        e != cudaSuccess)
        return e;
    if (uint64_t(grid.x) * grid.y * grid.z > uint64_t(per_sm) * d->properties().multiProcessorCount)
        return record(Error::CooperativeLaunchTooLarge);
    return cudaLaunchKernel(func, grid, block, args, shared, stream);
}

// ---- pitched and 3D memory

cudaError_t cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    if (!ptr || !pitch) return record(Error::InvalidValue);
    // Rows aligned as a GPU aligns them (512 bytes): pitch arithmetic in
    // programs works the same.
    const size_t p = (std::max<size_t>(width, 1) + 511) / 512 * 512;
    cudaError_t e = cudaMalloc(ptr, p * std::max<size_t>(height, 1));
    if (e == cudaSuccess) *pitch = p;
    return e;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* out, struct cudaExtent extent) {
    if (!out) return record(Error::InvalidValue);
    size_t pitch = 0;
    void* p = nullptr;
    cudaError_t e = cudaMallocPitch(&p, &pitch, extent.width, extent.height * std::max<size_t>(extent.depth, 1));
    if (e == cudaSuccess) *out = make_cudaPitchedPtr(p, pitch, extent.width, extent.height);
    return e;
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                              enum cudaMemcpyKind kind, cudaStream_t stream) {
    if (width > dpitch || width > spitch) return record(Error::InvalidValue);
    if (!width || !height) return cudaSuccess;
    // Contiguous rows: one copy.
    if (dpitch == width && spitch == width)
        return cudaMemcpyAsync(dst, src, width * height, kind, stream);
    for (size_t y = 0; y < height; ++y)
        if (cudaError_t e = cudaMemcpyAsync(static_cast<char*>(dst) + y * dpitch,
                                            static_cast<const char*>(src) + y * spitch, width, kind, stream);
            e != cudaSuccess)
            return e;
    return cudaSuccess;
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                         enum cudaMemcpyKind kind) {
    if (cudaError_t e = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, nullptr); e != cudaSuccess)
        return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream) {
    if (!p || p->srcArray || p->dstArray) return record(p ? Error::NotSupported : Error::InvalidValue);
    const cudaPitchedPtr& s = p->srcPtr;
    const cudaPitchedPtr& t = p->dstPtr;
    const size_t sh = s.ysize ? s.ysize : p->extent.height, th = t.ysize ? t.ysize : p->extent.height;
    for (size_t z = 0; z < p->extent.depth; ++z) {
        const char* from = static_cast<const char*>(s.ptr) + ((p->srcPos.z + z) * sh + p->srcPos.y) * s.pitch + p->srcPos.x;
        char* to = static_cast<char*>(t.ptr) + ((p->dstPos.z + z) * th + p->dstPos.y) * t.pitch + p->dstPos.x;
        if (cudaError_t e = cudaMemcpy2DAsync(to, t.pitch, from, s.pitch, p->extent.width, p->extent.height, p->kind,
                                              stream);
            e != cudaSuccess)
            return e;
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p) {
    if (cudaError_t e = cudaMemcpy3DAsync(p, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemset2DAsync(void* ptr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
    if (width > pitch) return record(Error::InvalidValue);
    if (pitch == width) return cudaMemsetAsync(ptr, value, width * height, stream);
    for (size_t y = 0; y < height; ++y)
        if (cudaError_t e = cudaMemsetAsync(static_cast<char*>(ptr) + y * pitch, value, width, stream); e != cudaSuccess)
            return e;
    return cudaSuccess;
}

cudaError_t cudaMemset2D(void* ptr, size_t pitch, int value, size_t width, size_t height) {
    if (cudaError_t e = cudaMemset2DAsync(ptr, pitch, value, width, height, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr p, int value, struct cudaExtent extent, cudaStream_t stream) {
    const size_t rows = p.ysize ? p.ysize : extent.height;
    for (size_t z = 0; z < extent.depth; ++z)
        if (cudaError_t e = cudaMemset2DAsync(static_cast<char*>(p.ptr) + z * rows * p.pitch, p.pitch, value,
                                              extent.width, extent.height, stream);
            e != cudaSuccess)
            return e;
    return cudaSuccess;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr p, int value, struct cudaExtent extent) {
    if (cudaError_t e = cudaMemset3DAsync(p, value, extent, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dst_device, const void* src, int src_device, size_t count,
                                cudaStream_t stream) {
    if (dst_device != 0 || src_device != 0) return record(Error::InvalidDevice);
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

cudaError_t cudaMemcpyPeer(void* dst, int dst_device, const void* src, int src_device, size_t count) {
    if (cudaError_t e = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, nullptr); e != cudaSuccess)
        return e;
    return cudaStreamSynchronize(nullptr);
}

// Stream-ordered allocation: the allocation itself is immediate; the free
// waits for the stream's earlier work.
cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t) { return cudaMalloc(ptr, size); }
cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
    if (cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return e;
    return cudaFree(ptr);
}

// ---- pointers, flags and settings

// Managed memory is visible everywhere at all times: nothing to attach.
cudaError_t cudaStreamAttachMemAsync(cudaStream_t, void*, size_t, unsigned int) { return cudaSuccess; }

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* a, const void* ptr) {
    if (!a) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    *a = {};
    a->device = 0;
    if (cpusim::cudart::pinned_contains(ptr, 1)) {  // page-locked: usable by kernels in place
        a->type = cudaMemoryTypeHost;
        a->hostPointer = const_cast<void*>(ptr);
        a->devicePointer = d->is_managed_pointer(ptr) ? const_cast<void*>(ptr) : nullptr;
    } else if (d->is_managed_pointer(ptr)) {
        a->type = cudaMemoryTypeManaged;
        a->devicePointer = a->hostPointer = const_cast<void*>(ptr);
    } else if (d->is_device_pointer(ptr)) {
        a->type = cudaMemoryTypeDevice;
        a->devicePointer = const_cast<void*>(ptr);
    } else {
        a->type = cudaMemoryTypeUnregistered;
        a->device = -2;
    }
    return cudaSuccess;
}

cudaError_t cudaHostGetFlags(unsigned int* flags, void* host) {
    if (!flags) return record(Error::InvalidValue);
    REQUIRE_DEVICE(d);
    if (d->is_managed_pointer(host)) *flags = cudaHostAllocMapped;
    else if (cpusim::cudart::pinned_contains(host, 1)) *flags = cudaHostAllocDefault;
    else return record(Error::InvalidValue);
    return cudaSuccess;
}

namespace {
unsigned device_flags = 0;
cudaFuncCache cache_config = cudaFuncCachePreferNone;
}  // namespace

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    device_flags = flags;
    return cudaSuccess;
}
cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    if (!flags) return record(Error::InvalidValue);
    *flags = device_flags | cudaDeviceMapHost;  // host memory is always mapped
    return cudaSuccess;
}
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache config) {
    cache_config = config;
    return cudaSuccess;
}
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* config) {
    if (!config) return record(Error::InvalidValue);
    *config = cache_config;
    return cudaSuccess;
}
cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp*) {
    if (!device) return record(Error::InvalidValue);
    *device = 0;
    return cudaSuccess;
}

// ---- __device__ and __constant__ variables

namespace {
// The VRAM address of a registered variable, its module loaded.
cudaError_t symbol_address(const void* symbol, uint64_t* address, size_t* size) {
    REQUIRE_DEVICE(d);
    Registry& r = registry();
    std::lock_guard lock(r.mutex);
    auto it = r.variables.find(symbol);
    if (it == r.variables.end() || !it->second.module->header) return record(Error::InvalidSymbol);
    Module& m = *it->second.module;
    if (Error e = load_generic(m, d); e != Error::Success) return record(e);
    for (const auto& s : m.symbols)
        if (s.name == it->second.name) {
            *address = s.address;
            if (size) *size = s.size;
            return cudaSuccess;
        }
    return record(Error::InvalidSymbol);  // the compiler dropped it: no device code uses it
}
}  // namespace

cudaError_t cudaGetSymbolAddress(void** dev_ptr, const void* symbol) {
    if (!dev_ptr) return record(Error::InvalidValue);
    uint64_t a;
    if (cudaError_t e = symbol_address(symbol, &a, nullptr); e != cudaSuccess) return e;
    *dev_ptr = reinterpret_cast<void*>(a);
    return cudaSuccess;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    if (!size) return record(Error::InvalidValue);
    uint64_t a;
    return symbol_address(symbol, &a, size);
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset,
                                    cudaMemcpyKind kind, cudaStream_t stream) {
    uint64_t a;
    size_t size;
    if (cudaError_t e = symbol_address(symbol, &a, &size); e != cudaSuccess) return e;
    if (offset + count > size) return record(Error::InvalidValue);
    if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyHostToHost) return record(Error::InvalidMemcpyDirection);
    return cudaMemcpyAsync(reinterpret_cast<void*>(a + offset), src, count, kind, stream);
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset,
                                      cudaMemcpyKind kind, cudaStream_t stream) {
    uint64_t a;
    size_t size;
    if (cudaError_t e = symbol_address(symbol, &a, &size); e != cudaSuccess) return e;
    if (offset + count > size) return record(Error::InvalidValue);
    if (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyHostToHost) return record(Error::InvalidMemcpyDirection);
    return cudaMemcpyAsync(dst, reinterpret_cast<const void*>(a + offset), count, kind, stream);
}

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) {
    if (cudaError_t e = cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, nullptr); e != cudaSuccess) return e;
    return cudaStreamSynchronize(nullptr);
}

cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind) {
    if (cudaError_t e = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, nullptr); e != cudaSuccess)
        return e;
    return cudaStreamSynchronize(nullptr);
}

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

const char* cudaGetErrorString(cudaError_t error) { return cpusim::error_string(static_cast<Error>(error)); }
const char* cudaGetErrorName(cudaError_t error) { return cpusim::error_name(static_cast<Error>(error)); }

}  // extern "C"
