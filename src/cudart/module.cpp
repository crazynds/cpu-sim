// The CUDA runtime: the modules the compiler registers (fatbin, manifest),
// their code for each block shape (ahead of time or JIT), and their
// __device__ and __constant__ variables.
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

Registry& registry() {
    static Registry r;
    return r;
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
    // The CPU it compiles for: a cache shared by machines (a home on NFS)
    // must not hand one machine's code to another.
    h = (h ^ cpusim::cpu_signature()) * 1099511628211ull;
    return h ^ CPUSIM_KERNEL_ABI;
}

// Path of the specialized shared object, compiling it if it is not cached.
bool jit_object(const Module& m, const std::array<uint32_t, 3>& block, const std::string& kernel, std::string* path) {
#if defined(CPUSIM_DEVC_PATH) && defined(CPUSIM_CLANG_PATH)
    const auto* base = reinterpret_cast<const uint8_t*>(m.header);
    const uint8_t* bc = base + m.header->bitcode_offset;
    const uint64_t n = m.header->bitcode_size;
    const uint64_t h = bitcode_hash(m);

    char name[128];
    std::snprintf(name, sizeof(name), "/%016llx-%ux%ux%u-%016llx", (unsigned long long)h, block[0], block[1], block[2],
                  (unsigned long long)std::hash<std::string>{}(kernel));
    std::string stem = jit_cache_dir() + name;
    *path = stem + ".so";
    if (access(path->c_str(), R_OK) == 0) return true;

    std::string tmp = stem + "." + std::to_string(getpid());
    {
        std::ofstream f(tmp + ".bc", std::ios::binary);
        f.write(reinterpret_cast<const char*>(bc), std::streamsize(n));
        if (!f) return false;
    }
    // block {0, 0, c}: the class of shapes c (cpusim_shape_class).
    char spec[48];
    if (block[0]) std::snprintf(spec, sizeof(spec), "%u,%u,%u", block[0], block[1], block[2]);
    else std::snprintf(spec, sizeof(spec), "%u", block[2]);
    auto start = std::chrono::steady_clock::now();
    bool ok = run_tool({CPUSIM_DEVC_PATH, tmp + ".bc", "-o", tmp + ".o", "--manifest", tmp + ".manifest",
                        block[0] ? "--specialize" : "--specialize-class", spec, "--kernels", kernel}) &&
              run_tool({CPUSIM_CLANG_PATH, "-shared", "-nostdlib", "-o", tmp + ".so", tmp + ".o", "-Wl,--as-needed",
                        "-lmvec", "-lm", "-L" CPUSIM_LIB_DIR_PATH, "-lcpusim_tensor",
                        "-Wl,-rpath," CPUSIM_LIB_DIR_PATH}) &&
              std::rename((tmp + ".so").c_str(), path->c_str()) == 0;
    for (const char* ext : {".bc", ".o", ".manifest", ".so"}) std::remove((tmp + ext).c_str());
    if (std::getenv("CPUSIM_VERBOSE"))
        std::fprintf(stderr, "cpusim: JIT of %s for block (%u,%u,%u): %s in %.1f s\n", kernel.c_str(), block[0],
                     block[1], block[2], ok ? "compiled" : "FAILED",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return ok;
#else
    (void)m, (void)block, (void)kernel, (void)path;
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
    // block {0, 0, c}: the module of the class of shapes c.
    for (uint64_t i = 0; i < m.header->specialized_count; ++i) {
        const auto& e = table[i];
        if (block[0] ? e.block[0] == block[0] && e.block[1] == block[1] && e.block[2] == block[2]
                     : e.block[0] == 0 && e.shape_class == block[2])
            return &e;
    }
    return nullptr;
}

uint64_t jit_function(Module& m, const KernelDesc& k, const std::array<uint32_t, 3>& shape, cpusim::Device* d) {
    if (!specialization_enabled() || (!m.header->bitcode_size && !m.header->specialized_count)) return 0;
    // A kernel without barriers runs its class's code when the shape has a
    // class (one module for every shape of the class): key {0, 0, class}.
    const uint32_t cls = k.barriers || warp_size() < 32 ? 0 : cpusim_shape_class(shape[0], shape[1], shape[2]);
    const std::array<uint32_t, 3> block = cls ? std::array<uint32_t, 3>{0, 0, cls} : shape;
    auto lookup = [&](JitModule& jm) -> uint64_t {
        if (!jm.ok) return 0;
        auto it = jm.functions.find(k.name);
        if (it != jm.functions.end()) return it->second;
        uint64_t fn = 0;
        if (d->get_function(jm.handle, k.name, &fn) != Error::Success) fn = 0;
        jm.functions[k.name] = fn;
        return fn;
    };
    // The program's variables, already in the VRAM (the generic module is
    // loaded first).
    auto bind = [&](JitModule& jm) {
        if (jm.ok && !m.symbols.empty()) jm.ok = d->bind_symbols(jm.handle, m.symbol_table(), false) == Error::Success;
    };
    // Compiled ahead of time for this shape (or class).
    if (const cpusim_fatbin_specialized* s = specialized_object(m, block)) {
        JitModule& j = m.jit[block];
        if (!j.tried) {
            j.tried = true;
            j.ok = d->load_module(reinterpret_cast<const uint8_t*>(m.header) + s->offset, s->size, &j.handle) ==
                   Error::Success;
            bind(j);
            if (std::getenv("CPUSIM_VERBOSE"))
                std::fprintf(stderr, "cpusim: block (%u,%u,%u): specialized ahead of time%s\n", block[0], block[1],
                             block[2], j.ok ? "" : ", but it does not load");
        }
        if (uint64_t fn = lookup(j)) return fn;
    }
    // Otherwise the JIT: this kernel for this shape (compiling every kernel
    // of the file for each shape cost minutes on files with many kernels).
    if (!m.header->bitcode_size) return 0;
    JitModule& r = m.jit_kernel[{block, k.name}];
    if (!r.tried) {
        r.tried = true;
        std::string path;
        if (jit_object(m, block, k.name, &path)) {
            std::ifstream f(path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            r.ok = !bytes.empty() && d->load_module(bytes.data(), bytes.size(), &r.handle) == Error::Success;
        }
        bind(r);
        if (!r.ok && std::getenv("CPUSIM_VERBOSE"))
            std::fprintf(stderr, "cpusim: JIT unavailable, using the generic kernels\n");
    }
    return lookup(r);
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
    // cpu=<signature>: the module was compiled for "this machine" (cucc's
    // default, --mcpu=native); it runs only on the same kind of CPU.
    if (auto pos = line.find("cpu="); pos != std::string::npos)
        m->foreign = std::strtoull(line.c_str() + pos + 4, nullptr, 16) != cpusim::cpu_signature();
    // warp=N: the warp size of the module; every module of a program must agree.
    if (auto pos = line.find("warp="); pos != std::string::npos) {
        int w = std::atoi(line.c_str() + pos + 5), expected = 0;
        if (!program_warp_size.compare_exchange_strong(expected, w) && expected != w) {
            std::fprintf(stderr, "cpusim: kernels compiled with warps of %d and of %d in one program\n", expected, w);
            return false;
        }
    }
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
            else if (opt == "indep") k.lane_independent = true;
            else if (opt == "barriers") k.barriers = true;
        m->kernels[k.name] = k;
    }
    return true;
}

}  // namespace cpusim::rt
#pragma GCC visibility pop

extern "C" {

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


}  // extern "C"
