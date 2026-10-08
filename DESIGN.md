# CudaSimulator: design

Documento de design consolidado a partir da fase de planejamento (2026-10-08). Serve de ponto de partida para quem continuar a implementação: o que já foi decidido, por quê, e em que ordem construir.

Ainda não há nome para o projeto nem para as ferramentas. Nomes usados aqui (`cpusim`, `cpu-smi`) são provisórios.

---

## 1. Objetivo

Pegar um programa CUDA que funciona numa GPU NVIDIA e, **sem alterar o código-fonte**, compilá-lo para rodar na CPU, com cada warp executado nas lanes de registradores AVX-512.

- **Compilação AOT**, não interpretação. O kernel escrito "por thread" vira código nativo que executa um warp inteiro em SIMD. Escalonamento, barreiras e controle de blocos ficam no código gerado mais um runtime nosso.
- **Driver residente**, que se comporta como o driver da NVIDIA: reserva cores e memória ("VRAM"), recebe módulos de kernel, parâmetros e launches.
- **Nada da NVIDIA é usado nem distribuído**: headers CUDA, runtime e driver são nossos. A versão de CUDA simulada é a **13**.
- **AVX-512 é obrigatório.** Sem fallback por enquanto, mas o código deve detectar a ausência e emitir uma mensagem clara.

Primeiro marco: uma **prova de conceito básica que funcione de ponta a ponta** (seção 9).

---

## 2. Estado atual

- `samples/`: 10 programas CUDA autovalidados (veja `samples/README.md`). Rodam na GPU real e comparam com referência em CPU. **Todos passam na RTX 5060.** Eles são a suíte de testes diferenciais: a saída do nosso compilador precisa continuar imprimindo `PASS`.
- Nada do simulador foi implementado ainda.

### Ambiente da máquina de desenvolvimento

| Item | Valor |
|---|---|
| CPU | AMD Ryzen 9 8940HX (Zen 4), 16 cores / 32 threads, 2 CCDs |
| AVX-512 | F, DQ, BW, VL, CD, VNNI, BF16, IFMA, VBMI, VBMI2, BITALG, VPOPCNTDQ. **Sem FP16.** Unidades de 256 bits ("double-pumped") |
| Caches | L1d 32 KB/core, L2 1 MB/core, L3 2 x 32 MB |
| GPU de referência | NVIDIA GeForce RTX 5060 Laptop (compute capability 12.0) |
| CUDA SDK | 13.0 em `/usr/local/cuda-13.0` (usado **só** para rodar os samples na GPU real) |
| Toolchain | GCC 13.3, CMake 3.28. **Clang e LLVM ainda não instalados** (precisam de `apt install clang llvm-dev`, com sudo) |

---

## 3. Visão geral

```text
                 programa.cu (sem alterações)
                          │
                ┌─────────┴──────────┐
                │ nosso compilador   │  Clang (só como parser de C++)
                │                    │  + headers CUDA 13 nossos
                └─────────┬──────────┘
          host code       │        device code (IR por thread)
              │           │              │
              │           │     nossos passes LLVM: vetorização SIMT,
              │           │     máscaras, barreiras, memória
              │           │              │
              ▼           │              ▼
      binário da app ◄────┴──── "fatbin": .so nativo + bitcode por thread
              │
              │ linka com nosso libcudart
              ▼
     ┌──────────────────────── interface de device ───────────────────────┐
     │  backend "dentro do processo"   │   backend "daemon" (driver)      │
     │  pool de workers local          │   socket + memória do driver     │
     └─────────────────────────────────┴──────────────────────────────────┘
                                         │
                                  cpu-smi (futuro)
```

---

## 4. Modelo de execução

### 4.1 Warp

- **Warp = 32 threads semanticamente** (`warpSize == 32`). Muito código real assume isso.
- Fisicamente: **2 registradores zmm por valor** (2 x 16 lanes de 32 bits) e máscara `__mmask32`. O compilador gera IR com `<32 x T>` / `<32 x i1>` e o backend do LLVM legaliza.
- O warp anda em lockstep. Não há *independent thread scheduling* (Volta+): código que depende de lanes do mesmo warp progredirem independentemente (spinlock entre lanes) não é suportado. Limitação documentada.

### 4.2 Pares de warps (K) e SMT

- Cada thread de hardware executa **um par de warps intercalados (K=2)** para preencher a pipeline (no Zen 4, ~1,5 warps de operações independentes saturam as unidades de FP). K=2 também deixa o caminho aberto para emular wave64 da AMD (mesmo layout físico, semântica diferente nos intrínsecos de warp).
- **Na POC usar K=1.** K=2 entra como otimização logo depois.
- A ideia de dividir os 32 zmm em 2 bancos de software com "troca rápida" foi descartada: o SMT faz isso em hardware, com o arquivo de registradores completo para cada thread.

### 4.3 Blocos e cores

- **Um bloco por core físico.** As **duas threads SMT do core** executam pares de warps **do mesmo bloco**.
- Shared memory e frames dos warps ficam num buffer local do bloco (quente no L1/L2 do core).
- Distribuição de blocos entre cores: contador atômico global de `blockIdx` (um `fetch_add` por bloco).
- Escalonamento dentro do bloco:
  ```text
  par = next_pair[fase].fetch_add(1)
  se par < num_pares: executa o par até barreira ou fim
  senão: espera na barreira de 2 participantes
  ```
  O contador só cresce, então não há estado "em execução". O reset entre fases usa *sense reversal* ou contadores alternados por fase.
- **Quem espera na barreira não pode girar livre**: rouba ciclos da thread irmã, que está terminando o último par. Usar `pause` no mínimo; o ideal é `MONITORX/MWAITX` (modo usuário na AMD).
- **Blocos pequenos** (menos de 2 pares): um bloco por thread HW em vez de um por core.
- Fixar as threads nos irmãos SMT lendo `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` (não assumir numeração).
- Sem escalonamento ciente de CCD (decidido).

### 4.4 Barreiras (`__syncthreads`)

- O kernel vira uma **máquina de estados**: cada trecho entre barreiras é um estado. Barreira dentro de loop funciona naturalmente (o loop volta a um estado anterior).
- Na barreira o par **salva apenas os valores vivos** num frame por warp (no buffer do bloco) e devolve o controle ao escalonador.
- Warps que terminaram (`return`) são marcados `done`, não contam para a barreira (semântica CUDA) e não são mais escalonados. Par com um warp `done` executa com a máscara dele zerada.
- Os dois warps de um par sempre estão no mesmo estado (saem e chegam às mesmas barreiras).

### 4.5 Divergência

- `if` divergente: executa os dois lados com máscara (linearização). **Atalho de branch uniforme obrigatório**: se a máscara de um lado é toda zero, pula o bloco inteiro.
- Loops divergentes: rodam enquanto houver lane ativa; a máscara perde lanes conforme saem.
- Máscara efetiva de cada instrução = máscara ativa do warp, combinada com predicados.
- Semântica padrão de escrita: **merge masking** (lane inativa preserva o valor). Zero masking só onde fizer sentido.

### 4.6 Futuro (fora da POC)

- **Memory yield**: em cadeias de loads dependentes (perseguição de ponteiros), emitir prefetch e passar a vez para outro par. Mesma máquina de estados das barreiras. Desligado por padrão; só vale onde medirmos ganho.
- Coarsening/intercalação além de K=2, se medições mostrarem ganho.

---

## 5. Compilador

### 5.1 Frontend

- **Clang como parser de C++**, em modo CUDA, com `-nocudainc -nocudalib`. Um `.cu` é C++ completo (templates, `constexpr`, biblioteca padrão no host): um parser próprio está fora de questão.
- **Headers CUDA nossos** (`cuda_runtime.h`, `vector_types.h`, funções matemáticas, intrínsecos de warp, `CUDART_VERSION 13000`). A versão de CUDA simulada é a que esses headers declaram; o suporte do Clang ao SDK da NVIDIA deixa de importar.
- Host: `<<<>>>` vira chamadas de launch que o nosso `libcudart` implementa; registro de kernels via o mecanismo que o Clang emite.
- Device: IR LLVM **por thread**, com `threadIdx` etc. como intrínsecos. Fazemos o retarget para x86-64 (address spaces global/shared/constant/local).

### 5.2 Pipeline de passes (nossos)

1. **Classificação de valores** (seção 5.3)
2. **Linearização do CFG** com máscaras + atalho de branch uniforme
3. **Alargamento** para `<32 x T>` (warp)
4. **Memória**: escolhe load/store contíguo, strided ou gather/scatter
5. **Máquina de estados** nas barreiras (frames com valores vivos)
6. (depois) Intercalação K=2, memory yield
7. O3 padrão + codegen `-mcpu=znver4`

Não depender do autovetorizador do LLVM: ele não vetoriza uma função de kernel inteira. O IR gerado é explícito (`llvm.masked.load/store/gather/scatter`, intrínsecos `llvm.x86.avx512.*` onde for preciso controle exato). Alocação de registradores e scheduling ficam com o LLVM. Usar `llvm-mca` e inspeção de assembly para conferir.

Referências do mesmo tipo de transformação: ISPC (SPMD em SIMD), Region Vectorizer / Whole-Function Vectorization (Moll, Hack, Karrenberg).

### 5.3 Classificação de valores

| Classe | Exemplo | Onde fica |
|---|---|---|
| Constante | `2.0f`, `TILE` | imediato |
| Uniforme | `blockIdx.x`, ponteiros de parâmetro, contador de loop uniforme | registrador escalar (GPR) ou broadcast |
| Afim | `blockIdx.x*blockDim.x + threadIdx.x` | `base uniforme + stride * lane` |
| Variável | `indices[i]`, `x * y` com `x` variável | zmm |

A classe afim decide o acesso à memória: stride 1 = load/store contíguo mascarado; stride constante = load + permute ou gather; variável = gather/scatter. "Uniforme" existe em níveis (grid, bloco, warp); para gerar código importa o nível warp.

### 5.4 Memória

- Gather/scatter AVX-512 usam índices int32 + base escalar. Ponteiro por lane de 64 bits exigiria `i64gather` (8 lanes por instrução, 4 por warp): **reconhecer "base uniforme + offset int32"** é otimização obrigatória.
- Gather/scatter mascarados não acessam lanes inativas (sem fault).
- Conflito em scatter (duas lanes no mesmo endereço): AVX-512 escreve em ordem crescente de lane (a maior vence). Em CUDA é indefinido, então qualquer ordem é válida.
- Gather é lento no Zen 4 e scatter mais ainda. Contíguo sempre que possível.
- Não existe gather de byte: loads de `uint8_t` variáveis viram gather de dword + extração.

### 5.5 Arrays locais por thread

Arrays privados de cada thread (`float acc[4]`) ficam na pilha do warp em **layout SoA** (`acc[i][lane]`), como a *local memory* da NVIDIA (intercalada por thread). Índice uniforme = load contíguo; índice variável = gather dentro desse layout. É uma decisão interna, invisível ao usuário.

### 5.6 Funções, ponto flutuante e matemática

- **Funções de device: inline de tudo** no começo. Depois, versões vetorizadas com a máscara como parâmetro. Recursão e ponteiros de função: muito depois.
- **Ponto flutuante**: padrão igual ao nvcc (**FMA ligado**; o Zen 4 tem FMA), com flag equivalente a `-fmad=false`. Meta: dentro do ULP documentado pela NVIDIA, não bit a bit idêntico. Os samples usam `-fmad=false` para comparação exata.
- **Biblioteca matemática vetorial própria** (sem SLEEF), escrita em IR/intrínsecos para ser inlined no kernel (não `.s` separado, que exigiria convenção de chamada e salvar zmm). Validar contra a RTX 5060.
- `half`/`bf16` e qualquer tipo sem suporte nativo: emular (converter para float). Fica para depois.

### 5.7 Intrínsecos de warp

Primeiro: `__shfl_sync` e variantes (permutes entre as 2 metades, `vpermt2ps`), `__ballot_sync` (a própria máscara), `__any_sync`/`__all_sync`, `__activemask`, `__popc`, `__clz`, `__syncwarp` (no-op). `__match_any_sync` via `vpconflictd`. **Cooperative groups por último.**

### 5.8 Atômicas

- **Padrão / fallback**: loop escalar só sobre as lanes ativas (`tzcnt` + `blsr` na máscara), um `lock` por lane:

  | CUDA | x86 |
  |---|---|
  | `atomicAdd` int, resultado ignorado | `lock add` |
  | `atomicAdd` int, resultado usado | `lock xadd` |
  | `atomicAdd` float/double | loop `lock cmpxchg` |
  | `atomicCAS` | `lock cmpxchg` |
  | `atomicExch` | `xchg` |
  | `atomicAnd/Or/Xor` sem resultado | `lock and/or/xor` |
  | com resultado, `atomicMin/Max` | loop `lock cmpxchg` |

- **Otimização planejada**: endereço uniforme (global ou shared) = redução no SIMD + **um** `lock xadd` + soma de prefixo exclusiva para os valores de retorno.
- **Shared memory também precisa de `lock`**: as duas threads SMT executam warps do mesmo bloco ao mesmo tempo. Sem `lock` só quando o bloco roda numa única thread HW (blocos pequenos); o compilador pode gerar as duas variantes e o runtime escolhe no launch.
- `vpconflictd` para casos tipo histograma, só se medirmos ganho.
- As atômicas de CUDA são *relaxed*; o `lock` do x86 é mais forte, o que é correto.

### 5.9 Ordem de memória

x86 é TSO. O compilador não pode reordenar acessos através de barreiras e de `__threadfence`; `volatile` respeitado.

### 5.10 `printf` e `assert` no device

Prioridade alta (sem isso fica difícil depurar o simulador). Buffer por bloco, descarregado no `cudaDeviceSynchronize`, como na GPU.

### 5.11 Recursos não suportados

Sempre **erro de compilação**, nunca resultado errado em silêncio. Isso inclui recursos do sm_120 que reportarmos (seção 6.6) mas não suportarmos.

---

## 6. Driver e runtime

### 6.1 Dois backends, uma interface

Uma **interface de device** (estilo CUDA driver API) com duas implementações, **construídas e testadas juntas**:

- **Dentro do processo**: workers no próprio processo da aplicação.
- **Daemon**: processo residente que reserva cores e VRAM; a aplicação fala com ele.

Operações mínimas da interface: abrir device / consultar propriedades, alocar e liberar, memcpy (H2D, D2H, D2D), memset, carregar módulo, lançar kernel, sincronizar, consultar erro.

Todo teste roda nos dois backends. Se passa num e falha no outro, o problema está no transporte, não no compilador.

### 6.2 VRAM

- **Exclusiva do driver** (decidido): ponteiros de device só valem dentro do driver, como numa GPU dedicada sem memória unificada. Todo dado passa por `cudaMemcpy`.
- **Alocador com pools**, preparado para depois:
  ```text
  Device
  ├── pool "private"  (vram_private)  só mapeado no driver       ← POC
  └── pool "shared"   (vram_shared)   memfd no MESMO endereço     ← futuro
                                      virtual no driver e no cliente
  ```
- Backing com hugepages (2 MB ou 1 GB; ajuda o TLB nos gathers). `MFD_HUGETLB` exige `vm.nr_hugepages`; alternativa THP com `madvise`.

### 6.3 Transporte (daemon)

- **Controle**: socket Unix (`SOCK_SEQPACKET`) para requisições. Descritores (módulos, memfd) passam por `SCM_RIGHTS`. Depois: anel em memória compartilhada para launch de baixa latência.
- **memcpy**: o driver usa `process_vm_readv/writev` direto na memória do cliente (uma cópia, papel do DMA). O Ubuntu usa `ptrace_scope=1`, então o cliente chama `prctl(PR_SET_PTRACER, pid_do_driver)` na inicialização. **Fallback**: anel de staging em memória compartilhada (duas cópias).
- Depois: threads de cópia dedicadas no driver (*copy engines*) para `cudaMemcpyAsync` sobrepor cópia e kernel.
- **Um driver por usuário**: o driver executa código enviado pelo cliente.
- Kernel com acesso inválido pode derrubar o driver (não há validação barata por gather). Tratar o sinal como erro tipo `cudaErrorIllegalAddress` quando possível.

### 6.4 Módulos ("fatbin") e ABI de kernel

- A aplicação embute, por módulo:
  - **`.so` nativo** (usado agora). Enviado ao driver como memfd e carregado com `dlopen("/proc/self/fd/N")`.
  - **Bitcode LLVM do kernel por thread** (papel do PTX). Embutido desde já, usado no futuro para recompilar no launch com especialização (`blockDim`, `gridDim` como constantes) e cache em disco.
- **Contrato estável** entre código gerado e runtime (definir cedo, é o que mais custa mudar):
  - **Versão da ABI**: o runtime recusa módulo incompatível.
  - **Metadados por kernel**: nome, tamanho/alinhamento de cada parâmetro, shared memory estática, tamanho do frame por warp (valores vivos), número de estados.
  - **Ponto de entrada**: executa **um par de warps** a partir do estado salvo até a próxima barreira ou o fim, e retorna o motivo (`BARRIER` / `EXIT`). Recebe um contexto com `blockIdx`, dimensões, base da shared memory, frames do par, máscaras e parâmetros. O loop de escalonamento (seção 4.3) fica no runtime.
- Antes do compilador existir, **2 ou 3 kernels escritos à mão nesse formato** servem para testar runtime e driver.

### 6.5 Configuração

Arquivo provisório `~/.config/cpusim/device.toml`:

```toml
[device]
cores        = 12        # "SMs" = cores físicos reservados (com os 2 threads SMT)
vram_private = "16GiB"
vram_shared  = "0GiB"    # > 0 habilita o pool compartilhado (futuro)
hugepages    = "2MiB"    # 2MiB, 1GiB ou none
```

### 6.6 Propriedades do device

- Base: **RTX 5060 (compute capability 12.0)**, configurável no driver. Para obter os valores, rodar uma vez um programa de consulta com o CUDA real e salvar os campos de `cudaGetDeviceProperties` (limites de threads por bloco, shared memory, etc.).
- `multiProcessorCount` = cores configurados; `totalGlobalMem` = `vram_private`; `warpSize` = 32.
- Com `vram_shared = 0` (agora): `integrated = 0`, `canMapHostMemory = 0`, `managedMemory = 0`, `concurrentManagedAccess = 0`. `cudaMallocManaged` e `cudaHostAlloc` mapeado retornam `cudaErrorNotSupported`.

### 6.7 Runtime API

Primeira superfície (a mesma que os samples usam): `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaGetErrorString`, launch (`<<<>>>`). Streams: tudo serializado no stream padrão no começo.

---

## 7. Memória unificada (futuro)

Quando `vram_shared > 0`, o device se apresenta como **integrado**, como uma GPU integrada com memória dedicada mais memória compartilhada:

- Pool compartilhado = memfd mapeado no **mesmo endereço virtual** no driver e no cliente. O cliente reserva cedo com `mmap(MAP_FIXED_NOREPLACE)`; se colidir, esse cliente cai para modo só-cópia.
- `cudaMallocManaged`, `cudaHostAlloc` = zero-copy nessa região. `cudaMemPrefetchAsync`/`cudaMemAdvise` viram no-op.
- Coerência vem de graça (é a mesma CPU); sinais de launch e sync com semântica release/acquire.
- Globais `__device__`/`__managed__` colocadas pelo compilador na região compartilhada.
- Opcional: substituir `malloc`/`new` do cliente para o heap inteiro ficar compartilhado (estilo HMM). Quebra programas que usam `fork()` esperando copy-on-write, por isso é flag.
- Pilha e globais comuns do host nunca são compartilháveis entre processos. Só o modo dentro do processo dá SVM completo.

---

## 8. Ordem de construção

1. **Base do driver**: interface de device, configuração, alocação (pools), comunicação (socket, memcpy via `process_vm_*`), carregamento de módulo e launch. Backends daemon e dentro do processo juntos.
2. **Runtime de execução**: pool de workers fixados nos irmãos SMT, escalonamento de blocos e pares, barreira de 2 participantes com `MWAITX`, frames. Testado com os kernels escritos à mão (seção 6.4).
3. **`libcudart` mínimo** sobre a interface de device.
4. **Compilador**, desenvolvido testando junto com o driver: frontend com headers nossos, classificação, linearização, alargamento, memória, máquina de estados.
5. Samples da POC passando nos dois backends.

---

## 9. Prova de conceito (primeiro marco)

Fatia vertical mínima, ponta a ponta:

| Camada | Entra na POC | Fica para depois |
|---|---|---|
| Frontend | Clang + headers nossos mínimos | API completa |
| Compilador | uniforme/afim/variável, `if` e loops com máscara, warp 32, load/store/gather/scatter, `__syncthreads`, K=1 | K=2, memory yield, intrínsecos de warp, atômicas, math |
| Runtime e driver | `cudaMalloc/Memcpy/Memset/Free`, launch, sync, nos dois backends; pares de warps em SMT | streams, `cpu-smi`, config completa |
| Samples-alvo | `matmul`, `transpose`, `heat2d`, `scan`, `mandelbrot`, `bitonic_sort` | `nbody` (math), `reduction` e `spmv` (shuffle), `histogram` (atômicas) |

**Critério de aceite**: os 6 samples, sem alteração no fonte, compilados pelo nosso compilador, imprimem `PASS` nos dois backends.

---

## 10. Backlog (depois da POC, aproximadamente nesta ordem)

- K=2 (intercalação de pares)
- Intrínsecos de warp, atômicas com agregação, biblioteca matemática própria → `nbody`, `reduction`, `spmv`, `histogram`
- `printf`/`assert` no device (prioridade alta, pode entrar antes se a depuração exigir)
- Shared memory dinâmica (`extern __shared__`)
- Profiler: tempo por kernel e **eficiência SIMD** (lanes ativas / lanes totais)
- `cpu-smi` (ocupação, VRAM privada e compartilhada, processos)
- Streams com execução concorrente; vários processos no driver (fila única primeiro, depois divisão de cores ou revezamento)
- Modo determinístico (ordem fixa de blocos)
- Pool de VRAM compartilhada (seção 7)
- JIT com especialização no launch a partir do bitcode embutido
- Memory yield com prefetch
- Emulação de `half`/`bf16` e demais tipos sem suporte
- Funções de device não inlined, recursão, ponteiros de função
- Cooperative groups (por último)

## 11. Fora de escopo

Thrust, CUB, cuBLAS e demais bibliotecas; PTX inline (`asm()`); Tensor Cores; paralelismo dinâmico; texturas; nomes de projeto e ferramentas (por enquanto).

## 12. Riscos e pontos a verificar primeiro

- **Clang com `-nocudainc -nocudalib` e headers próprios**: confirmar logo no início compilando `samples/matmul.cu` (separar host e device, obter IR do device por thread).
- Desempenho de gather/scatter no Zen 4 (scatter é microcodificado): medir cedo, define o quanto a classificação afim importa.
- Custo da troca de estado nas barreiras e da barreira entre irmãos SMT.
- `process_vm_readv` com `ptrace_scope` e `PR_SET_PTRACER`.

## 13. Referências

ISPC; Region Vectorizer / Whole-Function Vectorization (Karrenberg, Moll, Hack); GPU Ocelot (PTX para x86); MCUDA; CuPBoP; COX (warp functions em CPU); AdaptiveCpp (barreiras em CPU por compilador); POCL; AMAC / interleaving com corrotinas (para memory yield); ZLUDA (reimplementação da API CUDA).
