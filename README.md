# GA HNN nDxnD Engine Archive

Complete evolution archive of the **GA + HNN nDxnD** engine family — 28 versions spanning 5 architectural eras, from Father/Son hierarchical GA to the **Lane2048 Living Topology** engine.

## Purpose

This archive enables deep code review and C++ portability assessment for the **Living Silicon** engine (V26 Antigravity runtime). It documents the full evolution from Python GA+HNN prototypes to a production bare-metal C++20 AVX2 engine with self-evolving topology.

## Structure

```
versions/                          # All 28 .txt versions + 3 .zip archives (001-039)
docs/                              # Audit documents and analysis
reference/
  living_silicon/                  # Lane2048 Living Topology engine (CURRENT)
    living_silicon.hpp             # Lane2048 struct, Genome, Engine API
    living_silicon.cpp             # AVX2 block-loop tick, topology mutation, GA
  pde_engines/                     # Legacy PDE/tensor engines
    tensor_nd.hpp                  # nD tensor operations
    pde_base.hpp                   # PDE solver base
    ks_ch_torsion2d_ac.hpp         # Kuramoto-Sivashinsky / Cahn-Hilliard
```

## Architecture Evolution (This Repo)

### Era 1-5: Python Prototypes (versions/001-039)

| Era | Versions | Architecture |
|---|---|---|
| 1. KO Father/Son | 001-014 | Hierarchical dual-process GA, DEAP, PPO, AI voting |
| 2. GA HHN nD | 015-018 | Holographic nD FFT, complex64 holograms, LUT quantization |
| 3. GA HNN nDxnD | 019-025 | Generic Conv{N}d/BatchNorm{N}d, self-evolving fitness |
| 4. nDxnD nbit | 026-029 | Per-layer quantization, RandomForest surrogate, niche penalty |
| 5. nbit FRAKTAL | 030-039 | Fractal compression, CUDA IFS kernel, sparse/dense LUT |

### Era 6: C++ Living Silicon (reference/)

Three commits document the architectural evolution from scalar nD to living topology:

| Commit | Architecture | Key Insight |
|---|---|---|
| `d7686bd` — nD Scalar | Per-node coordinate math every tick | Working but slow — division/modulo per node |
| `9c4712a` — NdVectorKernel | Precomputed offset/weight plan + AVX2 executor | Fast but still compiler/executor split |
| **`9d4f3f0` — Lane2048** 🏆 | **Living topology — offsets ARE the field** | No plan, no coordinates, GA evolves offsets directly |

## Lane2048: Living Topology Engine

The current engine (`reference/living_silicon/`) implements a paradigm where **topology is not metadata — it IS the living field**.

### Core Structure

```cpp
struct Lane2048 {
    // Living field state (2048 nodes)
    int16_t  mag[2048];          // magnitude (activation)
    uint16_t ph[2048];           // phase (oscillation angle)
    int16_t  omega[2048];        // Kuramoto natural frequency
    int8_t   ei[2048];           // excitatory/inhibitory identity

    // Topology as living tissue (128 SIMD blocks × 4 neighbors)
    int16_t  off0..off3[128];    // neighbor offsets per block
    int16_t  w0..w3[128];        // Q15 weights per block
    uint16_t block_flags[128];   // AVX2 safety + activity bits

    // Per-node evolved parameters
    int16_t  g_coupling[2048];   // local coupling strength
    int16_t  g_blend[2048];      // local diffusion rate
    int16_t  g_decay[2048];      // local energy decay

    // Telemetry
    int16_t  stress[2048];       // local stress signal
    int16_t  novelty[2048];      // local novelty metric
};
```

### Key Design Principles

1. **nD is not runtime math** — No coordinate calculations, no `rank`, no `dims` at tick time
2. **nD is evolving offset geometry** — GA mutates `off0..off3` and `w0..w3` directly
3. **One universal AVX2 loop** — Same block-loop for any topology (1D ring, 2D grid, 4D hypercube, or irregular)
4. **No build/plan step** — Topology lives inside Lane2048, not in an external `NdVectorKernel`
5. **Per-node gene fields** — `g_coupling`, `g_blend`, `g_decay` are arrays, not global scalars

### Tick Operator (Pseudocode)

```
for each of 128 blocks (b = 0..127):
    i = b * 16                           // 16-node SIMD block
    load center mag[i..i+15], ph[i..i+15]
    load neighbors via off0[b]..off3[b]  // precomputed offsets
    weighted accumulate with w0[b]..w3[b] // Q15 multiply
    blend, couple, decay
    store back
```

### Topology Evolution

```
// GA doesn't set rank/dims. It mutates offsets directly:
epoch:
    for hot blocks:
        off0[b] = mutate_offset(off0[b])   // could become -1, -64, -512...
        w1[b]   = mutate_weight(w1[b])      // strengthen/weaken connection
        recompute block_flags[b]            // update AVX2 safety

// System DISCOVERS its own geometry through evolution
```

### Performance

| Metric | Value |
|---|---|
| Nodes | 2,048 |
| SIMD blocks | 128 (16 nodes each) |
| Neighbors per block | 4 slots |
| Population lanes | 8 |
| Tick speed | 26,456 ticks/s (benchmark) |
| Per-tick latency | ~38 μs (epoch-level) |
| Mutations/s | 385 |
| Instruction set | AVX2 (256-bit, int16) |
| Compiler | MSVC C++20 `/arch:AVX2` |

### Initialization

System starts as a 1D ring:
```
off0 = -1  (left neighbor)
off1 = +1  (right neighbor)
off2 = 0   (unused)
off3 = 0   (unused)
w0 = 16384 (Q15 half)
w1 = 16384 (Q15 half)
w2 = 0
w3 = 0
```

GA can evolve this into 2D, 3D, 4D, or irregular topologies over epochs.

## Top 5 Historical Versions

1. **v024** (58KB) — AMP float16 + int8/4bit, full nD dimension tracking
2. **v036** (36KB) — Fractal entropy bonus, PSNR-compression tradeoff
3. **v033** (21KB) — Tested: 485:1 compression, 35.2 dB PSNR, 0.93 fitness
4. **v039** (22KB) — nD IFS kernel, tournament selection, diversity scoring
5. **v018** (77KB) — Production-grade async framework (Pydantic, structlog)

## Related Repositories

- **Production V26**: [V26_test_V6_integruotas](https://github.com/Karpavicius82/V26_test_V6_integruotas)
  - Lane2048 reference copy at `reference/living_silicon_lane2048/`
  - Production engine at `src/control/living_silicon.*` (stable, 40/40 tests)

## Build

```powershell
# Compile-check (standalone, no CMake needed)
cl /nologo /std:c++20 /EHsc /arch:AVX2 /c reference\living_silicon\living_silicon.cpp /I reference\living_silicon
```

## License

Proprietary — Antigravity Core V26 © 2026
