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

### AVX2 Tick — Block Loop (from `living_silicon.cpp:279`)

```cpp
for (std::uint32_t b = 0; b < kBlocks; ++b) {
    const std::uint32_t i = b * kBlockLanes;
    if ((lane.block_flags[b] & 0x1U) == 0U) {
        // Boundary block — scalar fallback
        for (std::size_t j = i; j < i + kBlockLanes; ++j)
            advance_node_scalar(lane, j, delta, coupling, blend, decay, tick_energy);
        continue;
    }

    // Load center block (16 int16 = 256-bit AVX2)
    const __m256i m = _mm256_load_si256(lane.scratch_mag.data() + i);
    __m256i p = _mm256_load_si256(lane.scratch_ph.data() + i);
    const __m256i omega = _mm256_load_si256(lane.omega.data() + i);
    p = _mm256_add_epi16(p, _mm256_add_epi16(delta_v, omega));

    // Load this block's living topology
    const std::int16_t offsets[]{lane.off0[b], lane.off1[b], lane.off2[b], lane.off3[b]};
    const std::int16_t weights[]{lane.w0[b], lane.w1[b], lane.w2[b], lane.w3[b]};

    // Gather-accumulate over neighbor slots
    for (std::size_t slot = 0; slot < kMaxNeighbors; ++slot) {
        if (weights[slot] == 0) continue;
        const auto base = static_cast<std::ptrdiff_t>(i) + offsets[slot];
        const __m256i weight_v = _mm256_set1_epi32(weights[slot]);
        const __m256i nm = _mm256_loadu_si256(lane.scratch_mag.data() + base);
        // ... int32 widen → multiply → accumulate → pack back to int16 ...
    }
    // Blend, couple, decay, store back to lane.mag/ph
}
```

### Topology Evolution (from `living_silicon.cpp:111`)

```cpp
void mutate_topology_blocks(Lane2048& lane, std::uint32_t& rng) {
    for (std::size_t b = 0; b < kBlocks; ++b) {
        if (!should_mutate_block(lane, b, rng))
            continue;
        lane.off0[b] = mutate_offset(lane.off0[b], rng);
        lane.off1[b] = mutate_offset(lane.off1[b], rng);
        lane.w0[b]   = mutate_weight(lane.w0[b], rng);
        lane.w1[b]   = mutate_weight(lane.w1[b], rng);
        if ((xorshift32(rng) & 0x3U) == 0U) {
            lane.off2[b] = mutate_offset(lane.off2[b], rng);
            lane.w2[b]   = mutate_weight(lane.w2[b], rng);
        }
        if ((xorshift32(rng) & 0x7U) == 0U) {
            lane.off3[b] = mutate_offset(lane.off3[b], rng);
            lane.w3[b]   = mutate_weight(lane.w3[b], rng);
        }
        recompute_block_flags(lane, b);
    }
}
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

### Initialization (from `living_silicon.cpp:74`)

```cpp
void initialize_ring_topology(Lane2048& lane) {
    for (std::size_t b = 0; b < kBlocks; ++b) {
        lane.off0[b] = -1;      // left neighbor
        lane.off1[b] =  1;      // right neighbor
        lane.off2[b] =  0;      // unused slot
        lane.off3[b] =  0;      // unused slot
        lane.w0[b]   = 16384;   // Q15 half-weight
        lane.w1[b]   = 16384;   // Q15 half-weight
        lane.w2[b]   = 0;
        lane.w3[b]   = 0;
        recompute_block_flags(lane, b);
    }
}
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
