# GA HNN nDxnD Engine Archive

Complete evolution archive of the **GA + HNN nDxnD** engine family — 28 versions spanning 5 architectural eras, from Father/Son hierarchical GA to Fractal+LUT+IFS compression engines.

## Purpose

This archive enables Codex and other agents to perform deep code review and C++ portability assessment for integrating nDxnD features into the **Living Silicon** engine (V26 Antigravity runtime).

## Structure

```
versions/           # All 28 .txt versions + 3 .zip archives (001-039)
docs/               # Audit documents and analysis
reference/
  living_silicon/   # Current Living Silicon C++ engine (target for integration)
  pde_engines/      # Legacy PDE/tensor engines (tensor_nd.hpp, pde_base.hpp, torsion2d)
```

## Evolution Timeline

| Era | Versions | Architecture |
|---|---|---|
| 1. KO Father/Son | 001-014 | Hierarchical dual-process GA, DEAP, PPO, AI voting |
| 2. GA HHN nD | 015-018 | Holographic nD FFT, complex64 holograms, LUT quantization |
| 3. GA HNN nDxnD | 019-025 | Generic Conv{N}d/BatchNorm{N}d, self-evolving fitness |
| 4. nDxnD nbit | 026-029 | Per-layer quantization, RandomForest surrogate, niche penalty |
| 5. nbit FRAKTAL | 030-039 | Fractal compression, CUDA IFS kernel, sparse/dense LUT |

## Top 5 Versions (Initial Audit)

1. **v024** (58KB) — AMP float16 + int8/4bit, full nD dimension tracking
2. **v036** (36KB) — Fractal entropy bonus, PSNR-compression tradeoff
3. **v033** (21KB) — Tested: 485:1 compression, 35.2 dB PSNR, 0.93 fitness
4. **v039** (22KB) — nD IFS kernel, tournament selection, diversity scoring
5. **v018** (77KB) — Production-grade async framework (Pydantic, structlog)

## Living Silicon Integration Target

The Living Silicon engine is a high-performance C++20 GNN core:
- 1D ring of 2048 coupled oscillators
- AVX2 SIMD, `alignas(64)`, int16 fixed-point
- 8-lane GA evolution with PI homeostatic control
- Hot path budget: ≤500ns per tick
- No heap allocation in tick loop

See `reference/living_silicon/` for current implementation.

## Related Repository

Main project: [V26_test_V6_integruotas](https://github.com/Karpavicius82/V26_test_V6_integruotas)
