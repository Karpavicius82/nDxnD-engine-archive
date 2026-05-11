# nDxnD Codex Audit

Scope: audited the 28 text files `versions/012*.txt` through `versions/039*.txt`, plus `docs/nd_engine_inventory.md` and `reference/living_silicon/{living_silicon.hpp,living_silicon.cpp}`.

Note: `versions/` contains more than 28 `.txt` files. The exact 28-file range is `012-039`; earlier `001-011` and the alternate `006` files were not treated as part of this requested 28-file audit.

## Executive Findings

- The inventory's era model is broadly correct: `012-017` are holographic/FFT/LUT sketches, `018-025` are production-style PyTorch HNN builders, `026-029` introduce per-layer n-bit architecture evolution, and `030-039` add fractal/LUT/IFS compression ideas.
- The Top 5 needs one correction in wording: `v033` is not validated as a real benchmark. The file itself says the report is a simulation. Keep it in the Top 5 as a strong concept/prototype, but do not cite the 485:1 result as measured evidence.
- `v028` really has duplicate `HNNBuilder` definitions at lines `172` and `1065`. The second one is not just a duplicate; it is an interrupted/contaminated paste and should not be used as the authoritative implementation.
- The highest C++ portability candidates for `reference/living_silicon` are GA control ideas, not PyTorch model builders: adaptive mutation, multi-channel fitness, niche/diversity penalty, unit-sphere parameter normalization, and lightweight LUT/fractal block descriptors.
- The low-portability ideas are dynamic `Conv{N}d`, PyTorch quantization, Numba CUDA kernels, sklearn surrogate/KMeans, and large sparse LUT tensors. These conflict with Living Silicon's fixed-size, int16, cache-resident, AVX2-friendly design.

## Per-Version Audit

| Version | Python classes identified | GA operators | NN architecture | Quantization | Fitness formula | Unique innovations / notes |
|---|---|---|---|---|---|---|
| 012 | No complete class declarations in extracted code; discusses `Genome._init_hnn()` and `HNN_Core` | Conceptual genome mutation of architecture/controller params | Direct tensor-weight HNN, avoids `nn.Linear`/`nn.Sequential` in favor of low-level tensors | LUT generated from `quantization_bits` | Not fully implemented | Transitional design note: lower-level PyTorch tensors for HNN weights and LUT quantization. |
| 013 | `HNN_Core`, `VRAMMonitor`, `HolographicMemory` | Not a complete GA loop | 2D complex hologram, phase encoding, CUDA FTH-style kernel | Not central | Not present | Early CUDA holographic memory / Fourier-transform holography sketch. |
| 014 | `Config`, `ResourceManager`, `Genome`, `HNN_Core`, `GeneticArchitect`, `BinanceStreamManager`, `AIAgentsManager` | Tournament/rank selection, genome crossover, mutation of HNN architecture, weights, quantization, pipeline | nD hologram tensor with `torch.fft.fftn`, flatten/embed/decode workflow | 4/8-bit LUT phase quantization | `normalized_mse * hnn_accuracy_weight + normalized_vram_efficiency * vram_weight + normalized_compression_ratio * compression_weight + ai_bonus - compute_penalty` | First substantial nD holographic GA with AI-agent weights and self-mutating fitness weights. |
| 015 | Same main set as `014` | Same family: tournament/rank, crossover, mutation | nD FFT holographic core, linear embedding only after placeholder removal | 4/8-bit LUT phase quantization | Similar MSE/VRAM/compression/AI formula with stronger truncation/compute penalty intent | Cleans up placeholders and emphasizes padding/truncation penalty and smarter VRAM reduction. |
| 016 | No complete executable class set; design-response file | Conceptual crossover improvements | nD hologram embedding discussion | Not implemented | Not implemented | Design note for dimensional projection/interpolation crossover between incompatible hologram shapes. |
| 017 | `ResourceManager`, `ComplexQuantizer`, `Genome` | Mutation/crossover rates as genome parameters | Complex nD hologram state, no full HNN builder | Quantizes real/imag complex state into integer levels; `state_quant_bits` | Accuracy/VRAM/compression/complexity weights are initialized | Introduces explicit complex-state quantizer and GA-controlled hologram dimensions. |
| 018 | `SystemConfig`, `HNNConfig`, `GAConfig`, `AIAgentsConfig`, `FullConfig`, `Genome`, `HNNBuilder`, `HNN_Core`, `GeneticArchitect`, managers | Configured mutation/crossover, encoder-layer crossover, quantization and fitness-weight mutation | Linear encoder/decoder via `nn.Sequential`; production-style config/logging/async | 16-bit `.half()`, dynamic int8 quantization; 4-bit logged as not implemented in core code | `accuracy_score*hnn_accuracy + vram_score*vram_efficiency + compression_score*compression_ratio_impact + ai_bonus*ai_guidance_bonus` | Best framework hygiene: Pydantic config, YAML shape, structlog, async evaluation. Less nD than later builders. |
| 019 | Same framework classes as `018` | Deepcopy architecture crossover, mutation of quantization/latent/layers | Mostly linear HNN builder despite nDxnD title; begins generic shape support | 4/8/16-bit paths, int8 dynamic, 4-bit bitsandbytes conversion attempt | Same four-channel weighted score as `018` | First nDxnD branch; compression calculation accounts for latent bit width. |
| 020 | Adds `HNNLayer` | Min-length crossover, mutation, config validation | Typed layer definitions; still mainly linear builder in code | int8 dynamic and recursive 4-bit conversion attempt | Same weighted accuracy/VRAM/compression/AI formula | Cleaner typed layer model and recursive quantization mechanics. |
| 021 | Same as `020` | Same | Linear typed HNN builder | Same | Same | Stability/bug-fix variant of `020`. |
| 022 | Same as `020` | Same, uses `copy.copy` in crossover | Linear typed HNN builder | Same | Same | Stability variant; no major new architecture feature found. |
| 023 | Same as `020` | Adds structural mutation probability / max layers per inventory; min-length crossover | Linear typed HNN builder with stricter layer logging | Same | Same | Stronger structural mutation boundaries. |
| 024 | Same as `020` | Same | Mature linear HNN builder with AMP/training-time float16 path and better shape handling | quantization bits affect latent byte calculation; int8 and 4-bit post-training paths are present in code family | Same weighted score; compression uses `latent_dim_flat * quantization_bits / 8` | Strongest non-fractal production-style nDxnD candidate. Inventory ranking as #1 is defensible if framework maturity matters. |
| 025 | Same as `020` | Same | Simplified typed linear HNN | Same | Same | Simplified/stability branch; less distinctive than `024`. |
| 026 | `SystemConfig`, `HNNLayer`, `Genome`, `HNN_Core`, `HNNBuilder`, `GeneticArchitect` | Parent selection, architecture/layer-bit crossover, mutation | 1D conv/linear/dropout/batchnorm/pool autoencoder style | Per-layer `layer_bits`; QAT bits; large-bit penalty | Accuracy, VRAM, compression, and formula type (`linear`, `exponential`, `multiplicative`, `quadratic`) | First clear n-bit per-layer architecture evolution. |
| 027 | Same as `026` | Selection, crossover of architecture and `layer_bits`, mutation with multiple attempts | Better 1D shape tracking, `_adjust_output_size` | Per-layer bits with length repair after crossover | Same multi-channel n-bit score | More robust 1D conv/pool dimension handling than `026`. |
| 028 | `SystemConfig`, `HNNLayer`, `Genome`, `HNN_Core`, duplicate `HNNBuilder`, `GeneticArchitect` | Two partially duplicated GA sections; selection, crossover of architecture and bits, mutation | First builder: generic `getattr(nn, f"Conv{N}d")` / `BatchNorm{N}d` / pool, shape stack, final adjust; second builder: 1D-3D class arrays | Per-layer bits, QAT toggle, custom quantize, large-bit penalty | Accuracy from validation loss, VRAM score, compression, large-bit penalty, formula type | Important but dirty. Contains the best generic nD builder idea, plus a broken duplicated second builder. |
| 029 | `SystemConfig`, `LayerDefinition`, `Genome`, `HNN_Core`, `HNNBuilder`, `GeneticArchitect` | Parent selection, one-child crossover, mutation | Simplified generic nD conv/linear builder using `Conv{N}d`/BN/pool | Per-layer bits; supported QAT bits `[8]` | Accuracy/VRAM/compression formula with large-bit penalty | Cleaner post-`028` simplification. |
| 030 | Same as `029` | RandomForest surrogate, KMeans init, niche radius, stagnation/adaptive mutation, crossover/mutation | Generic nD conv/linear builder | Per-layer n-bit, QAT bits `[8]` | Accuracy/VRAM/compression with large-bit penalty and formula type | First fractal-era GA controls: surrogate, KMeans, niche, stagnation. Fractal naming is still light in actual model. |
| 031 | Same core classes as `030` | RandomForest, KMeans, niche, surrogate, adaptive mutation | Fractal encode/decode concept with LUT/IFS | Per-layer bits; LUT size/type evolves | Adds PSNR into fitness | Mostly analysis plus code; explicitly critiques LUT/fractal coupling as theoretical. |
| 032 | `FractalParams`, `TrainingParams`, `FractalHNN`, `FractalProcessor`, `Genome`, `GeneticOptimizer`, `FractalAutoencoderSystem`, plus later core classes | Genetic optimizer crossover/mutation; RandomForest/KMeans appears again | Separate fractal autoencoder system, CUDA encode/decode kernels | LUT enum, sparse LUT, quantization in later code | PSNR, compression, VRAM penalty | Valuable because it self-identifies fake/simplified fractal kernels and warns PSNR could be artifact. |
| 033 | `SystemConfig`, `LayerDefinition`, `Genome`, `HNN_Core`, `HNNBuilder`, `GeneticArchitect` | RandomForest, KMeans, niche/surrogate/stagnation params, crossover/mutation | Fractal encode/decode replaces regular HNN internals; LUT sparse/dense | Supported QAT bits `[8,4,2]`, per-layer bits, custom quantize | `accuracy*w + vram_score*w + compression_ratio*w + psnr/100*w + complexity_cost*negative_w` | Strong concept report with 485:1/35.2 dB claim, but the file says the report is simulated. |
| 034 | Same as `033` | RandomForest, KMeans with `n_init=10`, crossover/mutation | Similar fractal/LUT HNN | `[8,4,2]`, per-layer bits | Same family | Claims 0.95/0.97 in top report text, but also simulation-style; less important than `033`/`036`. |
| 035 | Same as `033` | KMeans, stagnation, crossover/mutation; RandomForest removed | Simplified fractal/LUT HNN | `[8,4,2]`, per-layer bits | Same family, no fractal entropy bonus | Smaller simplified branch. |
| 036 | Same as `033` | RandomForest, KMeans `n_init=10`, mutation/crossover of layer bits and fractal params | Fractal/LUT HNN with CPU Numba fallback and LUT matmul shape correction | `[8,4,2]`, per-layer bits, custom quantize | `accuracy*w + vram*w + compression*w + psnr*w + complexity_penalty + (1-fractal_entropy)*fractal_entropy_bonus`; dynamic PSNR threshold | Strongest fractal implementation candidate: entropy bonus, compression-PSNR tradeoff, CPU fallback, better LUT shape. |
| 037 | Adds `ResidualBlock` | KMeans/stagnation/crossover/mutation | Fractal/LUT plus 2D residual block | `[8,4,2]`, smaller LUT range to avoid OOM | PSNR-heavy weighted score | Interesting residual CNN addition, but less portable and less general. |
| 038 | Same as `033` | RandomForest/KMeans, tournament parent selection, crossover/mutation | Fractal/LUT with CUDA IFS-like kernel | Supported QAT bits `[8]`, per-layer bits | Includes diversity score in fitness | Compact tournament/diversity variant. |
| 039 | Same as `038` | RandomForest/KMeans, tournament selection, crossover/mutation; mutation/crossover rates become genome params | nD IFS kernel, fractal encode/decode, serialized actuator buffer | Supported QAT bits `[8]`, custom quantize, sparse/dense LUT | `accuracy*w + vram*w + compression*w + psnr*w + nD_complexity*w + diversity*0.1` | Unique nD IFS kernel and actuator serialization. High idea value, medium code quality. |

## Top 5 Ranking Validation

Inventory Top 5:

1. `v024`
2. `v036`
3. `v033`
4. `v039`
5. `v018`

Validation:

- `v024` as #1 is defensible for a production-oriented PyTorch HNN branch. It is not the most radical architecture, but it has the cleanest mature framework among non-fractal versions and a coherent fitness/compression calculation.
- `v036` as #2 is defensible and arguably the best fractal-era implementation. It adds the most concrete improvements: CPU fallback, LUT shape correction, fractal entropy bonus, dynamic PSNR-compression threshold, and KMeans fixes.
- `v033` should be demoted in evidence strength but can remain in Top 5 for concept clarity. Its report says "simuliacija", so the 485:1 result is not validated experimental evidence.
- `v039` belongs in Top 5 for uniqueness: nD IFS kernel, tournament selection, diversity term, and serialized actuator state. Code quality is weaker than `v036`.
- `v018` belongs in Top 5 for framework quality: Pydantic/structlog/YAML/async patterns are better production scaffolding than most later files.

Recommended evidence-aware ranking:

| Rank | Version | Reason |
|---|---|---|
| 1 | `v024` | Best mature non-fractal HNN framework and quantized compression accounting. |
| 2 | `v036` | Best fractal-era implementation, with entropy and PSNR tradeoff mechanisms. |
| 3 | `v018` | Best production scaffolding; useful for C++ config/test discipline even if architecture is simpler. |
| 4 | `v039` | Most unique nD IFS/diversity/actuator ideas. |
| 5 | `v033` | Best narrative target and simulated performance report; not a measured proof. |

If ranking by conceptual novelty instead of implementation maturity, keep the inventory order except annotate `v033` as "simulated".

## v028 Duplicate `HNNBuilder`

Confirmed.

- First `HNNBuilder`: `versions/028  GA HNN nDxnD nbit.txt:172`.
- Second `HNNBuilder`: `versions/028  GA HNN nDxnD nbit.txt:1065`.

Assessment:

- The first builder is the useful one. It uses dynamic `getattr(nn, f"Conv{N}d")`, `BatchNorm{N}d`, `MaxPool{N}d`, tracks `current_shape`, keeps a `spatial_shapes_stack`, and final-adjusts output back to `initial_input_shape`.
- The second builder is not a clean improved replacement. It switches to explicit 1D/2D/3D class arrays and introduces `conv_dim`, which reduces generic nD scope. It also becomes contaminated around line `1219` with prose text inserted into the method body, so it is syntactically broken as code.
- Recommendation: if porting ideas from `v028`, extract only the first builder's shape algebra and generic class lookup. Do not preserve the duplicated second builder.

## v033 Compression 485:1 Validation

Not validated as a real benchmark.

Evidence from `v033`:

- Lines `8-9` call the test report a simulation.
- Lines `15-18` state fitness `0.93`, PSNR `35.2 dB`, compression `485:1`, VRAM `1850 MB`.
- Lines `35-37` describe generation 4 and generation 8 behavior.
- Lines `354-358` compute compression as `input_size_bits / latent_size_bits`, where `latent_size_bits = len(encoded.flatten()) * random.choice(genome.layer_bits['encoder'])`.

Problems:

- The file's own report is simulated, not a reproduced run artifact.
- The compression denominator ignores model parameters, LUT memory, actuator state, and fractal metadata. A sparse LUT of `750000` entries is described in the report and can dominate real storage.
- `random.choice(genome.layer_bits['encoder'])` makes the compression score nondeterministic for the same encoded tensor.
- `genome.accuracy` is initialized but not meaningfully trained as a real task accuracy in this branch, so the final fitness is not a robust validation metric.

Conclusion: 485:1 is a target or simulated result. It is plausible only as latent-tensor compression under a narrow formula, not as end-to-end system compression.

## C++ Portability vs Living Silicon

Living Silicon reference traits:

- Fixed-size C++ engine: `kNodes=2048`, `kThreads=8`.
- Per-lane `Genome` uses atomic int16/int32 fields: `delta`, `coupling`, `threshold`, `blend`, `decay`, `inject_rate`, `omega_width`, `ei_balance`, `fitness`, etc.
- Hot path is cache-resident arrays of `mag`, `ph`, `omega`, `ei`, and `nd`.
- SIMD AVX2 path exists for phase/magnitude updates.
- Fitness is multi-channel and local: population term, energy delta/homeostasis, coherence band-pass, drive response.
- GA crossover is conservative: every few epochs, blend best lane into worst stagnating lane.

High-portability ideas:

| Idea | Source | C++ fit | Suggested Living Silicon mapping |
|---|---|---|---|
| Adaptive mutation on stagnation | `030-039`, already partially in LS | High | Extend existing `stagnation_epochs` logic with genome-level mutation strength or field-specific boost. |
| Niche/diversity penalty | `026-039` | High | Compute cheap genome-distance across 8 lanes using int16 fields; penalize crossover donor selection when lanes are too similar. |
| Self-evolving fitness weights | `014+` | Medium-high | Add small fixed-point weights for energy/coherence/population/drive, mutate slowly, normalize by sum or power-of-two shifts. |
| Unit-sphere parameter normalization | `030+` | Medium | For C++, use fixed-point vector normalization only in slow epoch path, not in per-tick AVX2 path. |
| KMeans-style clustered init | `026+` | Medium | Replace sklearn KMeans with deterministic seeded presets or tiny integer k-medoids over genome fields. |
| Fractal block descriptors | `033+` | Medium | Store small block schedules and reuse counters; avoid tensor LUTs. Could map to `nd[]`/membrane summaries. |
| PSNR-compression tradeoff | `036` | Medium | Port the control idea as quality-vs-latency or coherence-vs-energy threshold adaptation; not literal PSNR unless image data exists. |
| Father/son hierarchy | inventory earlier era | Medium | Map to lane roles: mentor/probe lanes rather than flat best/worst crossover. |

Low-portability ideas:

| Idea | Why not directly portable |
|---|---|
| PyTorch `Conv{N}d`, `BatchNorm{N}d`, `MaxPool{N}d` builders | Dynamic allocation, tensor framework dependency, and arbitrary shapes conflict with fixed `kNodes` arrays. |
| Numba CUDA IFS kernels | Living Silicon is currently C++/AVX2 with scalar fallback, not CUDA-first. |
| `RandomForestRegressor` surrogate | sklearn is too heavy; a tiny hand-coded linear/ridge surrogate or table of recent genome deltas would fit better. |
| Large sparse/dense LUT tensors up to 1M | Breaks cache budget and contradicts fixed small-state engine design. |
| Runtime Python quantization and bitsandbytes | Not relevant to int16 fixed-point C++ engine. |
| Simulated PSNR image metrics | No direct image reconstruction loop in Living Silicon reference. |

Recommended C++ import order:

1. Add diversity/niche-aware donor selection to `maybe_crossover()`.
2. Add fixed-point self-evolving fitness weights to `Genome` and `GenomeSnapshot`.
3. Add a tiny slow-path surrogate score based on recent genome snapshots and observed fitness deltas.
4. Add role-aware lanes (mentor/probe/explorer) only after the first three changes are measurable.
5. Treat fractal/LUT/IFS as a separate experimental branch, not a hot-path change.

## Final Assessment

The inventory is useful but overstates the evidentiary value of `v033`. The most implementation-worthy material is split between `v024`/`v018` for framework discipline, `v036`/`v039` for fractal-era concepts, and `v028` for generic nD shape algebra after removing duplicated broken code. For Living Silicon, the right path is to port GA control mechanisms and fixed-point summaries, not the Python neural-network builders themselves.
