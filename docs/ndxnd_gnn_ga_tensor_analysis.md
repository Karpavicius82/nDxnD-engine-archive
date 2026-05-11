# nDxnD GNN+GA Tensor Engine Deep Code Analysis

Scope: `versions/012*.txt` through `versions/039*.txt`, `reference/pde_engines/tensor_nd.hpp`, and `reference/living_silicon/{living_silicon.cpp,living_silicon.hpp}`.

Goal: choose the best nDxnD foundation for a Living Silicon GNN+GA engine using fixed-point, cache-aligned int16 arrays and AVX2 SIMD.

## Executive Recommendation

The best foundation is a hybrid:

- Graph/topology: use `v028` generic nD spatial shape algebra, but not its duplicated/broken second `HNNBuilder`.
- Message passing: keep Living Silicon's current 1D ring hot path and generalize it to nD strided-neighbor stencils. This is cleaner and more SIMD-compatible than the Python PyTorch builders.
- Tensor ops: port the `v028` `Conv{N}d`/`MaxPool{N}d`/`AvgPool{N}d` concepts into fixed nD stencil kernels, and selectively reuse the float32 operation taxonomy from `reference/pde_engines/tensor_nd.hpp`.
- GA evolution: combine Living Silicon's conservative lane graft with `v036` adaptive compression/quality mutation and `v039` tournament/diversity ideas.
- Fractal/IFS: keep `v039` nD block split and IFS as a slow-path descriptor/evaluation feature, not in the per-node hot path.

Best single source version: `v028` for tensor-native nD topology and operation vocabulary.

Best combination: `v028` + `v036` + `v039` + current Living Silicon.

## Task 1 - Version-to-GNN+GA Mapping

Legend for SIMD fit: Yes = direct int16/Q-shift port; Partial = usable concept with rewrite; No = too float/PyTorch/FFT-heavy for the hot path.

| Version | Graph structure | Message passing rule | Node update function | Edge-weight evolution | Aggregation | Tensor operations | int16 AVX2 fit |
|---|---|---|---|---|---|---|---|
| 012 | Conceptual low-level HNN tensor, no clean executable graph. Nodes are tensor cells. | Implicit holographic tensor interaction. | LUT/quantized HNN transform concept. | Conceptual genome mutation of HNN/controller params. | Not explicit. | Low-level tensors, LUT quantization. | Partial: LUTs can port; graph/message rules are underspecified. |
| 013 | 2D holographic memory plane. Nodes are complex pixels/cells. | Fourier-domain global propagation, not local neighbor passing. | Phase encode/decode through hologram. | None in complete GA sense. | Global Fourier superposition. | Complex 2D hologram, FFT/FTH-style CUDA. | Partial/No: FFT/global complex ops are not a good int16 hot path. |
| 014 | Evolvable nD hologram tensor `hologram_dims`; graph is dense/global nD field. | Global interference via nD FFT, not edge-local GNN. | Embed input, FFT, phase quantize, inverse FFT. | GA mutates hologram dims, weights/state, quantization, pipeline; no explicit edge weights. | Superposition/FFT global reduction. | `torch.fft.fftn`, LUT phase quantization, pad/truncate reshape. | Partial: topology evolution useful; FFT and complex tensors are not hot-path friendly. |
| 015 | nD hologram tensor with tighter linear embedding. | Same as `014`: global nD holographic propagation. | Linear embed, nD FFT, quantization, reconstruction. | Similar hologram/quantization mutation. | Global Fourier aggregation. | nD FFT, padding/truncation penalty. | Partial: control logic portable, tensor ops not ideal. |
| 016 | Design note for incompatible nD hologram shapes. | Conceptual dimensional projection/interpolation between holograms. | Not complete. | Crossover improvements for shape mismatch. | Projection/interpolation. | Shape projection concept. | Partial: valuable for slow GA crossover only. |
| 017 | Complex nD hologram state. | Implicit global holographic interaction. | Complex quantization of real/imag state. | Genome controls hologram dimensions and mutation/crossover rates. | Global complex-state combination. | Complex-state quantizer, nD state. | Partial: quantizer maps to fixed point; global complex model does not. |
| 018 | Production HHN framework, mostly flattened linear latent graph. | Encoder/decoder feed-forward, not neighbor message passing. | `nn.Sequential` linear encode/decode plus quantization. | Crossover/mutation of layers, quantization, latent dims, fitness weights. | Linear layer weighted sums. | Dynamic int8, half precision, linear layers. | Partial: framework/GA discipline useful; tensors are not lattice-native. |
| 019 | Early nDxnD, mostly linear in code with generic shape ambitions. | Feed-forward layer communication. | Linear/shape-aware HNN builder. | Architecture, quantization, latent mutation. | Weighted sums. | Linear layers, shape accounting. | Partial. |
| 020 | Typed layer graph, still mostly linear builder. | Feed-forward layer communication. | Typed HNN layers. | Architecture/quantization mutation. | Weighted sums. | Linear, recursive quantization. | Partial. |
| 021 | Stability variant of `020`. | Same as `020`. | Same as `020`. | Same as `020`. | Same as `020`. | Same. | Partial. |
| 022 | Stability variant of `020`, shallow-copy crossover. | Same. | Same. | Same. | Same. | Same. | Partial. |
| 023 | Typed linear HNN with structural mutation limits. | Same feed-forward layer graph. | Linear layers with stricter validation. | Adds structural mutation bounds. | Weighted sums. | Linear, quantization. | Partial. |
| 024 | Mature non-fractal HNN, primarily linear latent graph. | Feed-forward encoder/decoder. | Linear encode/decode, optional quantized inference. | Layer crossover/mutation, latent dim mutation, self-normalized fitness weights. | Weighted sums. | Linear, int8 dynamic quant, 4-bit bitsandbytes, float16. | Partial: high framework quality but not native lattice message passing. |
| 025 | Simplified `024` branch. | Feed-forward. | Linear HNN. | Same family. | Weighted sums. | Linear/quantization. | Partial. |
| 026 | 1D conv/linear/pool autoencoder style. | Local convolution where conv layers exist. | Conv/linear/dropout/batchnorm/pool. | Per-layer bit evolution and architecture mutation. | Conv sum, pool max/mean. | 1D conv/pool, n-bit QAT concept. | Yes/Partial: 1D conv/pool maps well, but code is PyTorch. |
| 027 | More robust 1D conv/pool shape tracking. | Explicit 1D convolution/pooling neighbor communication. | Conv/pool/adjust output size. | Per-layer bits with repair after crossover. | Conv sum, pool. | 1D conv, pool, output adjust. | Yes/Partial: good for current 1D ring extension. |
| 028 | Generic nD tensor graph: `Conv{N}d`, `BatchNorm{N}d`, `MaxPool{N}d`, `AvgPool{N}d`; nodes are tensor cells, edges are convolution neighborhoods. | Explicit neighborhood stencil through nD convolution/pooling. | Conv/transpose-conv/pool/linear/activation with shape stack. | GA mutates layer types, kernel, stride, bit width, latent dim, fitness weights. No explicit per-edge weights, but conv kernels act as edge weights. | Conv weighted sum; max/avg pool; linear reductions. | Generic nD conv, pool, unflatten/upsample, quantization. | Yes/Partial: best tensor fit; needs fixed-shape C++ rewrite. |
| 029 | Cleaner simplified generic nD conv/linear builder. | Generic nD conv where present. | Conv/linear builder. | Per-layer bits and architecture mutation. | Conv sum/pool. | Generic `Conv{N}d`, per-layer bits. | Yes/Partial: cleaner than `028`, less rich. |
| 030 | Generic nD architecture plus GA meta-graph over genome params. | Conv/linear message passing where model has conv; surrogate predicts fitness. | HNN builder plus surrogate/niche controls. | Adaptive mutation, niche penalty, KMeans init, surrogate usage; topology not deeply coupled. | Conv/weighted sum; population distance reduction. | Conv/linear, RandomForest, KMeans. | Partial: GA controls useful; sklearn not portable. |
| 031 | Fractal/LUT concept graph, blocks as nodes. | Block transform through LUT/fractal encode. | Fractal encode/decode concept. | Evolves LUT/fractal params. | PSNR/mean block reduction. | LUT, fractal blocks, PSNR. | Partial: block descriptors can port; large LUT bad for L1. |
| 032 | Separate fractal autoencoder/block graph. | CUDA fractal block encoding. | Fractal encode/decode kernels. | Genetic optimizer mutates fractal params. | Block means/reconstruction score. | CUDA kernels, LUT, PSNR. | Partial/No for hot path. |
| 033 | Fractal/LUT HNN; blocks are graph nodes. | LUT transform per block; mean latent. | Fractal encode/decode. | Evolves layer bits, LUT, fractal params. | Mean over block transforms, PSNR/compression fitness. | Sparse/dense LUT, matmul, PSNR. | Partial: simulated report; block mean and LUT can port if cache-bounded. |
| 034 | Similar to `033`. | Same. | Same. | Same. | Same. | Same. | Partial. |
| 035 | Simplified fractal/LUT. | Same block transform style. | Simplified fractal encode/decode. | KMeans/stagnation/crossover/mutation. | Mean/fitness reductions. | LUT/fractal blocks. | Partial. |
| 036 | Best fractal implementation candidate: blocks, LUT transforms, entropy/PSNR feedback. | Block-to-LUT transform and latent mean. | Matmul block vector by LUT matrix, quantize latent, decode. | Adaptive mutation of latent dim, layer bits, fractal blocks, IFS coeffs, LUT, PSNR threshold. | Mean over transformed blocks; histogram entropy; PSNR. | Block split, LUT matmul, mean, entropy, PSNR. | Partial: slow-path GA/fitness excellent; LUT matmul must be tiny/cache-bounded for AVX2. |
| 037 | Fractal/LUT plus residual 2D CNN block. | Residual 2D conv message passing. | Residual block plus fractal encode/decode. | Evolves LUT/fractal/bit params. | Conv sum, residual add, PSNR. | 2D residual conv, LUT. | Partial: residual add/2D conv portable; less nD-general. |
| 038 | Fractal/LUT with tournament/diversity. | Block LUT transform and IFS-like kernel. | Fractal encode/decode. | Tournament selection, diversity score, mutation/crossover. | Mean, diversity std. | LUT, CUDA IFS concept. | Partial: tournament/diversity useful. |
| 039 | Most explicit nD fractal graph: nD blocks from Cartesian product and nD IFS recurrence. | nD block split, transform per block, IFS neighbor-dimensional recurrence `z[d]` uses adjacent dimensions. | Fractal encode/decode with actuator serialization and CPU fallback. | Crossover averages IFS coeffs/LUT size; mutation changes blocks, coeffs, LUT, PSNR threshold. | Mean over block latents, diversity std, nD complexity term. | nD block split, IFS kernel, LUT transform, mean. | Partial/Yes for slow path: nD topology unique; IFS recurrence can be int16 AVX2 but not current hot path. |

## Task 2 - Scoring Matrix

0-10 criteria. `Total` is unweighted sum out of 80.

| Version | Graph generality | Message clarity | Tensor native ops | Fixed-point compatibility | GA integration | Cache friendliness | Code quality | Unique value | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 012 | 3 | 2 | 4 | 5 | 3 | 6 | 2 | 4 | 29 |
| 013 | 3 | 2 | 6 | 2 | 1 | 2 | 3 | 5 | 24 |
| 014 | 8 | 3 | 7 | 3 | 7 | 2 | 5 | 7 | 42 |
| 015 | 7 | 3 | 7 | 3 | 6 | 3 | 5 | 5 | 39 |
| 016 | 5 | 2 | 3 | 4 | 5 | 7 | 2 | 6 | 34 |
| 017 | 7 | 3 | 5 | 5 | 6 | 4 | 4 | 6 | 40 |
| 018 | 4 | 4 | 5 | 5 | 7 | 6 | 8 | 6 | 45 |
| 019 | 5 | 4 | 5 | 5 | 6 | 6 | 6 | 5 | 42 |
| 020 | 5 | 4 | 5 | 5 | 6 | 6 | 6 | 5 | 42 |
| 021 | 5 | 4 | 5 | 5 | 6 | 6 | 6 | 4 | 41 |
| 022 | 5 | 4 | 5 | 5 | 6 | 6 | 6 | 4 | 41 |
| 023 | 5 | 4 | 5 | 5 | 7 | 6 | 6 | 5 | 43 |
| 024 | 5 | 4 | 5 | 6 | 7 | 6 | 8 | 6 | 47 |
| 025 | 4 | 4 | 4 | 6 | 6 | 6 | 6 | 3 | 39 |
| 026 | 4 | 7 | 7 | 7 | 7 | 7 | 5 | 6 | 50 |
| 027 | 4 | 8 | 7 | 7 | 7 | 7 | 6 | 6 | 52 |
| 028 | 9 | 8 | 9 | 7 | 8 | 7 | 4 | 9 | 61 |
| 029 | 8 | 7 | 8 | 7 | 7 | 7 | 6 | 7 | 57 |
| 030 | 7 | 6 | 7 | 6 | 9 | 5 | 5 | 8 | 53 |
| 031 | 7 | 5 | 7 | 5 | 8 | 4 | 4 | 7 | 47 |
| 032 | 7 | 5 | 7 | 4 | 8 | 3 | 4 | 7 | 45 |
| 033 | 7 | 6 | 8 | 5 | 8 | 3 | 5 | 8 | 50 |
| 034 | 7 | 6 | 8 | 5 | 8 | 3 | 5 | 7 | 49 |
| 035 | 6 | 5 | 6 | 5 | 7 | 4 | 5 | 5 | 43 |
| 036 | 7 | 7 | 8 | 6 | 9 | 4 | 7 | 9 | 57 |
| 037 | 6 | 7 | 8 | 6 | 8 | 4 | 5 | 7 | 51 |
| 038 | 7 | 7 | 7 | 6 | 8 | 4 | 5 | 7 | 51 |
| 039 | 9 | 8 | 7 | 6 | 8 | 5 | 5 | 10 | 58 |

Ranking by GNN+GA tensor suitability:

1. `v028` - best direct nD conv/pool tensor vocabulary and topology.
2. `v039` - best unique nD topology/block/IFS idea.
3. `v029` - cleaner generic nD builder, less rich than `v028`.
4. `v036` - best adaptive fractal/GA quality logic.
5. `v030` - useful GA meta-control: surrogate, niche, stagnation, unit-sphere mutation.
6. `v027` - practical 1D conv/pool stepping stone for the current ring.
7. `v026` - same family as `v027`, less robust.
8. `v024` - best production scaffold, but not lattice-native.

## Task 3 - Extracted Operations and C++ AVX2 Port Sketches

### 3.1 nD Convolution Operation

Best source: `versions/028  GA HNN nDxnD nbit.txt:182-230`.

```python
182:     def calculate_output_dim(input_dims: Tuple[int, ...], kernel_size: Tuple[int, ...], stride: Tuple[int, ...], padding: Tuple[int, ...], 
183:                              output_padding: Tuple[int, ...] = None, dilation: Tuple[int, ...] = None, transposed: bool = False) -> Tuple[int, ...]:
188:         output_dims = []
189:         for i, dim in enumerate(input_dims):
190:             if transposed:
191:                 out = (dim - 1) * stride[i] - 2 * padding[i] + dilation[i] * (kernel_size[i] - 1) + output_padding[i] + 1
192:             else:
193:                 out = (dim + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
194:             output_dims.append(max(1, out))
195:         return tuple(output_dims)
224:                 elif layer_def.type in ["conv", "conv_transpose"]:
225:                     conv_class = getattr(nn, f'ConvTranspose{N}d') if layer_def.type == "conv_transpose" else getattr(nn, f'Conv{N}d')
227:                     encoder_modules.append(conv_class(current_shape[0], layer_def.out_features, kernel_size_tuple, stride_tuple, padding_tuple, 
229:                     current_shape = (layer_def.out_features, *HNNBuilder.calculate_output_dim(current_shape[1:], kernel_size_tuple, stride_tuple, padding_tuple, 
```

C++ int16 AVX2 equivalent for Living Silicon should not implement arbitrary PyTorch dynamic conv. It should implement a fixed local stencil over flattened nD lattice coordinates.

```cpp
template <size_t N>
struct alignas(64) LaneTensor {
  std::array<int16_t, N> mag;
  std::array<uint16_t, N> ph;
  std::array<int16_t, N> omega;
  std::array<int8_t, N> ei;
};

// 1D ring conv: out[i] = (w0*x[i-1] + w1*x[i] + w2*x[i+1]) >> q
void conv1d_ring3_q15(const int16_t* x, int16_t* out, size_t n,
                      int16_t w_l, int16_t w_c, int16_t w_r, int q) {
  __m256i wl = _mm256_set1_epi16(w_l);
  __m256i wc = _mm256_set1_epi16(w_c);
  __m256i wr = _mm256_set1_epi16(w_r);
  for (size_t i = 0; i < n; i += 16) {
    __m256i xl = load_ring_shift_left(x, i, n);   // neighbor i-1
    __m256i xc = _mm256_load_si256((const __m256i*)(x + i));
    __m256i xr = load_ring_shift_right(x, i, n);  // neighbor i+1
    __m256i acc = madd_epi16_to_q15(xl, wl, xc, wc, xr, wr, q);
    _mm256_store_si256((__m256i*)(out + i), acc);
  }
}
```

Intrinsics: `_mm256_load_si256`, `_mm256_mullo_epi16` for small weights or `_mm256_madd_epi16` with 32-bit accumulation, `_mm256_add_epi32`, `_mm256_srai_epi32`, `_mm256_packs_epi32`, `_mm256_store_si256`.

Output: one aligned `int16_t[N]` tensor channel.

Estimated cycle budget: 2048-node 3-tap 1D stencil is roughly 0.8-2.5 us per lane on a modern AVX2 CPU if aligned and L1-resident; about 0.4-1.3 ns per node. A 2D/3D flattened stencil with more gathers/strides is more like 2-8 us per 2048 cells unless dimensions are tiled.

### 3.2 nD Pooling/Reduction Operation

Best source for generic pool: `versions/028  GA HNN nDxnD nbit.txt:239-247`.

```python
239:                 elif layer_def.type == "max_pool":
240:                     pool_class = getattr(nn, f'MaxPool{N}d')
241:                     encoder_modules.append(pool_class(kernel_size_tuple, stride_tuple, padding_tuple))
242:                     current_shape = (current_shape[0], *HNNBuilder.calculate_output_dim(current_shape[1:], kernel_size_tuple, stride_tuple, padding_tuple))
244:                 elif layer_def.type == "avg_pool":
245:                     pool_class = getattr(nn, f'AvgPool{N}d')
246:                     encoder_modules.append(pool_class(kernel_size_tuple, stride_tuple, padding_tuple))
247:                     current_shape = (current_shape[0], *HNNBuilder.calculate_output_dim(current_shape[1:], kernel_size_tuple, stride_tuple, padding_tuple))
```

Best source for block reduction: `versions/036  GA HNN nDxnD nbit FRAKTAL.txt:207-208`.

```python
207:         # Stack ir mean - dabar [latent_dim]
208:         latent = torch.stack(latent).mean(dim=0)
```

C++ AVX2 max/avg pool sketch:

```cpp
void avgpool2_q15_pairs(const int16_t* x, int16_t* y, size_t n) {
  const __m256i one = _mm256_set1_epi16(1);
  for (size_t i = 0, o = 0; i < n; i += 32, o += 16) {
    __m256i a = _mm256_load_si256((const __m256i*)(x + i));
    __m256i b = _mm256_load_si256((const __m256i*)(x + i + 16));
    // For adjacent-pair pool, use shuffle/unpack in real implementation.
    __m256i sum = _mm256_adds_epi16(a, b);
    __m256i avg = _mm256_srai_epi16(_mm256_adds_epi16(sum, one), 1);
    _mm256_store_si256((__m256i*)(y + o), avg);
  }
}

void maxpool_q15_vec(const int16_t* a, const int16_t* b, int16_t* y, size_t n) {
  for (size_t i = 0; i < n; i += 16) {
    __m256i va = _mm256_load_si256((const __m256i*)(a + i));
    __m256i vb = _mm256_load_si256((const __m256i*)(b + i));
    _mm256_store_si256((__m256i*)(y + i), _mm256_max_epi16(va, vb));
  }
}
```

Intrinsics: `_mm256_max_epi16`, `_mm256_adds_epi16`, `_mm256_srai_epi16`, plus shuffle/unpack for true window pooling.

Output: downsampled summary tensor or scalar block reductions.

Estimated cycle budget: pairwise pool for 2048 cells is about 0.2-0.8 us; nD window pool with boundary handling is 1-5 us depending on layout.

### 3.3 Fitness/Loss Calculation

Best source for fractal-aware fitness: `versions/036  GA HNN nDxnD nbit FRAKTAL.txt:418-425`, `466-492`.

```python
418:     def _compute_psnr(self, original: torch.Tensor, decoded: torch.Tensor) -> float:
419:         mse = nn.MSELoss()(original, decoded)
420:         if mse.item() < 1e-10:
421:             return 100.0
424:         max_val = original.max().item() if original.max().item() > 0 else 1.0
425:         return 20 * math.log10(max_val / math.sqrt(mse.item()))
472:         input_size_bits = math.prod(self.initial_input_shape) * 32 * genome.batch_size
473:         latent_size_bits = encoded.flatten().numel() * random.choice(genome.layer_bits['encoder'])
475:         genome.compression_ratio = input_size_bits / latent_size_bits if latent_size_bits > 0 else 0.0
477:         complexity_cost = len(genome.hnn_architecture['encoder_layers']) + len(genome.hnn_architecture['decoder_layers']) + genome.lut_size / int(1e6)
479:         vram_score = 1.0 - genome.vram_used_mb / genome.vram_limit_mb
484:         genome.fitness = (genome.accuracy * genome.fitness_weights["hnn_accuracy"] +
485:                           vram_score * genome.fitness_weights["vram_efficiency"] +
486:                           genome.compression_ratio * genome.fitness_weights["compression_ratio_impact"] +
487:                           (genome.psnr_score / 100) * genome.fitness_weights["psnr_score"] +
488:                           complexity_cost * genome.fitness_weights["complexity_penalty"] +
489:                           (1 - genome.fractal_entropy) * genome.fitness_weights["fractal_entropy_bonus"])
492:         dynamic_psnr_threshold = genome.psnr_threshold - (genome.compression_ratio / 100) * genome.compression_psnr_tradeoff
```

Living Silicon equivalent should be fixed-point and should replace PSNR with lattice reconstruction/coherence terms:

```cpp
struct FitnessWeightsQ8 {
  int16_t coherence = 64;
  int16_t energy = 64;
  int16_t population = 48;
  int16_t response = 48;
  int16_t compression = 16;
  int16_t diversity = 16;
};

int32_t fitness_q8(const Observation& o, const FitnessWeightsQ8& w,
                   int32_t compression_q8, int32_t diversity_q8) {
  int32_t coh8 = int32_t(o.coherence >> 8);
  int32_t coherence = 255 - abs(coh8 - 192);
  int32_t population = 256 - abs(o.nd_popcount - 128);
  int32_t energy = clamp_energy_homeostasis(o.energy);
  int32_t response = o.recent_drive > 16 ? clamp_response(o.energy) : 0;
  return (coherence*w.coherence + energy*w.energy + population*w.population +
          response*w.response + compression_q8*w.compression +
          diversity_q8*w.diversity) >> 8;
}
```

Intrinsics: mostly scalar slow-path at epoch boundary. For MSE/absolute-error reductions use `_mm256_subs_epi16`, `_mm256_abs_epi16`, `_mm256_madd_epi16`, horizontal sums.

Output: `int32_t fitness`.

Estimated cycle budget: epoch-only, 2048-cell reduction roughly 0.5-2 us; amortized per tick under 20 ns if evaluated every 128 ticks.

### 3.4 GA Crossover Mechanism

Best architecture crossover source: `versions/028  GA HNN nDxnD nbit.txt:489-530`.

```python
489:     def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
496:         for section_key in ["encoder_layers", "decoder_layers"]:
497:             min_len = min(len(parent1.hnn_architecture[section_key]), len(parent2.hnn_architecture[section_key]))
498:             crossover_point = random.randint(0, min_len)
499:             child1.hnn_architecture[section_key] = parent1.hnn_architecture[section_key][:crossover_point] + parent2.hnn_architecture[section_key][crossover_point:]
500:             child2.hnn_architecture[section_key] = parent2.hnn_architecture[section_key][:crossover_point] + parent1.hnn_architecture[section_key][crossover_point:]
506:             crossover_point_bits = random.randint(0, min_len_bits)
507:             child1.layer_bits[bits_section] = parent1_bits[:crossover_point_bits] + parent2_bits[crossover_point_bits:]
508:             child2.layer_bits[bits_section] = parent2_bits[:crossover_point_bits] + parent1_bits[crossover_point_bits:]
522:         for param in ["latent_dim_flat", "training_epochs", "learning_rate", "batch_size", "qat_enabled", "vram_limit_mb", "min_time_between_vram_clears_sec", "vram_warning_threshold", "large_bit_penalty_threshold", "large_bit_penalty_factor", "fitness_formula_type"]:
523:             setattr(child1, param, getattr(random.choice([parent1, parent2]), param))
526:         for key in parent1.fitness_weights:
527:             child1.fitness_weights[key] = random.choice([parent1.fitness_weights[key], parent2.fitness_weights[key]])
```

Best fractal crossover source: `versions/036  GA HNN nDxnD nbit FRAKTAL.txt:587-611` and `versions/039...:386-393`.

```python
587:     def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
589:         child.hnn_architecture['latent_dim_flat'] = random.choice([parent1.hnn_architecture['latent_dim_flat'], parent2.hnn_architecture['latent_dim_flat']])
592:         min_len_encoder = min(len(parent1.layer_bits['encoder']), len(parent2.layer_bits['encoder']))
593:         child.layer_bits['encoder'] = [random.choice(b) for b in zip(parent1.layer_bits['encoder'][:min_len_encoder], parent2.layer_bits['encoder'][:min_len_encoder])]
603:         child.compression_psnr_tradeoff = random.choice([parent1.compression_psnr_tradeoff, parent2.compression_psnr_tradeoff])
606:         child.fractal_blocks = random.choice([parent1.fractal_blocks, parent2.fractal_blocks])
607:         child.ifs_coeffs = random.choice([parent1.ifs_coeffs, parent2.ifs_coeffs])
608:         child.lut_type = random.choice([parent1.lut_type, parent2.lut_type])
609:         child.lut_size = random.choice([parent1.lut_size, parent2.lut_size])
```

C++ slow-path graft sketch:

```cpp
void crossover_worst_from_best(Genome& worst, const Genome& best, uint32_t rng) {
  // Conservative 25% blend for hot-path parameters.
  store_i16(worst.coupling, (3*load(worst.coupling) + load(best.coupling)) >> 2);
  store_i16(worst.blend,    (3*load(worst.blend)    + load(best.blend))    >> 2);
  if (rng & 1) store_i16(worst.delta, load(best.delta));

  // nD/tensor additions: inherit or blend slow-path descriptors.
  store_i16(worst.kernel_radius, choose_or_blend(worst.kernel_radius, best.kernel_radius, rng));
  store_i16(worst.nd_rank,       choose(worst.nd_rank, best.nd_rank, rng));
  store_i16(worst.block_size,    choose_or_blend(worst.block_size, best.block_size, rng));
}
```

Estimated cycle budget: slow epoch path under 200 ns for scalar genome fields.

### 3.5 GA Mutation Mechanism

Best source: `versions/036  GA HNN nDxnD nbit FRAKTAL.txt:563-585`.

```python
563:     def mutate(self, genome: Genome) -> None:
564:         mutation_strength_boost = 0.1
565:         if self.check_stagnation():
567:             mutation_strength_boost = 0.5
568:             genome.hnn_architecture['latent_dim_flat'] = max(16, genome.hnn_architecture['latent_dim_flat'] // 2)
569:             genome.layer_bits['encoder'] = [random.choice([2, 1]) for _ in genome.layer_bits['encoder']]
572:         if genome.psnr_score < genome.psnr_threshold:
573:             genome.psnr_threshold += random.uniform(0.1, 1.0)
577:         if genome.compression_ratio < 100:
578:             genome.hnn_architecture['latent_dim_flat'] = max(16, genome.hnn_architecture['latent_dim_flat'] // 2)
580:         genome.fractal_blocks = [max(2, min(32, b + random.choice([-2, 2]))) for b in genome.fractal_blocks]
581:         genome.ifs_coeffs += random.uniform(-mutation_strength_boost, mutation_strength_boost)
582:         genome.lut_size = max(1024, min(int(1e6), int(genome.lut_size * random.uniform(0.9, 1.1))))
584:         if random.random() < 0.1:
585:             genome.lut_type = random.choice(['sparse', 'dense'])
```

C++ fixed-point mutation:

```cpp
void mutate_nd_genome(Genome& g, ControllerState& c, uint32_t& rng) {
  int boost = c.stagnation_epochs > 8 ? 2 : 1;
  mutate_atomic_i16(g.coupling, 0, 200, rand_delta(rng, 8) * boost);
  mutate_atomic_i16(g.kernel_radius, 1, 4, rand_delta(rng, 1));
  mutate_atomic_i16(g.block_size, 2, 32, rand_delta(rng, 2));
  mutate_atomic_i16(g.fitness_w_coh, 1, 255, rand_delta(rng, 4));
  normalize_weight_sum_q8(g);
}
```

Estimated cycle budget: epoch-only, under 500 ns per mutated lane.

### 3.6 Topology Evolution / N_dims Changes

Best explicit topology mutation source: `versions/014 GA GA.txt:236-266`.

```python
236:     def mutate_hnn_architecture(self):
237:         if random.random() < 0.3:
238:             old_dims = list(self.components["hnn"]["hologram_dims"])
239:             new_dims = list(old_dims)
246:                 if mutation_type < 0.4:
247:                     idx_to_mutate = random.randint(0, len(new_dims) - 1)
248:                     change = random.choice([0.5, 1.5, 2.0])
249:                     new_dims[idx_to_mutate] = max(8, min(256, int(new_dims[idx_to_mutate] * change)))
250:                 elif mutation_type < 0.7 and len(new_dims) < 4:
251:                     new_dims.append(random.randint(8, 32))
252:                 elif len(new_dims) > 1:
253:                     new_dims.pop(random.randint(0, len(new_dims) - 1))
255:             new_dims = tuple(new_dims)
257:             max_total_elements = 256 * 256 * 2
259:             if np.prod(new_dims) > max_total_elements:
260:                 logging.warning(f"Attempted to create too large hologram: {new_dims}. Skipping mutation.")
263:             if new_dims != old_dims:
264:                 self.components["hnn"]["hologram_dims"] = new_dims
265:                 self.components["hnn"]["latent_dim"] = np.prod(new_dims)
266:                 self.components["hnn"]["hologram_state"] = self._initialize_hologram_state(new_dims)
```

Best nD runtime dimension query source: `versions/039...:34-35`.

```python
34:     def get_N_dims() -> int:
35:         return len(SystemConfig.initial_input_shape[1:])
```

C++ topology should not realloc hot arrays. Use a compile-time flat capacity with runtime shape metadata:

```cpp
struct NdShape {
  uint8_t rank;             // 1..4
  uint16_t dims[4];         // product <= kNodes
  uint16_t stride[4];       // row-major strides
};

bool mutate_shape(NdShape& s, uint32_t& rng) {
  NdShape next = s;
  // change dim, add rank, or remove rank, but product must remain <= kNodes.
  // Recompute strides. Arrays remain std::array<int16_t, kNodes>.
  return commit_if_valid(next, s);
}
```

Estimated cycle budget: topology mutation is slow path; under 1 us.

## Task 4 - `reference/pde_engines/tensor_nd.hpp`

The file is operation-rich but not directly Living Silicon-ready: it uses float32 buffers, NCHW layout, and only some AVX2. It is useful as a semantic reference for tensor operations, not as-is for the int16 hot path.

| Operation | Lines | What it does | AVX2? | Data types | Use as-is in Living Silicon? | Equivalent Python source |
|---|---:|---|---|---|---|---|
| `conv2d` | 23-57 | NCHW float32 2D convolution with stride/pad/dilation. | No explicit AVX2. | `float*`. | No. Needs int16/Q accumulation and aligned loads. | `v028` `Conv{N}d`, lines 224-230. |
| `conv2d_backward_data` | 60-94 | Data gradient for 2D conv. | No. | `float*`. | No; not needed for current online GA hot path. | PyTorch autograd equivalent in conv versions. |
| `conv2d_backward_weight` | 97-133 | Weight/bias gradient for conv. | No. | `float*`. | No; GA mutates parameters, not backprop in hot path. | PyTorch training loops in `v024+`. |
| `matmul` | 137-155 | `C[M,N]=A[M,K]*B[K,N]`. | Yes: `_mm256_set1_ps`, `_mm256_loadu_ps`, `_mm256_fmadd_ps`. | `float*`. | No as-is; port to int16/int32 for LUT/block transforms. | `v036` LUT matmul lines 202-205. |
| `matmul_backward` | 160-182 | Gradients for matmul. | No. | `float*`. | No. | PyTorch autograd equivalent. |
| `pool2d` | 185-213 | 2D max pool, optional argmax indices. | No. | `float*`, `int*`. | No as-is; rewrite int16 max pool. | `v028` MaxPool lines 239-242. |
| `upsample2d` | 216-230 | Nearest-neighbor 2D upsample. | No. | `float*`. | Maybe conceptually for slow decode; rewrite int16. | `v028` decoder upsample lines 312-313 and 641-642. |
| `batchnorm2d` | 233-279 | NCHW batch norm with computed mean/variance. | Partly yes for normalize/store. | `float*`. | No in hot path; use fixed-point EMA normalization if needed. | `v028` `BatchNorm{N}d`, lines 235-237. |
| `layer_norm` | 282-296 | Whole-vector normalization. | No. | `float*`. | No as-is; slow-path fixed-point normalization only. | `v030` unit-sphere params lines 96-98, 590-593. |
| `relu`, `gelu`, derivatives | 299-311 | Scalar activations. | Scalar. | `float`. | ReLU maps to int16; GELU does not belong in hot path. | `v028` activations lines 249-250. |
| `relu_inplace` | 314-322 | AVX2 ReLU over float vector. | Yes. | `float*`. | Rewrite to `_mm256_max_epi16`. | `v028` activation after conv/pool. |
| `gelu_inplace` | 325-327 | Scalar GELU over float vector. | No. | `float*`. | No; approximate with LUT only if needed. | `v028` GELU activation option. |
| `softmax` | 330-340 | Row softmax. | No. | `float*`. | No for current engine; attention would need fixed-point approximation. | No strong nDxnD hot-path equivalent. |
| `add_bias_2d` | 343-359 | Adds per-channel bias. | Yes. | `float*`. | Rewrite to int16 saturating add. | Conv layer bias equivalent. |
| `residual_add` | 362-370 | Vector residual add. | Yes. | `float*`. | Rewrite to `_mm256_adds_epi16`; useful. | `v037` residual block concept. |

## Task 5 - Optimal Hybrid Engine Design

### 5.1 Graph Topology

Use `v028` generic nD tensor topology as the model, but implement it as fixed-capacity flattened arrays:

- `rank`: 1..4 runtime rank.
- `dims[4]`: product must be `<= kNodes`.
- `stride[4]`: row-major flat index strides.
- Edges: immediate plus optional radius-2 neighbors along each axis, with ring wrap or clamped boundary selected by genome.

The current Living Silicon 1D ring is `rank=1`, `dims={2048}`, `stride={1}`, wrap enabled.

### 5.2 Message Passing

Use stencil message passing:

```text
msg[i] = sum_d (w_neg[d] * state[i - stride[d]] + w_pos[d] * state[i + stride[d]])
new_state[i] = blend(self_state, msg) + phase/EI/coupling update
```

This directly maps to GNN neighborhood aggregation and preserves the current `advance_lane()` semantics.

### 5.3 GA Evolution

Use three tiers:

- Per-epoch Living Silicon mutation: scalar int16 fields, stagnation boost.
- Every few epochs: current best-to-worst conservative graft.
- Slower topology epochs: `v036`/`v039` mutate `rank`, `dims`, `kernel_radius`, `block_size`, `fitness_weights`, `compression_tradeoff`, and optional IFS coefficients.

### 5.4 Tensor Ops

Critical tensor ops to implement in C++:

- `conv_nd_stencil_q15`: nD neighbor aggregation.
- `pool_nd_q15`: max/avg reduction for summaries.
- `reduce_abs_q15`: energy, MSE, entropy-lite histogram.
- `residual_add_q15`: residual state updates.
- Optional tiny `matmul_q15`: for cache-bounded block/LUT transforms.

Avoid dynamic PyTorch-like arbitrary channels in the hot path. One to four channels (`mag`, `ph`, `omega`, optional `membrane`) is enough.

### 5.5 Data Layout

Recommended minimal addition:

```cpp
struct alignas(64) NdShape {
  uint8_t rank{1};
  uint16_t dims[4]{2048, 1, 1, 1};
  uint16_t stride[4]{1, 2048, 2048, 2048};
  uint8_t wrap_mask{1};
};

struct alignas(64) ThreadState {
  alignas(64) std::array<int16_t, kNodes> mag;
  alignas(64) std::array<uint16_t, kNodes> ph;
  alignas(64) std::array<int16_t, kNodes> omega;
  alignas(64) std::array<int8_t, kNodes> ei;
  alignas(64) std::array<int16_t, kNodes> scratch; // required for non-1D nD stencil
  std::array<uint64_t, 4> nd;
  NdShape shape;
  uint32_t rng;
  uint64_t tick_counter;
};
```

Important cache note: `mag` alone is 4096 bytes, `ph` 4096 bytes, `omega` 4096 bytes, `ei` 2048 bytes, `scratch` 4096 bytes. One lane is about 18-20 KB plus metadata, so a single active lane fits in a 32 KB L1 data cache. All 8 lanes do not fit simultaneously; process lane-by-lane as the current code does.

### 5.6 Tick Budget

Target in request: <=500 ns per node update. This is loose for AVX2 int16. The realistic target should be <=500 ns for a 16-node SIMD vector, or <=5 us per 2048-node lane tick.

Expected:

- Current 1D hot path: likely low microseconds per full lane tick when L1-resident.
- 1D generalized stencil: 1-3 us per 2048 nodes.
- 2D/3D nD stencil with two-axis/four-axis neighbors: 3-10 us per 2048 active cells.
- Epoch GA/fitness: amortized negligible if every 128 ticks.

### 5.7 SIMD Critical Path

Critical intrinsics:

- Aligned load/store: `_mm256_load_si256`, `_mm256_store_si256`.
- Neighbor shifts: `_mm256_alignr_epi8`, plus boundary loads for row/tile edges.
- Q8/Q15 blend: `_mm256_unpacklo_epi16`, `_mm256_unpackhi_epi16`, `_mm256_mullo_epi32`, `_mm256_srai_epi32`, `_mm256_packs_epi32`.
- Saturating add/sub: `_mm256_adds_epi16`, `_mm256_subs_epi16`.
- Pool/reduce: `_mm256_max_epi16`, `_mm256_abs_epi16`, `_mm256_madd_epi16`.
- Sign/EI: `_mm256_cvtepi8_epi16`, `_mm256_sign_epi16`.

## Task 6 - Living Silicon Comparison

### 6.1 `advance_lane()`

Current source: `reference/living_silicon/living_silicon.cpp:167-436`.

Current graph: 1D ring of `kNodes=2048`, defined in `living_silicon.hpp:11`. Message passing uses only next-neighbor values:

- AVX2 path loads `mag`, `ph`, `omega`, computes shifted neighbor vectors with `_mm256_alignr_epi8` at lines 260-263.
- Phase and magnitude are blended with Q8 arithmetic at lines 203-237 and 265-269.
- Coupling is phase-sign-dependent at lines 271-292.
- Scalar fallback uses `next = (j + 1U) & (kNodes - 1U)` at lines 309-328.

Compared with `v028`, current LS is a hard-coded 1D `Conv1d` with kernel effectively over self and next neighbor. `v028` generalizes this to nD `Conv{N}d`/pool/transpose-conv shape algebra. The minimal upgrade is to replace the hard-coded next neighbor with shape-derived `stride[d]` neighbors while preserving the existing Q8 blend and EI coupling.

### 6.2 `epoch_boundary()` / GA

There is no function named `epoch_boundary()`. Epoch behavior is split:

- `is_epoch = (state.tick_counter & kEpochMask) == 0` at `living_silicon.cpp:172`.
- `maybe_mutate()` runs only at epoch boundary, lines 438-577.
- `maybe_crossover()` runs every `kCrossoverEpochs`, lines 595-641.

Current GA:

- Fitness formula at lines 448-474: population, energy delta, energy homeostasis, coherence band-pass, drive response.
- Mutation at lines 500-543: one scalar genome field is changed, with stagnation boost.
- Crossover at lines 621-638: conservative 25% graft from best to worst, plus optional delta inheritance.

Compared with `v036`, LS is already stronger for real-time fixed-point control, but missing adaptive quality/compression weights, topology mutation, and diversity/niche protection. Compared with `v039`, LS lacks tournament selection and explicit diversity score.

### 6.3 `Genome` Fields to Add

Current fields are in `living_silicon.hpp:16-29`.

Minimal additions:

```cpp
std::atomic<std::int16_t> nd_rank{1};
std::atomic<std::int16_t> kernel_radius{1};
std::atomic<std::int16_t> boundary_mode{1};      // 1=wrap, 0=clamp
std::atomic<std::int16_t> block_size{8};
std::atomic<std::int16_t> compression_tradeoff{128}; // Q8
std::atomic<std::int16_t> fitness_w_energy{64};
std::atomic<std::int16_t> fitness_w_coh{64};
std::atomic<std::int16_t> fitness_w_pop{48};
std::atomic<std::int16_t> fitness_w_drive{48};
std::atomic<std::int16_t> fitness_w_diversity{16};
```

Optional experimental additions:

```cpp
std::array<std::atomic<std::int16_t>, 4> ifs_coeff_q12;
std::atomic<std::int16_t> lut_mode{0};
std::atomic<std::int16_t> topology_epoch_period{16};
```

### 6.4 `ThreadState` Arrays to Add

Current fields are in `living_silicon.hpp:83-91`.

Add:

- `scratch`: required for nD stencil updates without corrupting neighbor reads.
- `shape`: runtime nD metadata.
- Optional `edge_w_pos[4]`/`edge_w_neg[4]` in genome or controller, not full edge arrays.
- Optional `block_summary[256]` for slow block reductions, not hot path.

Do not add full adjacency lists. For 2048 nodes, explicit edges would waste cache and hurt SIMD. Shape-derived strides are enough.

### 6.5 `ControllerState` Feedback Loops Missing

Current controller fields are in `living_silicon.hpp:71-81`.

Missing nDxnD-inspired loops:

- Diversity/niche pressure from `v030` and `v039`.
- Compression/quality tradeoff from `v036`.
- Topology stagnation counter separate from scalar genome stagnation.
- Tensor reduction telemetry: pooled energy by block, per-axis coherence, boundary disagreement.
- Fitness weight adaptation: slow mutation/renormalization of fixed-point weights.

### 6.6 Minimal Change Plan

Minimal change that supports an nD lattice while keeping the AVX2 hot path:

1. Add `NdShape` and `scratch` to `ThreadState`.
2. Keep current 1D AVX2 path as the fast specialization when `rank == 1`.
3. Add a generic nD scalar or partially vectorized path for `rank > 1`.
4. Once correct, vectorize the common 2D tiled case with contiguous row segments.
5. Add slow-path topology mutation at epoch boundaries only.
6. Add diversity-aware donor selection to `maybe_crossover()`.

This avoids destabilizing the proven 1D ring and gives a clear path to nD tensors.

## Final Foundation Choice

Use `v028` as the tensor/GNN vocabulary source, `v036` as the adaptive GA/fitness source, and `v039` as the nD topology/fractal exploration source. Living Silicon remains the execution foundation because it already has the required fixed-size arrays, int16 arithmetic, AVX2 hot path, epoch GA, and cache discipline.

The redesign should not port PyTorch. It should port the ideas:

- `Conv{N}d` becomes fixed nD stencil message passing.
- `MaxPool{N}d`/`AvgPool{N}d` become block summaries and fitness telemetry.
- Layer-bit evolution becomes Q-shift/kernel/weight evolution.
- Fractal blocks become slow-path compression/topology descriptors.
- PSNR/compression tradeoff becomes coherence/energy/compression tradeoff.

