#include "living_silicon.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace antigravity::control::living {
namespace {

std::uint32_t xorshift32(std::uint32_t &rng) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}

std::uint64_t exc_inh(const std::uint64_t current,
                      const std::uint64_t excitatory,
                      const std::uint64_t inhibitory) {
  return (current & ~inhibitory) | (excitatory & (~current) & (~inhibitory));
}

int clamp_i16(const int value) { return std::clamp(value, -32768, 32767); }

std::uint16_t phase_distance(const std::uint16_t a, const std::uint16_t b) {
  const std::uint32_t d = a >= b ? static_cast<std::uint32_t>(a - b)
                                 : static_cast<std::uint32_t>(b - a);
  return static_cast<std::uint16_t>(std::min<std::uint32_t>(d, 65536U - d));
}

#if defined(__AVX2__)
std::int64_t hsum_abs16(__m256i v) {
  const __m256i a = _mm256_abs_epi16(v);
  const __m256i lo = _mm256_unpacklo_epi16(a, _mm256_setzero_si256());
  const __m256i hi = _mm256_unpackhi_epi16(a, _mm256_setzero_si256());
  const __m256i s32 = _mm256_add_epi32(lo, hi);
  const __m128i l = _mm256_castsi256_si128(s32);
  const __m128i h = _mm256_extracti128_si256(s32, 1);
  __m128i s = _mm_add_epi32(l, h);
  s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
  s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
  return _mm_cvtsi128_si32(s);
}

__m256i loadu256(const void *ptr) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
}
#endif

void recompute_block_flags(Lane2048 &lane, const std::size_t b) {
  const auto i = static_cast<int>(b * kBlockLanes);
  const std::int16_t offsets[kMaxNeighbors]{lane.off0[b], lane.off1[b],
                                            lane.off2[b], lane.off3[b]};
  const std::int16_t weights[kMaxNeighbors]{lane.w0[b], lane.w1[b], lane.w2[b],
                                            lane.w3[b]};
  bool safe = true;
  bool active = false;
  for (std::size_t slot = 0; slot < kMaxNeighbors; ++slot) {
    active = active || weights[slot] != 0;
    const auto begin = i + static_cast<int>(offsets[slot]);
    const auto end = begin + static_cast<int>(kBlockLanes);
    safe = safe && begin >= 0 && end <= static_cast<int>(kNodes);
  }
  lane.block_flags[b] = static_cast<std::uint16_t>((safe ? 0x1U : 0U) |
                                                   (active ? 0x2U : 0U));
}

void initialize_ring_topology(Lane2048 &lane) {
  for (std::size_t b = 0; b < kBlocks; ++b) {
    lane.off0[b] = -1;
    lane.off1[b] = 1;
    lane.off2[b] = 0;
    lane.off3[b] = 0;
    lane.w0[b] = 16384;
    lane.w1[b] = 16384;
    lane.w2[b] = 0;
    lane.w3[b] = 0;
    recompute_block_flags(lane, b);
  }
}

std::int16_t mutate_offset(const std::int16_t current, std::uint32_t &rng) {
  const int step = static_cast<int>(xorshift32(rng) & 0x1FU) - 16;
  const int next = std::clamp<int>(static_cast<int>(current) + step, -256, 256);
  return static_cast<std::int16_t>(next == 0 ? (step < 0 ? -1 : 1) : next);
}

std::int16_t mutate_weight(const std::int16_t current, std::uint32_t &rng) {
  const int step = static_cast<int>(xorshift32(rng) & 0x0FFFU) - 2048;
  return static_cast<std::int16_t>(
      std::clamp<int>(static_cast<int>(current) + step, -32768, 32767));
}

bool should_mutate_block(const Lane2048 &lane, const std::size_t b,
                         std::uint32_t &rng) {
  const auto first = b * kBlockLanes;
  int stress = 0;
  for (std::size_t i = first; i < first + kBlockLanes; ++i) {
    stress += std::abs(static_cast<int>(lane.stress[i]));
  }
  const std::uint32_t chance = stress > 8192 ? 0x07U : 0x1FU;
  return (xorshift32(rng) & chance) == 0U;
}

void mutate_topology_blocks(Lane2048 &lane, std::uint32_t &rng) {
  for (std::size_t b = 0; b < kBlocks; ++b) {
    if (!should_mutate_block(lane, b, rng)) {
      continue;
    }
    lane.off0[b] = mutate_offset(lane.off0[b], rng);
    lane.off1[b] = mutate_offset(lane.off1[b], rng);
    lane.w0[b] = mutate_weight(lane.w0[b], rng);
    lane.w1[b] = mutate_weight(lane.w1[b], rng);
    if ((xorshift32(rng) & 0x3U) == 0U) {
      lane.off2[b] = mutate_offset(lane.off2[b], rng);
      lane.w2[b] = mutate_weight(lane.w2[b], rng);
    }
    if ((xorshift32(rng) & 0x7U) == 0U) {
      lane.off3[b] = mutate_offset(lane.off3[b], rng);
      lane.w3[b] = mutate_weight(lane.w3[b], rng);
    }
    recompute_block_flags(lane, b);
  }
}

void copy_genome(const Genome &src, Genome &dst) {
  dst.delta.store(src.delta.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  dst.coupling.store(src.coupling.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
  dst.threshold.store(src.threshold.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
  dst.blend.store(src.blend.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  dst.decay.store(src.decay.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  dst.inject_rate.store(src.inject_rate.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
  dst.omega_width.store(src.omega_width.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
  dst.ei_balance.store(src.ei_balance.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
  dst.generation.store(src.generation.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
  dst.fitness.store(src.fitness.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
  dst.best_fitness.store(src.best_fitness.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  dst.total_mutations.store(src.total_mutations.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
}

void advance_node_scalar(Lane2048 &lane, const std::size_t j,
                         const std::int16_t delta,
                         const std::int16_t coupling,
                         const std::int16_t blend,
                         const std::int16_t decay, std::int64_t &tick_energy) {
  const auto b = j / kBlockLanes;
  const std::int16_t offsets[kMaxNeighbors]{lane.off0[b], lane.off1[b],
                                            lane.off2[b], lane.off3[b]};
  const std::int16_t weights[kMaxNeighbors]{lane.w0[b], lane.w1[b], lane.w2[b],
                                            lane.w3[b]};
  std::int32_t mag_acc = 0;
  std::uint32_t ph_acc = 0;
  for (std::size_t slot = 0; slot < kMaxNeighbors; ++slot) {
    if (weights[slot] == 0) {
      continue;
    }
    auto n = static_cast<int>(j) + static_cast<int>(offsets[slot]);
    n %= static_cast<int>(kNodes);
    if (n < 0) {
      n += static_cast<int>(kNodes);
    }
    const auto idx = static_cast<std::size_t>(n);
    mag_acc += static_cast<int>(lane.scratch_mag[idx]) * weights[slot];
    ph_acc += static_cast<std::uint32_t>(
        static_cast<std::uint16_t>(lane.scratch_ph[idx] + delta +
                                   lane.omega[idx]) *
        static_cast<std::uint32_t>(static_cast<std::uint16_t>(weights[slot])));
  }

  const auto p = static_cast<std::uint16_t>(lane.scratch_ph[j] + delta +
                                           lane.omega[j]);
  const auto pn = static_cast<std::uint16_t>(ph_acc >> 15);
  const auto m = lane.scratch_mag[j];
  const auto mn = static_cast<std::int16_t>(clamp_i16(mag_acc >> 15));
  const auto node_blend = std::clamp<int>(
      blend + (static_cast<int>(lane.g_blend[j]) >> 8), 0, 256);
  const auto inv_blend = 256 - node_blend;
  const auto pb = static_cast<std::uint16_t>(
      (static_cast<std::uint32_t>(p) * node_blend +
       static_cast<std::uint32_t>(pn) * inv_blend) >>
      8);
  auto mb = static_cast<int>(
      (static_cast<int>(m) * node_blend + static_cast<int>(mn) * inv_blend) >>
      8);
  const auto node_coupling =
      std::clamp<int>(coupling + (static_cast<int>(lane.g_coupling[j]) >> 8),
                      -32768, 32767);
  const int sign = (static_cast<std::uint16_t>(pb + 16384U) >> 15) ? -1 : 1;
  mb = clamp_i16(mb + sign * static_cast<int>(lane.ei[j]) * node_coupling);
  const auto node_decay =
      std::clamp<int>(decay + (static_cast<int>(lane.g_decay[j]) >> 12), 0, 8);
  if (node_decay > 0) {
    mb -= (mb >> node_decay);
  }

  lane.mag[j] = static_cast<std::int16_t>(clamp_i16(mb));
  lane.ph[j] = pb;
  lane.stress[j] = static_cast<std::int16_t>(
      clamp_i16(std::abs(static_cast<int>(lane.mag[j]) - static_cast<int>(m))));
  lane.novelty[j] = static_cast<std::int16_t>(
      clamp_i16(static_cast<int>(phase_distance(lane.ph[j], lane.scratch_ph[j]))));
  tick_energy += std::abs(static_cast<int>(lane.mag[j]));
}

std::int64_t advance_lane_blocks(Lane2048 &lane, const Genome &genome,
                                 const std::int16_t coupling,
                                 const std::int16_t blend) {
  const auto delta = genome.delta.load(std::memory_order_relaxed);
  const auto decay = genome.decay.load(std::memory_order_relaxed);
  std::copy(lane.mag.begin(), lane.mag.end(), lane.scratch_mag.begin());
  std::copy(lane.ph.begin(), lane.ph.end(), lane.scratch_ph.begin());

  std::int64_t tick_energy = 0;
#if defined(__AVX2__)
  const __m256i zero_v = _mm256_setzero_si256();
  const __m256i delta_v = _mm256_set1_epi16(delta);
  const __m256i coupling_v = _mm256_set1_epi16(coupling);
  const __m256i blend32_v = _mm256_set1_epi32(static_cast<int>(blend));
  const __m256i inv_blend32_v =
      _mm256_set1_epi32(static_cast<int>(256 - blend));
  const auto sign_extend_i16_to_i32 =
      [&](const __m256i v, __m256i &lo, __m256i &hi) noexcept {
        const __m256i sign = _mm256_cmpgt_epi16(zero_v, v);
        lo = _mm256_unpacklo_epi16(v, sign);
        hi = _mm256_unpackhi_epi16(v, sign);
      };
  const auto zero_extend_u16_to_i32 =
      [&](const __m256i v, __m256i &lo, __m256i &hi) noexcept {
        lo = _mm256_unpacklo_epi16(v, zero_v);
        hi = _mm256_unpackhi_epi16(v, zero_v);
      };
  const auto blend_phase_q8 = [&](const __m256i p,
                                  const __m256i pn) noexcept -> __m256i {
    __m256i p_lo, p_hi, pn_lo, pn_hi;
    zero_extend_u16_to_i32(p, p_lo, p_hi);
    zero_extend_u16_to_i32(pn, pn_lo, pn_hi);
    const __m256i sum_lo = _mm256_add_epi32(
        _mm256_mullo_epi32(p_lo, blend32_v),
        _mm256_mullo_epi32(pn_lo, inv_blend32_v));
    const __m256i sum_hi = _mm256_add_epi32(
        _mm256_mullo_epi32(p_hi, blend32_v),
        _mm256_mullo_epi32(pn_hi, inv_blend32_v));
    return _mm256_packus_epi32(_mm256_srli_epi32(sum_lo, 8),
                               _mm256_srli_epi32(sum_hi, 8));
  };
  const auto blend_mag_q8 = [&](const __m256i m,
                                const __m256i mn) noexcept -> __m256i {
    __m256i m_lo, m_hi, mn_lo, mn_hi;
    sign_extend_i16_to_i32(m, m_lo, m_hi);
    sign_extend_i16_to_i32(mn, mn_lo, mn_hi);
    const __m256i sum_lo = _mm256_add_epi32(
        _mm256_mullo_epi32(m_lo, blend32_v),
        _mm256_mullo_epi32(mn_lo, inv_blend32_v));
    const __m256i sum_hi = _mm256_add_epi32(
        _mm256_mullo_epi32(m_hi, blend32_v),
        _mm256_mullo_epi32(mn_hi, inv_blend32_v));
    return _mm256_packs_epi32(_mm256_srai_epi32(sum_lo, 8),
                              _mm256_srai_epi32(sum_hi, 8));
  };

  for (std::uint32_t b = 0; b < kBlocks; ++b) {
    const std::uint32_t i = b * kBlockLanes;
    if ((lane.block_flags[b] & 0x1U) == 0U) {
      for (std::size_t j = i; j < i + kBlockLanes; ++j) {
        advance_node_scalar(lane, j, delta, coupling, blend, decay,
                            tick_energy);
      }
      continue;
    }

    const __m256i m = _mm256_load_si256(
        reinterpret_cast<const __m256i *>(lane.scratch_mag.data() + i));
    __m256i p = _mm256_load_si256(
        reinterpret_cast<const __m256i *>(lane.scratch_ph.data() + i));
    const __m256i omega = _mm256_load_si256(
        reinterpret_cast<const __m256i *>(lane.omega.data() + i));
    p = _mm256_add_epi16(p, _mm256_add_epi16(delta_v, omega));

    const std::int16_t offsets[kMaxNeighbors]{lane.off0[b], lane.off1[b],
                                              lane.off2[b], lane.off3[b]};
    const std::int16_t weights[kMaxNeighbors]{lane.w0[b], lane.w1[b],
                                              lane.w2[b], lane.w3[b]};
    __m256i acc_mag_lo = _mm256_setzero_si256();
    __m256i acc_mag_hi = _mm256_setzero_si256();
    __m256i acc_phase_lo = _mm256_setzero_si256();
    __m256i acc_phase_hi = _mm256_setzero_si256();

    for (std::size_t slot = 0; slot < kMaxNeighbors; ++slot) {
      if (weights[slot] == 0) {
        continue;
      }
      const auto base = static_cast<std::ptrdiff_t>(i) + offsets[slot];
      const __m256i weight_v = _mm256_set1_epi32(weights[slot]);
      const __m256i nm = loadu256(lane.scratch_mag.data() + base);
      __m256i np = loadu256(lane.scratch_ph.data() + base);
      const __m256i nomega = loadu256(lane.omega.data() + base);
      np = _mm256_add_epi16(np, _mm256_add_epi16(delta_v, nomega));

      __m256i nm_lo, nm_hi, np_lo, np_hi;
      sign_extend_i16_to_i32(nm, nm_lo, nm_hi);
      zero_extend_u16_to_i32(np, np_lo, np_hi);
      acc_mag_lo =
          _mm256_add_epi32(acc_mag_lo, _mm256_mullo_epi32(nm_lo, weight_v));
      acc_mag_hi =
          _mm256_add_epi32(acc_mag_hi, _mm256_mullo_epi32(nm_hi, weight_v));
      acc_phase_lo = _mm256_add_epi32(
          acc_phase_lo, _mm256_mullo_epi32(np_lo, weight_v));
      acc_phase_hi = _mm256_add_epi32(
          acc_phase_hi, _mm256_mullo_epi32(np_hi, weight_v));
    }

    const __m256i mn = _mm256_packs_epi32(_mm256_srai_epi32(acc_mag_lo, 15),
                                          _mm256_srai_epi32(acc_mag_hi, 15));
    const __m256i pn =
        _mm256_packus_epi32(_mm256_srli_epi32(acc_phase_lo, 15),
                            _mm256_srli_epi32(acc_phase_hi, 15));
    const __m256i pb = blend_phase_q8(p, pn);
    __m256i mb = blend_mag_q8(m, mn);
    const __m256i quarter_v = _mm256_set1_epi16(16384);
    const __m256i sg = _mm256_srai_epi16(_mm256_add_epi16(pb, quarter_v), 15);
    const __m256i c = _mm256_sub_epi16(_mm256_xor_si256(coupling_v, sg), sg);
    const __m256i ei =
        _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(
            lane.ei.data() + i)));
    mb = _mm256_adds_epi16(mb, _mm256_sign_epi16(c, ei));
    if (decay > 0) {
      const __m128i decay_count = _mm_cvtsi32_si128(decay);
      mb = _mm256_sub_epi16(mb, _mm256_sra_epi16(mb, decay_count));
    }

    _mm256_store_si256(reinterpret_cast<__m256i *>(lane.mag.data() + i), mb);
    _mm256_store_si256(reinterpret_cast<__m256i *>(lane.ph.data() + i), pb);
    tick_energy += hsum_abs16(mb);
  }
#else
  for (std::uint32_t b = 0; b < kBlocks; ++b) {
    const std::uint32_t i = b * kBlockLanes;
    for (std::size_t j = i; j < i + kBlockLanes; ++j) {
      advance_node_scalar(lane, j, delta, coupling, blend, decay, tick_energy);
    }
  }
#endif
  return tick_energy;
}

} // namespace

Engine::Engine() { initialize(0xC0FFEEULL); }

void Engine::initialize(const std::uint64_t seed) {
  std::scoped_lock lock(mutex_);
  global_tick_ = 0;
  membrane_.fill(0);

  std::uint32_t rng = static_cast<std::uint32_t>(seed == 0 ? 0xC0FFEEU : seed);
  for (std::size_t lane_index = 0; lane_index < kThreads; ++lane_index) {
    auto &genome = genomes_[lane_index];
    genome.delta.store(17, std::memory_order_relaxed);
    genome.coupling.store(64, std::memory_order_relaxed);
    genome.threshold.store(8000, std::memory_order_relaxed);
    genome.blend.store(192, std::memory_order_relaxed);
    genome.decay.store(0, std::memory_order_relaxed);
    genome.inject_rate.store(64, std::memory_order_relaxed);
    genome.omega_width.store(8, std::memory_order_relaxed);
    genome.ei_balance.store(204, std::memory_order_relaxed);
    genome.generation.store(0, std::memory_order_relaxed);
    genome.fitness.store(0, std::memory_order_relaxed);
    genome.best_fitness.store(0, std::memory_order_relaxed);
    genome.total_mutations.store(0, std::memory_order_relaxed);

    auto &state = data_[lane_index];
    state = Lane2048{};
    state.rng = (0xCAFE0000U + static_cast<std::uint32_t>(lane_index)) ^ rng;
    initialize_ring_topology(state);
    for (auto &word : state.nd) {
      word =
          (static_cast<std::uint64_t>(xorshift32(rng)) << 32) | xorshift32(rng);
    }
    for (std::size_t i = 0; i < kNodes; ++i) {
      state.mag[i] =
          static_cast<std::int16_t>((xorshift32(rng) & 0x7FFFU) - 0x4000U);
      state.ph[i] = static_cast<std::uint16_t>(xorshift32(rng));
      const int spread = static_cast<int>(xorshift32(rng) & 0x0FU) - 8;
      state.omega[i] = static_cast<std::int16_t>(std::clamp(spread, -32, 32));
      state.ei[i] = (static_cast<int>(xorshift32(rng) & 0xFFU) < 204)
                        ? static_cast<std::int8_t>(1)
                        : static_cast<std::int8_t>(-1);
    }
    obs_[lane_index] = Observation{};
    ctrl_[lane_index] = ControllerState{};
  }
}

void Engine::inject(const std::size_t lane, const std::int16_t *signal,
                    std::size_t n) {
  if (lane >= kThreads || signal == nullptr || n == 0) {
    return;
  }
  std::scoped_lock lock(mutex_);
  n = std::min(n, kNodes);
  std::copy_n(signal, n, data_[lane].mag.begin());

  std::int64_t drive = 0;
  for (std::size_t i = 0; i < n; ++i) {
    drive += std::abs(static_cast<int>(signal[i]));
  }
  drive = n > 0 ? (drive / static_cast<std::int64_t>(n)) : 0;
  ctrl_[lane].recent_drive =
      static_cast<std::int32_t>(std::clamp<std::int64_t>(drive >> 6, 0, 1024));
}

void Engine::tick(const std::uint64_t ticks) {
  std::scoped_lock lock(mutex_);
  tick_locked(ticks, (1ULL << kThreads) - 1ULL);
}

void Engine::tick_lanes(const std::uint64_t ticks, std::uint64_t lane_mask) {
  std::scoped_lock lock(mutex_);
  lane_mask &= (1ULL << kThreads) - 1ULL;
  tick_locked(ticks, lane_mask);
}

void Engine::tick_lane(const std::size_t lane, const std::uint64_t ticks) {
  if (lane >= kThreads) {
    return;
  }
  tick_lanes(ticks, 1ULL << lane);
}

void Engine::tick_locked(const std::uint64_t ticks,
                         const std::uint64_t lane_mask) {
  if (lane_mask == 0) {
    return;
  }
  for (std::uint64_t i = 0; i < ticks; ++i) {
    if (enable_collective_) {
      compute_membrane();
    }
    for (std::size_t lane = 0; lane < kThreads; ++lane) {
      if ((lane_mask & (1ULL << lane)) != 0) {
        advance_lane(lane);
      }
    }
    ++global_tick_;
    if (enable_collective_ && lane_mask == ((1ULL << kThreads) - 1ULL) &&
        (global_tick_ & kEpochMask) == 0) {
      maybe_crossover();
    }
  }
}

void Engine::advance_lane(const std::size_t lane) {
  auto &state = data_[lane];
  auto &obs = obs_[lane];
  auto &ctrl = ctrl_[lane];
  state.tick_counter += 1;
  const bool is_epoch = (state.tick_counter & kEpochMask) == 0;
  std::chrono::steady_clock::time_point epoch_start;
  if (is_epoch) {
    epoch_start = std::chrono::steady_clock::now();
  }

  auto &genome = genomes_[lane];
  const auto base_coupling = genome.coupling.load(std::memory_order_relaxed);
  const auto base_blend = genome.blend.load(std::memory_order_relaxed);
  const int mem = std::min<int>(ctrl.membrane_local, 4);
  const int stag = std::min<int>(ctrl.stagnation_epochs, 4);
  const auto coupling = static_cast<std::int16_t>(std::clamp<int>(
      base_coupling + mem * 8 + stag * 4 + ctrl.coupling_adapt, 0, 200));
  const auto blend = static_cast<std::int16_t>(
      std::clamp<int>(base_blend - mem * 8 + ctrl.blend_adapt, 32, 248));

  const auto tick_energy =
      advance_lane_blocks(state, genome, coupling, blend);
  obs.energy = tick_energy;

  if (is_epoch) {
    std::int64_t tick_coherence = 0;
    for (std::size_t b = 0; b < kBlocks; ++b) {
      const auto first = b * kBlockLanes;
      for (std::size_t j = first; j < first + kBlockLanes; ++j) {
        auto n = static_cast<int>(j) + static_cast<int>(state.off1[b]);
        n %= static_cast<int>(kNodes);
        if (n < 0) {
          n += static_cast<int>(kNodes);
        }
        tick_coherence +=
            phase_distance(state.ph[j], state.ph[static_cast<std::size_t>(n)]);
      }
    }
    const auto normalized =
        std::clamp<std::int64_t>(65535 - (tick_coherence >> 5), 0, 65535);
    obs.coherence = normalized;

    const auto coh8 = static_cast<std::int32_t>(obs.coherence >> 8);
    constexpr std::int32_t kCohTarget = 180;
    const auto error = kCohTarget - coh8;
    const auto step = std::clamp<int>(error / 8, -24, 24);
    ctrl.coupling_adapt = static_cast<std::int16_t>(
        std::clamp<int>(ctrl.coupling_adapt + step, -200, 200));
    ctrl.blend_adapt = static_cast<std::int16_t>(
        std::clamp<int>(ctrl.blend_adapt + step / 2, -32, 56));
  }

  const auto avg_abs =
      static_cast<std::int32_t>(tick_energy / static_cast<std::int64_t>(kNodes));
  const auto effective_threshold =
      static_cast<std::int16_t>(std::max<std::int32_t>(64, avg_abs * 4));
  std::uint64_t exc = 0;
  std::uint64_t inh = 0;
  for (int bit = 0; bit < 64; ++bit) {
    const auto idx = static_cast<std::size_t>(bit * 32);
    if (state.mag[idx] > effective_threshold) {
      exc |= (1ULL << bit);
    }
    if (state.mag[idx] < -effective_threshold) {
      inh |= (1ULL << bit);
    }
  }
  for (auto &word : state.nd) {
    word = exc_inh(word, exc, inh);
  }
  obs.nd_popcount = 0;
  for (const auto word : state.nd) {
    obs.nd_popcount += static_cast<std::int32_t>(std::popcount(word));
  }

  maybe_mutate(lane, tick_energy);

  if (enable_collective_) {
    std::uint32_t mem_sum = 0;
    for (std::size_t j = 0; j < kNodes; ++j) {
      mem_sum += membrane_[j];
    }
    ctrl.membrane_local = static_cast<std::uint16_t>(mem_sum / kNodes);
  } else {
    ctrl.membrane_local = 0;
  }

  ctrl.recent_drive = (ctrl.recent_drive * 7) / 8;
  obs.fitness_ema = ctrl.fitness_ema;
  obs.recent_drive = ctrl.recent_drive;
  obs.stagnation_epochs = ctrl.stagnation_epochs;
  obs.membrane_local = ctrl.membrane_local;
  obs.attention_hits = ctrl.attention_hits;

  if (is_epoch) {
    const auto elapsed = std::chrono::steady_clock::now() - epoch_start;
    const auto elapsed_ns = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    const auto elapsed_us =
        static_cast<std::int32_t>(std::max<std::int64_t>(0, elapsed_ns / 1000));
    obs.tick_ns = elapsed_ns;
    obs.tick_ns_ema = obs.tick_ns_ema <= 0
                          ? elapsed_ns
                          : ((obs.tick_ns_ema * 7) + elapsed_ns) / 8;
    obs.pressure =
        obs.pressure <= 0
            ? elapsed_us
            : static_cast<std::int32_t>(((obs.pressure * 7) + elapsed_us) / 8);
  }
}

void Engine::maybe_mutate(const std::size_t lane,
                          const std::int64_t tick_energy) {
  auto &state = data_[lane];
  auto &obs = obs_[lane];
  auto &ctrl = ctrl_[lane];
  if ((state.tick_counter & kEpochMask) != 0) {
    return;
  }

  const auto coh8 = static_cast<std::int32_t>(obs.coherence >> 8);
  constexpr std::int32_t kTargetCoherence = 192;
  const auto coherence_term = 255 - std::abs(coh8 - kTargetCoherence);
  const auto energy_bucket = static_cast<std::int32_t>(tick_energy >> 11);
  const auto energy_delta = std::abs(energy_bucket - ctrl.prev_energy_bucket);
  ctrl.prev_energy_bucket = energy_bucket;
  constexpr std::int32_t kEnergyLow = 32;
  constexpr std::int32_t kEnergyHigh = 128;
  const auto energy_mid = (kEnergyLow + kEnergyHigh) / 2;
  const auto energy_home =
      (energy_bucket >= kEnergyLow && energy_bucket <= kEnergyHigh)
          ? 64
          : std::max(0, 64 - std::abs(energy_bucket - energy_mid) / 2);
  const auto population_term = 256 - std::abs(obs.nd_popcount - 128);
  const auto drive_response =
      (ctrl.recent_drive > 16) ? std::min(energy_delta * 4, 128) : 0;
  const auto my_fitness = population_term + energy_delta + energy_home +
                          coherence_term + drive_response;

  auto &genome = genomes_[lane];
  const auto previous_fitness = genome.fitness.load(std::memory_order_relaxed);
  genome.fitness.store(my_fitness, std::memory_order_relaxed);
  const auto current_best = genome.best_fitness.load(std::memory_order_relaxed);
  ctrl.prev_fitness_ema = ctrl.fitness_ema;
  ctrl.fitness_ema = (7 * ctrl.fitness_ema + my_fitness) / 8;
  const auto slope = ctrl.fitness_ema - ctrl.prev_fitness_ema;

  if (my_fitness > previous_fitness) {
    ctrl.stagnation_epochs = 0;
  } else if (ctrl.stagnation_epochs < 255) {
    ctrl.stagnation_epochs += 1;
  }

  const bool explore = ctrl.stagnation_epochs >= 2;
  const bool should_mutate =
      explore || ((state.rng & 0x1FU) == 0U && slope <= 0);
  if (should_mutate) {
    auto mutate_field = [&](std::atomic<std::int16_t> &field, const int min_v,
                            const int max_v, const int delta_mask,
                            const int delta_bias) {
      const auto current = field.load(std::memory_order_relaxed);
      xorshift32(state.rng);
      const int stag_boost = (ctrl.stagnation_epochs > 8) ? 2 : 1;
      const auto raw_next =
          static_cast<int>(current) +
          ((static_cast<int>(state.rng & delta_mask) - delta_bias) *
           stag_boost);
      field.store(static_cast<std::int16_t>(
                      std::clamp(raw_next, min_v, max_v)),
                  std::memory_order_relaxed);
    };

    xorshift32(state.rng);
    switch (state.rng % 8U) {
    case 0:
      mutate_field(genome.delta, 1, 100, 0x0F, 8);
      break;
    case 1:
      mutate_field(genome.coupling, 0, 200, 0x0F, 8);
      break;
    case 2:
      mutate_field(genome.blend, 32, 224, 0x1F, 16);
      break;
    case 3:
      mutate_field(genome.omega_width, 1, 32, 0x07, 4);
      break;
    case 4:
      mutate_field(genome.decay, 0, 8, 0x03, 1);
      break;
    case 5:
      mutate_field(genome.inject_rate, 16, 256, 0x1F, 16);
      break;
    case 6:
      mutate_field(genome.ei_balance, 128, 240, 0x0F, 8);
      break;
    default:
      mutate_topology_blocks(state, state.rng);
      break;
    }
    if ((state.rng & 0x3U) == 0U) {
      mutate_topology_blocks(state, state.rng);
    }

    genome.generation.fetch_add(1, std::memory_order_relaxed);
    genome.total_mutations.fetch_add(1, std::memory_order_relaxed);
    obs.mutations += 1;

    int expected = current_best;
    while (my_fitness > expected &&
           !genome.best_fitness.compare_exchange_weak(
               expected, my_fitness, std::memory_order_relaxed)) {
    }
    if (my_fitness > current_best) {
      obs.improvements += 1;
    }
  }

  if ((state.rng & 0x3FU) == 0U) {
    const auto base_inject = genome.inject_rate.load(std::memory_order_relaxed);
    const auto effective_inject_rate =
        static_cast<std::uint32_t>(std::clamp<int>(
            base_inject + (ctrl.stagnation_epochs >= 2 ? 16 : 0), 16, 256));
    for (std::uint32_t i = 0; i < effective_inject_rate; ++i) {
      const auto idx = xorshift32(state.rng) & (kNodes - 1U);
      if (enable_collective_ && membrane_[idx] >= 2) {
        state.mag[idx] =
            static_cast<std::int16_t>((state.rng & 0x7FFFU) - 0x4000U);
        ctrl.attention_hits += 1;
      } else if (!enable_collective_) {
        state.mag[idx] =
            static_cast<std::int16_t>((state.rng & 0x7FFFU) - 0x4000U);
      }
    }
  }
}

void Engine::compute_membrane() {
  for (std::size_t i = 0; i < kNodes; ++i) {
    std::uint8_t hi = 0;
    for (std::size_t lane = 0; lane < kThreads; ++lane) {
      hi += static_cast<std::uint8_t>((data_[lane].ph[i] >> 15) & 1U);
    }
    membrane_[i] = static_cast<std::uint8_t>(
        std::min<int>(static_cast<int>(hi),
                      static_cast<int>(kThreads) - static_cast<int>(hi)));
  }
}

void Engine::maybe_crossover() {
  const auto epoch_num = global_tick_ >> 7;
  if ((epoch_num % kCrossoverEpochs) != 0) {
    return;
  }

  std::size_t best = 0;
  std::size_t worst = 0;
  auto lane_score = [&](std::size_t lane) -> std::int64_t {
    return static_cast<std::int64_t>(
        genomes_[lane].fitness.load(std::memory_order_relaxed));
  };
  for (std::size_t lane = 1; lane < kThreads; ++lane) {
    if (lane_score(lane) > lane_score(best)) {
      best = lane;
    }
    if (lane_score(lane) < lane_score(worst)) {
      worst = lane;
    }
  }
  if (best == worst || obs_[worst].stagnation_epochs < 4) {
    return;
  }

  auto &gb = genomes_[best];
  auto &gw = genomes_[worst];
  gw.coupling.store(static_cast<std::int16_t>(
                        (3 * gw.coupling.load(std::memory_order_relaxed) +
                         gb.coupling.load(std::memory_order_relaxed)) /
                        4),
                    std::memory_order_relaxed);
  gw.blend.store(
      static_cast<std::int16_t>((3 * gw.blend.load(std::memory_order_relaxed) +
                                 gb.blend.load(std::memory_order_relaxed)) /
                                4),
      std::memory_order_relaxed);
  if ((data_[worst].rng & 1U) == 0U) {
    gw.delta.store(gb.delta.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
  }
  if ((data_[worst].rng & 2U) == 0U) {
    data_[worst].off0 = data_[best].off0;
    data_[worst].off1 = data_[best].off1;
    data_[worst].off2 = data_[best].off2;
    data_[worst].off3 = data_[best].off3;
    data_[worst].w0 = data_[best].w0;
    data_[worst].w1 = data_[best].w1;
    data_[worst].w2 = data_[best].w2;
    data_[worst].w3 = data_[best].w3;
    data_[worst].block_flags = data_[best].block_flags;
  }
  obs_[worst].stagnation_epochs = 0;
}

Observation Engine::observation(const std::size_t lane) const {
  std::scoped_lock lock(mutex_);
  if (lane >= kThreads) {
    return {};
  }
  return obs_[lane];
}

GenomeSnapshot Engine::genome(const std::size_t lane) const {
  std::scoped_lock lock(mutex_);
  if (lane >= kThreads) {
    return {};
  }
  const auto &g = genomes_[lane];
  return GenomeSnapshot{
      .delta = g.delta.load(std::memory_order_relaxed),
      .coupling = g.coupling.load(std::memory_order_relaxed),
      .threshold = g.threshold.load(std::memory_order_relaxed),
      .blend = g.blend.load(std::memory_order_relaxed),
      .decay = g.decay.load(std::memory_order_relaxed),
      .inject_rate = g.inject_rate.load(std::memory_order_relaxed),
      .omega_width = g.omega_width.load(std::memory_order_relaxed),
      .ei_balance = g.ei_balance.load(std::memory_order_relaxed),
      .generation = g.generation.load(std::memory_order_relaxed),
      .fitness = g.fitness.load(std::memory_order_relaxed),
      .best_fitness = g.best_fitness.load(std::memory_order_relaxed),
      .total_mutations = g.total_mutations.load(std::memory_order_relaxed),
  };
}

std::uint32_t Engine::total_mutations() const {
  std::scoped_lock lock(mutex_);
  std::uint32_t total = 0;
  for (const auto &genome : genomes_) {
    total += genome.total_mutations.load(std::memory_order_relaxed);
  }
  return total;
}

LaneSnapshot Engine::snapshot(const std::size_t lane) const {
  std::scoped_lock lock(mutex_);
  if (lane >= kThreads) {
    return {};
  }
  const auto &g = genomes_[lane];
  return LaneSnapshot{
      .observation = obs_[lane],
      .genome =
          GenomeSnapshot{
              .delta = g.delta.load(std::memory_order_relaxed),
              .coupling = g.coupling.load(std::memory_order_relaxed),
              .threshold = g.threshold.load(std::memory_order_relaxed),
              .blend = g.blend.load(std::memory_order_relaxed),
              .decay = g.decay.load(std::memory_order_relaxed),
              .inject_rate = g.inject_rate.load(std::memory_order_relaxed),
              .omega_width = g.omega_width.load(std::memory_order_relaxed),
              .ei_balance = g.ei_balance.load(std::memory_order_relaxed),
              .generation = g.generation.load(std::memory_order_relaxed),
              .fitness = g.fitness.load(std::memory_order_relaxed),
              .best_fitness = g.best_fitness.load(std::memory_order_relaxed),
              .total_mutations =
                  g.total_mutations.load(std::memory_order_relaxed),
          },
  };
}

std::size_t Engine::read_magnitude(std::size_t lane, std::int16_t *out,
                                   std::size_t n) const {
  if (lane >= kThreads || !out) {
    return 0;
  }
  std::scoped_lock lock(mutex_);
  const auto count = std::min(n, kNodes);
  std::memcpy(out, data_[lane].mag.data(), count * sizeof(std::int16_t));
  return count;
}

std::size_t Engine::read_phase(std::size_t lane, std::uint16_t *out,
                               std::size_t n) const {
  if (lane >= kThreads || !out) {
    return 0;
  }
  std::scoped_lock lock(mutex_);
  const auto count = std::min(n, kNodes);
  std::memcpy(out, data_[lane].ph.data(), count * sizeof(std::uint16_t));
  return count;
}

void Engine::perturb(std::size_t lane, const std::int16_t *signal,
                     std::size_t n, std::int16_t gain_q8) {
  if (lane >= kThreads || !signal) {
    return;
  }
  std::scoped_lock lock(mutex_);
  const auto count = std::min(n, kNodes);
  for (std::size_t i = 0; i < count; ++i) {
    const int v = static_cast<int>(data_[lane].mag[i]) +
                  ((static_cast<int>(signal[i]) * gain_q8) >> 8);
    data_[lane].mag[i] = static_cast<std::int16_t>(clamp_i16(v));
  }
  std::int64_t drive = 0;
  for (std::size_t i = 0; i < count; ++i) {
    drive += std::abs((static_cast<int>(signal[i]) * gain_q8) >> 8);
  }
  drive = count > 0 ? (drive / static_cast<std::int64_t>(count)) : 0;
  ctrl_[lane].recent_drive = std::max(
      ctrl_[lane].recent_drive,
      static_cast<std::int32_t>(std::clamp<std::int64_t>(drive >> 6, 0, 1024)));
}

void Engine::clone_lane(std::size_t from, std::size_t to) {
  if (from >= kThreads || to >= kThreads || from == to) {
    return;
  }
  std::scoped_lock lock(mutex_);
  data_[to] = data_[from];
  ctrl_[to] = ctrl_[from];
  obs_[to] = obs_[from];
  copy_genome(genomes_[from], genomes_[to]);
}

void Engine::clone_lane_to(Engine &target, const std::size_t from,
                           const std::size_t to) const {
  if (from >= kThreads || to >= kThreads) {
    return;
  }
  if (this == &target) {
    target.clone_lane(from, to);
    return;
  }

  std::scoped_lock lock(mutex_, target.mutex_);
  target.data_[to] = data_[from];
  target.ctrl_[to] = ctrl_[from];
  target.obs_[to] = obs_[from];
  copy_genome(genomes_[from], target.genomes_[to]);
}

} // namespace antigravity::control::living
