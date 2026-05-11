#include "control/living_silicon.hpp"

#include <algorithm>
#include <array>
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

// Correct cyclic phase distance on uint16 ring
std::uint16_t phase_distance(const std::uint16_t a, const std::uint16_t b) {
  const std::uint32_t d = a >= b ? static_cast<std::uint32_t>(a - b)
                                 : static_cast<std::uint32_t>(b - a);
  return static_cast<std::uint16_t>(std::min<std::uint32_t>(d, 65536U - d));
}

#if defined(__AVX2__)
// Pure SIMD horizontal absolute sum — matches V3 original (zero store-to-load
// penalty)
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
#endif

} // namespace

Engine::Engine() { initialize(0xC0FFEEULL); }

void Engine::initialize(const std::uint64_t seed) {
  std::scoped_lock lock(mutex_);

  std::uint32_t rng = static_cast<std::uint32_t>(seed == 0 ? 0xC0FFEEU : seed);
  for (std::size_t lane = 0; lane < kThreads; ++lane) {
    auto &genome = genomes_[lane];
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

    auto &state = data_[lane];
    auto &obs = obs_[lane];
    state.rng = 0xCAFE0000U + static_cast<std::uint32_t>(lane) ^ rng;
    state.tick_counter = 0;
    for (auto &word : state.nd) {
      word =
          (static_cast<std::uint64_t>(xorshift32(rng)) << 32) | xorshift32(rng);
    }
    for (std::size_t i = 0; i < kNodes; ++i) {
      state.mag[i] =
          static_cast<std::int16_t>((xorshift32(rng) & 0x7FFFU) - 0x4000U);
      state.ph[i] = static_cast<std::uint16_t>(xorshift32(rng));
      // Kuramoto: per-node natural frequency detuning
      const int spread = static_cast<int>(xorshift32(rng) & 0x0FU) - 8; // ±8
      state.omega[i] = static_cast<std::int16_t>(std::clamp(spread, -32, 32));
      // E/I identity: ~80% excitatory by default (ei_balance=204 → 204/256 ≈
      // 80%)
      state.ei[i] = (static_cast<int>(xorshift32(rng) & 0xFFU) < 204)
                        ? static_cast<std::int8_t>(1)
                        : static_cast<std::int8_t>(-1);
    }
    obs = Observation{};
    ctrl_[lane] = ControllerState{};
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

  // Track stimulus strength for fitness gating
  std::int64_t drive = 0;
  for (std::size_t i = 0; i < n; ++i) {
    drive += std::abs(static_cast<int>(signal[i]));
  }
  drive = static_cast<std::int64_t>(n) > 0
              ? (drive / static_cast<std::int64_t>(n))
              : 0;
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
  if (lane >= kThreads)
    return;
  tick_lanes(ticks, 1ULL << lane);
}

void Engine::tick_locked(const std::uint64_t ticks,
                         const std::uint64_t lane_mask) {
  if (lane_mask == 0)
    return;
  for (std::uint64_t i = 0; i < ticks; ++i) {
    // Compute collective membrane BEFORE lane advances
    if (enable_collective_) {
      compute_membrane();
    }
    for (std::size_t lane = 0; lane < kThreads; ++lane) {
      if ((lane_mask & (1ULL << lane)) != 0) {
        advance_lane(lane);
      }
    }
    ++global_tick_;
    // Epoch-based crossover after all lanes advanced
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
  if (is_epoch)
    epoch_start = std::chrono::steady_clock::now();

  auto &genome = genomes_[lane];
  const auto delta = genome.delta.load(std::memory_order_relaxed);
  const auto base_coupling = genome.coupling.load(std::memory_order_relaxed);
  const auto base_blend = genome.blend.load(std::memory_order_relaxed);
  const auto decay = genome.decay.load(std::memory_order_relaxed);

  // Runtime modulation: effective_* from base genome + controller state
  const int mem = std::min<int>(ctrl.membrane_local, 4);
  const int stag = std::min<int>(ctrl.stagnation_epochs, 4);
  const auto coupling = static_cast<std::int16_t>(
      std::clamp<int>(base_coupling + mem * 8 + stag * 4 + ctrl.coupling_adapt, 0, 200));
  // Homeostatic blend: increase diffusion when system is decoherent
  const auto blend =
      static_cast<std::int16_t>(std::clamp<int>(base_blend - mem * 8 + ctrl.blend_adapt, 32, 248));

  std::int64_t tick_energy = 0;

// AVX2 path — Kuramoto omega + E/I coupling + correct Q8 blend
#if defined(__AVX2__)
  const __m256i delta_v = _mm256_set1_epi16(delta);
  const __m256i coupling_v = _mm256_set1_epi16(coupling);
  // 32-bit widened blend — exact match to scalar Q8 math (no 16-bit overflow)
  const __m256i blend32_v = _mm256_set1_epi32(static_cast<int>(blend));
  const __m256i inv_blend32_v = _mm256_set1_epi32(static_cast<int>(256 - blend));
  const __m256i zero_v = _mm256_setzero_si256();

  // Phase blend: unsigned 16→32, multiply, >>8, pack back to uint16
  const auto blend_phase_q8 = [&](const __m256i p,
                                  const __m256i pn) noexcept -> __m256i {
      const __m256i p_lo = _mm256_unpacklo_epi16(p, zero_v);
      const __m256i p_hi = _mm256_unpackhi_epi16(p, zero_v);
      const __m256i pn_lo = _mm256_unpacklo_epi16(pn, zero_v);
      const __m256i pn_hi = _mm256_unpackhi_epi16(pn, zero_v);
      const __m256i sum_lo = _mm256_add_epi32(
          _mm256_mullo_epi32(p_lo, blend32_v),
          _mm256_mullo_epi32(pn_lo, inv_blend32_v));
      const __m256i sum_hi = _mm256_add_epi32(
          _mm256_mullo_epi32(p_hi, blend32_v),
          _mm256_mullo_epi32(pn_hi, inv_blend32_v));
      return _mm256_packus_epi32(_mm256_srli_epi32(sum_lo, 8),
                                 _mm256_srli_epi32(sum_hi, 8));
  };

  // Magnitude blend: signed 16→32, multiply, >>8 (arithmetic), pack to int16
  const auto blend_mag_q8 = [&](const __m256i m,
                                const __m256i mn) noexcept -> __m256i {
      const __m256i m_sign = _mm256_cmpgt_epi16(zero_v, m);
      const __m256i mn_sign = _mm256_cmpgt_epi16(zero_v, mn);
      const __m256i m_lo = _mm256_unpacklo_epi16(m, m_sign);
      const __m256i m_hi = _mm256_unpackhi_epi16(m, m_sign);
      const __m256i mn_lo = _mm256_unpacklo_epi16(mn, mn_sign);
      const __m256i mn_hi = _mm256_unpackhi_epi16(mn, mn_sign);
      const __m256i sum_lo = _mm256_add_epi32(
          _mm256_mullo_epi32(m_lo, blend32_v),
          _mm256_mullo_epi32(mn_lo, inv_blend32_v));
      const __m256i sum_hi = _mm256_add_epi32(
          _mm256_mullo_epi32(m_hi, blend32_v),
          _mm256_mullo_epi32(mn_hi, inv_blend32_v));
      return _mm256_packs_epi32(_mm256_srai_epi32(sum_lo, 8),
                                _mm256_srai_epi32(sum_hi, 8));
  };

  for (std::size_t j = 0; j < kNodes; j += 32) {
    auto *mag_ptr = reinterpret_cast<__m256i *>(state.mag.data() + j);
    auto *ph_ptr = reinterpret_cast<__m256i *>(state.ph.data() + j);
    auto *mag_ptr_1 = reinterpret_cast<__m256i *>(state.mag.data() + j + 16);
    auto *ph_ptr_1 = reinterpret_cast<__m256i *>(state.ph.data() + j + 16);

    __m256i m0 = _mm256_load_si256(mag_ptr);
    __m256i m1 = _mm256_load_si256(mag_ptr_1);
    __m256i p0 = _mm256_load_si256(ph_ptr);
    __m256i p1 = _mm256_load_si256(ph_ptr_1);

    // Kuramoto: load per-node omega and add to delta
    const auto *omega_ptr =
        reinterpret_cast<const __m256i *>(state.omega.data() + j);
    const auto *omega_ptr_1 =
        reinterpret_cast<const __m256i *>(state.omega.data() + j + 16);
    const __m256i omega0 = _mm256_load_si256(omega_ptr);
    const __m256i omega1 = _mm256_load_si256(omega_ptr_1);
    p0 = _mm256_add_epi16(p0, _mm256_add_epi16(delta_v, omega0));
    p1 = _mm256_add_epi16(p1, _mm256_add_epi16(delta_v, omega1));

    const __m256i pn0 = _mm256_alignr_epi8(p1, p0, 2);
    const __m256i pn1 = _mm256_alignr_epi8(p0, p1, 2);
    const __m256i mn0 = _mm256_alignr_epi8(m1, m0, 2);
    const __m256i mn1 = _mm256_alignr_epi8(m0, m1, 2);

        // 32-bit widened blend — exact scalar match
        __m256i pb0 = blend_phase_q8(p0, pn0);
        __m256i pb1 = blend_phase_q8(p1, pn1);
        __m256i mb0 = blend_mag_q8(m0, mn0);
        __m256i mb1 = blend_mag_q8(m1, mn1);

    // Phase sign for coupling direction
    const __m256i quarter_v = _mm256_set1_epi16(16384);
    const __m256i sg0 = _mm256_srai_epi16(_mm256_add_epi16(pb0, quarter_v), 15);
    const __m256i sg1 = _mm256_srai_epi16(_mm256_add_epi16(pb1, quarter_v), 15);
    // sign * coupling
    const __m256i c0 = _mm256_sub_epi16(_mm256_xor_si256(coupling_v, sg0), sg0);
    const __m256i c1 = _mm256_sub_epi16(_mm256_xor_si256(coupling_v, sg1), sg1);

    // E/I: load per-node ei[] (int8 -> widen to int16), apply as sign to
    // coupling ei is int8_t (+1/-1), pack 32 values = two __m128i = one __m256i
    // of bytes
    const auto *ei_ptr = reinterpret_cast<const __m128i *>(state.ei.data() + j);
    const auto *ei_ptr_1 =
        reinterpret_cast<const __m128i *>(state.ei.data() + j + 16);
    const __m256i ei0_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ei_ptr));
    const __m256i ei1_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ei_ptr_1));
    // c_ei = c * ei (sign flip for inhibitory)
    const __m256i c_ei0 = _mm256_sign_epi16(c0, ei0_16);
    const __m256i c_ei1 = _mm256_sign_epi16(c1, ei1_16);

    mb0 = _mm256_adds_epi16(mb0, c_ei0);
    mb1 = _mm256_adds_epi16(mb1, c_ei1);

    if (decay > 0) {
      const __m256i dec0 = _mm256_srai_epi16(mb0, decay);
      const __m256i dec1 = _mm256_srai_epi16(mb1, decay);
      mb0 = _mm256_sub_epi16(mb0, dec0);
      mb1 = _mm256_sub_epi16(mb1, dec1);
    }

    _mm256_store_si256(mag_ptr, mb0);
    _mm256_store_si256(mag_ptr_1, mb1);
    _mm256_store_si256(ph_ptr, pb0);
    _mm256_store_si256(ph_ptr_1, pb1);

    tick_energy += hsum_abs16(mb0) + hsum_abs16(mb1);
  }
#else
  for (std::size_t j = 0; j < kNodes; ++j) {
    const std::size_t next = (j + 1U) & (kNodes - 1U);
    // Kuramoto: each node rotates at delta + its own omega
    const auto p =
        static_cast<std::uint16_t>(state.ph[j] + delta + state.omega[j]);
    const auto pn =
        static_cast<std::uint16_t>(state.ph[next] + delta + state.omega[next]);
    const auto m = state.mag[j];
    const auto mn = state.mag[next];

    const auto pb = static_cast<std::uint16_t>(
        (static_cast<std::uint32_t>(p) * blend +
         static_cast<std::uint32_t>(pn) * (256 - blend)) >>
        8);
    auto mb = static_cast<int>(
        (static_cast<int>(m) * blend + static_cast<int>(mn) * (256 - blend)) >>
        8);
    const int sign = (static_cast<std::uint16_t>(pb + 16384U) >> 15) ? -1 : 1;
    // E/I: excitatory nodes attract, inhibitory nodes repel
    mb = clamp_i16(mb + sign * static_cast<int>(state.ei[j]) * coupling);
    if (decay > 0) {
      mb -= (mb >> decay);
    }

    state.mag[j] = static_cast<std::int16_t>(clamp_i16(mb));
    state.ph[j] = pb;
    tick_energy += std::abs(static_cast<int>(state.mag[j]));
  }
#endif

  obs.energy = tick_energy;

  // Coherence: only at epoch boundaries (every 128 ticks) — matches V3 original
  if (is_epoch) {
    std::int64_t tick_coherence = 0;
    for (std::size_t j = 0; j < kNodes; ++j) {
      const std::size_t next = (j + 1U) & (kNodes - 1U);
      tick_coherence += phase_distance(state.ph[j], state.ph[next]);
    }
    const auto normalized =
        std::clamp<std::int64_t>(65535 - (tick_coherence >> 5), 0, 65535);
    obs.coherence = normalized;
  }

  // Homeostatic plasticity: proportional negative feedback on coherence.
  // Keeps the lane near the critical band while genome mutation remains slow.
  if (is_epoch) {
    const auto coh8 = static_cast<std::int32_t>(obs.coherence >> 8);
    constexpr std::int32_t kCohTarget = 180;  // target coherence band center (~46080 raw)

    // Proportional error: positive when too chaotic, negative when too ordered
    const auto error = kCohTarget - coh8;  // >0 means need more coupling
    // Proportional response: step = error / 8, clamped to [-24..+24] per epoch
    const auto step = std::clamp<int>(error / 8, -24, 24);

    ctrl.coupling_adapt =
        static_cast<std::int16_t>(std::clamp<int>(ctrl.coupling_adapt + step, -200, 200));
    // Blend adaptation: when chaotic, increase blend (more diffusion to fight detuning)
    ctrl.blend_adapt =
        static_cast<std::int16_t>(std::clamp<int>(ctrl.blend_adapt + step / 2, -32, 56));
  }

  // nd[] and nd_popcount BEFORE fitness/mutation
  // P2: Adaptive threshold — scales to actual magnitude level instead of
  //     fixed 8000 which is 150x above average magnitude (~52).
  const auto avg_abs = static_cast<std::int32_t>(tick_energy / kNodes);
  const auto effective_threshold =
      static_cast<std::int16_t>(std::max<std::int32_t>(64, avg_abs * 4));
  std::uint64_t exc = 0;
  std::uint64_t inh = 0;
  for (int bit = 0; bit < 64; ++bit) {
    const auto idx = static_cast<std::size_t>((bit * 32) & (kNodes - 1));
    if (state.mag[idx] > effective_threshold)
      exc |= (1ULL << bit);
    if (state.mag[idx] < -effective_threshold)
      inh |= (1ULL << bit);
  }
  for (auto &word : state.nd) {
    word = exc_inh(word, exc, inh);
  }
  obs.nd_popcount = 0;
  for (const auto word : state.nd) {
    obs.nd_popcount += static_cast<std::int32_t>(std::popcount(word));
  }

  // Mutate AFTER nd/obs are current (correct fitness calculation)
  maybe_mutate(lane, tick_energy);

  // Update controller membrane_local
  if (enable_collective_) {
    std::uint32_t mem_sum = 0;
    for (std::size_t j = 0; j < kNodes; ++j) {
      mem_sum += membrane_[j];
    }
    ctrl.membrane_local = static_cast<std::uint16_t>(mem_sum / kNodes);
  } else {
    ctrl.membrane_local = 0;
  }

  // Decay recent_drive (stimulus memory fades)
  ctrl.recent_drive = (ctrl.recent_drive * 7) / 8;

  // Mirror controller state to observation for telemetry
  obs.fitness_ema = ctrl.fitness_ema;
  obs.recent_drive = ctrl.recent_drive;
  obs.stagnation_epochs = ctrl.stagnation_epochs;
  obs.membrane_local = ctrl.membrane_local;
  obs.attention_hits = ctrl.attention_hits;

  // Timing: only at epoch boundaries to avoid syscall overhead in hot-path
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

  // Fitness: multi-channel proprioceptive signal.
  // P1: Band-pass coherence — reward critical zone, not dead ground state
  const auto coh8 = static_cast<std::int32_t>(obs.coherence >> 8);
  constexpr std::int32_t kTargetCoherence = 192;
  const auto coherence_term = 255 - std::abs(coh8 - kTargetCoherence);

  // P3: Energy homeostasis — reward energy in viable band
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

  // P4: Drive-gated responsiveness — amplify energy response when stimulus
  // present
  const auto drive_response =
      (ctrl.recent_drive > 16) ? std::min(energy_delta * 4, 128) : 0;

  const auto my_fitness = population_term + energy_delta + energy_home +
                          coherence_term + drive_response;

  auto &genome = genomes_[lane];
  const auto previous_fitness = genome.fitness.load(std::memory_order_relaxed);
  genome.fitness.store(my_fitness, std::memory_order_relaxed);
  const auto current_best = genome.best_fitness.load(std::memory_order_relaxed);

  // Update fitness EMA (alpha = 1/8)
  ctrl.prev_fitness_ema = ctrl.fitness_ema;
  ctrl.fitness_ema = (7 * ctrl.fitness_ema + my_fitness) / 8;
  const auto slope = ctrl.fitness_ema - ctrl.prev_fitness_ema;

  // Stagnation tracking
  const bool improving = my_fitness > previous_fitness;
  if (improving) {
    ctrl.stagnation_epochs = 0;
  } else if (ctrl.stagnation_epochs < 255) {
    ctrl.stagnation_epochs += 1;
  }

  // Single exploration mechanism: stagnation-based
  const bool explore = ctrl.stagnation_epochs >= 2;
  const bool should_mutate =
      explore || ((state.rng & 0x1FU) == 0U &&
                  slope <= 0); // rare random only when NOT trending up

  if (should_mutate) {
    auto mutate_field = [&](std::atomic<std::int16_t> &field, const int min_v,
                            const int max_v, const int delta_mask,
                            const int delta_bias) {
      const auto current = field.load(std::memory_order_relaxed);
      xorshift32(state.rng);
      // Wider mutations when deeply stagnant
      const int stag_boost = (ctrl.stagnation_epochs > 8) ? 2 : 1;
      const auto raw_next =
          static_cast<int>(current) +
          static_cast<int>(((state.rng & delta_mask) - delta_bias) *
                           stag_boost);
      const auto next =
          static_cast<std::int16_t>(std::clamp(raw_next, min_v, max_v));
      field.store(next, std::memory_order_relaxed);
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
      break; // Kuramoto spread
    case 4:
      mutate_field(genome.decay, 0, 8, 0x03, 1);
      break;
    case 5:
      mutate_field(genome.inject_rate, 16, 256, 0x1F, 16);
      break;
    case 6:
      mutate_field(genome.ei_balance, 128, 240, 0x0F, 8);
      break; // E/I ratio
    default:
      mutate_field(genome.coupling, 0, 200, 0x0F, 8);
      break;
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

  // Injection: focused when collective, with effective_inject_rate
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

// ── Collective mechanisms ───────────────────────────────────────────

void Engine::compute_membrane() {
  // Disagreement score per node: how many lanes are in each half-period.
  // 0 = all agree, kThreads/2 = maximum split.
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
  // Only every kCrossoverEpochs epochs
  const auto epoch_num = global_tick_ >> 7; // divide by kEpochMask+1
  if ((epoch_num % kCrossoverEpochs) != 0)
    return;

  // P6: Find best and worst lanes by fitness only (not raw coherence)
  //     Coherence is now embedded in fitness via band-pass term.
  std::size_t best = 0, worst = 0;
  auto lane_score = [&](std::size_t lane) -> std::int64_t {
    return static_cast<std::int64_t>(
        genomes_[lane].fitness.load(std::memory_order_relaxed));
  };
  for (std::size_t lane = 1; lane < kThreads; ++lane) {
    if (lane_score(lane) > lane_score(best))
      best = lane;
    if (lane_score(lane) < lane_score(worst))
      worst = lane;
  }

  if (best == worst)
    return;
  // Only graft if worst is actually stagnating
  if (obs_[worst].stagnation_epochs < 4)
    return;

  // Conservative 25% graft — don't kill the worst, nudge it
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
  // 50% chance to inherit delta directly
  if ((data_[worst].rng & 1U) == 0U) {
    gw.delta.store(gb.delta.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
  }
  // Reset worst stagnation so it gets a fair chance
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
  const auto &genome = genomes_[lane];
  return GenomeSnapshot{
      .delta = genome.delta.load(std::memory_order_relaxed),
      .coupling = genome.coupling.load(std::memory_order_relaxed),
      .threshold = genome.threshold.load(std::memory_order_relaxed),
      .blend = genome.blend.load(std::memory_order_relaxed),
      .decay = genome.decay.load(std::memory_order_relaxed),
      .inject_rate = genome.inject_rate.load(std::memory_order_relaxed),
      .omega_width = genome.omega_width.load(std::memory_order_relaxed),
      .ei_balance = genome.ei_balance.load(std::memory_order_relaxed),
      .generation = genome.generation.load(std::memory_order_relaxed),
      .fitness = genome.fitness.load(std::memory_order_relaxed),
      .best_fitness = genome.best_fitness.load(std::memory_order_relaxed),
      .total_mutations = genome.total_mutations.load(std::memory_order_relaxed),
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

// ── Perception Layer (ported from xray_expert) ──

std::size_t Engine::read_magnitude(std::size_t lane, std::int16_t *out,
                                   std::size_t n) const {
  if (lane >= kThreads || !out)
    return 0;
  std::scoped_lock lock(mutex_);
  const auto count = std::min(n, kNodes);
  std::memcpy(out, data_[lane].mag.data(), count * sizeof(std::int16_t));
  return count;
}

std::size_t Engine::read_phase(std::size_t lane, std::uint16_t *out,
                               std::size_t n) const {
  if (lane >= kThreads || !out)
    return 0;
  std::scoped_lock lock(mutex_);
  const auto count = std::min(n, kNodes);
  std::memcpy(out, data_[lane].ph.data(), count * sizeof(std::uint16_t));
  return count;
}

// ── Hypothesis Probing Layer ──

void Engine::perturb(std::size_t lane, const std::int16_t *signal,
                     std::size_t n, std::int16_t gain_q8) {
  if (lane >= kThreads || !signal)
    return;
  std::scoped_lock lock(mutex_);
  const auto count = std::min(n, kNodes);
  for (std::size_t i = 0; i < count; ++i) {
    const int v = static_cast<int>(data_[lane].mag[i]) +
                  ((static_cast<int>(signal[i]) * gain_q8) >> 8);
    data_[lane].mag[i] = static_cast<std::int16_t>(clamp_i16(v));
  }
  // Drive accounting — perturb is the primary signal path in continuous mode
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
  if (from >= kThreads || to >= kThreads || from == to)
    return;
  std::scoped_lock lock(mutex_);
  // Clone ThreadState (mag, phase, nd, rng, tick_counter)
  data_[to] = data_[from];
  // Clone ControllerState (fitness_ema, drive, stagnation)
  ctrl_[to] = ctrl_[from];
  // Clone Observation (energy, coherence, pressure — all read-only metrics)
  obs_[to] = obs_[from];
  // Clone Genome (atomic-safe copy for probe isolation)
  // During probe, all lanes MUST start with identical genome parameters.
  // Without this, score differences come from lane evolution, not hypotheses.
  auto &src = genomes_[from];
  auto &dst = genomes_[to];
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

void Engine::clone_lane_to(Engine &target, const std::size_t from,
                           const std::size_t to) const {
  if (from >= kThreads || to >= kThreads)
    return;
  if (this == &target) {
    target.clone_lane(from, to);
    return;
  }

  std::scoped_lock lock(mutex_, target.mutex_);
  target.data_[to] = data_[from];
  target.ctrl_[to] = ctrl_[from];
  target.obs_[to] = obs_[from];

  const auto &src = genomes_[from];
  auto &dst = target.genomes_[to];
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

} // namespace antigravity::control::living
