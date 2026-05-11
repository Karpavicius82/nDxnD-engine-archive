#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace antigravity::control::living::one_layer {

inline constexpr std::size_t kNodes = 2048;
inline constexpr std::size_t kVectorLanes = 16;
inline constexpr std::size_t kBlocks = kNodes / kVectorLanes;
inline constexpr std::size_t kPorts = 4;

struct alignas(64) BlockTopology2048 {
  // One-layer topology: each SIMD block owns four live graph ports.
  // No external plan object is used by the tick path.
  alignas(64) std::array<std::int16_t, kBlocks> off0{};
  alignas(64) std::array<std::int16_t, kBlocks> off1{};
  alignas(64) std::array<std::int16_t, kBlocks> off2{};
  alignas(64) std::array<std::int16_t, kBlocks> off3{};
  alignas(64) std::array<std::int16_t, kBlocks> w0{};
  alignas(64) std::array<std::int16_t, kBlocks> w1{};
  alignas(64) std::array<std::int16_t, kBlocks> w2{};
  alignas(64) std::array<std::int16_t, kBlocks> w3{};
  alignas(64) std::array<std::uint16_t, kBlocks> flags{};
};

struct alignas(64) Lane2048 {
  alignas(64) std::array<std::int16_t, kNodes> mag{};
  alignas(64) std::array<std::uint16_t, kNodes> ph{};
  alignas(64) std::array<std::int16_t, kNodes> omega{};
  alignas(64) std::array<std::int8_t, kNodes> ei{};

  alignas(64) std::array<std::int16_t, kNodes> g_coupling{};
  alignas(64) std::array<std::int16_t, kNodes> g_blend{};
  alignas(64) std::array<std::int16_t, kNodes> g_decay{};
  alignas(64) std::array<std::int16_t, kNodes> stress{};
  alignas(64) std::array<std::int16_t, kNodes> novelty{};

  BlockTopology2048 topology{};

  alignas(64) std::array<std::int16_t, kNodes> scratch_mag{};
  alignas(64) std::array<std::uint16_t, kNodes> scratch_ph{};

  std::uint32_t rng{0xC0FFEEu};
  std::uint64_t tick{0};
};

inline std::uint32_t xorshift32(std::uint32_t &rng) noexcept {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}

inline std::int16_t sat16(const int v) noexcept {
  return static_cast<std::int16_t>(std::clamp(v, -32768, 32767));
}

inline std::int16_t clamp_offset_for_block(std::size_t block,
                                           std::int32_t offset) noexcept {
  const auto base = static_cast<std::int32_t>(block * kVectorLanes);
  const auto lo = base + offset;
  const auto hi = lo + static_cast<std::int32_t>(kVectorLanes - 1);
  if (lo >= 0 && hi < static_cast<std::int32_t>(kNodes)) {
    return static_cast<std::int16_t>(offset);
  }
  if (base - offset >= 0 &&
      base - offset + static_cast<std::int32_t>(kVectorLanes - 1) <
          static_cast<std::int32_t>(kNodes)) {
    return static_cast<std::int16_t>(-offset);
  }
  return 0;
}

inline void normalize_block_weights(BlockTopology2048 &t,
                                    std::size_t block) noexcept {
  int weights[kPorts] = {std::max<int>(1, t.w0[block]),
                         std::max<int>(1, t.w1[block]),
                         std::max<int>(1, t.w2[block]),
                         std::max<int>(1, t.w3[block])};
  const int sum = weights[0] + weights[1] + weights[2] + weights[3];
  t.w0[block] = static_cast<std::int16_t>((weights[0] * 32768) / sum);
  t.w1[block] = static_cast<std::int16_t>((weights[1] * 32768) / sum);
  t.w2[block] = static_cast<std::int16_t>((weights[2] * 32768) / sum);
  const int acc = static_cast<int>(t.w0[block]) + static_cast<int>(t.w1[block]) +
                  static_cast<int>(t.w2[block]);
  t.w3[block] = static_cast<std::int16_t>(32768 - acc);
}

inline void repair_topology_block(Lane2048 &lane, std::size_t block) noexcept {
  auto &t = lane.topology;
  t.off0[block] = clamp_offset_for_block(block, t.off0[block]);
  t.off1[block] = clamp_offset_for_block(block, t.off1[block]);
  t.off2[block] = clamp_offset_for_block(block, t.off2[block]);
  t.off3[block] = clamp_offset_for_block(block, t.off3[block]);
  normalize_block_weights(t, block);
}

inline void seed_ndxnD_topology(Lane2048 &lane, std::uint8_t rank,
                                const std::uint16_t dims_in[4],
                                std::uint8_t radius,
                                bool /*wrap*/) noexcept {
  rank = static_cast<std::uint8_t>(std::clamp<int>(rank, 1, 4));
  radius = static_cast<std::uint8_t>(std::clamp<int>(radius, 1, 4));

  std::uint16_t dims[4]{1, 1, 1, 1};
  std::uint16_t stride[4]{1, 1, 1, 1};
  std::size_t total = 1;
  for (int axis = 0; axis < 4; ++axis) {
    dims[axis] = axis < rank ? std::max<std::uint16_t>(1, dims_in[axis]) : 1;
    if (axis < rank && total * dims[axis] > kNodes) {
      dims[axis] = static_cast<std::uint16_t>(std::max<std::size_t>(1, kNodes / total));
    }
    if (axis < rank) {
      total *= dims[axis];
    }
  }
  stride[0] = 1;
  for (int axis = 1; axis < 4; ++axis) {
    stride[axis] = static_cast<std::uint16_t>(stride[axis - 1] * dims[axis - 1]);
  }

  for (std::size_t b = 0; b < kBlocks; ++b) {
    std::int16_t *offs[kPorts] = {&lane.topology.off0[b], &lane.topology.off1[b],
                                  &lane.topology.off2[b], &lane.topology.off3[b]};
    std::int16_t *ws[kPorts] = {&lane.topology.w0[b], &lane.topology.w1[b],
                                &lane.topology.w2[b], &lane.topology.w3[b]};
    for (std::size_t p = 0; p < kPorts; ++p) {
      const int axis = static_cast<int>((b + p / 2) % rank);
      const int dir = (p & 1U) == 0 ? -1 : 1;
      const int r = 1 + static_cast<int>((b + p) % radius);
      *offs[p] = clamp_offset_for_block(
          b, dir * r * static_cast<int>(stride[axis]));
      *ws[p] = 8192;
    }
    lane.topology.flags[b] = static_cast<std::uint16_t>(rank | (radius << 8));
    normalize_block_weights(lane.topology, b);
  }
}

inline void mutate_topology_block(Lane2048 &lane, std::size_t block) noexcept {
  auto &rng = lane.rng;
  auto &t = lane.topology;
  const auto r = xorshift32(rng);
  std::int16_t *offsets[kPorts] = {&t.off0[block], &t.off1[block],
                                   &t.off2[block], &t.off3[block]};
  std::int16_t *weights[kPorts] = {&t.w0[block], &t.w1[block], &t.w2[block],
                                   &t.w3[block]};
  const auto port = static_cast<std::size_t>(r & 3U);
  if ((r & 4U) == 0) {
    const int delta_blocks = static_cast<int>((r >> 8) & 7U) - 3;
    *offsets[port] = sat16(static_cast<int>(*offsets[port]) +
                           delta_blocks * static_cast<int>(kVectorLanes));
  } else {
    *weights[port] = sat16(static_cast<int>(*weights[port]) +
                           static_cast<int>((r >> 8) & 1023U) - 512);
  }
  repair_topology_block(lane, block);
}

inline void copy_to_scratch(Lane2048 &lane) noexcept {
  std::memcpy(lane.scratch_mag.data(), lane.mag.data(),
              kNodes * sizeof(std::int16_t));
  std::memcpy(lane.scratch_ph.data(), lane.ph.data(),
              kNodes * sizeof(std::uint16_t));
}

inline std::int64_t advance_lane2048_scalar(Lane2048 &lane,
                                            std::int16_t global_delta = 17) noexcept {
  copy_to_scratch(lane);
  std::int64_t energy = 0;
  for (std::size_t b = 0; b < kBlocks; ++b) {
    const auto base = b * kVectorLanes;
    const int off[kPorts] = {lane.topology.off0[b], lane.topology.off1[b],
                             lane.topology.off2[b], lane.topology.off3[b]};
    const int w[kPorts] = {lane.topology.w0[b], lane.topology.w1[b],
                           lane.topology.w2[b], lane.topology.w3[b]};
    for (std::size_t lane_i = 0; lane_i < kVectorLanes; ++lane_i) {
      const auto i = base + lane_i;
      int acc_m = 0;
      int acc_p = 0;
      for (std::size_t p = 0; p < kPorts; ++p) {
        const auto n = static_cast<std::size_t>(static_cast<int>(i) + off[p]);
        acc_m += static_cast<int>(lane.scratch_mag[n]) * w[p];
        acc_p += static_cast<int>(static_cast<std::uint16_t>(
                     lane.scratch_ph[n] + global_delta + lane.omega[n])) *
                 w[p];
      }
      const int mn = acc_m >> 15;
      const auto pn = static_cast<std::uint16_t>(acc_p >> 15);
      const int blend = std::clamp<int>(lane.g_blend[i], 0, 256);
      const int m = lane.scratch_mag[i];
      const auto ph = static_cast<std::uint16_t>(lane.scratch_ph[i] +
                                                global_delta + lane.omega[i]);
      int mb = ((m * blend) + (mn * (256 - blend))) >> 8;
      const auto pb = static_cast<std::uint16_t>(
          ((static_cast<int>(ph) * blend) +
           (static_cast<int>(pn) * (256 - blend))) >> 8);
      const int sign = (static_cast<std::uint16_t>(pb + 16384U) >> 15) ? -1 : 1;
      mb = sat16(mb + sign * static_cast<int>(lane.ei[i]) *
                         static_cast<int>(lane.g_coupling[i]));
      const int decay = std::clamp<int>(lane.g_decay[i], 0, 8);
      if (decay > 0) {
        mb -= (mb >> decay);
      }
      lane.mag[i] = static_cast<std::int16_t>(mb);
      lane.ph[i] = pb;
      energy += std::abs(mb);
    }
  }
  ++lane.tick;
  return energy;
}

#if defined(__AVX2__)
namespace detail {
inline __m256i loadu256(const void *p) noexcept {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
}
inline void i16_to_i32(__m256i v, __m256i &lo, __m256i &hi) noexcept {
  const __m128i vlo = _mm256_castsi256_si128(v);
  const __m128i vhi = _mm256_extracti128_si256(v, 1);
  lo = _mm256_cvtepi16_epi32(vlo);
  hi = _mm256_cvtepi16_epi32(vhi);
}
inline void u16_to_i32(__m256i v, __m256i &lo, __m256i &hi) noexcept {
  const __m128i vlo = _mm256_castsi256_si128(v);
  const __m128i vhi = _mm256_extracti128_si256(v, 1);
  lo = _mm256_cvtepu16_epi32(vlo);
  hi = _mm256_cvtepu16_epi32(vhi);
}
inline __m256i pack_i32_to_i16(__m256i lo, __m256i hi) noexcept {
  return _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xD8);
}
inline __m256i pack_i32_to_u16(__m256i lo, __m256i hi) noexcept {
  return _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8);
}
inline std::int64_t hsum_abs16(__m256i v) noexcept {
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
} // namespace detail

inline std::int64_t advance_lane2048_avx2(Lane2048 &lane,
                                          std::int16_t global_delta = 17) noexcept {
  copy_to_scratch(lane);
  std::int64_t energy = 0;
  const __m256i zero = _mm256_setzero_si256();
  const __m256i delta_v = _mm256_set1_epi16(global_delta);
  const __m256i q8_v = _mm256_set1_epi32(256);
  const __m256i quarter_v = _mm256_set1_epi16(16384);

  for (std::size_t b = 0; b < kBlocks; ++b) {
    const auto i = b * kVectorLanes;
    const __m256i m = detail::loadu256(lane.scratch_mag.data() + i);
    const __m256i p0 = detail::loadu256(lane.scratch_ph.data() + i);
    const __m256i omega = detail::loadu256(lane.omega.data() + i);
    const __m256i p = _mm256_add_epi16(p0, _mm256_add_epi16(delta_v, omega));

    __m256i acc_m_lo = _mm256_setzero_si256();
    __m256i acc_m_hi = _mm256_setzero_si256();
    __m256i acc_p_lo = _mm256_setzero_si256();
    __m256i acc_p_hi = _mm256_setzero_si256();

    const std::int16_t offsets[kPorts] = {lane.topology.off0[b], lane.topology.off1[b],
                                          lane.topology.off2[b], lane.topology.off3[b]};
    const std::int16_t weights[kPorts] = {lane.topology.w0[b], lane.topology.w1[b],
                                          lane.topology.w2[b], lane.topology.w3[b]};
    for (std::size_t port = 0; port < kPorts; ++port) {
      const auto base = static_cast<std::ptrdiff_t>(i) + offsets[port];
      const __m256i nm = detail::loadu256(lane.scratch_mag.data() + base);
      const __m256i np0 = detail::loadu256(lane.scratch_ph.data() + base);
      const __m256i no = detail::loadu256(lane.omega.data() + base);
      const __m256i np = _mm256_add_epi16(np0, _mm256_add_epi16(delta_v, no));
      const __m256i w = _mm256_set1_epi32(static_cast<int>(weights[port]));

      __m256i nm_lo, nm_hi, np_lo, np_hi;
      detail::i16_to_i32(nm, nm_lo, nm_hi);
      detail::u16_to_i32(np, np_lo, np_hi);
      acc_m_lo = _mm256_add_epi32(acc_m_lo, _mm256_mullo_epi32(nm_lo, w));
      acc_m_hi = _mm256_add_epi32(acc_m_hi, _mm256_mullo_epi32(nm_hi, w));
      acc_p_lo = _mm256_add_epi32(acc_p_lo, _mm256_mullo_epi32(np_lo, w));
      acc_p_hi = _mm256_add_epi32(acc_p_hi, _mm256_mullo_epi32(np_hi, w));
    }

    const __m256i mn = detail::pack_i32_to_i16(_mm256_srai_epi32(acc_m_lo, 15),
                                               _mm256_srai_epi32(acc_m_hi, 15));
    const __m256i pn = detail::pack_i32_to_u16(_mm256_srli_epi32(acc_p_lo, 15),
                                               _mm256_srli_epi32(acc_p_hi, 15));

    const __m256i blend16 = detail::loadu256(lane.g_blend.data() + i);
    __m256i blend_lo, blend_hi, inv_blend_lo, inv_blend_hi;
    detail::i16_to_i32(blend16, blend_lo, blend_hi);
    blend_lo = _mm256_max_epi32(zero, _mm256_min_epi32(q8_v, blend_lo));
    blend_hi = _mm256_max_epi32(zero, _mm256_min_epi32(q8_v, blend_hi));
    inv_blend_lo = _mm256_sub_epi32(q8_v, blend_lo);
    inv_blend_hi = _mm256_sub_epi32(q8_v, blend_hi);

    __m256i m_lo, m_hi, mn_lo, mn_hi, p_lo, p_hi, pn_lo, pn_hi;
    detail::i16_to_i32(m, m_lo, m_hi);
    detail::i16_to_i32(mn, mn_lo, mn_hi);
    detail::u16_to_i32(p, p_lo, p_hi);
    detail::u16_to_i32(pn, pn_lo, pn_hi);

    __m256i mb_lo = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(m_lo, blend_lo),
                         _mm256_mullo_epi32(mn_lo, inv_blend_lo)),
        8);
    __m256i mb_hi = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(m_hi, blend_hi),
                         _mm256_mullo_epi32(mn_hi, inv_blend_hi)),
        8);
    __m256i pb_lo = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(p_lo, blend_lo),
                         _mm256_mullo_epi32(pn_lo, inv_blend_lo)),
        8);
    __m256i pb_hi = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(p_hi, blend_hi),
                         _mm256_mullo_epi32(pn_hi, inv_blend_hi)),
        8);

    __m256i mb = detail::pack_i32_to_i16(mb_lo, mb_hi);
    const __m256i pb = detail::pack_i32_to_u16(pb_lo, pb_hi);

    const __m256i gc = detail::loadu256(lane.g_coupling.data() + i);
    const __m256i sg = _mm256_srai_epi16(_mm256_add_epi16(pb, quarter_v), 15);
    const __m256i signed_c = _mm256_sub_epi16(_mm256_xor_si256(gc, sg), sg);
    const __m256i ei = _mm256_cvtepi8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(lane.ei.data() + i)));
    mb = _mm256_adds_epi16(mb, _mm256_sign_epi16(signed_c, ei));

    const __m256i gd16 = detail::loadu256(lane.g_decay.data() + i);
    __m256i gd_lo, gd_hi;
    detail::i16_to_i32(gd16, gd_lo, gd_hi);
    gd_lo = _mm256_max_epi32(zero, _mm256_min_epi32(_mm256_set1_epi32(8), gd_lo));
    gd_hi = _mm256_max_epi32(zero, _mm256_min_epi32(_mm256_set1_epi32(8), gd_hi));
    detail::i16_to_i32(mb, mb_lo, mb_hi);
    mb_lo = _mm256_sub_epi32(mb_lo, _mm256_srav_epi32(mb_lo, gd_lo));
    mb_hi = _mm256_sub_epi32(mb_hi, _mm256_srav_epi32(mb_hi, gd_hi));
    mb = detail::pack_i32_to_i16(mb_lo, mb_hi);

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(lane.mag.data() + i), mb);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(lane.ph.data() + i), pb);
    energy += detail::hsum_abs16(mb);
  }

  ++lane.tick;
  return energy;
}
#endif

inline std::int64_t advance_lane2048(Lane2048 &lane,
                                     std::int16_t global_delta = 17) noexcept {
#if defined(__AVX2__)
  return advance_lane2048_avx2(lane, global_delta);
#else
  return advance_lane2048_scalar(lane, global_delta);
#endif
}

} // namespace antigravity::control::living::one_layer
