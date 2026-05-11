#include "topology2048.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace antigravity::control::living {
namespace {

std::uint32_t topo_xorshift32(std::uint32_t& rng) noexcept {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

std::size_t topo_node_count(const NdShape& shape) noexcept {
    std::size_t total = 1;
    const auto rank = std::clamp<int>(shape.rank, 1, 4);
    for (int axis = 0; axis < rank; ++axis) {
        total *= std::max<std::uint16_t>(shape.dims[axis], 1);
    }
    return std::min<std::size_t>(total, kNodes);
}

int topo_clamp_i16(const int value) noexcept {
    return std::clamp(value, -32768, 32767);
}

bool block_offset_is_safe(const std::size_t block,
                          const std::size_t active_nodes,
                          const std::int16_t offset) noexcept {
    const auto begin = static_cast<std::ptrdiff_t>(block * kVectorLanes);
    const auto end = static_cast<std::ptrdiff_t>(block * kVectorLanes + kVectorLanes - 1);
    const auto first = begin + static_cast<std::ptrdiff_t>(offset);
    const auto last = end + static_cast<std::ptrdiff_t>(offset);
    return first >= 0 && last < static_cast<std::ptrdiff_t>(active_nodes);
}

std::int16_t safe_or_self_offset(const std::size_t block,
                                 const std::size_t active_nodes,
                                 const std::int16_t offset,
                                 bool& compressed) noexcept {
    if (block_offset_is_safe(block, active_nodes, offset)) {
        return offset;
    }
    compressed = true;
    return 0;
}

void set_block_fan_in(Topology2048& topology,
                      const std::size_t block,
                      const std::array<std::int16_t, kTopologyFanIn>& offsets) noexcept {
    bool compressed = false;
    int live = 0;
    for (std::size_t fan = 0; fan < kTopologyFanIn; ++fan) {
        const auto safe_offset = safe_or_self_offset(block, topology.active_nodes, offsets[fan], compressed);
        topology.block_offset[fan][block] = safe_offset;
        if (safe_offset != 0) {
            ++live;
        }
    }

    if (live == 0) {
        for (std::size_t fan = 0; fan < kTopologyFanIn; ++fan) {
            topology.block_weight_q15[fan][block] = 0;
        }
        topology.block_flags[block] = static_cast<std::uint16_t>(kTopoFlagLive | kTopoFlagBoundaryCompressed);
        return;
    }

    const int base = 32768 / live;
    int remainder = 32768 - base * live;
    for (std::size_t fan = 0; fan < kTopologyFanIn; ++fan) {
        if (topology.block_offset[fan][block] == 0) {
            topology.block_weight_q15[fan][block] = 0;
            continue;
        }
        topology.block_weight_q15[fan][block] = static_cast<std::int16_t>(base + (remainder > 0 ? 1 : 0));
        if (remainder > 0) {
            --remainder;
        }
    }

    topology.block_flags[block] = kTopoFlagLive;
    if (compressed) {
        topology.block_flags[block] |= kTopoFlagBoundaryCompressed;
    }
}

std::array<std::int16_t, kTopologyFanIn> choose_axis_offsets(const NdShape& shape,
                                                             const std::uint8_t rank,
                                                             const std::uint8_t radius) noexcept {
    std::array<std::int16_t, kTopologyFanIn> offsets{};
    const auto r = static_cast<int>(std::max<std::uint8_t>(radius, 1));

    // fan0/fan1: fastest axis. This preserves the 1D ring specialization in a
    // form the generic AVX2 kernel can consume.
    offsets[0] = static_cast<std::int16_t>(-r * static_cast<int>(shape.stride[0]));
    offsets[1] = static_cast<std::int16_t>( r * static_cast<int>(shape.stride[0]));

    // fan2/fan3: next active axes. For rank 2 this is the Y axis; for rank 3/4
    // it intentionally keeps the fan-in fixed and evolves which two axes matter.
    const int axis2 = rank > 1 ? 1 : 0;
    const int axis3 = rank > 2 ? 2 : axis2;
    offsets[2] = static_cast<std::int16_t>(-r * static_cast<int>(shape.stride[axis2]));
    offsets[3] = static_cast<std::int16_t>( r * static_cast<int>(shape.stride[axis3]));
    return offsets;
}

#if defined(__AVX2__)
__m256i loadu256(const void* ptr) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
}

void sign_extend_i16_to_i32(const __m256i v, __m256i& lo, __m256i& hi) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sign = _mm256_cmpgt_epi16(zero, v);
    lo = _mm256_unpacklo_epi16(v, sign);
    hi = _mm256_unpackhi_epi16(v, sign);
}

void zero_extend_u16_to_i32(const __m256i v, __m256i& lo, __m256i& hi) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    lo = _mm256_unpacklo_epi16(v, zero);
    hi = _mm256_unpackhi_epi16(v, zero);
}

std::int64_t hsum_abs16_local(const __m256i v) noexcept {
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

Topology2048 make_topology2048(const NdShape& shape, const std::int16_t radius) {
    Topology2048 topology{};
    topology.rank = static_cast<std::uint8_t>(std::clamp<int>(shape.rank, 1, 4));
    topology.radius = static_cast<std::uint8_t>(std::clamp<int>(radius, 1, 4));
    topology.active_nodes = static_cast<std::uint16_t>(topo_node_count(shape));
    topology.active_blocks = static_cast<std::uint16_t>((topology.active_nodes + kVectorLanes - 1) / kVectorLanes);

    const auto offsets = choose_axis_offsets(shape, topology.rank, topology.radius);
    for (std::size_t block = 0; block < topology.active_blocks; ++block) {
        set_block_fan_in(topology, block, offsets);
    }
    return topology;
}

void mutate_topology2048(Topology2048& topology, std::uint32_t& rng, const std::uint16_t heat_q8) {
    const auto heat = std::min<std::uint16_t>(heat_q8, 255);
    for (std::size_t block = 0; block < topology.active_blocks; ++block) {
        if ((topo_xorshift32(rng) & 0xffU) >= heat) {
            continue;
        }
        const auto fan = topo_xorshift32(rng) % kTopologyFanIn;
        const int dir = (topo_xorshift32(rng) & 1U) ? 1 : -1;
        const int step = 1 << static_cast<int>(topo_xorshift32(rng) & 3U);
        const auto mutated = static_cast<std::int16_t>(topology.block_offset[fan][block] + dir * step);
        if (block_offset_is_safe(block, topology.active_nodes, mutated)) {
            topology.block_offset[fan][block] = mutated;
            topology.block_flags[block] |= kTopoFlagMutated;
        }

        int live = 0;
        for (std::size_t f = 0; f < kTopologyFanIn; ++f) {
            live += topology.block_offset[f][block] != 0 ? 1 : 0;
        }
        const int base = live > 0 ? 32768 / live : 0;
        int remainder = live > 0 ? 32768 - base * live : 0;
        for (std::size_t f = 0; f < kTopologyFanIn; ++f) {
            if (topology.block_offset[f][block] == 0) {
                topology.block_weight_q15[f][block] = 0;
            } else {
                topology.block_weight_q15[f][block] = static_cast<std::int16_t>(base + (remainder > 0 ? 1 : 0));
                if (remainder > 0) {
                    --remainder;
                }
            }
        }
    }
}

std::int64_t advance_topology2048_scalar(ThreadState& state,
                                         const Topology2048& topology,
                                         const std::int16_t delta,
                                         const std::int16_t coupling,
                                         const std::int16_t blend,
                                         const std::int16_t decay) {
    const auto active_nodes = static_cast<std::size_t>(topology.active_nodes);
    std::copy_n(state.mag.begin(), active_nodes, state.scratch.begin());
    std::copy_n(state.ph.begin(), active_nodes, state.scratch_phase.begin());

    std::int64_t energy = 0;
    for (std::size_t block = 0; block < topology.active_blocks; ++block) {
        const auto begin = block * kVectorLanes;
        const auto end = std::min<std::size_t>(begin + kVectorLanes, active_nodes);
        for (std::size_t i = begin; i < end; ++i) {
            std::int32_t acc_mag = 0;
            std::int32_t acc_phase = 0;
            for (std::size_t fan = 0; fan < kTopologyFanIn; ++fan) {
                const auto base = static_cast<std::ptrdiff_t>(i) + topology.block_offset[fan][block];
                const auto n = static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(base, 0, static_cast<std::ptrdiff_t>(active_nodes - 1)));
                const int w = topology.block_weight_q15[fan][block];
                acc_mag += static_cast<int>(state.scratch[n]) * w;
                acc_phase += static_cast<int>(static_cast<std::uint16_t>(state.scratch_phase[n] + delta + state.omega[n])) * w;
            }

            const auto m = state.scratch[i];
            const auto p = static_cast<std::uint16_t>(state.scratch_phase[i] + delta + state.omega[i]);
            const auto mn = static_cast<std::int16_t>(acc_mag >> 15);
            const auto pn = static_cast<std::uint16_t>(static_cast<std::uint32_t>(acc_phase) >> 15);
            const auto pb = static_cast<std::uint16_t>((static_cast<std::uint32_t>(p) * blend + static_cast<std::uint32_t>(pn) * (256 - blend)) >> 8);
            auto mb = static_cast<int>((static_cast<int>(m) * blend + static_cast<int>(mn) * (256 - blend)) >> 8);
            const int sign = (static_cast<std::uint16_t>(pb + 16384U) >> 15) ? -1 : 1;
            mb = topo_clamp_i16(mb + sign * static_cast<int>(state.ei[i]) * coupling);
            if (decay > 0) {
                mb -= (mb >> decay);
            }
            state.mag[i] = static_cast<std::int16_t>(topo_clamp_i16(mb));
            state.ph[i] = pb;
            energy += std::abs(static_cast<int>(state.mag[i]));
        }
    }
    return energy;
}

#if defined(__AVX2__)
std::int64_t advance_topology2048_avx2(ThreadState& state,
                                       const Topology2048& topology,
                                       const std::int16_t delta,
                                       const std::int16_t coupling,
                                       const std::int16_t blend,
                                       const std::int16_t decay) {
    const auto active_nodes = static_cast<std::size_t>(topology.active_nodes);
    std::copy_n(state.mag.begin(), active_nodes, state.scratch.begin());
    std::copy_n(state.ph.begin(), active_nodes, state.scratch_phase.begin());

    std::int64_t energy = 0;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i delta_v = _mm256_set1_epi16(delta);
    const __m256i coupling_v = _mm256_set1_epi16(coupling);
    const __m256i blend32_v = _mm256_set1_epi32(static_cast<int>(blend));
    const __m256i inv_blend32_v = _mm256_set1_epi32(static_cast<int>(256 - blend));
    const __m256i quarter_v = _mm256_set1_epi16(16384);

    for (std::size_t block = 0; block < topology.active_blocks; ++block) {
        const auto i = block * kVectorLanes;
        if (i + kVectorLanes > active_nodes) {
            break;
        }

        const __m256i m = loadu256(state.scratch.data() + i);
        __m256i p = loadu256(state.scratch_phase.data() + i);
        const __m256i omega = loadu256(state.omega.data() + i);
        p = _mm256_add_epi16(p, _mm256_add_epi16(delta_v, omega));

        __m256i acc_mag_lo = _mm256_setzero_si256();
        __m256i acc_mag_hi = _mm256_setzero_si256();
        __m256i acc_phase_lo = _mm256_setzero_si256();
        __m256i acc_phase_hi = _mm256_setzero_si256();

        for (std::size_t fan = 0; fan < kTopologyFanIn; ++fan) {
            const auto offset = static_cast<std::ptrdiff_t>(topology.block_offset[fan][block]);
            const __m256i weight_v = _mm256_set1_epi32(static_cast<int>(topology.block_weight_q15[fan][block]));
            const auto base = static_cast<std::ptrdiff_t>(i) + offset;
            const __m256i nm = loadu256(state.scratch.data() + base);
            __m256i np = loadu256(state.scratch_phase.data() + base);
            const __m256i nomega = loadu256(state.omega.data() + base);
            np = _mm256_add_epi16(np, _mm256_add_epi16(delta_v, nomega));

            __m256i nm_lo, nm_hi, np_lo, np_hi;
            sign_extend_i16_to_i32(nm, nm_lo, nm_hi);
            zero_extend_u16_to_i32(np, np_lo, np_hi);
            acc_mag_lo = _mm256_add_epi32(acc_mag_lo, _mm256_mullo_epi32(nm_lo, weight_v));
            acc_mag_hi = _mm256_add_epi32(acc_mag_hi, _mm256_mullo_epi32(nm_hi, weight_v));
            acc_phase_lo = _mm256_add_epi32(acc_phase_lo, _mm256_mullo_epi32(np_lo, weight_v));
            acc_phase_hi = _mm256_add_epi32(acc_phase_hi, _mm256_mullo_epi32(np_hi, weight_v));
        }

        const __m256i mn = _mm256_packs_epi32(_mm256_srai_epi32(acc_mag_lo, 15), _mm256_srai_epi32(acc_mag_hi, 15));
        const __m256i pn = _mm256_packus_epi32(_mm256_srli_epi32(acc_phase_lo, 15), _mm256_srli_epi32(acc_phase_hi, 15));

        __m256i p_lo, p_hi, pn_lo, pn_hi;
        zero_extend_u16_to_i32(p, p_lo, p_hi);
        zero_extend_u16_to_i32(pn, pn_lo, pn_hi);
        const __m256i pb_lo = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p_lo, blend32_v), _mm256_mullo_epi32(pn_lo, inv_blend32_v)), 8);
        const __m256i pb_hi = _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p_hi, blend32_v), _mm256_mullo_epi32(pn_hi, inv_blend32_v)), 8);
        const __m256i pb = _mm256_packus_epi32(pb_lo, pb_hi);

        __m256i m_lo, m_hi, mn_lo, mn_hi;
        sign_extend_i16_to_i32(m, m_lo, m_hi);
        sign_extend_i16_to_i32(mn, mn_lo, mn_hi);
        const __m256i mb_lo = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(m_lo, blend32_v), _mm256_mullo_epi32(mn_lo, inv_blend32_v)), 8);
        const __m256i mb_hi = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(m_hi, blend32_v), _mm256_mullo_epi32(mn_hi, inv_blend32_v)), 8);
        __m256i mb = _mm256_packs_epi32(mb_lo, mb_hi);

        const __m256i sg = _mm256_srai_epi16(_mm256_add_epi16(pb, quarter_v), 15);
        const __m256i c = _mm256_sub_epi16(_mm256_xor_si256(coupling_v, sg), sg);
        const __m256i ei = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(state.ei.data() + i)));
        mb = _mm256_adds_epi16(mb, _mm256_sign_epi16(c, ei));
        if (decay > 0) {
            mb = _mm256_sub_epi16(mb, _mm256_srai_epi16(mb, decay));
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(state.mag.data() + i), mb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(state.ph.data() + i), pb);
        energy += hsum_abs16_local(mb);
    }

    return energy;
}
#endif

} // namespace antigravity::control::living
