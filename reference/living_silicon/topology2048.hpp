#pragma once

#include "living_silicon.hpp"

#include <array>
#include <cstdint>

namespace antigravity::control::living {

// Single-layer nDxnD topology embedded as lane-local vector execution data.
//
// This is intentionally not a compiler-style plan object.  The topology is the
// executable geometry of the 2048 field itself: one SIMD block == 16 nodes, and
// every block carries its own fan-in offsets, Q15 weights, and flags.  The tick
// kernel consumes this directly; epoch/mutation code changes this structure
// directly.  No per-node coordinate math is needed on the hot path.
inline constexpr std::size_t kTopologyFanIn = 4;
inline constexpr std::uint16_t kTopoFlagLive = 1u << 0;
inline constexpr std::uint16_t kTopoFlagBoundaryCompressed = 1u << 1;
inline constexpr std::uint16_t kTopoFlagMutated = 1u << 2;

struct alignas(64) Topology2048 {
    std::uint8_t rank{1};
    std::uint8_t radius{1};
    std::uint16_t active_nodes{2048};
    std::uint16_t active_blocks{kVectorBlocks};

    // block_offset[f][b] is added to the base node index of SIMD block b.
    // A single offset must be safe for all 16 lanes in that block.  Boundary
    // blocks encode clamp/wrap policy here, not in a separate scalar executor.
    std::array<std::array<std::int16_t, kVectorBlocks>, kTopologyFanIn> block_offset{};
    std::array<std::array<std::int16_t, kVectorBlocks>, kTopologyFanIn> block_weight_q15{};
    std::array<std::uint16_t, kVectorBlocks> block_flags{};
};

// Build executable lane topology directly from NdShape.  This is not a hot-path
// planner; it is the canonical initialization of the living topology field.
Topology2048 make_topology2048(const NdShape& shape, std::int16_t radius);

// Mutate the executable topology itself.  This is the one-layer replacement for
// rebuilding an external NdVectorKernel plan at every epoch.
void mutate_topology2048(Topology2048& topology, std::uint32_t& rng, std::uint16_t heat_q8);

// Scalar reference executor with the same block-local topology semantics.
std::int64_t advance_topology2048_scalar(ThreadState& state,
                                         const Topology2048& topology,
                                         std::int16_t delta,
                                         std::int16_t coupling,
                                         std::int16_t blend,
                                         std::int16_t decay);

#if defined(__AVX2__)
// Single hot-path vector kernel.  Every full block uses the same AVX2 shape; the
// nD geometry is already encoded in Topology2048::block_offset/weight.
std::int64_t advance_topology2048_avx2(ThreadState& state,
                                       const Topology2048& topology,
                                       std::int16_t delta,
                                       std::int16_t coupling,
                                       std::int16_t blend,
                                       std::int16_t decay);
#endif

} // namespace antigravity::control::living
