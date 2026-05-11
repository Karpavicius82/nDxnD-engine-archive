#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace antigravity::control::living {

inline constexpr std::size_t kNodes = 2048;
inline constexpr std::size_t kThreads = 8;
inline constexpr std::uint64_t kEpochMask = 0x7f;
inline constexpr std::size_t kCrossoverEpochs = 4;  // crossover every N epochs

struct alignas(64) Genome {
    std::atomic<std::int16_t> delta{17};
    std::atomic<std::int16_t> coupling{64};
    std::atomic<std::int16_t> threshold{8000};
    std::atomic<std::int16_t> blend{192};
    std::atomic<std::int16_t> decay{0};
    std::atomic<std::int16_t> inject_rate{64};
    std::atomic<std::int16_t> omega_width{8};     // Kuramoto: natural frequency spread [1..32]
    std::atomic<std::int16_t> ei_balance{204};     // E/I ratio threshold [128..240] (~80% excitatory)
    std::atomic<std::int16_t> d_rank{1};           // active lattice rank [1..4]
    std::atomic<std::int16_t> kernel_radius{1};    // nD stencil radius [1..4]
    std::atomic<std::int16_t> boundary_mode{1};    // 1=wrap/toroidal, 0=clamp
    std::atomic<std::int16_t> d_dim0{2048};
    std::atomic<std::int16_t> d_dim1{1};
    std::atomic<std::int16_t> d_dim2{1};
    std::atomic<std::int16_t> d_dim3{1};
    std::atomic<std::uint32_t> generation{0};
    std::atomic<std::int32_t> fitness{0};
    std::atomic<std::int32_t> best_fitness{0};
    std::atomic<std::uint32_t> total_mutations{0};
};

struct GenomeSnapshot {
    std::int16_t delta{17};
    std::int16_t coupling{64};
    std::int16_t threshold{8000};
    std::int16_t blend{192};
    std::int16_t decay{0};
    std::int16_t inject_rate{64};
    std::int16_t omega_width{8};
    std::int16_t ei_balance{204};
    std::int16_t d_rank{1};
    std::int16_t kernel_radius{1};
    std::int16_t boundary_mode{1};
    std::int16_t d_dim0{2048};
    std::int16_t d_dim1{1};
    std::int16_t d_dim2{1};
    std::int16_t d_dim3{1};
    std::uint32_t generation{0};
    std::int32_t fitness{0};
    std::int32_t best_fitness{0};
    std::uint32_t total_mutations{0};
};

struct alignas(64) Observation {
    std::int64_t energy{0};
    std::int64_t coherence{0};
    std::int32_t nd_popcount{0};
    std::int32_t pressure{0};      // backward-compatible EMA in microseconds
    std::int64_t tick_ns{0};       // last lane tick duration in nanoseconds
    std::int64_t tick_ns_ema{0};   // EMA of lane tick duration in nanoseconds
    std::uint32_t mutations{0};
    std::uint32_t improvements{0};
    // Controller telemetry (mirrored from ControllerState)
    std::int32_t fitness_ema{0};
    std::int32_t recent_drive{0};       // stimulus strength (0=idle)
    std::uint16_t stagnation_epochs{0};
    std::uint16_t membrane_local{0};
    std::uint32_t attention_hits{0};
};

// Atomic snapshot — observation + genome read under single mutex lock.
// Prevents race when background tick thread is active.
struct LaneSnapshot {
    Observation observation;
    GenomeSnapshot genome;
};

// Internal per-lane runtime state — reacts fast, does not touch Genome
struct alignas(64) ControllerState {
    std::int32_t fitness_ema{0};
    std::int32_t prev_fitness_ema{0};
    std::int32_t prev_energy_bucket{0};
    std::int32_t recent_drive{0};        // EMA of inject signal strength
    std::uint16_t stagnation_epochs{0};
    std::uint16_t membrane_local{0};
    std::int16_t coupling_adapt{0};      // homeostatic coupling adjustment [-128..+128]
    std::int16_t blend_adapt{0};          // homeostatic blend adjustment [-32..+48]
    std::uint32_t attention_hits{0};
};

struct alignas(64) NdShape {
    std::uint8_t rank{1};
    std::uint16_t dims[4]{2048, 1, 1, 1};
    std::uint16_t stride[4]{1, 2048, 2048, 2048};
    std::uint8_t wrap_mask{0x0F};
};

struct alignas(64) ThreadState {
    std::array<std::int16_t, kNodes> mag{};
    std::array<std::uint16_t, kNodes> ph{};
    std::array<std::int16_t, kNodes> omega{};  // per-node natural frequency detuning (Kuramoto)
    std::array<std::int8_t, kNodes> ei{};      // per-node E/I identity: +1 excitatory, -1 inhibitory
    NdShape shape{};
    alignas(64) std::array<std::int16_t, kNodes> scratch{};
    alignas(64) std::array<std::uint16_t, kNodes> scratch_phase{};
    std::array<std::uint64_t, 4> nd{};
    std::uint32_t rng{0};
    std::uint64_t tick_counter{0};
};

class Engine {
public:
    Engine();

    void initialize(std::uint64_t seed);
    void inject(std::size_t lane, const std::int16_t* signal, std::size_t n);
    void tick(std::uint64_t ticks);
    void tick_lanes(std::uint64_t ticks, std::uint64_t lane_mask);
    void tick_lane(std::size_t lane, std::uint64_t ticks);

    void set_collective(bool enabled) {
        enable_collective_.store(enabled, std::memory_order_release);
    }
    [[nodiscard]] bool collective_enabled() const {
        return enable_collective_.load(std::memory_order_acquire);
    }

    [[nodiscard]] Observation observation(std::size_t lane) const;
    [[nodiscard]] GenomeSnapshot genome(std::size_t lane) const;
    [[nodiscard]] LaneSnapshot snapshot(std::size_t lane) const;
    [[nodiscard]] std::uint32_t total_mutations() const;

    // ── Perception Layer (ported from xray_expert) ──
    // Read back per-node magnitude/phase state after ticks.
    std::size_t read_magnitude(std::size_t lane, std::int16_t* out, std::size_t n) const;
    std::size_t read_phase(std::size_t lane, std::uint16_t* out, std::size_t n) const;

    // ── Hypothesis Probing Layer (NEW) ──
    // Additive perturbation: mag[i] += (signal[i] * gain_q8) >> 8
    // Unlike inject() which OVERWRITES, this BLENDS with existing state.
    void perturb(std::size_t lane, const std::int16_t* signal, std::size_t n,
                 std::int16_t gain_q8 = 64);

    // Clone FULL lane state for probe isolation:
    // ThreadState + ControllerState + Observation + Genome (atomic-safe)
    void clone_lane(std::size_t from, std::size_t to);
    void clone_lane_to(Engine& target, std::size_t from, std::size_t to) const;

private:
    void tick_locked(std::uint64_t ticks, std::uint64_t lane_mask);
    void advance_lane(std::size_t lane);
    void maybe_mutate(std::size_t lane, std::int64_t tick_energy);
    void compute_membrane();
    void maybe_crossover();

    std::atomic<bool> enable_collective_{true};  // P5: proprioceptive default — enables membrane + attention
    std::uint64_t global_tick_{0};
    std::array<std::uint8_t, kNodes> membrane_{};  // disagreement field

    mutable std::mutex mutex_;
    std::array<Genome, kThreads> genomes_{};
    std::array<Observation, kThreads> obs_{};
    std::array<ThreadState, kThreads> data_{};
    std::array<ControllerState, kThreads> ctrl_{};
};

} // namespace antigravity::control::living
