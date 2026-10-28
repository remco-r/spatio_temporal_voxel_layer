#include <benchmark/benchmark.h>

#include "vertical_fov_check.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

constexpr std::size_t kPointCount = 100000;
constexpr double kVerticalFov = 30.0 * 3.14159265358979323846 / 180.0;
constexpr double kPadding = 0.02;

struct TestPoint
{
  double z;
  double radial_distance_squared;
};

std::vector<TestPoint> MakePoints(const double offset)
{
  const double tan_half_fov = std::tan(kVerticalFov / 2.0);
  const double tan_offset = std::tan(offset);
  std::vector<TestPoint> points;
  points.reserve(kPointCount);

  for (std::size_t i = 0; i < kPointCount; ++i) {
    const double range = 1.0 + static_cast<double>(i % 1000) * 0.01;
    const double range_squared = range * range;
    const bool outside = i % 5 == 0;
    const double deviation = (outside ? 1.5 : 0.25) * range * tan_half_fov;
    const double signed_deviation = i % 2 == 0 ? deviation : -deviation;
    points.push_back({range * tan_offset + signed_deviation, range_squared});
  }

  return points;
}

void LegacyCheck(benchmark::State & state)
{
  const double offset = static_cast<double>(state.range(0)) *
    3.14159265358979323846 / 180.0;
  const double tan_half_fov = std::tan(kVerticalFov / 2.0);
  const double tan_half_fov_squared = tan_half_fov * tan_half_fov;
  const std::vector<TestPoint> points = MakePoints(offset);

  for (auto _ : state) {
    std::size_t inside_count = 0;
    for (const TestPoint & point : points) {
      const double z_rel =
        point.z - std::sqrt(point.radial_distance_squared) * std::tan(offset);
      const double v_padded = std::fabs(z_rel) + kPadding;
      inside_count +=
        v_padded * v_padded / point.radial_distance_squared <=
        tan_half_fov_squared;
    }
    benchmark::DoNotOptimize(inside_count);
  }

  state.SetItemsProcessed(
    static_cast<std::int64_t>(state.iterations()) *
    static_cast<std::int64_t>(points.size()));
}

void CachedCheck(benchmark::State & state)
{
  const double offset = static_cast<double>(state.range(0)) *
    3.14159265358979323846 / 180.0;
  const double tan_half_fov = std::tan(kVerticalFov / 2.0);
  const double tan_half_fov_squared = tan_half_fov * tan_half_fov;
  const double tan_offset = std::tan(offset);
  const std::vector<TestPoint> points = MakePoints(offset);

  for (auto _ : state) {
    std::size_t inside_count = 0;
    for (const TestPoint & point : points) {
      inside_count += geometry::detail::IsInsideVerticalFov(
        point.z, point.radial_distance_squared, kPadding,
        tan_half_fov_squared, offset, tan_offset);
    }
    benchmark::DoNotOptimize(inside_count);
  }

  state.SetItemsProcessed(
    static_cast<std::int64_t>(state.iterations()) *
    static_cast<std::int64_t>(points.size()));
}

BENCHMARK(LegacyCheck)->Arg(0)->Arg(5)->Unit(benchmark::kNanosecond);
BENCHMARK(CachedCheck)->Arg(0)->Arg(5)->Unit(benchmark::kNanosecond);

}  // namespace

BENCHMARK_MAIN();
