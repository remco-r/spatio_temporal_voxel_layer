#ifndef SPATIO_TEMPORAL_VOXEL_LAYER__FRUSTUM_MODELS__VERTICAL_FOV_CHECK_HPP_
#define SPATIO_TEMPORAL_VOXEL_LAYER__FRUSTUM_MODELS__VERTICAL_FOV_CHECK_HPP_

#include <cmath>

namespace geometry
{
namespace detail
{

inline bool IsInsideVerticalFov(
  const double z, const double radial_distance_squared, const double vFOVPadding,
  const double tan_vFOVhalf_squared, const double vFOVOffset,
  const double tan_vFOVOffset)
{
  const double z_rel = vFOVOffset == 0.0 ?
    z :
    z - std::sqrt(radial_distance_squared) * tan_vFOVOffset;
  const double v_padded = std::fabs(z_rel) + vFOVPadding;

  return v_padded * v_padded <=
         radial_distance_squared * tan_vFOVhalf_squared;
}

}  // namespace detail
}  // namespace geometry

#endif  // SPATIO_TEMPORAL_VOXEL_LAYER__FRUSTUM_MODELS__VERTICAL_FOV_CHECK_HPP_
