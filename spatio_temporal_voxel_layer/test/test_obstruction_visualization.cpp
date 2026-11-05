/*
 * Software License Agreement
 *
 *  Copyright (c) 2018, Simbe Robotics, Inc.
 *  Copyright (c) 2021, Samsung Research America
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Simbe Robotics, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "spatio_temporal_voxel_layer/obstruction_visualization.hpp"
#include "gtest/gtest.h"

namespace stvl = spatio_temporal_voxel_layer;
namespace obstruction_viz = stvl::obstruction_visualization;

namespace
{

constexpr double kPi = 3.14159265358979323846;
const geometry::AngularPolygon kTriangle{{0.0, 0.0}, {kPi / 2.0, 0.0}, {kPi / 2.0, kPi / 4.0}};

double pointRadius(const geometry_msgs::msg::Point &point)
{
  return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

} // namespace

TEST(ObstructionVisualization, SamplesClosedBoundaryOnRequestedRadius)
{
  constexpr double range = 2.5;
  const auto points = obstruction_viz::sampleObstructionBoundary(kTriangle, range);

  ASSERT_EQ(points.size(), 37U);
  EXPECT_NEAR(points.front().x, range, 1e-12);
  EXPECT_NEAR(points.front().y, 0.0, 1e-12);
  EXPECT_NEAR(points.front().z, 0.0, 1e-12);
  EXPECT_NEAR(points[12].x, 0.0, 1e-12);
  EXPECT_NEAR(points[12].y, range, 1e-12);
  EXPECT_NEAR(points[12].z, 0.0, 1e-12);

  for (const auto &point : points) {
    EXPECT_NEAR(pointRadius(point), range, 1e-12);
  }
  EXPECT_DOUBLE_EQ(points.front().x, points.back().x);
  EXPECT_DOUBLE_EQ(points.front().y, points.back().y);
  EXPECT_DOUBLE_EQ(points.front().z, points.back().z);

  // The midpoint of the first edge follows its spherical arc, not a flat chord.
  EXPECT_NEAR(points[6].x, range / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(points[6].y, range / std::sqrt(2.0), 1e-12);
}

TEST(ObstructionVisualization, EmptyPolygonProducesNoBoundaryPoints)
{
  EXPECT_TRUE(obstruction_viz::sampleObstructionBoundary({}, 1.0).empty());
}

TEST(ObstructionVisualization, BuildsOuterAndInnerRangeMarkers)
{
  const auto markers = obstruction_viz::buildObstructionMarkers(
      "[[0,0, 1.5707963267948966,0, 1.5707963267948966,0.7853981633974483]]", {1.25},
      "sensor_frame", "front_lidar", 5.0);

  ASSERT_EQ(markers.size(), 2U);
  EXPECT_EQ(markers[0].header.frame_id, "sensor_frame");
  EXPECT_EQ(markers[0].ns, "front_lidar");
  EXPECT_EQ(markers[0].id, 0);
  EXPECT_EQ(markers[1].id, 1);
  for (const auto &marker : markers) {
    EXPECT_EQ(marker.type, visualization_msgs::msg::Marker::LINE_STRIP);
    EXPECT_EQ(marker.action, visualization_msgs::msg::Marker::ADD);
    EXPECT_TRUE(marker.frame_locked);
    EXPECT_DOUBLE_EQ(marker.pose.orientation.w, 1.0);
    EXPECT_FLOAT_EQ(marker.scale.x, 0.02F);
    EXPECT_FLOAT_EQ(marker.color.r, 1.0F);
    EXPECT_FLOAT_EQ(marker.color.g, 0.55F);
    ASSERT_EQ(marker.points.size(), 37U);
    EXPECT_DOUBLE_EQ(marker.points.front().x, marker.points.back().x);
    EXPECT_DOUBLE_EQ(marker.points.front().y, marker.points.back().y);
    EXPECT_DOUBLE_EQ(marker.points.front().z, marker.points.back().z);
  }
  EXPECT_FLOAT_EQ(markers[0].color.a, 1.0F);
  EXPECT_FLOAT_EQ(markers[1].color.a, 0.35F);
  EXPECT_NEAR(pointRadius(markers[0].points.front()), 5.0, 1e-12);
  EXPECT_NEAR(pointRadius(markers[1].points.front()), 1.25, 1e-12);
}

TEST(ObstructionVisualization, UsesDefaultOuterOnlyForMissingOrNonPositiveMinRange)
{
  const std::string polygon = "[[0,0, 1,0, 1,0.5]]";

  const auto missing_range =
      obstruction_viz::buildObstructionMarkers(polygon, {}, "frame", "source", 3.0);
  ASSERT_EQ(missing_range.size(), 1U);
  EXPECT_EQ(missing_range[0].id, 0);
  EXPECT_FLOAT_EQ(missing_range[0].color.a, 1.0F);
  EXPECT_NEAR(pointRadius(missing_range[0].points.front()), 3.0, 1e-12);

  const auto non_positive_range =
      obstruction_viz::buildObstructionMarkers(polygon, {-1.0}, "frame", "source", 3.0);
  ASSERT_EQ(non_positive_range.size(), 1U);
  EXPECT_EQ(non_positive_range[0].id, 0);
  EXPECT_FLOAT_EQ(non_positive_range[0].color.a, 1.0F);
}

TEST(ObstructionVisualization, HandlesMultiplePolygonsWithStableMarkerIds)
{
  const auto markers = obstruction_viz::buildObstructionMarkers(
      "[[0,0, 1,0, 1,0.5], [2,0, 2.5,0, 2.5,0.5]]", {1.0, 0.0}, "frame", "source", 3.0);

  ASSERT_EQ(markers.size(), 3U);
  EXPECT_EQ(markers[0].id, 0);
  EXPECT_EQ(markers[1].id, 1);
  EXPECT_EQ(markers[2].id, 2);
  EXPECT_FLOAT_EQ(markers[0].color.a, 1.0F);
  EXPECT_FLOAT_EQ(markers[1].color.a, 0.35F);
  EXPECT_FLOAT_EQ(markers[2].color.a, 1.0F);
}

TEST(ObstructionVisualization, EmptyParametersOrFrameProducesNoMarkers)
{
  EXPECT_TRUE(obstruction_viz::buildObstructionMarkers("", {}, "frame", "source", 3.0).empty());
  EXPECT_TRUE(
      obstruction_viz::buildObstructionMarkers("not a polygon", {}, "", "source", 3.0).empty());
}

TEST(ObstructionVisualization, MalformedNonEmptyPolygonIsReported)
{
  EXPECT_THROW(obstruction_viz::buildObstructionMarkers("[[0,0, 1,0]]", {}, "frame", "source", 3.0),
               std::runtime_error);
}
