/*********************************************************************
 * Tests for obstruction_polygons.hpp - 3D spherical half-plane
 * representation (SphericalPoint/AngularPolygon, ConvexCone,
 * ValidatedPolygon/validatePolygons, parsePolygonsFromString,
 * ObstructionFilter).
 *********************************************************************/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/logger.hpp"
#include "spatio_temporal_voxel_layer/obstruction_polygons.hpp"

namespace
{

// (azimuth, elevation) -> unit 3D direction, matching ConvexCone's own
// convention exactly (cos(el)*cos(az), cos(el)*sin(az), sin(el)), so tests
// can independently build/inspect directions without depending on any
// private state of the class under test.
geometry::Vec3D toDir(const geometry::SphericalPoint & p)
{
  const double cos_el = std::cos(p.elevation);
  return {cos_el * std::cos(p.azimuth), cos_el * std::sin(p.azimuth), std::sin(p.elevation)};
}

// Inverse of toDir: unit 3D direction -> (azimuth in [0, 2pi], elevation).
geometry::SphericalPoint toAngular(const geometry::Vec3D & d)
{
  double az = std::atan2(d.y, d.x);
  if (az < 0.0) {
    az += 2.0 * M_PI;
  }
  const double el = std::atan2(d.z, std::hypot(d.x, d.y));
  return {az, el};
}

geometry::Vec3D normalize(geometry::Vec3D v)
{
  const double n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return {v.x / n, v.y / n, v.z / n};
}

// Minimal fake satisfying ObstructionFilter::fromParams's duck-typed NodeT
// interface (has_parameter/declare_parameter/get_parameter), so fromParams
// can be exercised without a real rclcpp::Node.
class FakeParamNode
{
public:
  explicit FakeParamNode(std::string polygons_param) : polygons_param_(std::move(polygons_param)) {}

  bool has_parameter(const std::string &) const { return true; }
  void declare_parameter(const std::string &, const std::string &) {}
  void get_parameter(const std::string &, std::string & out) const { out = polygons_param_; }

private:
  std::string polygons_param_;
};

bool isObstructedAt(const geometry::ObstructionFilter & filter, const geometry::SphericalPoint & p)
{
  const geometry::Vec3D d = toDir(p);
  return filter.isObstructed(
    static_cast<float>(d.x), static_cast<float>(d.y), static_cast<float>(d.z));
}

}  // namespace

class ObstructionPolygonsTest : public ::testing::Test
{
protected:
  rclcpp::Logger logger_ = rclcpp::get_logger("test_obstruction_polygons");

  // A small square well clear of the azimuth wraparound and the poles, so
  // spherical distortion is small and the shape is unambiguously convex.
  geometry::AngularPolygon makeSquareCcw(double az_min, double el_min, double az_max, double el_max)
  {
    return {
      {az_min, el_min},
      {az_max, el_min},
      {az_max, el_max},
      {az_min, el_max},
    };
  }

  geometry::AngularPolygon makeSquareCw(double az_min, double el_min, double az_max, double el_max)
  {
    return {
      {az_min, el_min},
      {az_min, el_max},
      {az_max, el_max},
      {az_max, el_min},
    };
  }

  geometry::AngularPolygon makeTriangle()
  {
    return {
      {1.0, 0.0},
      {1.2, 0.0},
      {1.1, 0.1},
    };
  }

  geometry::AngularPolygon makeTriangleWithDuplicateVertex()
  {
    return {
      {1.0, -0.1},
      {1.0, -0.1},  // duplicate of the previous vertex -> zero-length edge
      {1.2, 0.1},
    };
  }

  // Square with one vertex pulled inward on the top edge, producing a
  // concave notch: this must never be accepted as a single convex
  // half-plane cone.
  geometry::AngularPolygon makeDentPentagon()
  {
    return {
      {1.0, -0.1}, {1.2, -0.1},
      {1.2, 0.1},  {1.1, 0.02},  // dent, pulled in from the top edge toward the interior
      {1.0, 0.1},
    };
  }
};

// ============================================================
// ConvexCone::fromAngularVertices tests
// ============================================================

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsFewerThanThreeVertices)
{
  EXPECT_FALSE(geometry::ConvexCone::fromAngularVertices({}).has_value());
  EXPECT_FALSE(geometry::ConvexCone::fromAngularVertices({{1.0, 0.0}}).has_value());
  EXPECT_FALSE(geometry::ConvexCone::fromAngularVertices({{1.0, 0.0}, {1.2, 0.0}}).has_value());
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsConvexSquareCcw)
{
  auto cone = geometry::ConvexCone::fromAngularVertices(makeSquareCcw(1.0, -0.1, 1.2, 0.1));
  ASSERT_TRUE(cone.has_value());
  EXPECT_EQ(cone->normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsConvexSquareCw)
{
  // Same four corners, reversed winding - exercises the centroid-based
  // auto-orientation-flip (the mechanism that replaced the old 2D version's
  // shoelace winding check).
  auto cone = geometry::ConvexCone::fromAngularVertices(makeSquareCw(1.0, -0.1, 1.2, 0.1));
  ASSERT_TRUE(cone.has_value());
  EXPECT_EQ(cone->normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ConvexConeNormalsPointInwardForAllVertices)
{
  auto poly = makeSquareCcw(1.0, -0.1, 1.2, 0.1);
  auto cone = geometry::ConvexCone::fromAngularVertices(poly);
  ASSERT_TRUE(cone.has_value());

  std::vector<geometry::Vec3D> dirs;
  for (const auto & v : poly) {
    dirs.push_back(toDir(v));
  }

  constexpr double kEps = 1e-9;
  for (const auto & normal : cone->normals()) {
    for (const auto & dir : dirs) {
      const double dot = normal.x * dir.x + normal.y * dir.y + normal.z * dir.z;
      EXPECT_GE(dot, -kEps);
    }
  }
}

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsDegenerateEdge)
{
  auto cone = geometry::ConvexCone::fromAngularVertices(makeTriangleWithDuplicateVertex());
  EXPECT_FALSE(cone.has_value());
}

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsNonConvexPolygon)
{
  auto cone = geometry::ConvexCone::fromAngularVertices(makeDentPentagon());
  EXPECT_FALSE(cone.has_value());
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsMinimalTriangle)
{
  auto cone = geometry::ConvexCone::fromAngularVertices(makeTriangle());
  ASSERT_TRUE(cone.has_value());
  EXPECT_EQ(cone->normals().size(), 3u);
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsVertexOnEdgeBoundary)
{
  auto square = makeSquareCcw(1.0, -0.1, 1.2, 0.1);

  // Insert a vertex exactly on the great-circle boundary of the edge between
  // square[2] and square[3]: the normalized sum of two unit directions
  // always lies exactly in the plane spanned by them, i.e. exactly on that
  // edge's half-plane boundary. This is a robust, non-flaky way to hit the
  // kConvexEps tolerance - flat (azimuth, elevation) collinearity is NOT
  // exactly preserved under the spherical projection, so a flat midpoint
  // would not reliably land on the boundary.
  const geometry::Vec3D d2 = toDir(square[2]);
  const geometry::Vec3D d3 = toDir(square[3]);
  const geometry::Vec3D mid = normalize({d2.x + d3.x, d2.y + d3.y, d2.z + d3.z});

  geometry::AngularPolygon with_boundary_vertex = {
    square[0], square[1], square[2], toAngular(mid), square[3]};

  auto cone = geometry::ConvexCone::fromAngularVertices(with_boundary_vertex);
  EXPECT_TRUE(cone.has_value());
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsPolygonAtHighElevation)
{
  // Elevation up to ~80 degrees - far enough from the pole to stay
  // well-defined, close enough to stress the cos(el) projection.
  auto cone = geometry::ConvexCone::fromAngularVertices(makeSquareCcw(0.5, 1.2, 1.0, 1.4));
  EXPECT_TRUE(cone.has_value());
}

// ============================================================
// angularArea tests
// ============================================================

TEST_F(ObstructionPolygonsTest, AngularAreaComputesShoelaceAreaForSquare)
{
  auto poly = makeSquareCcw(1.0, -0.1, 1.2, 0.1);
  EXPECT_NEAR(geometry::angularArea(poly), 0.04, 1e-9);
}

TEST_F(ObstructionPolygonsTest, AngularAreaIndependentOfWindingDirection)
{
  auto ccw = makeSquareCcw(1.0, -0.1, 1.2, 0.1);
  auto cw = makeSquareCw(1.0, -0.1, 1.2, 0.1);
  EXPECT_NEAR(geometry::angularArea(ccw), geometry::angularArea(cw), 1e-9);
}

// ============================================================
// validatePolygons tests
// ============================================================

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsTooFewVertices)
{
  geometry::AngularPolygon poly = {{1.0, 0.0}, {1.2, 0.0}};
  auto valid = geometry::validatePolygons({poly}, logger_);
  EXPECT_TRUE(valid.empty());
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsNaNVertices)
{
  geometry::AngularPolygon poly = {{1.0, 0.0}, {1.2, NAN}, {1.1, 0.1}};
  auto valid = geometry::validatePolygons({poly}, logger_);
  EXPECT_TRUE(valid.empty());
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsInfVertices)
{
  geometry::AngularPolygon poly = {{1.0, 0.0}, {INFINITY, 0.1}, {1.1, 0.1}};
  auto valid = geometry::validatePolygons({poly}, logger_);
  EXPECT_TRUE(valid.empty());
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsAzimuthBelowZero)
{
  geometry::AngularPolygon poly = {{-0.1, 0.0}, {1.0, 0.0}, {0.5, 0.1}};
  auto valid = geometry::validatePolygons({poly}, logger_);
  EXPECT_TRUE(valid.empty());
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsAzimuthAbove2Pi)
{
  geometry::AngularPolygon poly = {{2.0 * M_PI + 0.1, 0.0}, {1.0, 0.0}, {0.5, 0.1}};
  auto valid = geometry::validatePolygons({poly}, logger_);
  EXPECT_TRUE(valid.empty());
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsNonConvexPolygon)
{
  auto valid = geometry::validatePolygons({makeDentPentagon()}, logger_);
  EXPECT_TRUE(valid.empty());
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsGoodPolygon)
{
  auto valid = geometry::validatePolygons({makeSquareCcw(1.0, -0.1, 1.2, 0.1)}, logger_);
  ASSERT_EQ(valid.size(), 1u);
  EXPECT_NEAR(valid[0].angular_area, 0.04, 1e-9);
  EXPECT_EQ(valid[0].cone.normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsFiltersMultipleKeepsOnlyValid)
{
  auto good = makeSquareCcw(1.0, -0.1, 1.2, 0.1);
  geometry::AngularPolygon bad = {{-0.1, 0.0}, {1.0, 0.0}, {0.5, 0.1}};  // negative azimuth

  auto valid = geometry::validatePolygons({good, bad}, logger_);
  ASSERT_EQ(valid.size(), 1u);
  EXPECT_NEAR(valid[0].angular_area, 0.04, 1e-9);
}

// ============================================================
// parsePolygonsFromString tests
// ============================================================

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringParsesValidString)
{
  std::string input = "[[1.0,-0.1, 1.2,-0.1, 1.1,0.1], [2.0,-0.1, 2.2,-0.1, 2.2,0.1, 2.0,0.1]]";
  auto polys = geometry::parsePolygonsFromString(input, logger_);
  ASSERT_EQ(polys.size(), 2u);
  EXPECT_EQ(polys[0].size(), 3u);
  EXPECT_EQ(polys[1].size(), 4u);
  EXPECT_DOUBLE_EQ(polys[0][0].azimuth, 1.0);
  EXPECT_DOUBLE_EQ(polys[1][3].elevation, 0.1);
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringEmptyStringReturnsEmpty)
{
  auto polys = geometry::parsePolygonsFromString("", logger_);
  EXPECT_TRUE(polys.empty());
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringInvalidFormatReturnsEmpty)
{
  auto polys = geometry::parsePolygonsFromString("not a polygon", logger_);
  EXPECT_TRUE(polys.empty());
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringTooFewValuesSkipped)
{
  // Only 2 values (need >= 6 for a triangle).
  auto polys = geometry::parsePolygonsFromString("[[1.0, 2.0]]", logger_);
  EXPECT_TRUE(polys.empty());
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringOddValueCountSkipped)
{
  // 7 values - not divisible by 2.
  auto polys = geometry::parsePolygonsFromString("[[1.0,0.0, 1.2,0.0, 1.1,0.1, 0.5]]", logger_);
  EXPECT_TRUE(polys.empty());
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringHandlesExtraWhitespace)
{
  std::string input = "  [  [ 1.0 , -0.1 , 1.2 , -0.1 , 1.1 , 0.1 ]  ]  ";
  auto polys = geometry::parsePolygonsFromString(input, logger_);
  ASSERT_EQ(polys.size(), 1u);
  EXPECT_EQ(polys[0].size(), 3u);
}

// ============================================================
// ObstructionFilter tests
// ============================================================

TEST_F(ObstructionPolygonsTest, ObstructionFilterEmptyFilterReportsEmpty)
{
  geometry::ObstructionFilter filter;
  EXPECT_TRUE(filter.empty());
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterIsObstructedFalseWhenEmpty)
{
  geometry::ObstructionFilter filter;
  EXPECT_FALSE(filter.isObstructed(1.0f, 0.0f, 0.0f));
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterIsObstructedTrueInsideConeFalseOutside)
{
  FakeParamNode node("[[1.0,-0.1, 1.2,-0.1, 1.2,0.1, 1.0,0.1]]");
  auto filter = geometry::ObstructionFilter::fromParams(&node, "test", logger_);
  ASSERT_NE(filter, nullptr);

  EXPECT_TRUE(isObstructedAt(*filter, {1.1, 0.0}));   // inside the polygon
  EXPECT_FALSE(isObstructedAt(*filter, {0.0, 0.0}));  // well outside
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterIsObstructedTrueInAnyOfMultiplePolygons)
{
  FakeParamNode node(
    "[[1.0,-0.1, 1.2,-0.1, 1.2,0.1, 1.0,0.1], "
    "[2.0,-0.1, 2.2,-0.1, 2.2,0.1, 2.0,0.1]]");
  auto filter = geometry::ObstructionFilter::fromParams(&node, "test", logger_);
  ASSERT_NE(filter, nullptr);

  EXPECT_TRUE(isObstructedAt(*filter, {1.1, 0.0}));   // inside first polygon
  EXPECT_TRUE(isObstructedAt(*filter, {2.1, 0.0}));   // inside second polygon
  EXPECT_FALSE(isObstructedAt(*filter, {1.6, 0.0}));  // between the two
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterFromParamsParsesValidParameterString)
{
  FakeParamNode node("[[1.0,-0.1, 1.2,-0.1, 1.2,0.1, 1.0,0.1]]");
  auto filter = geometry::ObstructionFilter::fromParams(&node, "test", logger_);
  ASSERT_NE(filter, nullptr);
  EXPECT_FALSE(filter->empty());
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterFromParamsReturnsNullWhenParamEmpty)
{
  FakeParamNode node("");
  auto filter = geometry::ObstructionFilter::fromParams(&node, "test", logger_);
  EXPECT_EQ(filter, nullptr);
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterFromParamsReturnsNullWhenAllPolygonsInvalid)
{
  // Azimuth out of [0, 2pi] range -> fails validation -> no valid polygons.
  FakeParamNode node("[[-0.1,-0.1, 1.0,-0.1, 0.5,0.1]]");
  auto filter = geometry::ObstructionFilter::fromParams(&node, "test", logger_);
  EXPECT_EQ(filter, nullptr);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
