/*********************************************************************
 * Tests for obstruction_polygons.hpp - 3D spherical half-plane
 * representation (SphericalPoint/AngularPolygon, ConvexCone,
 * ValidatedPolygon/validatePolygons, parsePolygonsFromString,
 * ObstructionFilter).
 *
 * Parsing and validation are fail-fast: every rejection path throws
 * std::runtime_error rather than warning and skipping, so a misconfigured
 * obstruction_polygons parameter can never silently degrade into a
 * partially-masked frustum.
 *********************************************************************/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

// Assert that fn() throws std::runtime_error and that the message names the
// reason. The messages are the only diagnostic a misconfigured deployment
// gets - a throw carrying the wrong reason is a real defect, so the tests
// pin the message, not just the exception type.
template <typename Callable>
void expectThrowsSaying(Callable && fn, const std::string & needle)
{
  try {
    fn();
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
      << "expected message containing \"" << needle << "\", got \"" << e.what() << "\"";
    return;
  } catch (const std::exception & e) {
    FAIL() << "expected std::runtime_error, got a different exception: " << e.what();
  }
  FAIL() << "expected std::runtime_error containing \"" << needle << "\", nothing was thrown";
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

// Half-plane containment against a bare cone, for cases that need to inspect a
// single cone's geometry before it is flattened into an ObstructionFilter.
bool isInsideCone(const geometry::ConvexCone & cone, const geometry::SphericalPoint & p)
{
  const geometry::Vec3D d = toDir(p);
  for (const auto & n : cone.normals()) {
    if (n.x * d.x + n.y * d.y + n.z * d.z < 0.0) {
      return false;
    }
  }
  return true;
}

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

  // A square with one corner duplicated: unlike the degenerate triangle above
  // this still has full angular area, so it reaches the degenerate-edge check
  // in ConvexCone instead of being caught earlier by the area check.
  geometry::AngularPolygon makeSquareWithDuplicateVertex()
  {
    // vertices 2 and 3 are identical -> zero-length edge between them
    return {
      {1.0, -0.1}, {1.2, -0.1}, {1.2, 0.1}, {1.2, 0.1}, {1.0, 0.1},
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
  expectThrowsSaying(
    [] { geometry::ConvexCone::fromAngularVertices({}); }, "0 vertices, needs at least 3");
  expectThrowsSaying(
    [] { geometry::ConvexCone::fromAngularVertices({{1.0, 0.0}}); },
    "1 vertices, needs at least 3");
  expectThrowsSaying(
    [] { geometry::ConvexCone::fromAngularVertices({{1.0, 0.0}, {1.2, 0.0}}); },
    "2 vertices, needs at least 3");
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsConvexSquareCcw)
{
  const auto cone = geometry::ConvexCone::fromAngularVertices(makeSquareCcw(1.0, -0.1, 1.2, 0.1));
  EXPECT_EQ(cone.normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsConvexSquareCw)
{
  // Same four corners, reversed winding - exercises the centroid-based
  // auto-orientation-flip (the mechanism that replaced the old 2D version's
  // shoelace winding check).
  const auto cone = geometry::ConvexCone::fromAngularVertices(makeSquareCw(1.0, -0.1, 1.2, 0.1));
  EXPECT_EQ(cone.normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ConvexConeNormalsPointInwardForAllVertices)
{
  auto poly = makeSquareCcw(1.0, -0.1, 1.2, 0.1);
  const auto cone = geometry::ConvexCone::fromAngularVertices(poly);

  std::vector<geometry::Vec3D> dirs;
  for (const auto & v : poly) {
    dirs.push_back(toDir(v));
  }

  constexpr double kEps = 1e-9;
  for (const auto & normal : cone.normals()) {
    for (const auto & dir : dirs) {
      const double dot = normal.x * dir.x + normal.y * dir.y + normal.z * dir.z;
      EXPECT_GE(dot, -kEps);
    }
  }
}

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsDegenerateEdge)
{
  auto poly = makeTriangleWithDuplicateVertex();
  expectThrowsSaying(
    [&poly] { geometry::ConvexCone::fromAngularVertices(poly); },
    "degenerate edge between vertices 0 and 1");
}

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsNonConvexPolygon)
{
  auto poly = makeDentPentagon();
  expectThrowsSaying(
    [&poly] { geometry::ConvexCone::fromAngularVertices(poly); }, "is not spherically convex");
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsMinimalTriangle)
{
  const auto cone = geometry::ConvexCone::fromAngularVertices(makeTriangle());
  EXPECT_EQ(cone.normals().size(), 3u);
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

  EXPECT_NO_THROW(geometry::ConvexCone::fromAngularVertices(with_boundary_vertex));
}

TEST_F(ObstructionPolygonsTest, ConvexConeAcceptsPolygonAtHighElevation)
{
  // Elevation up to ~80 degrees - far enough from the pole to stay
  // well-defined, close enough to stress the cos(el) projection.
  EXPECT_NO_THROW(geometry::ConvexCone::fromAngularVertices(makeSquareCcw(0.5, 1.2, 1.0, 1.4)));
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
  expectThrowsSaying(
    [&poly] { geometry::validatePolygons({poly}); },
    "obstruction polygon 0 has 2 vertices, needs at least 3");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsNaNVertices)
{
  geometry::AngularPolygon poly = {{1.0, 0.0}, {1.2, NAN}, {1.1, 0.1}};
  expectThrowsSaying(
    [&poly] { geometry::validatePolygons({poly}); }, "obstruction polygon 0 has a NaN or inf");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsInfVertices)
{
  geometry::AngularPolygon poly = {{1.0, 0.0}, {INFINITY, 0.1}, {1.1, 0.1}};
  expectThrowsSaying(
    [&poly] { geometry::validatePolygons({poly}); }, "obstruction polygon 0 has a NaN or inf");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsAzimuthBelowZero)
{
  geometry::AngularPolygon poly = {{-0.1, 0.0}, {1.0, 0.0}, {0.5, 0.1}};
  expectThrowsSaying([&poly] { geometry::validatePolygons({poly}); }, "outside [0, 2pi]");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsAzimuthAbove2Pi)
{
  geometry::AngularPolygon poly = {{2.0 * M_PI + 0.1, 0.0}, {1.0, 0.0}, {0.5, 0.1}};
  expectThrowsSaying([&poly] { geometry::validatePolygons({poly}); }, "outside [0, 2pi]");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsResolvesSeamSpanningPolygonToShortArc)
{
  // Azimuths are always in [0, 2pi], so a pair like 6.2 and 0.1 is ambiguous on
  // its face: the two vertices are joined either by the short arc through 2pi/0
  // or by the long way round. The 3D representation removes the ambiguity for
  // free - a cone is the intersection of its edges' great-circle half-spaces,
  // and an intersection of half-spaces is convex by construction, so only the
  // short arc is representable. Nothing needs to detect or reject this.
  geometry::AngularPolygon seam = {{6.2, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {6.2, 0.1}};
  auto valid = geometry::validatePolygons({seam});
  ASSERT_EQ(valid.size(), 1u);

  // The masked region is the narrow band straddling 0, not the 6.1 rad
  // complement: directions just either side of the seam are in, the far side is
  // out, and a direction on the long way round is out.
  EXPECT_TRUE(isInsideCone(valid[0].cone, {0.0, 0.0}));
  EXPECT_TRUE(isInsideCone(valid[0].cone, {6.25, 0.0}));
  EXPECT_TRUE(isInsideCone(valid[0].cone, {0.05, 0.0}));
  EXPECT_FALSE(isInsideCone(valid[0].cone, {M_PI, 0.0}));
  EXPECT_FALSE(isInsideCone(valid[0].cone, {0.2, 0.0}));
  EXPECT_FALSE(isInsideCone(valid[0].cone, {6.0, 0.0}));

  // The stored area, however, is the shoelace of the raw angular values, which
  // the seam inflates to ~1.22 against a true masked solid angle of ~0.037 sr.
  // angular_area only orders polygons in flattenAndSortPolygons (largest first,
  // a hot-path early-out heuristic), so this costs ordering, never a wrong mask.
  EXPECT_NEAR(valid[0].angular_area, 1.22, 1e-9);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsElevationBelowMinusHalfPi)
{
  geometry::AngularPolygon poly = {{1.0, -M_PI_2 - 0.1}, {1.2, 0.0}, {1.1, 0.1}};
  expectThrowsSaying([&poly] { geometry::validatePolygons({poly}); }, "outside [-pi/2, pi/2]");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsElevationAboveHalfPi)
{
  geometry::AngularPolygon poly = {{1.0, 0.0}, {1.2, M_PI_2 + 0.1}, {1.1, 0.1}};
  expectThrowsSaying([&poly] { geometry::validatePolygons({poly}); }, "outside [-pi/2, pi/2]");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsElevationAtRangeBoundaries)
{
  // The range is inclusive: +/- pi/2 exactly must not be rejected by the range
  // check (a polygon whose apex is straight up or straight down is legal
  // input). Each pole is tested in its own polygon - a single polygon touching
  // both poles has antipodal vertices, whose cross product vanishes, and would
  // be rejected as a degenerate edge for a reason unrelated to the range.
  geometry::AngularPolygon apex_up = {{1.0, 0.0}, {1.2, 0.0}, {1.1, M_PI_2}};
  geometry::AngularPolygon apex_down = {{1.0, 0.0}, {1.2, 0.0}, {1.1, -M_PI_2}};
  EXPECT_NO_THROW(geometry::validatePolygons({apex_up}));
  EXPECT_NO_THROW(geometry::validatePolygons({apex_down}));
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsCollinearZeroAreaPolygon)
{
  // Three points on one line in the angular domain: masks nothing, so it is
  // rejected by the minimum-angular-area check before reaching ConvexCone.
  geometry::AngularPolygon poly = {{1.0, 0.0}, {1.1, 0.05}, {1.2, 0.1}};
  expectThrowsSaying([&poly] { geometry::validatePolygons({poly}); }, "near-zero angular area");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsCoincidentZeroAreaPolygon)
{
  auto poly = makeTriangleWithDuplicateVertex();
  expectThrowsSaying([&poly] { geometry::validatePolygons({poly}); }, "near-zero angular area");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsDegenerateEdgeWithNonZeroArea)
{
  // Full-area square with a duplicated corner: passes the area check, then
  // fails inside ConvexCone. The message must be prefixed with the polygon
  // index so the failure is attributable to a specific input polygon.
  auto poly = makeSquareWithDuplicateVertex();
  expectThrowsSaying(
    [&poly] { geometry::validatePolygons({poly}); },
    "obstruction polygon 0 has a degenerate edge between vertices 2 and 3");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsNonConvexPolygon)
{
  expectThrowsSaying(
    [this] { geometry::validatePolygons({makeDentPentagon()}); },
    "obstruction polygon 0 is not spherically convex");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsGoodPolygon)
{
  auto valid = geometry::validatePolygons({makeSquareCcw(1.0, -0.1, 1.2, 0.1)});
  ASSERT_EQ(valid.size(), 1u);
  EXPECT_NEAR(valid[0].angular_area, 0.04, 1e-9);
  EXPECT_EQ(valid[0].cone.normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsMultipleGoodPolygons)
{
  auto valid = geometry::validatePolygons(
    {makeSquareCcw(1.0, -0.1, 1.2, 0.1), makeSquareCcw(2.0, -0.1, 2.4, 0.1), makeTriangle()});
  ASSERT_EQ(valid.size(), 3u);
  EXPECT_NEAR(valid[0].angular_area, 0.04, 1e-9);
  EXPECT_NEAR(valid[1].angular_area, 0.08, 1e-9);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsWholeSetWhenAnyPolygonInvalid)
{
  // No partial acceptance: one bad polygon rejects the whole parameter, so a
  // typo cannot silently shrink the mask. The index in the message identifies
  // which polygon was at fault.
  auto good = makeSquareCcw(1.0, -0.1, 1.2, 0.1);
  geometry::AngularPolygon bad = {{-0.1, 0.0}, {1.0, 0.0}, {0.5, 0.1}};  // negative azimuth

  expectThrowsSaying(
    [&good, &bad] { geometry::validatePolygons({good, bad}); },
    "obstruction polygon 1 has azimuth -0.100000 outside [0, 2pi]");
}

// ============================================================
// parsePolygonsFromString tests
// ============================================================

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringParsesValidString)
{
  std::string input = "[[1.0,-0.1, 1.2,-0.1, 1.1,0.1], [2.0,-0.1, 2.2,-0.1, 2.2,0.1, 2.0,0.1]]";
  auto polys = geometry::parsePolygonsFromString(input);
  ASSERT_EQ(polys.size(), 2u);
  EXPECT_EQ(polys[0].size(), 3u);
  EXPECT_EQ(polys[1].size(), 4u);
  EXPECT_DOUBLE_EQ(polys[0][0].azimuth, 1.0);
  EXPECT_DOUBLE_EQ(polys[1][3].elevation, 0.1);
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsEmptyString)
{
  // An empty parameter is handled one level up in fromParams (it means "no
  // blind spots"); reaching the parser with one is a programming error.
  expectThrowsSaying([] { geometry::parsePolygonsFromString(""); }, "expected a bracketed list");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsInvalidFormat)
{
  expectThrowsSaying(
    [] { geometry::parsePolygonsFromString("not a polygon"); }, "expected a bracketed list");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsBracketsWithoutPolygons)
{
  expectThrowsSaying([] { geometry::parsePolygonsFromString("[]"); }, "no polygons found");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsTooFewValues)
{
  // Only 2 values (need >= 6 for a triangle).
  expectThrowsSaying(
    [] { geometry::parsePolygonsFromString("[[1.0, 2.0]]"); },
    "obstruction polygon 0 has 2 values, needs an even count of at least 6");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsOddValueCount)
{
  // 7 values - not divisible by 2.
  expectThrowsSaying(
    [] { geometry::parsePolygonsFromString("[[1.0,0.0, 1.2,0.0, 1.1,0.1, 0.5]]"); },
    "obstruction polygon 0 has 7 values, needs an even count of at least 6");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsUnreadableValue)
{
  expectThrowsSaying(
    [] { geometry::parsePolygonsFromString("[[1.0,0.0, oops,0.0, 1.1,0.1]]"); },
    "obstruction polygon 0 has unreadable value");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringRejectsWholeSetWhenLaterPolygonBad)
{
  // The second group is unreadable: the first, well-formed group must not
  // survive on its own, otherwise a single typo silently halves the mask.
  expectThrowsSaying(
    [] { geometry::parsePolygonsFromString("[[1.0,-0.1, 1.2,-0.1, 1.1,0.1], [2.0,-0.1, 2.2]]"); },
    "obstruction polygon 1 has 3 values");
}

TEST_F(ObstructionPolygonsTest, ParsePolygonsFromStringHandlesExtraWhitespace)
{
  std::string input = "  [  [ 1.0 , -0.1 , 1.2 , -0.1 , 1.1 , 0.1 ]  ]  ";
  auto polys = geometry::parsePolygonsFromString(input);
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
  // Unset parameter is the common case: no blind spots to mask, not an error.
  FakeParamNode node("");
  auto filter = geometry::ObstructionFilter::fromParams(&node, "test", logger_);
  EXPECT_EQ(filter, nullptr);
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterFromParamsThrowsOnInvalidPolygon)
{
  // Azimuth out of [0, 2pi] range -> fails validation. fromParams no longer
  // degrades to a null (unmasked) filter; it names the parameter and rethrows
  // so the layer fails to configure instead of running with a wrong mask.
  FakeParamNode node("[[-0.1,-0.1, 1.0,-0.1, 0.5,0.1]]");
  expectThrowsSaying(
    [&node, this] { geometry::ObstructionFilter::fromParams(&node, "test", logger_); },
    "Invalid 'test.obstruction_polygons': obstruction polygon 0 has azimuth -0.100000 "
    "outside [0, 2pi]");
}

TEST_F(ObstructionPolygonsTest, ObstructionFilterFromParamsThrowsOnUnparsableString)
{
  FakeParamNode node("this is not a polygon list");
  expectThrowsSaying(
    [&node, this] { geometry::ObstructionFilter::fromParams(&node, "test", logger_); },
    "Invalid 'test.obstruction_polygons': expected a bracketed list");
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
