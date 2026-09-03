/*********************************************************************
 * Tests for obstruction_polygons.hpp - 3D spherical half-plane
 * representation (SphericalPoint/AngularPolygon, solidAngleFromNormals,
 * ConvexCone, ValidatedPolygon/validatePolygons, parsePolygonsFromString,
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

TEST_F(ObstructionPolygonsTest, ConvexConeFlipMakesWindingIrrelevantToTheMask)
{
  // What the flip is for. A convex polygon's normals are all inward or all outward
  // together - the sign follows the traversal, not the individual edge - so writing
  // the corners the other way round yields the negated normal set, and isObstructed's
  // "all dots non-negative" test would mask the exact complement if left uncorrected.
  // One global negation is the whole fix, and the two windings must be indistinguishable.
  const auto ccw = geometry::ConvexCone::fromAngularVertices(makeSquareCcw(1.0, -0.1, 1.2, 0.1));
  const auto cw = geometry::ConvexCone::fromAngularVertices(makeSquareCw(1.0, -0.1, 1.2, 0.1));

  EXPECT_NEAR(cw.getSolidAngle(), ccw.getSolidAngle(), 1e-15);
  for (const auto & p : std::vector<geometry::SphericalPoint>{
         {1.1, 0.0},     // interior
         {1.0, -0.1},    // a corner
         {1.5, 0.0},     // outside in azimuth
         {1.1, 0.5},     // outside in elevation
         {4.2, 0.0}}) {  // the antipode of the interior
    EXPECT_EQ(isInsideCone(ccw, p), isInsideCone(cw, p))
      << "winding changed the mask at az " << p.azimuth << " el " << p.elevation;
  }
  EXPECT_TRUE(isInsideCone(ccw, {1.1, 0.0}));
  EXPECT_FALSE(isInsideCone(ccw, {4.2, 0.0}));
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

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsAntipodalEdgeAsDegenerate)
{
  // The degenerate-edge threshold is on |d_i x d_j|^2 = sin^2(arc), which vanishes at
  // an arc of pi as well as 0. Two opposite directions lie on one line through the
  // sensor, and a line does not define a plane, so the edge has no half-space to
  // contribute - infinitely many great circles pass through both, all the same length.
  // Same rejection as a duplicated vertex, and the message says so.
  expectThrowsSaying(
    [] { geometry::ConvexCone::fromAngularVertices({{0.0, 0.0}, {M_PI, 0.0}, {0.5, 0.3}}); },
    "the two directions are coincident or 180 deg apart");
}

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsVertexDirectionsThatCancelOut)
{
  // Orientation is decided against the summed vertex directions, which only lands
  // inside the cone while the vertices fit within one hemisphere. Spread them evenly
  // around a great circle and the sum is the zero vector: there is no interior
  // direction to reference, so there is nothing to orient against.
  expectThrowsSaying(
    [] {
      geometry::ConvexCone::fromAngularVertices(
        {{0.0, 0.0}, {2.0 * M_PI / 3.0, 0.0}, {4.0 * M_PI / 3.0, 0.0}});
    },
    "has vertex directions that cancel out");
  expectThrowsSaying(
    [] {
      geometry::ConvexCone::fromAngularVertices(
        {{0.0, 0.0}, {M_PI_2, 0.0}, {M_PI, 0.0}, {3.0 * M_PI_2, 0.0}});
    },
    "has vertex directions that cancel out");
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
// solidAngleFromNormals tests
// ============================================================
//
// Girard's theorem: the interior angles of a spherical polygon exceed the flat
// (n-2)*pi by exactly the area it covers, which reduces to
// 2*pi - sum(acos(n[i] . n[i+1])) over the edge-plane normals. Exercised here on
// hand-written normals, so the arithmetic is pinned independently of whether any
// (az, el) polygon can produce those normals.

TEST_F(ObstructionPolygonsTest, SolidAngleFromNormalsMeasuresOneOctant)
{
  // The reference case, exact and derivable without measurement: the triangle
  // spanning the x, y and z axes is one eighth of the sphere, so 4*pi/8 = pi/2.
  // Its three edge planes are the coordinate planes, so the normals are the axes.
  const std::vector<geometry::Vec3D> axes = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  EXPECT_NEAR(geometry::solidAngleFromNormals(axes), M_PI_2, 1e-12);
}

TEST_F(ObstructionPolygonsTest, SolidAngleFromNormalsIsIndependentOfNormalSign)
{
  // Each term is a dot product between two normals, so negating every normal
  // leaves every term unchanged. That is why this measures a polygon's size
  // without settling its winding, and why it can run either side of the flip.
  std::vector<geometry::Vec3D> axes = {{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  const double as_written = geometry::solidAngleFromNormals(axes);
  for (auto & n : axes) {
    n = {-n.x, -n.y, -n.z};
  }
  EXPECT_NEAR(geometry::solidAngleFromNormals(axes), as_written, 1e-15);
}

TEST_F(ObstructionPolygonsTest, SolidAngleFromNormalsReportsHemisphereWhenAllNormalsAreParallel)
{
  // Every edge sharing one plane means the cone is a single half-space, i.e. a
  // hemisphere, 2*pi sr. This is the shape that used to slip through: the chart
  // shoelace saw a healthy area and the convexity check was vacuous, so the mask
  // silently swallowed half the sensor's view. Here it is simply a number, which
  // is what lets fromAngularVertices reject it.
  const std::vector<geometry::Vec3D> parallel = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};
  EXPECT_NEAR(geometry::solidAngleFromNormals(parallel), 2.0 * M_PI, 1e-12);
}

TEST_F(ObstructionPolygonsTest, SolidAngleFromNormalsReportsZeroWhenOneNormalIsAntiparallel)
{
  // The other flat case: the wedge is pinched shut and encloses nothing.
  const std::vector<geometry::Vec3D> pinched = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
  EXPECT_NEAR(geometry::solidAngleFromNormals(pinched), 0.0, 1e-12);
}

TEST_F(ObstructionPolygonsTest, SolidAngleMatchesSmallPatchApproximation)
{
  // A patch small enough to be near-flat should come out at width * height *
  // cos(el) - the cos factor being exactly what a flat (az, el) area misses.
  constexpr double kAz = 0.2;
  constexpr double kEl = 0.15;
  constexpr double kCentre = 0.6;
  const auto cone = geometry::ConvexCone::fromAngularVertices(
    {{1.0, kCentre - kEl / 2},
     {1.0 + kAz, kCentre - kEl / 2},
     {1.0 + kAz, kCentre + kEl / 2},
     {1.0, kCentre + kEl / 2}});
  EXPECT_NEAR(cone.getSolidAngle(), kAz * kEl * std::cos(kCentre), 1e-3);
}

TEST_F(ObstructionPolygonsTest, ConvexConeReportsSolidAngleForOctantTriangle)
{
  // Same reference case, but driven from (az, el) through the whole pipeline, so
  // the vertex-to-direction conversion and the normal construction are covered too.
  const auto cone =
    geometry::ConvexCone::fromAngularVertices({{0.0, 0.0}, {M_PI_2, 0.0}, {0.0, M_PI_2}});
  EXPECT_NEAR(cone.getSolidAngle(), M_PI_2, 1e-12);
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
  //
  // Note this polygon has the same signature as the over-wide band in
  // ValidatePolygonsMasksComplementWhenAzimuthSpanExceedsPi - an azimuth extent well
  // over pi - but the opposite intent. That is exactly why no extent-based rule can
  // separate them, and why neither is rejected.
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

  // The stored size is the solid angle the cone really covers, ~0.037 sr. A flat
  // shoelace over the raw angular values read 1.22 here, inflated 33x by the seam
  // jump from 6.2 to 0.1, which mis-sorted this tiny polygon ahead of far larger
  // ones in flattenAndSortPolygons. Measuring on the sphere removes the seam's
  // effect entirely - see SolidAngleOrdersSeamPolygonBelowAGenuinelyLargerOne.
  EXPECT_NEAR(valid[0].solid_angle, 0.036678, 1e-6);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsMasksComplementWhenAzimuthSpanExceedsPi)
{
  // The trap. Edges are geodesics and take the short way, so once a band's azimuth
  // span passes pi the short way is the OTHER way, across 0/2pi, and the cone covers
  // the complement of what was written. Nothing rejects it: the complement band is
  // itself a perfectly valid convex cone, so the convexity check passes.
  //
  // Pinned either side of the threshold. Do not "fix" this by rejecting wide spans -
  // see the must-stay-accepted cases below, which are indistinguishable from it.
  //
  // The band has to be thin in elevation to reach this at all now. Combined with a
  // near-pi azimuth span, a 0.1 rad tall band bows almost to the pole and trips the
  // solid-angle cap instead - covered by ConvexConeRejectsBandWhoseBowMakesItEnormous.
  auto span = [](double s) {
    return geometry::AngularPolygon{
      {0.05, -0.01}, {0.05 + s, -0.01}, {0.05 + s, 0.01}, {0.05, 0.01}};
  };

  const auto under = geometry::validatePolygons({span(3.1)});
  ASSERT_EQ(under.size(), 1u);
  EXPECT_TRUE(isInsideCone(under[0].cone, {1.6, 0.0}));   // inside the written band
  EXPECT_FALSE(isInsideCone(under[0].cone, {4.5, 0.0}));  // outside it

  const auto over = geometry::validatePolygons({span(3.25)});
  ASSERT_EQ(over.size(), 1u);
  EXPECT_FALSE(isInsideCone(over[0].cone, {1.6, 0.0}));  // written band NOT masked
  EXPECT_TRUE(isInsideCone(over[0].cone, {4.5, 0.0}));   // the complement is
}

TEST_F(ObstructionPolygonsTest, ConvexConeGeodesicEdgeBowsAwayFromTheEquator)
{
  // Even below pi the region is not the flat az/el box it looks like: a geodesic
  // between two vertices at equal elevation bows away from the equator, reaching
  // atan(tan(el) / cos(d_az / 2)).
  const auto cone =
    geometry::ConvexCone::fromAngularVertices({{0.0, -0.1}, {2.8, -0.1}, {2.8, 0.1}, {0.0, 0.1}});

  // For el 0.1 over 2.8 rad of azimuth that peak is 0.5333 rad, so a band drawn
  // 0.1 tall masks out to ~31 deg of elevation.
  const double peak = std::atan(std::tan(0.1) / std::cos(1.4));
  EXPECT_NEAR(peak, 0.5333, 1e-4);

  EXPECT_TRUE(isInsideCone(cone, {1.4, 0.0}));    // drawn interior
  EXPECT_TRUE(isInsideCone(cone, {1.4, 0.3}));    // far above anything drawn
  EXPECT_TRUE(isInsideCone(cone, {1.4, 0.50}));   // still inside, near the bow peak
  EXPECT_FALSE(isInsideCone(cone, {1.4, 0.60}));  // just past it
  EXPECT_FALSE(isInsideCone(cone, {4.0, 0.0}));   // outside the azimuth span
}

TEST_F(ObstructionPolygonsTest, ConvexConeRejectsBandWhoseBowMakesItEnormous)
{
  // Same band 0.2 rad wider in azimuth. Nothing about the written numbers looks
  // alarming - 0.2 rad of elevation - but the bowed geodesics carry it to 3.81 sr,
  // over a quarter of the sphere, so it is rejected rather than masking that much.
  expectThrowsSaying(
    [] {
      geometry::ConvexCone::fromAngularVertices({{0.0, -0.1}, {3.0, -0.1}, {3.0, 0.1}, {0.0, 0.1}});
    },
    "far too large for a blind spot");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsSeamStraddlingBandAnExtentRuleWouldReject)
{
  // Guard rail for the test above. This band straddles 0/2pi, so its azimuth extent
  // is 6.10 - far over pi - yet it is correct today and is exactly what the docs
  // recommend for a blind spot sitting on sensor-forward. Any rule keyed on azimuth
  // extent would reject the documented remedy, which is why none is applied.
  const auto valid =
    geometry::validatePolygons({{{6.2, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {6.2, 0.1}}});
  ASSERT_EQ(valid.size(), 1u);
  EXPECT_TRUE(isInsideCone(valid[0].cone, {0.0, 0.0}));
  EXPECT_TRUE(isInsideCone(valid[0].cone, {6.25, 0.0}));
  EXPECT_FALSE(isInsideCone(valid[0].cone, {M_PI, 0.0}));
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsPoleRingingCap)
{
  // A cap around the nadir or zenith - masking straight down or straight up, e.g. the
  // sensor's own mount - used to be impossible. Ringing a pole means azimuth runs all
  // the way round, which the old flat (az, el) shoelace read as a retraced line with
  // zero area, so the minimum-area gate rejected it however the elevations were
  // written. Measuring the cone itself has no such blind spot: a hexagon inscribed in
  // a cap 0.17 rad from the south pole is an ordinary convex cone of 0.0759 sr.
  //
  // Note most +-20-40 elevation deg lidars cannot see a pole at all - the frustum's
  // vFOV check discards those directions first - so this only becomes reachable on a
  // wide-angle sensor.
  geometry::AngularPolygon cap;
  for (int i = 0; i < 6; ++i) {
    cap.push_back({i * 2.0 * M_PI / 6.0, -1.4});
  }
  const auto valid = geometry::validatePolygons({cap});
  ASSERT_EQ(valid.size(), 1u);
  EXPECT_NEAR(valid[0].solid_angle, 0.075880, 1e-6);

  // It masks the pole it rings and nothing near the equator.
  EXPECT_TRUE(isInsideCone(valid[0].cone, {0.0, -M_PI_2}));
  EXPECT_TRUE(isInsideCone(valid[0].cone, {3.0, -1.45}));
  EXPECT_FALSE(isInsideCone(valid[0].cone, {0.0, 0.0}));
  EXPECT_FALSE(isInsideCone(valid[0].cone, {0.0, -1.0}));
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsPoleRingingCapThatZigzagsInElevation)
{
  // Alternating elevations around the pole makes a star, not a cap: the vertices at
  // el -1.2 sit further from the pole than those at -1.4, so the outline has reflex
  // corners.
  geometry::AngularPolygon zigzag;
  for (int i = 0; i < 6; ++i) {
    zigzag.push_back({i * 2.0 * M_PI / 6.0, (i % 2 == 0) ? -1.4 : -1.2});
  }
  expectThrowsSaying(
    [&zigzag] { geometry::validatePolygons({zigzag}); }, "is not spherically convex");
}

TEST_F(ObstructionPolygonsTest, SlicingAWideBandIntoStripsRestoresTheIntendedMask)
{
  // The documented remedy: express a wide blind spot as several narrow polygons.
  // isObstructed is a union over polygons, so slicing is lossless - and each strip's
  // edges are short enough that neither the flip nor the bow bites.
  constexpr double kSpan = 4.0;
  constexpr int kSlices = 8;
  std::vector<geometry::AngularPolygon> strips;
  for (int i = 0; i < kSlices; ++i) {
    const double a0 = 0.05 + i * (kSpan / kSlices);
    const double a1 = 0.05 + (i + 1) * (kSpan / kSlices);
    strips.push_back({{a0, -0.1}, {a1, -0.1}, {a1, 0.1}, {a0, 0.1}});
  }
  const auto valid = geometry::validatePolygons(strips);
  ASSERT_EQ(valid.size(), static_cast<size_t>(kSlices));

  auto masked_by_any = [&valid](const geometry::SphericalPoint & p) {
    for (const auto & v : valid) {
      if (isInsideCone(v.cone, p)) {
        return true;
      }
    }
    return false;
  };

  // The written band is masked across its whole width, and only it.
  EXPECT_TRUE(masked_by_any({0.10, 0.0}));
  EXPECT_TRUE(masked_by_any({2.00, 0.0}));
  EXPECT_TRUE(masked_by_any({3.90, 0.0}));
  EXPECT_FALSE(masked_by_any({4.50, 0.0}));
  EXPECT_FALSE(masked_by_any({6.00, 0.0}));
  // And the bow is contained: nothing masked far above the 0.1 rad the strips span.
  EXPECT_FALSE(masked_by_any({2.00, 0.5}));
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

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsChartCollinearVerticesAsAThinSliver)
{
  // BEHAVIOUR CHANGE, pinned deliberately. These three points are collinear in the
  // flat (az, el) chart, so the old shoelace gate called them zero-area and rejected
  // them. On the sphere they are not degenerate at all - a straight line in the chart
  // is not a great circle - so they bound a real, if very thin, cone of 3.8e-5 sr.
  //
  // It is accepted because kMinSolidAngle (1e-9 sr) exists to catch cones that
  // enclose nothing, not to second-guess small ones. If chart-collinear vertices
  // should instead be treated as a likely typo, that is a threshold decision:
  // raising kMinSolidAngle to ~1e-4 sr would reject this while leaving every
  // realistic mask (a 30x20 deg mast is 0.19 sr) untouched.
  geometry::AngularPolygon poly = {{1.0, 0.0}, {1.1, 0.05}, {1.2, 0.1}};
  const auto valid = geometry::validatePolygons({poly});
  ASSERT_EQ(valid.size(), 1u);
  EXPECT_GT(valid[0].solid_angle, 0.00001);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsCoincidentVerticesAsADegenerateEdge)
{
  // A duplicated vertex used to be caught by the area gate. It is now caught one
  // step later, by the degenerate-edge check, which names the two vertices - a
  // strictly more useful message for the same input.
  auto poly = makeTriangleWithDuplicateVertex();
  expectThrowsSaying(
    [&poly] { geometry::validatePolygons({poly}); },
    "obstruction polygon 0 has a degenerate edge between vertices 0 and 1");
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsRejectsVerticesOnASingleGreatCircle)
{
  // The shape the old chart gate could not see and the convexity check could not
  // judge: every vertex on one great circle through the sensor. The cone is flat, so
  // every vertex lies in every edge plane, every dot product is zero, and the
  // convexity loop proves nothing. Both sub-cases were accepted before this check.
  //
  // Vertices are placed on a 45 deg inclined great circle, el = atan(sin(az)), which
  // is a sinusoid in the chart - not a straight line - so the old shoelace read a
  // healthy area for it.
  auto onGreatCircle = [](double az_deg) {
    const double az = az_deg * M_PI / 180.0;
    return geometry::SphericalPoint{az, std::atan(std::sin(az))};
  };

  // Spread over more than half the circle: all normals end up parallel, so the cone
  // is a half-space. This one used to mask an entire hemisphere in silence.
  geometry::AngularPolygon wrapping = {
    onGreatCircle(0.0), onGreatCircle(120.0), onGreatCircle(240.0)};
  expectThrowsSaying(
    [&wrapping] { geometry::validatePolygons({wrapping}); },
    "every vertex lies on one great circle through the sensor");

  // Spread over less than half: one normal is antiparallel and the cone is pinched
  // shut, so it used to be accepted while masking nothing.
  geometry::AngularPolygon pinched = {
    onGreatCircle(0.0), onGreatCircle(90.0), onGreatCircle(170.0)};
  expectThrowsSaying(
    [&pinched] { geometry::validatePolygons({pinched}); },
    "every vertex lies on one great circle through the sensor");
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
  // 0.2 x 0.2 rad astride the equator, so the solid angle is just over the 0.04 a
  // flat area would give - the geodesic edges bow outward slightly.
  EXPECT_GT(valid[0].solid_angle, 0.04);
  EXPECT_NEAR(valid[0].solid_angle, 0.041, 0.001);
  EXPECT_EQ(valid[0].cone.normals().size(), 4u);
}

TEST_F(ObstructionPolygonsTest, ValidatePolygonsAcceptsMultipleGoodPolygons)
{
  auto valid = geometry::validatePolygons(
    {makeSquareCcw(1.0, -0.1, 1.2, 0.1), makeSquareCcw(2.0, -0.1, 2.4, 0.1), makeTriangle()});
  ASSERT_EQ(valid.size(), 3u);
  EXPECT_GT(valid[0].solid_angle, 0.2 * 0.2);
  EXPECT_GT(valid[1].solid_angle, 0.4 * 0.2);
  EXPECT_GT(valid[2].solid_angle, 0.01);
}

TEST_F(ObstructionPolygonsTest, SolidAngleOrdersSeamPolygonBelowAGenuinelyLargerOne)
{
  // flattenAndSortPolygons orders polygons largest-first so isObstructed's early-out
  // hits sooner, and the key it sorts on used to be the flat chart shoelace. That key
  // was wrong by a wide margin for a seam-straddling polygon: the jump from az 6.2 to
  // 0.1 reads as 6.1 rad of extent, inflating a 0.037 sr sliver to 1.22 and sorting it
  // ahead of every genuinely larger mask. The solid angle has no seam to trip over.
  //
  // The filter's own spans are private, so this pins the key rather than reaching into
  // the ordering: what mattered was which of the two numbers is larger, and it flipped.
  geometry::AngularPolygon seam = {{6.2, -0.1}, {0.1, -0.1}, {0.1, 0.1}, {6.2, 0.1}};
  geometry::AngularPolygon larger = makeSquareCcw(2.0, -0.1, 2.4, 0.1);

  const auto valid = geometry::validatePolygons({seam, larger});
  ASSERT_EQ(valid.size(), 2u);
  EXPECT_NEAR(valid[0].solid_angle, 0.036, 1e-3);  // chart shoelace read 1.22
  EXPECT_NEAR(valid[1].solid_angle, 0.081, 1e-3);  // chart shoelace read 0.08
  EXPECT_GT(valid[1].solid_angle, valid[0].solid_angle);
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
