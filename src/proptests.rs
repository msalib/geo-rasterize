use geo::{
    map_coords::MapCoords, Coordinate, Geometry, Line, LineString, MultiLineString, MultiPoint,
    MultiPolygon, Point, Polygon, Rect, Triangle,
};
use pretty_assertions::assert_eq;
use proptest::prelude::*;

use crate::tests::compare;

// put everything in 17x19, so coordinates in -2..19, -2..21

#[rustfmt::skip]
prop_compose! {
    fn arb_point()(x in -2.0..19., y in -2.0..21.0) -> Point<f64> {
	Point::new(x, y)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_line()(start in arb_point(), end in arb_point()) -> Line<f64> {
	Line::new(start, end)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_linestring()(points in prop::collection::vec(arb_point(), 2..20)) -> LineString<f64> {
	points.into()
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_rect()(center in arb_point(),
		  width in 0.0..20., height in 0.0..20.) -> Rect<f64> {
	let min: Coordinate<f64> = (
	    center.x() - width / 2.,
	    center.y() - height / 2.).into();
	let max: Coordinate<f64> = (
	    center.x() + width / 2.,
	    center.y() + height/2.).into();
	Rect::new(min, max)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_linearring()(center in arb_point(),
		        exterior_points in 3..17,
		        radius in 0.000001..15.0) -> (Point<f64>, LineString<f64>) {
	let angles = (0..exterior_points)
	    .map(|idx| 2.0 * std::f64::consts::PI * (idx as f64) / (exterior_points as f64));
	let points: Vec<geo::Coordinate<f64>> = angles
	    .map(|angle_rad| angle_rad.sin_cos())
	    .map(|(sin, cos)| geo::Coordinate {
		x: center.x() + radius * cos,
		y: center.y() + radius * sin,
	    })
	    .collect();
	(center, LineString(points))
    }
}

fn shrink_ring(center: &Point<f64>, ring: &LineString<f64>, scale_factor: f64) -> LineString<f64> {
    let cx = center.x();
    let cy = center.y();
    use euclid::{Point2D, Transform2D, UnknownUnit, Vector2D};
    let transform: Transform2D<f64, UnknownUnit, UnknownUnit> = Transform2D::identity()
        .then_translate(Vector2D::new(-cx, -cy))
        .then_scale(scale_factor, scale_factor)
        .then_translate(Vector2D::new(cx, cy));
    ring.map_coords(|&(x, y)| transform.transform_point(Point2D::new(x, y)).to_tuple())
}

#[rustfmt::skip]
prop_compose! {
    fn arb_poly()(center_exterior in arb_linearring(),
                  include_hole in proptest::bool::ANY,
                  hole_scale_factor in 0.01..0.9) -> Polygon<f64> {
	let (center, exterior) = center_exterior;
	let holes = if include_hole {
	    vec![shrink_ring(&center, &exterior, hole_scale_factor)]
	} else {
	    vec![]
	};
	Polygon::new(exterior, holes)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_triangle()(a in arb_point(),
		      b in arb_point(),
		      c in arb_point()) -> Triangle<f64> {
	Triangle(a.0, b.0, c.0)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_multipoint()(points in proptest::collection::vec(arb_point(), 0..5)) -> MultiPoint<f64>{
	MultiPoint(points)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_multiline()(lines in proptest::collection::vec(arb_linestring(), 0..5)) -> MultiLineString<f64>{
	MultiLineString(lines)
    }
}

#[rustfmt::skip]
prop_compose! {
    fn arb_multipoly()(polys in proptest::collection::vec(arb_poly(), 0..5)) -> MultiPolygon<f64>{
	MultiPolygon(polys)
    }
}

fn arb_geo() -> impl Strategy<Value = Geometry<f64>> {
    prop_oneof![
        arb_point().prop_map(Geometry::Point),
        arb_line().prop_map(Geometry::Line),
        arb_poly().prop_map(Geometry::Polygon),
        arb_linestring().prop_map(Geometry::LineString),
        arb_rect().prop_map(Geometry::Rect),
        arb_triangle().prop_map(Geometry::Triangle),
        arb_multipoint().prop_map(Geometry::MultiPoint),
        arb_multiline().prop_map(Geometry::MultiLineString),
        arb_multipoly().prop_map(Geometry::MultiPolygon)
    ]
}

#[rustfmt::skip]
proptest! {
    #[test]
    fn match_gdal(shape in proptest::collection::vec(arb_geo(), 1..5)) {
	let shape = Geometry::GeometryCollection(geo::GeometryCollection(shape));
	let (actual, expected) = compare(17, 19, &shape).unwrap();
	assert_eq!(actual, expected);
    }
}
