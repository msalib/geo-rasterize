use geo::{Coordinate, Geometry, Line, LineString, Point, Polygon, Rect, Triangle};
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
    fn arb_poly()(center in arb_point(),
		  exterior_points in 3..17,
		  radius in 0.000001..15.0) -> Polygon<f64> {
	// FIXME: add holes!
	let angles = (0..exterior_points)
	    .map(|idx| 2.0 * std::f64::consts::PI * (idx as f64) / (exterior_points as f64));
	let points: Vec<geo::Coordinate<f64>> = angles
	    .map(|angle_rad| angle_rad.sin_cos())
	    .map(|(sin, cos)| geo::Coordinate {
		x: center.x() + radius * cos,
		y: center.y() + radius * sin,
	    })
	    .collect();

	Polygon::new(geo::LineString(points), vec![])
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

fn arb_geo() -> impl Strategy<Value = Geometry<f64>> {
    prop_oneof![
        arb_point().prop_map(Geometry::Point),
        arb_line().prop_map(Geometry::Line),
        arb_poly().prop_map(Geometry::Polygon),
        arb_linestring().prop_map(Geometry::LineString),
        arb_rect().prop_map(Geometry::Rect),
        arb_triangle().prop_map(Geometry::Triangle),
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
