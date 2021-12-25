use std::iter::once;

use geo::{coords_iter::CoordsIter, winding_order::Winding, LineString, Point};
use itertools::Itertools;
use smallvec::SmallVec;

use crate::{BinaryRasterizer, Rasterize};

fn y_coordinates<'a>(
    first: &'a LineString<f64>,
    rest: &'a [LineString<f64>],
) -> impl Iterator<Item = isize> + 'a {
    once(first).chain(rest).flat_map(|line_string| {
        line_string
            .points_iter()
            .map(|point| point.y().floor() as isize)
    })
}

type PointPair = (Point<f64>, Point<f64>);

/// Iterate over all points in a polygon (both the exterior and all
/// the holes) in clockwise order. GDAL copies all the coordinates
/// into a new structure whereas we only need to copy one `bool` per
/// ring and typically we won't need to allocate it on the heap.
struct CWIter<'a> {
    first: &'a LineString<f64>,
    rest: &'a [LineString<f64>],
    is_cw: SmallVec<[bool; 8]>,
}

impl<'a> CWIter<'a> {
    fn new(first: &'a LineString<f64>, rest: &'a [LineString<f64>]) -> CWIter<'a> {
        let is_cw = once(first)
            .chain(rest)
            .map(|line_string| line_string.is_cw())
            .collect();
        CWIter { first, rest, is_cw }
    }

    fn iter<'b: 'a>(&'b self) -> impl Iterator<Item = PointPair> + 'a {
        // We rely on the fact that all our inputs are closed, which
        // is asserted in `rasterize_polygon`; if they weren't, we'd
        // have to close them here.
        once(self.first)
            .chain(self.rest)
            .zip(self.is_cw.iter().copied())
            .flat_map(|(line_string, is_cw)| {
                if is_cw {
                    EitherIter::A(line_string.points_iter().tuple_windows())
                } else {
                    EitherIter::B(line_string.points_iter().rev().tuple_windows())
                }
            })
    }
}

/// A helper that allows us to return different iterator types from
/// different branches of an `impl Iterator` method. Taken from
/// `itertools`.
enum EitherIter<T, I1, I2>
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
{
    A(I1),
    B(I2),
}

impl<T, I1, I2> Iterator for EitherIter<T, I1, I2>
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EitherIter::A(iter) => iter.next(),
            EitherIter::B(iter) => iter.next(),
        }
    }
}

// FIXME: pull all (first, rest) into a struct with conversions from
// poly + LineString. then all the functions that take (first, rest)
// become methods.

// Most of this implementation came from `GDALdllImageFilledPolygon`
// in `gdal/alg/llrasterize.cpp`; that code fills the interior of the
// polygon. The bit at the end that rasterizes the exterior came from
// a call to `GDALdllImageLineAllTouched` that's called from
// `gv_rasterize_one_shape` in `gdal/alg/gdalrasterize.cpp`.

/// This will rasterize the interior of a polygon. We treat the
/// arguments as LinearRings, which don't exist in `geo`, so we use
/// closed `LineString`s instead.
pub fn rasterize_polygon(
    first: &LineString<f64>,
    rest: &[LineString<f64>],
    rasterizer: &mut BinaryRasterizer,
) {
    assert!(first.is_closed() && rest.iter().all(|ls| ls.is_closed()));
    let total_points = first.coords_count()
        + rest
            .iter()
            .map(|line_string| line_string.coords_count())
            .sum::<usize>();
    if total_points == 0 {
        return;
    }

    // We'll reuse this Vec for each scanline; it will never more than
    // `total_points` elements.
    let mut xs: Vec<isize> = Vec::with_capacity(total_points);

    // GDAL ensures that closed linear rings are always in CW order in
    // `GDALCollectRingsFromGeometry`.
    // These unwraps are safe since we would've returned if there were
    // no points, so there must be at least one point.
    let min_y = y_coordinates(first, rest).min().unwrap().max(0);
    let max_y = y_coordinates(first, rest)
        .max()
        .unwrap()
        .min(rasterizer.height() as isize - 1);
    let min_x = 0;
    let max_x = rasterizer.width() - 1;

    let cw_points = CWIter::new(first, rest);

    // iterate over scanlines
    for y in min_y..=max_y {
        let dy = 0.5 + (y as f64); // center height of line

        // iterate over ring point pairs
        for (ind1, ind2) in cw_points.iter() {
            // FIXME: we're assuming here that all loops are closed so
            // that the last point == the first point; the gdal code
            // makes this true.

            // ind1,ind2 are a pair of adjacent points looping back to
            // the front in the last case: (p0,p1), (p1,p2), (p2,p0)
            let mut dy1 = ind1.y();
            let mut dy2 = ind2.y();

            if (dy1 < dy && dy2 < dy) || (dy1 > dy && dy2 > dy) {
                continue;
            }

            let (dx1, dx2) = if dy1 < dy2 {
                (ind1.x(), ind2.x())
            } else if dy1 > dy2 {
                std::mem::swap(&mut dy1, &mut dy2);
                (ind2.x(), ind1.x())
            } else {
                if ind1.x() > ind2.x() {
                    let horizontal_x1 = (ind2.x() + 0.5).floor() as isize;
                    let horizontal_x2 = (ind1.x() + 0.5).floor() as isize;
                    if horizontal_x1 > (max_x as isize) || horizontal_x2 <= min_x {
                        continue;
                    }
                    rasterizer.fill_block(horizontal_x1..=(horizontal_x2 - 1), y..=y);
                }
                continue;
            };

            if dy < dy2 && dy >= dy1 {
                let intersect = (dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1;
                xs.push((intersect + 0.5).floor() as isize);
            }
        }

        xs.sort_unstable();
        for pair in xs[..].chunks_exact(2) {
            let x_start = pair[0].max(min_x);
            let x_end = pair[1].min((max_x + 1) as isize);
            if x_start <= (max_x as isize) && x_end > min_x {
                rasterizer.fill_block(x_start..=x_end - 1, y..=y);
            }
        }
        xs.clear();
    }

    // All the code above this point only deals with the interior of
    // the polygon but this bit handles the exterior border.
    once(first)
        .chain(rest)
        .for_each(|ls| ls.rasterize(rasterizer));
}
