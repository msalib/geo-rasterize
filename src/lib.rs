#![doc = include_str!("../README.md")]
use std::{
    fmt::Debug,
    ops::{Add, RangeInclusive},
};

use euclid::{Transform2D, UnknownUnit};
use geo::{
    algorithm::{
        coords_iter::CoordsIter,
        map_coords::{MapCoords, MapCoordsInplace},
    },
    Geometry, GeometryCollection, Line, LineString, MultiLineString, MultiPoint, MultiPolygon,
    Point, Polygon, Rect, Triangle,
};
use ndarray::s;
use ndarray::Array2;
use num_traits::{AsPrimitive, Num, NumCast};
use thiserror::Error;

mod poly;
use poly::rasterize_polygon;
mod line;
use line::rasterize_line;
#[cfg(test)]
mod proptests;

/// Affine transform that describes how to convert world-space
/// coordinates to pixel coordinates.
pub type Transform = Transform2D<f64, UnknownUnit, UnknownUnit>;
type EuclidPoint = euclid::Point2D<f64, UnknownUnit>;

/// Error type for this crate
#[derive(Error, Clone, Debug)]
pub enum RasterizeError {
    /// at least one coordinate of the supplied geometry is NaN or infinite
    #[error("at least one coordinate of the supplied geometry is NaN or infinite")]
    NonFiniteCoordinate,

    /// `width` is required in builder
    #[error("`width` is required in builder")]
    MissingWidth,

    /// `height` is required in builder
    #[error("`height` is required in builder")]
    MissingHeight,
}

/// Result type for this crate that uses [RasterizeError].
pub type Result<T> = std::result::Result<T, RasterizeError>;

/// A builder that can construct instances of [BinaryRasterizer], a
/// rasterizer that can rasterize shapes into a 2-dimensional array of
/// booleans.
///
/// ```rust
/// # use geo_rasterize::{Result, BinaryBuilder, BinaryRasterizer};
/// # fn main() -> Result<()> {
/// let rasterizer: BinaryRasterizer = BinaryBuilder::new().width(37).height(21).build()?;
/// # Ok(())}
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryBuilder {
    width: Option<usize>,
    height: Option<usize>,
    geo_to_pix: Option<Transform>,
}

impl BinaryBuilder {
    pub fn new() -> Self {
        BinaryBuilder::default()
    }

    pub fn width(mut self, width: usize) -> Self {
        self.width = Some(width);
        self
    }

    pub fn height(mut self, height: usize) -> Self {
        self.height = Some(height);
        self
    }

    pub fn geo_to_pix(mut self, geo_to_pix: Transform) -> Self {
        self.geo_to_pix = Some(geo_to_pix);
        self
    }

    pub fn build(self) -> Result<BinaryRasterizer> {
        match (self.width, self.height) {
            (None, _) => Err(RasterizeError::MissingWidth),
            (_, None) => Err(RasterizeError::MissingHeight),
            (Some(width), Some(height)) => BinaryRasterizer::new(width, height, self.geo_to_pix),
        }
    }
}

/// A rasterizer that burns shapes into a 2-dimensional boolean
/// array. It can be built either by calling
/// [new][BinaryRasterizer::new] or using [BinaryBuilder].
///
/// Each Rasterizer requires a `width` and `height` measured in pixels
/// that describe the shape of the output array. They can optionally
/// take an affine transform that describes how to convert world-space
/// coordinates into pixel-space coordinates. When a transformer is
/// supplied, all of its parameters must be finite or
/// [RasterizeError::NonFiniteCoordinate] will be returned.
///
/// ```rust
/// # use geo_rasterize::{Result, BinaryBuilder, BinaryRasterizer};
/// # fn main() -> Result<()> {
/// use geo::{Geometry, Line, Point};
/// use ndarray::array;
/// use geo_rasterize::BinaryBuilder;
///
/// let shapes: Vec<Geometry<i32>> =
///     vec![Point::new(3, 4).into(),
/// 	     Line::new((0, 3), (3, 0)).into()];
///
/// let mut r = BinaryBuilder::new().width(4).height(5).build()?;
/// for shape in shapes {
///     r.rasterize(&shape)?;
/// }
///
/// let pixels = r.finish();
/// assert_eq!(
///     pixels.mapv(|v| v as u8),
///     array![
///         [0, 0, 1, 0],
///         [0, 1, 1, 0],
///         [1, 1, 0, 0],
///         [1, 0, 0, 0],
///         [0, 0, 0, 1]
///     ]
/// );
/// # Ok(())}
/// ```
#[derive(Clone, Debug)]
pub struct BinaryRasterizer {
    pixels: Array2<bool>,
    geo_to_pix: Option<Transform>,
}

fn to_float<T>(coords: &(T, T)) -> (f64, f64)
where
    T: Into<f64> + Copy,
{
    (coords.0.into(), coords.1.into())
}

impl BinaryRasterizer {
    pub fn new(width: usize, height: usize, geo_to_pix: Option<Transform>) -> Result<Self> {
        let non_finite = geo_to_pix
            .map(|geo_to_pix| geo_to_pix.to_array().iter().any(|param| !param.is_finite()))
            .unwrap_or(false);
        if non_finite {
            Err(RasterizeError::NonFiniteCoordinate)
        } else {
            let pixels = Array2::default((width, height));
            Ok(BinaryRasterizer { pixels, geo_to_pix })
        }
    }

    fn width(&self) -> usize {
        self.pixels.shape()[0]
    }

    fn height(&self) -> usize {
        self.pixels.shape()[1]
    }

    fn fill_pixel<T1, T2>(&mut self, ix: T1, iy: T2)
    where
        T1: AsPrimitive<usize>,
        T2: AsPrimitive<usize>,
    {
        let ix: usize = ix.as_();
        let iy: usize = iy.as_();
        assert!(ix < self.width());
        assert!(iy < self.height());
        self.pixels.slice_mut(s![ix, iy]).fill(true);
    }

    fn fill_block<T1, T2>(&mut self, xs: RangeInclusive<T1>, ys: RangeInclusive<T2>)
    where
        T1: AsPrimitive<usize> + Debug,
        T2: AsPrimitive<usize>,
    {
        let (x_start, x_end) = xs.into_inner();
        let xs: RangeInclusive<usize> = (x_start.as_())..=(x_end.as_());
        let (y_start, y_end) = ys.into_inner();
        let ys: RangeInclusive<usize> = (y_start.as_())..=(y_end.as_());
        debug_assert!(x_start.as_() < self.width());
        debug_assert!(x_end.as_() < self.width());
        debug_assert!(y_start.as_() < self.height());
        debug_assert!(y_end.as_() < self.height());
        self.pixels.slice_mut(s![xs, ys]).fill(true);
    }

    /// Retrieve the transform.
    pub fn geo_to_pix(&self) -> Option<Transform> {
        self.geo_to_pix
    }

    pub fn set_transform(&mut self, geo_to_pix: Transform) {
        self.geo_to_pix = Some(geo_to_pix);
    }

    /// Rasterize one shape, which can be any type that [geo] provides
    /// using any coordinate numeric type that can be converted into
    /// `f64`.
    pub fn rasterize<Coord, InputShape, ShapeAsF64>(&mut self, shape: &InputShape) -> Result<()>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize + for<'a> CoordsIter<'a, Scalar = f64> + MapCoordsInplace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        // first, convert our input shape so that its coordinates are of type f64
        let mut float = shape.map_coords(to_float);

        // then ensure that all coordinates are finite or bail
        let all_finite = float
            .coords_iter()
            .all(|coordinate| coordinate.x.is_finite() && coordinate.y.is_finite());
        if !all_finite {
            return Err(RasterizeError::NonFiniteCoordinate);
        }

        // use `geo_to_pix` to convert geographic coordinates to image
        // coordinates, if it is available
        match self.geo_to_pix {
            None => float,
            Some(transform) => {
                float.map_coords_inplace(|&(x, y)| {
                    transform.transform_point(EuclidPoint::new(x, y)).to_tuple()
                });
                float
            }
        }
        .rasterize(self); // and then rasterize!

        Ok(())
    }

    /// Retrieve the completed raster array.
    pub fn finish(self) -> Array2<bool> {
        self.pixels.reversed_axes()
    }
}

#[doc(hidden)]
pub trait Rasterize {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer);
}

/// Conflict resolution strategy for cases where two shapes cover the
/// same pixel.
#[derive(Debug, Clone, Copy)]
pub enum MergeAlgorithm {
    /// Overwrite the pixel with the burn value associated with the
    /// last shape to be written to it. This is the default.
    Replace,

    /// Overwrite the pixel with the sum of the burn values associated
    /// with the shapes written to it.
    Add,
}

impl Default for MergeAlgorithm {
    fn default() -> Self {
        MergeAlgorithm::Replace
    }
}

/// A builder that constructs [LabelRasterizer]s. Whereas
/// [BinaryRasterizer] produces an array of booleans,
/// [LabelRasterizer] produces an array of some generic type (`Label`)
/// that implements [Copy][std::marker::Copy] and [Add][std::ops::Add]
/// though typically you'd use a numeric type.
///
/// [LabelBuilder] needs the `Label` type so the only way to make one
/// is to specify a `Label` value: the background. The `background` is
/// the value we'll use to initialize the raster array -- it
/// corresponds to the zero value, so you'll probably want to start
/// with `LabelBuilder::background(0)` or
/// `LabelBuilder::background(0f32)`.
///
/// In addition to `background`, `width`, and `height`, you can also
/// supply a [MergeAlgorithm] to specify what the rasterizer should do
/// when two different shapes fill the same pixel. If you don't supply
/// anything, the rasterizer will use
/// [Replace][MergeAlgorithm::Replace] by default.
///
/// ```rust
/// # fn main() -> geo_rasterize::Result<()> {
/// use geo_rasterize::LabelBuilder;
///
/// let mut rasterizer = LabelBuilder::background(0i32).width(4).height(5).build()?;
/// # Ok(())}
/// ```
#[derive(Debug, Clone, Default)]
pub struct LabelBuilder<Label> {
    background: Label,
    width: Option<usize>,
    height: Option<usize>,
    geo_to_pix: Option<Transform>,
    algorithm: Option<MergeAlgorithm>,
}

impl<Label> LabelBuilder<Label>
where
    Label: Copy + Add<Output = Label>,
{
    pub fn background(background: Label) -> Self {
        LabelBuilder {
            background,
            width: None,
            height: None,
            geo_to_pix: None,
            algorithm: None,
        }
    }

    pub fn width(mut self, width: usize) -> Self {
        self.width = Some(width);
        self
    }

    pub fn height(mut self, height: usize) -> Self {
        self.height = Some(height);
        self
    }

    pub fn geo_to_pix(mut self, geo_to_pix: Transform) -> Self {
        self.geo_to_pix = Some(geo_to_pix);
        self
    }

    pub fn algorithm(mut self, algorithm: MergeAlgorithm) -> Self {
        self.algorithm = Some(algorithm);
        self
    }

    pub fn build(self) -> Result<LabelRasterizer<Label>> {
        match (self.width, self.height) {
            (None, _) => Err(RasterizeError::MissingWidth),
            (_, None) => Err(RasterizeError::MissingHeight),
            (Some(width), Some(height)) => Ok(LabelRasterizer::new(
                width,
                height,
                self.geo_to_pix,
                self.algorithm.unwrap_or_default(),
                self.background,
            )),
        }
    }
}

/// [LabelRasterizer] rasterizes shapes like [BinaryRasterizer] but
/// instead of making a boolean array, it produces an array of some
/// generic type (`Label`) that implements [Copy][std::marker::Copy]
/// and [Add][std::ops::Add] though typically you'd use a numeric
/// type.
///
/// You can call [new][LabelRasterizer::new] or use [LabelBuilder] to
/// construct [LabelRasterizer] instances. Constructing one requires a
/// `width`, `height` and optional `geo_to_pix` transform in addition
/// to `background` which specifies the default `Label` value used to
/// fill the raster array. And you can provide a [MergeAlgorithm]
/// value to specify what the rasterizer should do when two different
/// shapes fill the same pixel. If you don't supply anything, the
/// rasterizer will use [Replace][MergeAlgorithm::Replace] by default.
///
/// ```rust
/// # use geo_rasterize::{Result, LabelBuilder, LabelRasterizer};
/// # fn main() -> Result<()> {
/// use geo::{Geometry, Line, Point};
/// use ndarray::array;
///
/// let point = Point::new(3, 4);
/// let line = Line::new((0, 3), (3, 0));
///
/// let mut rasterizer = LabelBuilder::background(0).width(4).height(5).build()?;
/// rasterizer.rasterize(&point, 7)?;
/// rasterizer.rasterize(&line, 3)?;
///
/// let pixels = rasterizer.finish();
/// assert_eq!(
///     pixels.mapv(|v| v as u8),
///     array![
///         [0, 0, 3, 0],
///         [0, 3, 3, 0],
///         [3, 3, 0, 0],
///         [3, 0, 0, 0],
///         [0, 0, 0, 7]
///     ]
/// );
/// # Ok(())}
/// ```
pub struct LabelRasterizer<Label> {
    // We store pixels transposed relative to [BinaryRasterizer],
    // which means we don't have to call `reversed_axes`.
    pixels: Array2<Label>,
    geo_to_pix: Option<Transform>,
    algorithm: MergeAlgorithm,
    background: Label,
}

impl<Label> LabelRasterizer<Label>
where
    Label: Copy + Add<Output = Label>,
{
    pub fn new(
        width: usize,
        height: usize,
        geo_to_pix: Option<Transform>,
        algorithm: MergeAlgorithm,
        background: Label,
    ) -> Self {
        let pixels = Array2::from_elem((height, width), background);
        LabelRasterizer {
            pixels,
            geo_to_pix,
            algorithm,
            background,
        }
    }

    fn width(&self) -> usize {
        self.pixels.shape()[1]
    }

    fn height(&self) -> usize {
        self.pixels.shape()[0]
    }

    fn binary(&self) -> Result<BinaryRasterizer> {
        BinaryRasterizer::new(self.width(), self.height(), self.geo_to_pix)
    }

    /// Rasterize one shape using the supplied foreground label.
    pub fn rasterize<Coord, InputShape, ShapeAsF64>(
        &mut self,
        shape: &InputShape,
        foreground: Label,
    ) -> Result<()>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize + for<'a> CoordsIter<'a, Scalar = f64> + MapCoordsInplace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        let mut binary = self.binary()?;
        binary.rasterize(shape)?;
        let binary_pixels = binary.finish();

        match self.algorithm {
            MergeAlgorithm::Add => {
                self.pixels
                    .zip_mut_with(&binary_pixels, |destination: &mut Label, &source| {
                        *destination =
                            *destination + if source { foreground } else { self.background };
                    })
            }
            MergeAlgorithm::Replace => {
                self.pixels
                    .zip_mut_with(&binary_pixels, |destination: &mut Label, &source| {
                        if source {
                            *destination = foreground;
                        }
                    })
            }
        };
        Ok(())
    }

    /// Retrieve the completed raster array.
    pub fn finish(self) -> Array2<Label> {
        self.pixels
    }
}

impl Rasterize for Point<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        if self.x() >= 0. && self.y() >= 0. {
            let x = self.x().floor() as usize;
            let y = self.y().floor() as usize;
            if x < rasterizer.width() && y < rasterizer.height() {
                rasterizer.pixels[(x, y)] = true;
            }
        }
    }
}

impl Rasterize for MultiPoint<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        self.iter().for_each(|point| point.rasterize(rasterizer));
    }
}

impl Rasterize for Rect<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        // Although it is tempting to make a really fast direct
        // implementation, we're going to convert to a polyon and rely
        // on that impl, in part because affine transforms can easily
        // rotate or shear the rectangle so that it is no longer axis
        // aligned.
        self.to_polygon().rasterize(rasterizer);
    }
}

impl Rasterize for Line<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        rasterize_line(self, rasterizer);
    }
}

impl Rasterize for LineString<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        // It is tempting to make this impl treat closed `LineString`s
        // as polygons without holes: to just fill them. GDAL seems
        // like it should do that (`gv_rasterize_one_shape` in
        // `gdalrasterize.cpp` has a default clause that just invokes
        // the polygon filling code), but in practice GDAL treats
        // closed `LinearRings` as `LineSegments` and doesn't fill
        // them and I'm not sure why. Perhaps `LinearRings` are more
        // of an internal implementation detail?
        self.lines().for_each(|line| line.rasterize(rasterizer));
    }
}

impl Rasterize for MultiLineString<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        self.iter()
            .for_each(|line_string| line_string.rasterize(rasterizer));
    }
}

impl Rasterize for Polygon<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        rasterize_polygon(self.exterior(), self.interiors(), rasterizer);
    }
}

impl Rasterize for MultiPolygon<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        self.iter().for_each(|poly| poly.rasterize(rasterizer));
    }
}

impl Rasterize for Triangle<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        self.to_polygon().rasterize(rasterizer)
    }
}

impl Rasterize for Geometry<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        match self {
            Geometry::Point(point) => point.rasterize(rasterizer),
            Geometry::Line(line) => line.rasterize(rasterizer),
            Geometry::LineString(ls) => ls.rasterize(rasterizer),
            Geometry::Polygon(poly) => poly.rasterize(rasterizer),
            Geometry::GeometryCollection(gc) => gc.rasterize(rasterizer),
            Geometry::MultiPoint(points) => points.rasterize(rasterizer),
            Geometry::MultiLineString(lines) => lines.rasterize(rasterizer),
            Geometry::MultiPolygon(polys) => polys.rasterize(rasterizer),
            Geometry::Rect(rect) => rect.rasterize(rasterizer),
            Geometry::Triangle(tri) => tri.rasterize(rasterizer),
        }
    }
}

impl Rasterize for GeometryCollection<f64> {
    fn rasterize(&self, rasterizer: &mut BinaryRasterizer) {
        self.iter().for_each(|thing| thing.rasterize(rasterizer));
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use anyhow::Result;
    use geo::{polygon, Coordinate};
    use ndarray::array;
    use pretty_assertions::assert_eq;

    /// Use `gdal`'s rasterizer to rasterize some shape into a
    /// (widith, height) window of u8.
    pub fn gdal_rasterize<Coord, InputShape, ShapeAsF64>(
        width: usize,
        height: usize,
        shape: &InputShape,
    ) -> Result<Array2<u8>>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize
            + for<'a> CoordsIter<'a, Scalar = f64>
            + Into<geo::Geometry<f64>>
            + MapCoordsInplace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        let float = shape.map_coords(to_float);
        let all_finite = float
            .coords_iter()
            .all(|coordinate| coordinate.x.is_finite() && coordinate.y.is_finite());
        assert!(all_finite);

        let geom: geo::Geometry<f64> = float.into();

        use gdal::{
            raster::{rasterize, RasterizeOptions},
            vector::ToGdal,
            Driver,
        };

        let gdal_geom = geom.to_gdal()?;
        let driver = Driver::get("MEM")?;
        let mut ds = driver.create_with_band_type::<u8, &str>(
            "some_filename",
            width as isize,
            height as isize,
            1,
        )?;
        let options = RasterizeOptions {
            all_touched: true,
            ..Default::default()
        };
        rasterize(&mut ds, &[1], &[gdal_geom], &[1.0], Some(options))?;
        ds.rasterband(1)?
            .read_as_array((0, 0), (width, height), (width, height), None)
            .map_err(|e| e.into())
    }

    pub fn compare<Coord, InputShape, ShapeAsF64>(
        width: usize,
        height: usize,
        shape: &InputShape,
    ) -> Result<(Array2<u8>, Array2<u8>)>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize
            + for<'a> CoordsIter<'a, Scalar = f64>
            + Into<geo::Geometry<f64>>
            + MapCoordsInplace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        let mut r = BinaryRasterizer::new(width, height, None)?;
        r.rasterize(shape)?;
        // Why switch to u8? Because it makes looking at arrays on the
        // console much easier.
        let actual = r.finish().mapv(|v| v as u8);

        let expected = gdal_rasterize(width, height, shape)?;
        if actual != expected {
            println!("{}\n\n{}", actual, expected);
        }
        Ok((actual, expected))
    }

    #[test]
    fn point() -> Result<()> {
        let (actual, expected) = compare(6, 5, &geo::Point::new(2, 3))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn rect() -> Result<()> {
        // let (actual, expected) = compare(6, 6, &geo::Rect::new((0, 1), (2, 3)))?;
        // assert_eq!(actual, expected);
        let (actual, expected) = compare(6, 5, &geo::Rect::new((1, 1), (3, 2)).to_polygon())?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn line() -> Result<()> {
        let (actual, expected) = compare(6, 5, &geo::Line::new((0, 0), (6, 6)))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn line2() -> Result<()> {
        let (actual, expected) = compare(6, 6, &geo::Line::new((6, 6), (0, 0)))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn line_vertical() -> Result<()> {
        let (actual, expected) = compare(5, 5, &geo::Line::new((2, 1), (2, 4)))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn line_horizontal() -> Result<()> {
        let (actual, expected) = compare(5, 5, &geo::Line::new((1, 2), (4, 2)))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn line_diag() -> Result<()> {
        let (actual, expected) = compare(5, 5, &geo::Line::new((1, 1), (3, 3)))?;
        assert_eq!(actual, expected);

        let (actual, expected) = compare(5, 5, &geo::Line::new((3, 1), (1, 3)))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn line3() -> Result<()> {
        let poly = polygon![
            (x:4, y:2),
            (x:2, y:0),
            (x:0, y:2),
            (x:2, y:4),
            (x:4, y:2),
        ];

        // FIXME: I think our problem here is that geo has no type to
        // represent LinearRings but GDAL does, so when we insist on
        // converting closed linestrings as linear rings, GDAL doesn't
        // do that automatically. maybe try updating the gdal
        // invocation function to covert geometries of that type into
        // GDAL LinearRings?

        let (actual, expected) = compare(5, 5, poly.exterior())?;
        assert_eq!(actual, expected);

        for line in poly.exterior().lines() {
            let (actual, expected) = compare(5, 5, &line)?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn poly() -> Result<()> {
        let poly = polygon![
            (x:4, y:2),
            (x:2, y:0),
            (x:0, y:2),
            (x:2, y:4),
            (x:4, y:2),
        ];

        let (actual, expected) = compare(5, 5, &poly)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn label_multiple() -> Result<()> {
        use ndarray::array;

        let point = Point::new(3, 4);
        let line = Line::new((0, 3), (3, 0));

        let mut rasterizer = LabelBuilder::background(0).width(4).height(5).build()?;
        rasterizer.rasterize(&point, 3)?;
        rasterizer.rasterize(&line, 7)?;

        let pixels = rasterizer.finish();
        assert_eq!(
            pixels.mapv(|v| v as u8),
            array![
                [0, 0, 7, 0],
                [0, 7, 7, 0],
                [7, 7, 0, 0],
                [7, 0, 0, 0],
                [0, 0, 0, 3]
            ]
        );
        Ok(())
    }

    #[test]
    fn heatmap() -> Result<()> {
        let lines = vec![Line::new((0, 0), (5, 5)), Line::new((5, 0), (0, 5))];

        let mut rasterizer = LabelBuilder::background(0)
            .width(5)
            .height(5)
            .algorithm(MergeAlgorithm::Add)
            .build()?;
        for line in lines {
            rasterizer.rasterize(&line, 1)?;
        }

        let pixels = rasterizer.finish();
        assert_eq!(
            pixels.mapv(|v| v as u8),
            array![
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 1],
                [0, 0, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 0, 1]
            ]
        );
        Ok(())
    }

    #[test]
    fn bad_line() -> Result<()> {
        let line = Line {
            start: Coordinate {
                x: -1.984208921953521,
                y: 17.310676190851567,
            },
            end: Coordinate {
                x: 0.0,
                y: 17.2410885032527,
            },
        };
        let (actual, expected) = compare(17, 19, &line)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn bad_rect() -> Result<()> {
        let r = Rect::new(
            Coordinate {
                x: -5.645366376556284,
                y: -8.757910782301106,
            },
            Coordinate {
                x: 5.645366376556284,
                y: 8.757910782301106,
            },
        );

        let (actual, expected) = compare(17, 19, &r)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn bad_poly() -> Result<()> {
        use geo::line_string;
        let r = Polygon::new(
            line_string![
                Coordinate {
                    x: 8.420838780938684,
                    y: 0.0,
                },
                Coordinate {
                    x: -4.21041939046934,
                    y: 7.292660305466085,
                },
                Coordinate {
                    x: -4.210419390469346,
                    y: -7.2926603054660815,
                },
                Coordinate {
                    x: 8.420838780938684,
                    y: 0.0,
                }
            ],
            vec![],
        );
        let (actual, expected) = compare(17, 19, &r)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn bad_poly2() -> Result<()> {
        use geo::line_string;
        let r = Polygon::new(
            line_string![
                Coordinate {
                    x: 19.88238653081379,
                    y: 0.0
                },
                Coordinate {
                    x: -0.3049020763576378,
                    y: 11.65513651155909
                },
                Coordinate {
                    x: -0.30490207635764666,
                    y: -11.655136511559085
                },
                Coordinate {
                    x: 19.88238653081379,
                    y: 0.0
                }
            ],
            vec![],
        );
        let (actual, expected) = compare(17, 19, &r)?;
        assert_eq!(actual, expected);
        Ok(())
    }
}

// trait Storage {
//     fn fill_pixel(&mut self, ix: usize, iy: usize);
//     fn fill_block(&mut self, xs: RangeInclusive<usize>, ys: RangeInclusive<usize>);
//     fn into_array(self) -> Result<Array2<bool>>;
//     fn into_subset(self) -> Result<(Array2<bool>, Rect<usize>)>;
// }

// TODO

// proptest with a gdal cache oracle that uses [kv] for persistence
