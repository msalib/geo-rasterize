#![doc = include_str!("../README.md")]
use std::{collections::HashSet, fmt::Debug, ops::Add};

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
use num_traits::{Num, NumCast};
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
#[derive(Error, Clone, Debug, PartialEq, Eq)]
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
///          Line::new((0, 3), (3, 0)).into()];
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
    inner: Rasterizer<u8>,
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
            let inner = Rasterizer::new(width, height, geo_to_pix, MergeAlgorithm::Replace, 0);
            Ok(BinaryRasterizer { inner })
        }
    }

    /// Retrieve the transform.
    pub fn geo_to_pix(&self) -> Option<Transform> {
        self.inner.geo_to_pix
    }

    /// Rasterize one shape, which can be any type that [geo] provides
    /// using any coordinate numeric type that can be converted into
    /// `f64`.
    pub fn rasterize<Coord, InputShape, ShapeAsF64>(&mut self, shape: &InputShape) -> Result<()>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize<u8> + for<'a> CoordsIter<'a, Scalar = f64> + MapCoordsInplace<f64>,
        Coord: Into<f64> + Copy + Debug + Num + NumCast + PartialOrd,
    {
        // first, convert our input shape so that its coordinates are of type f64
        self.inner.rasterize(shape, 1)
    }

    /// Retrieve the completed raster array.
    pub fn finish(self) -> Array2<bool> {
        self.inner
            .finish()
            .mapv(|v| if v == 1u8 { true } else { false })
    }
}

#[doc(hidden)]
pub trait Rasterize<Label>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>);
}

/// Conflict resolution strategy for cases where two shapes cover the
/// same pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A builder that constructs [Rasterizer]s. Whereas
/// [BinaryRasterizer] produces an array of booleans, [Rasterizer]
/// produces an array of some generic type (`Label`) that implements
/// [Copy][std::marker::Copy] and [Add][std::ops::Add] though
/// typically you'd use a numeric type.
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

    pub fn build(self) -> Result<Rasterizer<Label>> {
        match (self.width, self.height) {
            (None, _) => Err(RasterizeError::MissingWidth),
            (_, None) => Err(RasterizeError::MissingHeight),
            (Some(width), Some(height)) => Ok(Rasterizer::new(
                width,
                height,
                self.geo_to_pix,
                self.algorithm.unwrap_or_default(),
                self.background,
            )),
        }
    }
}

/// [Rasterizer] rasterizes shapes like [BinaryRasterizer] but instead
/// of making a boolean array, it produces an array of some generic
/// type (`Label`) that implements [Copy][std::marker::Copy] and
/// [Add][std::ops::Add] though typically you'd use a numeric type.
///
/// You can call [new][Rasterizer::new] or use [LabelBuilder] to
/// construct [Rasterizer] instances. Constructing one requires a
/// `width`, `height` and optional `geo_to_pix` transform in addition
/// to `background` which specifies the default `Label` value used to
/// fill the raster array. And you can provide a [MergeAlgorithm]
/// value to specify what the rasterizer should do when two different
/// shapes fill the same pixel. If you don't supply anything, the
/// rasterizer will use [Replace][MergeAlgorithm::Replace] by default.
///
/// ```rust
/// # use geo_rasterize::{Result, LabelBuilder, Rasterizer};
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
#[derive(Clone, Debug)]
pub struct Rasterizer<Label> {
    pixels: Array2<Label>,
    geo_to_pix: Option<Transform>,
    algorithm: MergeAlgorithm,
    foreground: Label,
    previous_burnt_points: HashSet<(usize, usize)>,
    current_burnt_points: HashSet<(usize, usize)>,
}

impl<Label> Rasterizer<Label>
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
        Rasterizer {
            pixels,
            geo_to_pix,
            algorithm,
            foreground: background,
            previous_burnt_points: HashSet::new(),
            current_burnt_points: HashSet::new(),
        }
    }

    fn width(&self) -> usize {
        self.pixels.shape()[1]
    }

    fn height(&self) -> usize {
        self.pixels.shape()[0]
    }

    /// Retrieve the transform.
    pub fn geo_to_pix(&self) -> Option<Transform> {
        self.geo_to_pix
    }

    // For MergeAlgorithm::Add, we have to ensure that we don't fill
    // in the same pixels twice for the same; this only matters for
    // the line drawing algorithm. To ensure that, we maintain a pair
    // of index sets: one describing the indices of pixels we've
    // filled in for the last line in the current line string and one
    // describing the indices of pixels we've filled in for the
    // current line of the current linestring. When we start a new
    // line string, we clear both. And when we advance from one line
    // to another within the same linestring, we swap them and clear
    // the current one. While drawing each line, for the
    // vertical/horizontal cases, we only refuse to fill a pixel if it
    // was filled in the previous iteration but double filling from
    // the current iteration is fine. But when drawing non-vertical
    // non-horizontal lines, we refuse to fill pixels if we've filled
    // them in either the current or previous iteration.

    // aka clear()
    fn new_linestring(&mut self) {
        self.previous_burnt_points.clear();
        self.current_burnt_points.clear();
    }

    fn new_line(&mut self) {
        std::mem::swap(
            &mut self.previous_burnt_points,
            &mut self.current_burnt_points,
        );
        self.current_burnt_points.clear();
    }

    fn fill_pixel(&mut self, ix: usize, iy: usize) {
        debug_assert!(ix < self.width());
        debug_assert!(iy < self.height());
        let mut slice = self.pixels.slice_mut(s![iy, ix]);
        match self.algorithm {
            MergeAlgorithm::Replace => slice.fill(self.foreground),
            MergeAlgorithm::Add => {
                slice.mapv_inplace(|v| v + self.foreground);
            }
        }
    }

    fn fill_pixel_no_repeat(&mut self, ix: usize, iy: usize, use_current_too: bool) {
        match self.algorithm {
            MergeAlgorithm::Replace => {
                self.fill_pixel(ix, iy);
            }
            MergeAlgorithm::Add => {
                let point = (ix, iy);
                let mut do_fill_pixel = !self.previous_burnt_points.contains(&point);
                if use_current_too {
                    do_fill_pixel = do_fill_pixel && !self.current_burnt_points.contains(&point);
                }
                if do_fill_pixel {
                    self.fill_pixel(ix, iy);
                    self.current_burnt_points.insert(point);
                }
            }
        }
    }

    // The rasterization algorithm's performance is extremely
    // sensitive to write ordering: it is focused on horizontal lines,
    // so it performs much better when pixels that are horizontally
    // adjacent are adjacent in memory (i.e., where the array we're
    // writing to has x as the last dimension).

    // Unlike the other fill_ methods, x_start and x_end are an
    // exclusive range (..), not inclusive (..=).
    fn fill_horizontal_line(&mut self, x_start: usize, x_end: usize, y: usize) {
        let mut slice = self.pixels.slice_mut(s![y, x_start..x_end]);
        match self.algorithm {
            MergeAlgorithm::Replace => slice.fill(self.foreground),
            MergeAlgorithm::Add => {
                slice.mapv_inplace(|v| v + self.foreground);
            }
        }
    }

    fn fill_horizontal_line_no_repeat(&mut self, x_start: usize, x_end: usize, y: usize) {
        for x in x_start..=x_end {
            self.fill_pixel_no_repeat(x, y, true);
        }
    }

    fn fill_vertical_line_no_repeat(&mut self, x: usize, y_start: usize, y_end: usize) {
        for y in y_start..=y_end {
            self.fill_pixel_no_repeat(x, y, false);
        }
    }

    /// Rasterize one shape, which can be any type that [geo] provides
    /// using any coordinate numeric type that can be converted into
    /// `f64`.
    pub fn rasterize<Coord, InputShape, ShapeAsF64>(
        &mut self,
        shape: &InputShape,
        foreground: Label,
    ) -> Result<()>
    where
        InputShape: MapCoords<Coord, f64, Output = ShapeAsF64>,
        ShapeAsF64: Rasterize<Label> + for<'a> CoordsIter<'a, Scalar = f64> + MapCoordsInplace<f64>,
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

        self.foreground = foreground;

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
    pub fn finish(self) -> Array2<Label> {
        self.pixels
    }
}

impl<Label> Rasterize<Label> for Point<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        if self.x() >= 0. && self.y() >= 0. {
            let x = self.x().floor() as usize;
            let y = self.y().floor() as usize;
            if x < rasterizer.width() && y < rasterizer.height() {
                rasterizer.fill_pixel(x, y);
            }
        }
    }
}

impl<Label> Rasterize<Label> for MultiPoint<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter().for_each(|point| point.rasterize(rasterizer));
    }
}

impl<Label> Rasterize<Label> for Rect<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        // Although it is tempting to make a really fast direct
        // implementation, we're going to convert to a polyon and rely
        // on that impl, in part because affine transforms can easily
        // rotate or shear the rectangle so that it is no longer axis
        // aligned.
        self.to_polygon().rasterize(rasterizer);
    }
}

impl<Label> Rasterize<Label> for Line<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        rasterizer.new_linestring();
        rasterize_line(self, rasterizer);
    }
}

impl<Label> Rasterize<Label> for LineString<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        // It is tempting to make this impl treat closed `LineString`s
        // as polygons without holes: to just fill them. GDAL seems
        // like it should do that (`gv_rasterize_one_shape` in
        // `gdalrasterize.cpp` has a default clause that just invokes
        // the polygon filling code), but in practice GDAL treats
        // closed `LinearRings` as `LineSegments` and doesn't fill
        // them and I'm not sure why. Perhaps `LinearRings` are more
        // of an internal implementation detail?
        rasterizer.new_linestring();
        self.lines().for_each(|line| {
            rasterizer.new_line();
            rasterize_line(&line, rasterizer);
        });
    }
}

impl<Label> Rasterize<Label> for MultiLineString<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter()
            .for_each(|line_string| line_string.rasterize(rasterizer));
    }
}

impl<Label> Rasterize<Label> for Polygon<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        rasterize_polygon(self.exterior(), self.interiors(), rasterizer);
    }
}

impl<Label> Rasterize<Label> for MultiPolygon<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter().for_each(|poly| poly.rasterize(rasterizer));
    }
}

impl<Label> Rasterize<Label> for Triangle<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.to_polygon().rasterize(rasterizer)
    }
}

impl<Label> Rasterize<Label> for Geometry<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
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

impl<Label> Rasterize<Label> for GeometryCollection<f64>
where
    Label: Copy + Add<Output = Label>,
{
    fn rasterize(&self, rasterizer: &mut Rasterizer<Label>) {
        self.iter().for_each(|thing| thing.rasterize(rasterizer));
    }
}

#[cfg(test)]
mod tests;
