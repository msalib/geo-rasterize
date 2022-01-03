# geo-rasterize: a pure-rust 2D rasterizer for geospatial applications

[![Crates.io][crates-badge]][crates-url]
[![Docs.rs][docs-badge]][docs-url]
[![CodeCov.io][codecov-badge]][codecov-url]
[![Build Status][actions-badge]][actions-url]
[![Python wrapper][py-badge]][py-pkg]

[crates-badge]: https://img.shields.io/crates/v/geo-rasterize.svg
[crates-url]: https://crates.io/crates/geo-rasterize
[docs-badge]: https://img.shields.io/docsrs/geo-rasterize
[docs-url]: https://docs.rs/geo-rasterize/latest/geo_rasterize/
[codecov-badge]: https://img.shields.io/codecov/c/github/msalib/geo-rasterize
[codecov-url]: https://app.codecov.io/gh/msalib/geo-rasterize/
[actions-badge]: https://github.com/msalib/geo-rasterize/actions/workflows/CI.yml/badge.svg
[actions-url]: https://github.com/msalib/geo-rasterize/actions?query=CI+branch%3Amain
[py-badge]: https://img.shields.io/pypi/v/geo-rasterize?style=plastic
[py-pkg]: https://pypi.org/project/geo-rasterize/

This crate is intended for folks who have some vector data (like a
`geo::Polygon`) and a raster source (like a GeoTiff perhaps opened
with `GDAL`) and who want to generate a boolean array representing
which bits of raster are filled in by the polygon. There's also a
[Python wrapper][py-pkg] available.

This implementation is based on `GDAL`'s `GDALRasterizeGeometries` and
allows you to rasterize any type supported by the `geo-types` package,
including:
* [Point](geo::Point)
* [Line](geo::Line)
* [LineString](geo::LineString)
* [Polygon](geo::Polygon)
* [Rect](geo::Rect) and [Triangle](geo::Triangle)
* [MultiPoint](geo::MultiPoint), [MultiLineString](geo::MultiLineString), and [MultiPolygon](geo::MultiPolygon)
* [Geometry](geo::Geometry) and [GeometryCollection](geo::GeometryCollection)

Those shapes can have any coordintes with any numeric type as long as
it can be converted into `f64`.

This crate matches GDAL's behavior when GDAL is supplied with the
`ALL_TOUCHED=TRUE` option. So you can use it as a drop-in replacement
for GDAL if you only need a GDAL-compatible rasterizer. Also, there's
no support for GDAL's `BURN_VALUE_FROM=Z`. But otherwise, this code
should produce identical results to GDAL's rasterizer -- the
rasterization algorithm is a direct port. We use
[proptest](https://crates.io/crates/proptest) to perform randomized
differential comparisons with GDAL in order bolster confidence about
our conformance.



<!-- there are different conventions for representing
image data in an array....image matrices are stored in blah blah order
-->


## Motivation: satellite imagery data analysis

Let's say you're interested in the free 10m resolution data from ESA's
[Sentinel-2](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2)
satellite mission. You might be especially interested in how farms
change over time, and you've got a bunch of farms represented as
polygons. You fetch some Sentinel-2 data from
[AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/). Since
Sentinel-2 tiles are so large (over 110 million pixels!) and since
they're stored as [Cloud Optimized GeoTiffs](https://www.cogeo.org/),
you convert your polygons into windows, and extract those windows from
the image tiles; that way you only have to download the pixels that
you care about.

But now you have a problem! Your polygons are not perfect rectangles
axis-aligned to the Sentinel-2 tiling system. So while you have small
field chips, you don't know which parts of those chips correspond to
your polygons. Even worse, some of your polygons have holes (for
example, to represent houses or ponds on the farms). That's where a
[geo-rasterize] comes in! Using [geo-rasterize], you can convert your
field polygons into a binary raster just like your Sentinel-2 field
chips. And you can use those mask chips to select which pixels of the
Sentinel-2 chips you care about. Filtering on the masks, you can now
generate time series for imagery, secure in the knowledge that you're
only examining pixels within your polygons!

## Binary rasterization

Let's say you want to rasterize a polygon into a grid 4 pixels wide by
5 pixels high. To that, you simply construct a [BinaryRasterizer]
using [BinaryBuilder], call [rasterize](BinaryRasterizer::rasterize)
with your polygon and call [finish](BinaryRasterizer::finish) to get
an [Array2<bool>](ndarray::Array2) of booleans.

```rust
# fn main() -> geo_rasterize::Result<()> {
use geo::polygon;
use ndarray::array;
use geo_rasterize::BinaryBuilder;

let poly = polygon![
    (x:4, y:2),
    (x:2, y:0),
    (x:0, y:2),
    (x:2, y:4),
    (x:4, y:2),
];

let mut r = BinaryBuilder::new().width(4).height(5).build()?;
r.rasterize(&poly)?;
let pixels = r.finish();

assert_eq!(
    pixels.mapv(|v| v as u8),
    array![
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 0]
    ]
);
# Ok(()) }

```

## ...with multiple shapes
But what if you want to rasterize several geometries? That's easy enough!

```rust
# fn main() -> geo_rasterize::Result<()> {
use geo::{Geometry, Line, Point};
use ndarray::array;
use geo_rasterize::BinaryBuilder;

let shapes: Vec<Geometry<i32>> =
    vec![Point::new(3, 4).into(),
	     Line::new((0, 3), (3, 0)).into()];

let mut r = BinaryBuilder::new().width(4).height(5).build()?;
for shape in shapes {
    r.rasterize(&shape)?;
}

let pixels = r.finish();
assert_eq!(
    pixels.mapv(|v| v as u8),
    array![
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
);
# Ok(())}
```

## Labeling (non-binary rasterization)

So far we've been generating binary arrays; what if you want to
rasterize different shapes to the same integer array, storing a
different value corresponding to each shape for each pixel? For that,
we have [Rasterizer] which we construct using
[LabelBuilder]. When you burn a shape with [Rasterizer] you
provide not just the shape, but also a foreground label. But before
you can burn anything, you have to specify a background label used to
fill the empty raster array.

```rust
# use geo_rasterize::{Result, LabelBuilder, Rasterizer};
# fn main() -> Result<()> {
use geo::{Geometry, Line, Point};
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
# Ok(())}
```

## Heatmaps

What happens if two shapes touch the same pixel? In the example above,
the last shape written wins. But you can change that behavior by
specifying a different value for [MergeAlgorithm] using
[LabelBuilder::algorithm]. In fact, using [MergeAlgorithm::Add], you
can easily make a heat map showing the shape density where each pixel value tells
you the number of shapes that landed on it!

```rust
# use geo_rasterize::{Result, LabelBuilder, Rasterizer, MergeAlgorithm};
# use geo::{Geometry, Line, Point};
# use ndarray::array;
# fn main() -> Result<()> {

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

# Ok(())}
```

Two lines cross at the center where you'll find `2`. Note that
[Rasterizer] is not limited to integers; any copyable type that
can be added will do. [Rasterizer] offers similar functionality
to [rasterio](https://rasterio.readthedocs.io/)'s
[features.rasterize](https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.rasterize)
function.


## Geographic transforms

All our examples so far have assumed that our shapes' coordinates are
in the image space. In other words, we've assumed that the `x`
coordinates will be in the range `0..width` and the `y` coordinates
will be in the range `0..height`. Alas, that is often not the case!

For satellite imagery (or remote sensing imagery in general), images
will almost always specify both a Coordinate Reference System
([CRS](https://en.wikipedia.org/wiki/Spatial_reference_system)) and an
affine transformation in their metadata. See [rasterio's
Georeferencing](https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html)
for more details.

In order to work with most imagery, you have to convert your vector
shapes from whatever their original CRS is (often `EPSG:4326` for
geographic longitude and latitude) into whatever CRS your data file
specifies (often a
[UTM](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system)
projection but there are so many choices). Then, you need to apply an
affine transformation to convert from world coordinates to pixel
coordinates. Since raster imagery usually specifies the inverse
transformation matrix (i.e. a `pix_to_geo` transform), you'll first
need to invert it to get a `geo_to_pix` transform before applying it
to the coordinates. And now you've got pixel coordinates appropriate
for your image data!

[BinaryRasterizer] and [Rasterizer] can ease this tedious process
by taking care of the affine transformation. Make sure to pass a
[Transform] object to [BinaryBuilder] or [LabelBuilder]. In either
case, that transform is a `geo_to_pix` transform, which means you'll
have to:

* extract the CRS from your image and convert your shapes into that
  CRS (probably using [the proj
  crate](https://docs.rs/proj/0.24.0/proj/index.html#integration-with-geo-types)
  and its integration with [geo types][geo],
* extract the `pix_to_geo` transform from your imagery metadata
* create a [Transform] instance from that data (GDAL represents these
  as a `[f64; 6]` array)
* call `transform.inverse` to get the corresponding `geo_to_pix`
  transform (since not all transforms are invertible, `inverse` gives
  you an `Option`)
* pass the resulting [Transform] to either [BinaryBuilder] or
  [LabelBuilder].

## Performance

For polygons, our runtime is `O(S * P * log(P))` where `S` is the
number of scanlines (the polygon's vertical extent in pixels) and `P`
is the number of coordinates in the polygon exterior and all its
holes. Memory consumption is approximately `P` machine words. Because
runtime depends so heavily on the number of coordinates, simplifying
polygons before rasterization can speed up rasterization dramatically,
especially in cases where polygons have very high resolution compared
to the pixel size.

For other shapes, runtime is proportional to the number of pixels
filled in.

<!-- say something about [Rasterizer] memory consumption -->
<!-- compare gdal performance -->

## Why not GDAL?

GDAL is the swiss army chainsaw of geospatial data processing. It
handles vector data (by wrapping `libgeos`) and many data formats. The
version that ships with Ubuntu 21.10 links to 115 shared libraries
which includes support for handling PDF files, Excel spreadsheets,
curl, Kerberos, ODBC, several XML libraries, a linear algebra solver,
several cryptographic packages, and on and on and on. GDAL is a giant
pile of C and C++ code slapped together in a fragile
assembly. Building GDAL is a rather unpleasant since even a stripped
down version depends on a bunch of other C and C++ packages. If you
want to quickly build and deploy a static binary for AWS Lambda, rust
makes that really easy, right up until you need GDAL. Then things get
really really difficult.

Speaking of rust, I've been bitten multiple times in my career now
with GDAL data race bugs that rust just forbids. I'm so tired.

Configuring GDAL is deeply unpleasant. Quick! Look at the [GDAL
configuration guide](https://gdal.org/user/configoptions.html) and
tell me which of the 170ish configuration knobs I need to adjust to
control GDAL's caching so that a lambda function that uses GDAL won't
leak memory due to image caching? Ha! That's a trick question because
you need multiple tunables to control the different caches. That's
what you expect for a 23 year old 2.5 MLOC software library.

For a more pythonic perspective on the noGDAL movement, check out
[Kipling Crossing](https://kipcrossing.github.io/2021-01-03-noGDAL/).

## Alternative crates

* [GDAL](https://docs.rs/gdal/latest/gdal/raster/fn.rasterize.html)
  can rasterize but then you'll need to bring in GDAL which is
  difficult to deal with.
* [raqote](https://crates.io/crates/raqote) is a powerful 2D
  rasterizer intended for graphics.
* [rasterize](https://crates.io/crates/rasterize) is another pure rust
  library, but less mature than `raqote`.


## Contributing

Contributions are welcome! Have a look at the
[issues](https://github.com/msalib/geo-rasterize/issues), and open a
pull request if you'd like to add an algorithm or some functionality.

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
