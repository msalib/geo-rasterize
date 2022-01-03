use std::{fs::File, io::BufReader};

use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use flatgeobuf::{FallibleStreamingIterator, FgbReader};
use geo::map_coords::MapCoordsInplace;
use geo::prelude::*;
use geo::Geometry;
use geo::Polygon;
use geo_rasterize::LabelBuilder;
use geo_rasterize::{MergeAlgorithm, Rasterize, Rasterizer};
use geozero::ToGeo;
use ndarray::Array2;
use rand::prelude::*;

fn load() -> Result<Vec<Polygon<f64>>> {
    let mut f = BufReader::new(File::open("benches/us-county-2018.fgb")?);
    let mut fgb = FgbReader::open(&mut f)?;
    let mut result = Vec::with_capacity(fgb.select_all()?);
    while let Some(feature) = fgb.next()? {
        match feature.to_geo()? {
            Geometry::Polygon(poly) => {
                result.push(poly);
            }
            Geometry::MultiPolygon(mp) => mp.iter().for_each(|poly| {
                result.push(poly.clone());
            }),
            _ => {}
        }
    }
    Ok(result)
}

const WIDTH: usize = 501;
const HEIGHT: usize = 500;
const LEN: usize = 1_00;

fn transform(shapes: &[Polygon<f64>]) -> Result<Vec<Polygon<f64>>> {
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let angle = rng.gen_range(0. ..360.);
        let mut shape = shape.rotate(angle);

        // recenter at origin
        let center = shape.centroid().unwrap();
        shape.translate_inplace(-center.x(), -center.y());

        // scale
        let bounds = shape.bounding_rect().unwrap();
        let scale_factor = rng.gen_range(0.01..10.) * (WIDTH.max(HEIGHT) as f64)
            / bounds.width().max(bounds.height());
        shape.map_coords_inplace(|&(x, y)| (scale_factor * x, scale_factor * y));
        shape.translate_inplace(
            rng.gen_range(0..WIDTH) as f64,
            rng.gen_range(0..HEIGHT) as f64,
        );
        if result.len() > LEN {
            break;
        }
        result.push(shape);
    }
    Ok(result)
}

fn i_rasterize(shapes: &[Polygon<f64>]) -> Result<Array2<f32>> {
    let mut r = LabelBuilder::background(0.)
        .width(WIDTH)
        .height(HEIGHT)
        .algorithm(MergeAlgorithm::Add)
        .build()?;
    shapes.iter().try_for_each(|shape| r.rasterize(shape, 1.))?;
    Ok(r.finish())
}

#[path = "../src/tests/utils.rs"]
mod utils;
use utils::gdal_rasterize;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yo");
    group.sample_size(10);

    let shapes = transform(&load().unwrap()).unwrap();

    group.bench_function("me", |b| b.iter(|| i_rasterize(&shapes)));
    group.bench_function("gdal", |b| {
        b.iter(|| gdal_rasterize(WIDTH, HEIGHT, &shapes, MergeAlgorithm::Add))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// used for profiling with perf+hotspt:

// pub fn main() {
//     let shapes = transform(&load().unwrap()).unwrap();
//     dbg!(i_rasterize(&shapes).unwrap().sum());
//}
