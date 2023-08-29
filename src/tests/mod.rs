use crate::*;
use anyhow::Result;
use geo::polygon;
use ndarray::array;
use pretty_assertions::assert_eq;

pub mod utils;
use utils::compare;

#[test]
fn point() -> Result<()> {
    let (actual, expected) = compare(6, 5, &[Point::new(2, 3)], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn rect() -> Result<()> {
    // let (actual, expected) = compare(6, 6, &geo::Rect::new((0, 1), (2, 3)))?;
    // assert_eq!(actual, expected);
    let (actual, expected) = compare(
        6,
        5,
        &[Rect::new((1, 1), (3, 2)).to_polygon()],
        MergeAlgorithm::Replace,
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn line() -> Result<()> {
    let (actual, expected) = compare(
        6,
        5,
        &[Line::new((0, 0), (6, 6))],
        MergeAlgorithm::Replace,
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn line2() -> Result<()> {
    let (actual, expected) = compare(
        6,
        6,
        &[Line::new((6, 6), (0, 0))],
        MergeAlgorithm::Replace,
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn line_vertical() -> Result<()> {
    let (actual, expected) = compare(
        5,
        5,
        &[Line::new((2, 1), (2, 4))],
        MergeAlgorithm::Replace,
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn line_horizontal() -> Result<()> {
    let (actual, expected) = compare(
        5,
        5,
        &[Line::new((1, 2), (4, 2))],
        MergeAlgorithm::Replace,
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn line_diag() -> Result<()> {
    let (actual, expected) = compare(
        5,
        5,
        &[Line::new((1, 1), (3, 3))],
        MergeAlgorithm::Replace,
    )?;
    assert_eq!(actual, expected);

    let (actual, expected) = compare(
        5,
        5,
        &[Line::new((3, 1), (1, 3))],
        MergeAlgorithm::Replace,
    )?;
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

    let (actual, expected) = compare(5, 5, &[poly.exterior().clone()], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);

    for line in poly.exterior().lines() {
        let (actual, expected) = compare(5, 5, &[line], MergeAlgorithm::Replace)?;
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

    let (actual, expected) = compare(5, 5, &[poly], MergeAlgorithm::Replace)?;
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
fn heatmap1() -> Result<()> {
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
fn heatmap_transform() -> Result<()> {
    let lines = vec![Line::new((0, 0), (5, 5)), Line::new((5, 0), (0, 5))];
    let transform = Transform::identity();

    let mut rasterizer = LabelBuilder::background(0)
        .width(5)
        .height(5)
        .algorithm(MergeAlgorithm::Add)
        .geo_to_pix(transform)
        .build()?;
    assert_eq!(rasterizer.geo_to_pix(), Some(transform));

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
        start: Coord {
            x: -1.984208921953521,
            y: 17.310676190851567,
        },
        end: Coord {
            x: 0.0,
            y: 17.2410885032527,
        },
    };
    let (actual, expected) = compare(17, 19, &[line], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn bad_rect() -> Result<()> {
    let r = Rect::new(
        Coord {
            x: -5.645366376556284,
            y: -8.757910782301106,
        },
        Coord {
            x: 5.645366376556284,
            y: 8.757910782301106,
        },
    );

    let (actual, expected) = compare(17, 19, &[r], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn bad_poly() -> Result<()> {
    use geo::line_string;
    let r = Polygon::new(
        line_string![
            Coord {
                x: 8.420838780938684,
                y: 0.0,
            },
            Coord {
                x: -4.21041939046934,
                y: 7.292660305466085,
            },
            Coord {
                x: -4.210419390469346,
                y: -7.2926603054660815,
            },
            Coord {
                x: 8.420838780938684,
                y: 0.0,
            }
        ],
        vec![],
    );
    let (actual, expected) = compare(17, 19, &[r], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn bad_poly2() -> Result<()> {
    use geo::line_string;
    let r = Polygon::new(
        line_string![
            Coord {
                x: 19.88238653081379,
                y: 0.0
            },
            Coord {
                x: -0.3049020763576378,
                y: 11.65513651155909
            },
            Coord {
                x: -0.30490207635764666,
                y: -11.655136511559085
            },
            Coord {
                x: 19.88238653081379,
                y: 0.0
            }
        ],
        vec![],
    );
    let (actual, expected) = compare(17, 19, &[r], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn bad_line2() -> Result<()> {
    let line = Line {
        start: Coord {
            x: 0.0,
            y: 0.995529841217325,
        },
        end: Coord {
            x: 14.345339055640835,
            y: 1.003085512751344,
        },
    };
    let (actual, expected) = compare(17, 19, &[line], MergeAlgorithm::Replace)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn stuff() -> Result<()> {
    BinaryBuilder::new()
        .width(3)
        .height(4)
        .geo_to_pix(Transform::identity())
        .build()?;

    let r = BinaryRasterizer::new(5, 7, None)?;
    assert_eq!(r.geo_to_pix(), None);
    let transform = Transform::identity();
    let r = BinaryRasterizer::new(5, 7, Some(transform))?;
    assert_eq!(r.geo_to_pix(), Some(transform));
    Ok(())
}

#[test]
fn check_errors() -> Result<()> {
    let point = Point::new(123., f64::NAN);
    assert_eq!(
        BinaryBuilder::new()
            .height(4)
            .width(7)
            .build()?
            .rasterize(&point),
        Err(RasterizeError::NonFiniteCoordinate)
    );

    assert_eq!(
        BinaryBuilder::new().width(4).build().err().unwrap(),
        RasterizeError::MissingHeight
    );

    assert_eq!(
        BinaryBuilder::new().height(4).build().err().unwrap(),
        RasterizeError::MissingWidth
    );

    assert_eq!(
        LabelBuilder::background(0).width(4).build().err().unwrap(),
        RasterizeError::MissingHeight
    );

    assert_eq!(
        LabelBuilder::background(0).height(4).build().err().unwrap(),
        RasterizeError::MissingWidth
    );

    assert_eq!(
        BinaryRasterizer::new(
            5,
            8,
            Some(Transform::from_array([0., 1., 2., 3., 4., f64::NAN]))
        )
        .err()
        .unwrap(),
        RasterizeError::NonFiniteCoordinate
    );
    Ok(())
}

#[test]
fn heatmap2() -> Result<()> {
    let lines = vec![Line::new((0, 0), (5, 5)), Line::new((5, 0), (0, 5))];
    let (actual, expected) = compare(5, 5, &lines, MergeAlgorithm::Add)?;
    assert_eq!(actual, expected);
    Ok(())
}
