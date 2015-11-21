extern crate csv;
extern crate gnuplot;
extern crate nalgebra;
extern crate num;
extern crate rand;

extern crate brasswheels;

use gnuplot::{Figure, Color};
use nalgebra::{Iterable};
use std::ops::Index;

use brasswheels::io::read_csv_f64;
use brasswheels::kmeans::KMeans;
use brasswheels::pca::PCA;

fn main() {
    // cargo build --example kmeans
    // ./target/debug/examples/kmeans

    // http://aima.cs.berkeley.edu/data/iris.csv
    let path = "./data/iris.csv";
    let mut reader = csv::Reader::from_file(path).unwrap().has_headers(false);
    let dx = read_csv_f64(&mut reader);

    // k-means
    let mut kmeans = KMeans::new(3, 300);
    kmeans.fit(&dx);

    println!("各クラスタの中心");
    for (_, cluster) in &kmeans.centroids {
        println!("{:?}", cluster.centroid.at);
    }

    let predicted = kmeans.predict(&dx);
    println!("結果\n{:?}", &predicted.at);

    // 主成分分析
    let mut pca = PCA::new(4, true);
    pca.fit(&dx);
    let transformed = pca.transform(&dx);

    // プロット
    // http://siegelord.github.io/RustGnuplot/doc/gnuplot/struct.Axes2D.html
    let mut fg = Figure::new();
    let colors = ["blue", "red", "green"];

    // fg.axes2d() によって fg が move するため、
    // 同一ブロック中で呼び出すと fg.show() が使えなくなる
    (0..kmeans.nclusters).fold(fg.axes2d(),
        |ax, c| {
            let mut xvals: Vec<f64> = vec![];
            let mut yvals: Vec<f64> = vec![];

            for (rownum, &predc) in predicted.iter().enumerate() {
                if predc == c {
                    xvals.push(*transformed.index((rownum, 0)));
                    yvals.push(*transformed.index((rownum, 1)));
                }
            }
            return ax.points(&xvals, &yvals, &[Color(colors[c])]);
        });

    fg.set_terminal("png", "kmeans.png");
    fg.show();
}
