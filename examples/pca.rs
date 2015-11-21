extern crate csv;
extern crate brasswheels;

use brasswheels::io::read_csv_f64;
use brasswheels::pca::PCA;
use brasswheels::mathfunc::round;

fn main() {
    // cargo build --example pca
    // ./target/debug/examples/pca

    // use "iris" data
    // http://aima.cs.berkeley.edu/data/iris.csv
    let path = "./data/iris.csv";
    let mut reader = csv::Reader::from_file(path).unwrap().has_headers(false);
    let dx = read_csv_f64(&mut reader);

    let ncols = dx.ncols();

    let mut pca = PCA::new(ncols, true);
    pca.fit(&dx);
    println!("Principal Components (center=true)\n{:?}", &round(&mut pca.rotation, 5));
    println!("Principal Component Scores\n{:?}", &round(&mut pca.transform(&dx), 5));

    let mut pca = PCA::new(ncols, false);
    pca.fit(&dx);
    println!("Principal Components (center=true)\n{:?}", &round(&mut pca.rotation, 5));
    println!("Principal Component Scores\n{:?}", &round(&mut pca.transform(&dx), 5));
}
