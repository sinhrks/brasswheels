extern crate csv;
extern crate nalgebra;

extern crate brasswheels;

use nalgebra::{DMat, DVec};

use brasswheels::io::read_csv_f64;
use brasswheels::lm::LinearModel;

fn main() {
    // cargo build --example lm
    // ./target/debug/examples/lm

    // Use R's "trees" data
    let data = "Girth,Height,Volume
8.3,70,10.3
8.6,65,10.3
8.8,63,10.2
10.5,72,16.4
10.7,81,18.8
10.8,83,19.7
11.0,66,15.6
11.0,75,18.2
11.1,80,22.6
11.2,75,19.9
11.3,79,24.2
11.4,76,21.0
11.4,76,21.4
11.7,69,21.3
12.0,75,19.1
12.9,74,22.2
12.9,85,33.8
13.3,86,27.4
13.7,71,25.7
13.8,64,24.9
14.0,78,34.5
14.2,80,31.7
14.5,74,36.3
16.0,72,38.3
16.3,77,42.6
17.3,81,55.4
17.5,82,55.7
17.9,80,58.3
18.0,80,51.5
18.0,80,51.0
20.6,87,77.0";

    // http://burntsushi.net/rustdoc/csv/
    let mut reader = csv::Reader::from_string(data).has_headers(true);
    let dx = read_csv_f64(&mut reader);

    let mut xvalues: Vec<f64> = vec![];
    let mut yvalues: Vec<f64> = vec![];
    for (i, &v) in dx.as_vec().iter().enumerate() {
        if i < dx.nrows() {
            yvalues.push(v);
        } else {
            xvalues.push(v);
        }
    }
    let dy = DVec::from_slice(dx.nrows(), &yvalues);
    let dx = DMat::from_col_vec(dx.nrows(), dx.ncols() - 1, &xvalues);

    // Linear regression
    let mut lm = LinearModel::new();
    lm.fit(&dx, &dy);
    println!("Coefs: {:?}", &lm.coefs);
    println!("Predicted:\n{:?}", &lm.predict(&dx));
    println!("Actual:\n{:?}", &dy);
}
