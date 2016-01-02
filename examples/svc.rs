extern crate csv;
extern crate gnuplot;
extern crate nalgebra;
extern crate num;
extern crate rand;

extern crate brasswheels;

use gnuplot::{Figure, Color};
use nalgebra::{DVec, DMat, Iterable};
use std::ops::Index;

use brasswheels::io::read_csv_f64;
use brasswheels::svm::{SVC, SVC2, SVMTrait};

fn main() {
    // cargo build --example svc
    // ./target/debug/examples/svc

    let path = "./data/svm.csv";
    let mut reader = csv::Reader::from_file(path).unwrap().has_headers(false);
    let dx = read_csv_f64(&mut reader);

    // ToDo:: make it a function (used in lm.rs also)
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

    let mut svc2 = SVC2::new(10., 0.00000001, 1000000);
    svc2.fit(&dx, &dy);
    println!("alpha {:?}", &svc2.alpha);
    println!("b {:?}", &svc2.b);

    let mut svc = SVC::new(10., 0.00000001, 1000000);
    svc.fit(&dx, &dy);
    println!("alpha {:?}", &svc.alpha);
    println!("b {:?}", &svc.b);

    let errors = svc.get_errors(&dy, &svc.alpha, &svc.b);
    println!("errors {:?}", &errors);

    for i in 0..dy.len() {
        let yi = dy[i];
        let mut ai = svc.alpha[i];
        let mut ei = errors[i];

        // KKT conditions
        if (yi * ei < -0.001 && ai < 10.) ||
           (yi * ei > 0.001 && ai > 0.) {
            println!("KKT out index={:?} y={:?} err={:?} alpha={:?}", &i, &yi, &(yi * ei), &ai);
        }
    }

    let mut w1 = 0.;
    let mut w2 = 0.;
    for i in 0..dy.len() {
        w1 += svc.alpha[i] * dy[i] * dx.index((i, 0));
        w2 += svc.alpha[i] * dy[i] * dx.index((i, 1));
    }
    let w = - w1 / w2;
    println!("w {:?}", &w);
    println!("b {:?}", &(-svc.b / w2));

}
