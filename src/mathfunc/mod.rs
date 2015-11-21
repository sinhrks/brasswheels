extern crate nalgebra;
extern crate num;

use nalgebra::{DVec, DMat, Iterable, ColSlice};
use num::{Num, Zero, Float, Signed};
use std::f64;
use std::ops::Sub;
use std::vec::Vec;

/// Get minimum values of each column of DVec
pub fn dvec_min(data: &DVec<f64>) -> f64 {
    // can't use normal min(a, b), because it can't handle NaN
    return data.iter().fold(f64::MAX, |a, b| a.min(*b));
}

/// Get maximum values of each column of DVec
pub fn dvec_max(data: &DVec<f64>) -> f64 {
    return data.iter().fold(f64::MIN, |a, b| a.max(*b));
}

/// Get minimum values of each column of DMat
pub fn dmat_min(data: &DMat<f64>) -> DVec<f64> {
    return DVec::from_fn(data.ncols(),
                         |i| dvec_min(&data.col_slice(i, 0, data.nrows())));
}

/// Get maximum values of each column of DMat
pub fn dmat_max(data: &DMat<f64>) -> DVec<f64> {
    return DVec::from_fn(data.ncols(),
                         |i| dvec_max(&data.col_slice(i, 0, data.nrows())));
}

/// Sum of squares
pub fn sum_square<T: Num + Zero + Sub + Copy>(vec1: &DVec<T>, vec2: &DVec<T>, m1: T, m2: T) -> T {
    let mut val: T = Zero::zero();
    for (v1, v2) in (&vec1).iter().zip(vec2.iter()) {
        val = val + (*v1 - m1) * (*v2 - m2);
    }
    return val;
}

// Euclid distance
pub fn euc_dist<T: Float + Signed>(vec1: &DVec<T>, vec2: &DVec<T>) -> T {
    let mut val: T = Zero::zero();
    for (v1, v2) in vec1.iter().zip(vec2.iter()) {
        val = val + num::pow(num::abs(*v1 - *v2), 2);
    }
    return val.sqrt();
}

/// Round DMat elements to specified decimals
pub fn round(data: &DMat<f64>, decimals: usize) -> DMat<f64> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    let d: f64 = num::pow(10., decimals);
    // add decimal handling to round()
    let vals: Vec<f64> = data.as_vec().iter().map(|&x| (x * d).round() / d).collect();
    return DMat::from_col_vec(nrows, ncols, &vals);
}
