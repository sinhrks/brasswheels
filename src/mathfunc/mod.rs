extern crate nalgebra;
extern crate num;

use nalgebra::{DVec, DMat, Iterable, ColSlice};
use num::{Num, Zero, Float, Signed};
use std::ops::Sub;
use std::vec::Vec;

/// Get minimum values of each column of DVec
pub fn dvec_min<T: Float>(data: &DVec<T>) -> T {
    // can't use normal min(a, b), because it can't handle NaN
    return data.iter().fold(T::infinity(), |a, b| a.min(*b));
}

/// Get maximum values of each column of DVec
pub fn dvec_max<T: Float>(data: &DVec<T>) -> T {
    return data.iter().fold(T::neg_infinity(), |a, b| a.max(*b));
}

/// Get minimum values of each column of DMat
pub fn dmat_min<T: Float>(data: &DMat<T>) -> DVec<T> {
    return DVec::from_fn(data.ncols(),
                         |i| dvec_min(&data.col_slice(i, 0, data.nrows())));
}

/// Get maximum values of each column of DMat
pub fn dmat_max<T: Float>(data: &DMat<T>) -> DVec<T> {
    return DVec::from_fn(data.ncols(),
                         |i| dvec_max(&data.col_slice(i, 0, data.nrows())));
}


/// Copy DVec
pub fn dvec_copy<T: Copy>(data: &DVec<T>) -> DVec<T> {
    let mut values: Vec<T> = vec![];
    for i in data.iter() {
        values.push(*i);
    }
    return DVec::from_slice(data.len(), &values);
}

/// Copy DMat
pub fn dmat_copy<T: Copy>(data: &DMat<T>) -> DMat<T> {
    let mut values: Vec<T> = vec![];
    for i in data.as_vec().iter() {
        values.push(*i);
    }
    return DMat::from_col_vec(data.nrows(), data.ncols(), &values);
}

/// Sum of squares
pub fn sum_square<T: Num + Zero + Sub + Copy>(vec1: &DVec<T>, vec2: &DVec<T>, m1: T, m2: T) -> T {
    let mut val: T = Zero::zero();
    for (v1, v2) in vec1.iter().zip(vec2.iter()) {
        val = val + (*v1 - m1) * (*v2 - m2);
    }
    return val;
}

// Euclid distance
pub fn euc_dist<T: Float + Signed>(vec1: &DVec<T>, vec2: &DVec<T>) -> T {
    let mut val: T = Zero::zero();
    assert!(vec1.len() == vec2.len());
    for (v1, v2) in vec1.iter().zip(vec2.iter()) {
        val = val + num::pow(num::abs(*v1 - *v2), 2);
    }
    return val.sqrt();
}

/// Inner Product
pub fn inner_product<T: Float + Signed>(vec1: &DVec<T>, vec2: &DVec<T>) -> T {
    let mut val: T = Zero::zero();
    assert!(vec1.len() == vec2.len());
    for (v1, v2) in vec1.iter().zip(vec2.iter()) {
        val = val + *v1 * *v2;
    }
    return val;
}

/// Round DMat elements to specified decimals
pub fn round(data: &DMat<f64>, decimals: usize) -> DMat<f64> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    // ToDo:: use generics?
    let d: f64 = num::pow(10., decimals);
    // add decimal handling to round()
    let vals: Vec<f64> = data.as_vec().iter().map(|&x| (x * d).round() / d).collect();
    return DMat::from_col_vec(nrows, ncols, &vals);
}


#[cfg(test)]
mod tests {
    use nalgebra::{DVec, DMat};
    use super::{dvec_min, dvec_max, dmat_min, dmat_max,
                sum_square, euc_dist, inner_product, round};

    #[test]
    fn test_dvec_minmax_float() {
        // dvec_min
        let v: DVec<f64> = DVec::from_slice(3, &vec![3., 1., 4.]);
        assert_eq!(1., dvec_min(&v));

        // dvec_max
        let v: DVec<f64> = DVec::from_slice(3, &vec![3., 1., 4.]);
        assert_eq!(4., dvec_max(&v));
    }

    #[test]
    fn test_dmat_minmax_float() {
        let m: DMat<f64> = DMat::from_row_vec(2, 2, &vec![3., 1., 4., 2.]);

        // dmat_min
        let exp: DVec<f64> = DVec::from_slice(2, &vec![3., 1.]);
        assert_eq!(exp, dmat_min(&m));

        // dmat_max
        let exp: DVec<f64> = DVec::from_slice(2, &vec![4., 2.]);
        assert_eq!(exp, dmat_max(&m));
    }

    #[test]
    fn test_sum_square() {
        let val1: Vec<f64> = vec![3., 4., 5.];
        let val2: Vec<f64> = vec![7., 8., 2.];
        let v1: DVec<f64> = DVec::from_slice(3, &val1);
        let v2: DVec<f64> = DVec::from_slice(3, &val2);

        assert_eq!(63., sum_square(&v1, &v2, 0., 0.));
        assert_eq!(-5., sum_square(&v1, &v2,
                                   val1.iter().fold(0., |a, b| a + b) / 3.0,
                                   val2.iter().fold(0., |a, b| a + b) / 3.0));
    }

    #[test]
    fn test_euq_dist() {
        let v1: DVec<f64> = DVec::from_slice(3, &vec![3., 4., 5.]);
        let v2: DVec<f64> = DVec::from_slice(3, &vec![7., 8., 2.]);

        assert_eq!(6.4031242374328485, euc_dist(&v1, &v2));

        let v1: DVec<f64> = DVec::from_slice(3, &vec![1., 2., 3.]);
        let v2: DVec<f64> = DVec::from_slice(3, &vec![5., 4., 2.]);

        assert_eq!(4.5825756949558398, euc_dist(&v1, &v2));
    }

    fn test_inner_product() {
        let v1: DVec<f64> = DVec::from_slice(3, &vec![3., 4., 5.]);
        let v2: DVec<f64> = DVec::from_slice(3, &vec![7., 8., 2.]);

        assert_eq!(63.0, inner_product(&v1, &v2));

        let v1: DVec<f64> = DVec::from_slice(3, &vec![1., 2., 3.]);
        let v2: DVec<f64> = DVec::from_slice(3, &vec![5., 4., 2.]);

        assert_eq!(19.0, inner_product(&v1, &v2));
    }

    #[test]
    fn test_dmat_round() {
        let m: DMat<f64> = DMat::from_row_vec(2, 2, &vec![5.555, 1.111, 4.444, 2.222]);

        let exp: DMat<f64> = DMat::from_row_vec(2, 2, &vec![5.6, 1.1, 4.4, 2.2]);
        assert_eq!(exp, round(&m, 1));

        let exp: DMat<f64> = DMat::from_row_vec(2, 2, &vec![6., 1., 4., 2.]);
        assert_eq!(exp, round(&m, 0));
    }

}
