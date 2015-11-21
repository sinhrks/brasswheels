extern crate nalgebra;
extern crate num;

use nalgebra::{DVec, DMat, Mat4, Mean, ColSlice, Transpose};
use std::vec::Vec;

use mathfunc::{sum_square};

pub struct PCA {
    center: bool,               // whether to center or not
    nfeatures: usize,           // feature dimensions
    pub rotation: DMat<f64>     // principal components
}

impl PCA {

    pub fn new(nfeatures: usize, center: bool) -> PCA {
        PCA {
            center: center,
            // 固有値計算に Mat4 を使うため、次元は4に固定
            nfeatures: match nfeatures {
                4 => nfeatures,
                _ => panic!("Number of features must be 4")
            },
            rotation: DMat::new_ones(nfeatures, nfeatures)
        }
    }

    fn get_centers(&self, data: &DMat<f64>) -> DVec<f64> {
        // センタリングに用いるベクトルを計算
        return match self.center {
            true => data.mean(),
            false => DVec::from_elem(self.nfeatures, 0.)
        };
    }

    pub fn fit(&mut self, data: &DMat<f64>) {
        let nrows = data.nrows();
        let centers = self.get_centers(&data);

        // 偏差平方和積和行列
        // Dmat::from_fn で、行番号 i, 列番号 j を引数とする関数から各要素の値を生成できる
        let smx = DMat::from_fn(self.nfeatures, self.nfeatures,
                                |i, j| sum_square(&data.col_slice(i, 0, nrows),
                                                  &data.col_slice(j, 0, nrows),
                                                  centers[i], centers[j]));

        // DMat では eigen_qr が使えないため Mat4 に変換
        let smxv = smx.to_vec();
        let smx4 = Mat4::new(smxv[0], smxv[1], smxv[2], smxv[3],
                             smxv[4], smxv[5], smxv[6], smxv[7],
                             smxv[8], smxv[9], smxv[10], smxv[11],
                             smxv[12], smxv[13], smxv[14], smxv[15]);
        // get eigenvectors and eigenvalues
        let (evec, _) = nalgebra::eigen_qr(&smx4, &1e-8, 10000);

        let mut vals: Vec<f64> = vec![];
        for row in evec.transpose().as_array() {
            vals.extend(row);
        }
        self.rotation = DMat::from_row_vec(self.nfeatures, self.nfeatures, &vals);
    }

    pub fn transform(&mut self, data: &DMat<f64>) -> DMat<f64> {
        let centers = self.get_centers(&data);
        let cdata = DMat::from_fn(data.nrows(), data.ncols(),
                                |i, j| data[(i, j)] - centers[j]);
        return cdata * &self.rotation;
    }
}

