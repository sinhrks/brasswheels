extern crate nalgebra;
extern crate num;

use nalgebra::{DMat, DVec, Inv, Mean, ColSlice};
use std::vec::Vec;

use super::mathfunc::{sum_square};

pub struct LinearModel {
    // 重回帰モデル
    pub coefs: Vec<f64>
}

impl LinearModel {

    pub fn new() -> LinearModel {
        LinearModel {
            coefs: vec![]
        }
    }

    pub fn fit(&mut self, data: &DMat<f64>, y: &DVec<f64>) {

        // 列ごとに平均値を取得
        let means = data.mean();
        let mean_y: f64 = (*y).at.iter().fold(0., |a, b| a + b) / (y.len() as f64);

        let nrows = data.nrows();
        let nfeatures = data.ncols();

        // 偏差平方和積和行列
        // Dmat::from_fn で、行番号 i, 列番号 j を引数とする関数から各要素の値を生成できる
        let smx = DMat::from_fn(nfeatures, nfeatures,
                                |i, j| sum_square(&data.col_slice(i, 0, nrows),
                                                  &data.col_slice(j, 0, nrows),
                                                  means[i], means[j]));
        // 偏差積和行列
        // DMat と Vec では演算ができないため、こちらも一列の DMat として生成
        let smy = DMat::from_fn(nfeatures, 1,
                                |i, _| sum_square(&data.col_slice(i, 0, nrows),
                                                  y, means[i], mean_y));
        // 偏回帰係数を計算し、Vec に変換
        let mut res = (smx.inv().unwrap() * smy).to_vec();
        println!("{:?}", res);

        // 切片を計算し、0 番目の要素として挿入
        let intercept = (0..means.len()).fold(mean_y, |m, i| m - res[i] * means[i]);
        res.insert(0, intercept);
        self.coefs = res;
    }

    pub fn predict(&self, data: &DMat<f64>) -> DVec<f64> {
        let coef_matrix: DMat<f64> = DMat::from_col_vec(self.coefs.len() - 1, 1,
                                                        &self.coefs[1..self.coefs.len()]);
        let result_m = data * coef_matrix + self.coefs[0];
        let result_v: DVec<f64> = DVec::from_slice(data.nrows(), &result_m.as_vec());
        return result_v;
    }
}
