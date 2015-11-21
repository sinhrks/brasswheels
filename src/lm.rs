extern crate nalgebra;
extern crate num;

use nalgebra::{DMat, Inv, Mean, ColSlice};
use std::vec::Vec;

use mathfunc::{sum_square};

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

    pub fn fit(&mut self, data: &DMat<f64>) {

        // 列ごとに平均値を取得
        let means = data.mean();

        let nrows = data.nrows();
        let nfeatures = data.ncols() - 1;

        // 偏差平方和積和行列
        // Dmat::from_fn で、行番号 i, 列番号 j を引数とする関数から各要素の値を生成できる
        let smx = DMat::from_fn(nfeatures, nfeatures,
                                |i, j| sum_square(&data.col_slice(i + 1, 0, nrows),
                                                  &data.col_slice(j + 1, 0, nrows),
                                                  means[i+1], means[j+1]));
        // 偏差積和行列
        // DMat と Vec では演算ができないため、こちらも一列の DMat として生成
        let smy = DMat::from_fn(nfeatures, 1,
                                |i, _| sum_square(&data.col_slice(i + 1, 0, nrows),
                                                  &data.col_slice(0, 0, nrows),
                                                  means[i], means[0]));
        // 偏回帰係数を計算し、Vec に変換
        let mut res = (smx.inv().unwrap() * smy).to_vec();

        // 切片を計算し、0 番目の要素として挿入
        let intercept = (1..means.len()).fold(means[0], |m, i| m - res[i-1]*means[i]);
        res.insert(0, intercept);
        self.coefs = res;
    }
}
