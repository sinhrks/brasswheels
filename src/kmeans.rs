extern crate nalgebra;
extern crate num;
extern crate rand;

use nalgebra::{DVec, DMat, RowSlice, Iterable};
use rand::sample;
use std::collections::HashMap;
use std::f64;

use mathfunc::{euc_dist};

pub struct KMeans {
    pub nclusters: usize,                   // クラスタ数
    max_iter: usize,                        // イテレーション回数
    pub centroids: HashMap<usize, Cluster>,
}

impl KMeans {

    pub fn new(nclusters: usize, max_iter: usize) -> KMeans {
        KMeans {
            nclusters: nclusters,
            max_iter: max_iter,
            centroids: HashMap::new(),
        }
    }

    pub fn fit(&mut self, data: &DMat<f64>) {
        let mut rng = rand::thread_rng();

        // データからクラスタの初期値をサンプリング (非復元抽出)
        let inits: Vec<usize> = sample(&mut rng, 0..data.nrows(), self.nclusters);
        for (i, rownum) in inits.into_iter().enumerate() {
            let mut c = Cluster::new(data.ncols());
            let row = data.row_slice(rownum, 0, data.ncols());
            c.add_element(row);
            self.centroids.insert(i, c);
        }

        let mut cindexer = self.predict(data);

        // 最大 max_iter 回繰り返し
        for _ in 0..self.max_iter {
            // 中心点の更新
            self.update_centroids(&data, &cindexer);
            // 各レコードを クラスタの中心点にもっとも近いものに分類
            let cindexer_new = self.predict(data);

            // Eq での比較結果は element-wise ではなく bool になる
            if cindexer_new == cindexer {
                // 変化がなくなったら終了
                break;
            } else {
                cindexer = cindexer_new;
            }
        }
    }

    /// 各レコードが所属するクラスタのベクトルを返す
    pub fn predict(&self, data: &DMat<f64>) -> DVec<usize> {
        return DVec::from_fn(data.nrows(),
                            |x| self.get_nearest(&data.row_slice(x, 0, data.ncols())));
    }

    /// レコードにもっとも近い中心点をもつクラスタのラベルを返す
    fn get_nearest(&self, values: &DVec<f64>) -> usize {

        let mut tmp_i = 0;
        let mut current_dist = f64::MAX;

        for (cnum, cluster) in &self.centroids {
            let d = euc_dist(values, &cluster.centroid);
            if d < current_dist {
                current_dist = d;
                tmp_i = *cnum;
            }
        }
        return tmp_i;
    }

    /// クラスタの中心点を更新する
    fn update_centroids(&mut self, data: &DMat<f64>, cindexer: &DVec<usize>) {
        self.centroids.clear();

        for i in 0..self.nclusters {
            let c = Cluster::new(data.ncols());
            self.centroids.insert(i, c);
        }

        for (rownum, cnum) in cindexer.iter().enumerate() {
            let row = data.row_slice(rownum, 0, data.ncols());
            let mut c = self.centroids.remove(&cnum).unwrap();
            c.add_element(row);
            self.centroids.insert(*cnum, c);
        }

        for cnum in 0..self.nclusters {
            let mut c = self.centroids.remove(&cnum).unwrap();
            c.finalize();
            self.centroids.insert(cnum, c);
        }
    }
}

pub struct Cluster {
    pub centroid: DVec<f64>,
    n: f64
}

impl Cluster {

    fn new(ncols: usize) -> Cluster {
        Cluster {
            centroid: DVec::from_elem(ncols, 0.),
            n: 0.
        }
    }

    /// 中心点を更新 (レコードを合計に追加する)
    fn add_element(&mut self, values: DVec<f64>) {
        self.centroid = self.centroid.clone() + values;
        self.n = self.n + 1.;
    }

    /// 中心点を更新 (レコード数で割り、中心点を求める)
    fn finalize(&mut self) {
        if self.n != 0. {
            self.centroid = self.centroid.clone() / self.n;
            self.n = 0.;
        }
    }
}
