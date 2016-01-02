extern crate nalgebra;
extern crate num;
extern crate rand;

use nalgebra::{DVec, DMat, RowSlice, Iterable};
use rand::sample;
use std::collections::HashMap;
use std::f64;
use std::ops::Index;
use std::process::exit;

use mathfunc::{inner_product, dvec_copy, dmat_copy};

pub struct SVC {
    c: f64,
    tolerance: f64,
    max_iter: usize,
    kernels: DMat<f64>,

    data: DMat<f64>,
    y: DVec<f64>,

    pub alpha: DVec<f64>,
    pub b: f64,

    beta: DVec<f64>,
}

pub trait SVMTrait {

    // 共通のメソッドを定義
    fn new(c: f64, tolerance: f64, max_iter: usize) -> Self;
    fn kernel(&self, i: usize, j: usize) -> f64;

    fn get_kernel_matrix(&mut self, data: &DMat<f64>) -> DMat<f64> {
        // build kernel matrix
        let mut values: Vec<f64> = vec![];
        for i in 0..data.nrows() {
            for j in 0..data.nrows() {
                if i >= j {
                    let xi = data.row_slice(i, 0, data.ncols());
                    let xj = data.row_slice(j, 0, data.ncols());
                    values.push(inner_product(&xi, &xj));
                } else {
                    values.push(0.0);
                }
            }
        }
        return DMat::from_row_vec(data.nrows(), data.nrows(), &values);
    }

    fn get_errors(&self, y: &DVec<f64>, alpha: &DVec<f64>, b: &f64) -> DVec<f64> {
        let mut errors = vec![];
        for i in 0..y.len() {
            let mut v = 0.0;
            for j in 0..y.len() {
                v += alpha[j] * y[j] * self.kernel(j, i);
            }
            v = v + b - y[i];
            errors.push(v);
        }
        return DVec::from_slice(y.len(), &errors);
    }
}

impl SVMTrait for SVC {

    fn new(c: f64, tolerance: f64, max_iter: usize) -> SVC {
        SVC {
            c: c,
            tolerance: tolerance,
            max_iter: max_iter,
            kernels: DMat::from_elem(1, 1, 0.0),

            data: DMat::from_elem(1, 1, 0.0),
            y: DVec::from_elem(1, 0.0),

            alpha: DVec::from_elem(1, 0.0),
            b: 0.0,

            beta: DVec::from_elem(1, 0.0),
        }
    }

    fn kernel(&self, i: usize, j: usize) -> f64 {
        if i >= j {
            return *self.kernels.index((i, j));
        } else {
            return *self.kernels.index((j, i));
        }
    }
}

impl SVC {

    pub fn fit(&mut self, data: &DMat<f64>, y: &DVec<f64>) {
        // build kernel matrix
        self.kernels = self.get_kernel_matrix(&data);

        // copy data
        self.data = dmat_copy(data);
        self.y = dvec_copy(y);

        // beta = alpha y
        self.beta = DVec::from_elem(data.nrows(), 0.0);

        // 式 7.15
        let mut u = DVec::from_slice(data.nrows(), &y.at);

        // 2 レコードをランダムサンプリング
        let mut rng = rand::thread_rng();
        let sample: Vec<usize> = sample(&mut rng, 0..data.nrows(), 2);
        let mut s = sample[0];
        let mut t = sample[1];

        // アルゴリズム 7.1: ステップ 4
        for _ in 0..self.max_iter {
            let delta = self.take_step(&y, &u, s, t);

            // 式 7.1
            for i in 0..u.len() {
                let new_u = u[i] - (self.kernel(i, s) - self.kernel(i, t)) * delta;
                u[i] = new_u;
            }

            let mut nexts = 0; // iup
            let mut nextt = 0; // ilow
            let mut nexts_u = f64::MIN;
            let mut nextt_u = f64::MAX;

            for i in 0..u.len() {
                let alpha = self.beta[i] / y[i];

                // I_UP
                if (0. < alpha && alpha < self.c) ||         // I0
                   (alpha == 0. && y[i] == 1.) ||            // I1
                   (alpha == self.c && y[i] == -1.) {        // I4
                    if u[i] >= nexts_u && i != s {
                        nexts_u = u[i];
                        nexts = i;
                    }
                }

                // I_LOW
                if (0. < alpha && alpha < self.c) ||         // I0
                   (alpha == 0. && y[i] == -1.) ||           // I2
                   (alpha == self.c && y[i] == 1.) {         // I3
                    if u[i] <= nextt_u && i != t {
                        nextt_u = u[i];
                        nextt = i;
                    }
                }
            }
            s = nexts;
            t = nextt;
        }

        self.alpha = DVec::from_fn(self.beta.len(), |i| self.beta[i] / y[i]);
        for i in 0..data.nrows() {
            if 0. < self.alpha[i] && self.alpha[i] < self.c {
                self.b = y[i];
                for j in 0..data.nrows() {
                    self.b -= self.alpha[j] * y[j] * self.kernel(i, j);
                }
                return;
            }
        }
    }

    fn take_step(&mut self, y: &DVec<f64>, u: &DVec<f64>,
                 s: usize, t: usize) -> f64 {

        let ys = y[s];
        let yt = y[t];

        let betas = self.beta[s];
        let betat = self.beta[t];

        // アルゴリズム 7.1: ステップ 5
        // 式 7.8
        let k = self.kernel(s, s) + self.kernel(t, t) + 2. * self.kernel(s, t);
        let mut delta = ys - yt;
        for i in 0..y.len() {
            delta -= self.beta[i] * (self.kernel(i, s) - self.kernel(i, t));
        }
        delta = delta / k;

        // 式7.6
        let mut l = 0.;
        let mut h = 0.;
        if ys == 1. && yt == 1. {
            // f64 は Ord トレイトを持たないため min / max 関数は利用できない
            l = (- betas).max(betat - self.c);
            h = (self.c - betas).min(betat);
        } else if ys == 1. && yt == -1. {
            l = (- betas).max(betat);
            h = (self.c - betas).min(self.c + betat);
        } else if ys == -1. && yt == 1. {
            l = (- self.c - betas).max(betat - self.c);
            h = (- betas).min(betat);
        } else if ys == -1. && yt == -1. {
            l = (- self.c - betas).max(betat);
            h = (- betas).min(self.c + betat);
        }

        if delta < l {
            delta = l;
        }
        if delta > h {
            delta = h;
        }

        // アルゴリズム 7.1: ステップ 6
        self.beta[s] = betas + delta;
        self.beta[t] = betat - delta;

        return delta;
    }
}

pub struct SVC2 {
    C: f64,
    tolerance: f64,
    max_iter: usize,
    kernels: DMat<f64>,
    pub alpha: DVec<f64>,
    pub b: f64,
}

impl SVMTrait for SVC2 {

    fn new(C: f64, tolerance: f64, max_iter: usize) -> SVC2 {
        SVC2 {
            C: C,
            tolerance: tolerance,
            max_iter: max_iter,
            kernels: DMat::from_elem(1, 1, 0.0),

            alpha: DVec::from_elem(1, 0.0),
            b: 0.0,
        }
    }

    fn kernel(&self, i: usize, j: usize) -> f64 {
        if i >= j {
            return *self.kernels.index((i, j));
        } else {
            return *self.kernels.index((j, i));
        }
    }
}

impl SVC2 {

    pub fn fit(&mut self, data: &DMat<f64>, y: &DVec<f64>) {
        // build kernel matrix
        self.kernels = self.get_kernel_matrix(&data);
        // init param
        self.alpha = DVec::from_elem(data.nrows(), 0.0);

        let mut updated = true;
        let mut errors = self.get_errors(&y, &self.alpha, &self.b);

        for t in 0..self.max_iter {

            updated = false;

            for i in 0..y.len() {
                let yi = y[i];
                let ai = self.alpha[i];
                let ei = errors[i];

                // KKT conditions
                if (yi * ei < -self.tolerance && ai < self.C) ||
                   (yi * ei > self.tolerance && ai > 0.) {
                    for j in 0..y.len() {
                        let updated_step = self.take_step(&y, &errors, i, j);
                        if updated_step {
                            updated = true;
                            errors = self.get_errors(&y, &self.alpha, &self.b);
                        }
                    }
                }
            }
            if !updated {
                break;
            }
        }
    }

    fn take_step(&mut self, y: &DVec<f64>, errors: &DVec<f64>,
                 i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }

        let yi = y[i];
        let ai = self.alpha[i];
        let ei = errors[i];

        let kii = self.kernel(i, i);
        let kjj = self.kernel(j, j);
        let kij = self.kernel(i, j);

        let yj = y[j];
        let aj = self.alpha[j];
        let ej = errors[j];

        let eta = kii + kjj - 2. * kij;

        if eta <= 0. {
            return false;
        }

        let mut aj_new = aj + yj * (ei - ej) / eta;
        // println!("temp {:?}", &aj);

        // clip
        let mut l = 0.0;
        let mut h = 0.0;
        if yi == yj {
            // same label
            // f64 は Ord トレイトを持たないため min / max 関数は利用できない
            l = (aj + ai - self.C).max(0.);
            h = self.C.min(aj + ai);
        } else {
            l = (aj - ai).max(0.);
            h = self.C.min(aj - ai + self.C);
        }
        if h == l {
            return false;
        }

        if aj_new > h {
            aj_new = h;
        }
        if aj_new < l {
            aj_new = l;
        }

        if (aj - aj_new).abs() < 0.001 {
            return false;
        }

        // update a
        let ai_new = ai + yi * yj * (aj - aj_new);
        self.alpha[i] = ai_new;
        self.alpha[j] = aj_new;

        // update b
        let b1_new = self.b - ei - yi * kii * (ai_new - ai) - yj * kij * (aj_new - aj);
        let b2_new = self.b - ej - yi * kij * (ai_new - ai) - yj * kjj * (aj_new - aj);

        if ai_new > 0. && ai_new < self.C {
            self.b = b1_new;
        } else if aj_new > 0. && aj_new < self.C {
            self.b = b2_new;
        } else {
            self.b = (b1_new + b2_new) / 2.;
        }
        return true;
    }
}

