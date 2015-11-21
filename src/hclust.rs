extern crate csv;
extern crate nalgebra;
extern crate num;
extern crate rand;

use nalgebra::{DMat, RowSlice, Iterable};
use num::Float;
use std::f64;
use std::ops::Index;

use mathfunc::{euc_dist};

pub enum ClusterDistance {
    Single,                 // Minimum or single-linkage clustering
    Complete,               // Maximum or complete-linkage clustering
    Average,                // Mean or average linkage clustering, or UPGMA
}

pub struct HClust {
    dist_mat: DMat<f64>,        // distance matrix
    method: ClusterDistance,    // linkage criteria
    clusters: Vec<Cluster>
}

impl HClust {

    pub fn new(method: ClusterDistance) -> HClust {
        HClust {
            // dummy
            dist_mat: DMat::from_elem(1, 1, 1.),
            method: method,
            clusters: vec![]
        }
    }

    pub fn fit(&mut self, data: &DMat<f64>) {
        self.dist_mat = self.get_dist_matrix(&data);

        // initialize clusters
        for i in 0..data.nrows() {
            let c = Cluster::from_nodes(vec![i]);
            self.clusters.push(c);
        }
        while self.clusters.len() > 1 {
            self.fit_step();
        }
    }

    /// merge closest clusters
    fn fit_step(&mut self) {
        let mut tmp_i = 0;
        let mut tmp_j = 0;
        let mut current_dist = f64::MAX;

        for i in 0..self.clusters.len() {
            for j in (i + 1)..self.clusters.len() {
                let d = self.get_cluster_dist(&self.clusters[i], &self.clusters[j]);
                if d < current_dist {
                    current_dist = d;
                    tmp_i = i;
                    tmp_j = j;
                }
            }
        }

        let mut new_clusters: Vec<Cluster> = vec![];
        for (i, n) in self.clusters.iter().enumerate() {
            if i != tmp_i && i != tmp_j {
                let n2 = Cluster::from_nodes(n.nodes.clone());
                new_clusters.push(n2);
            }
        }
        // take elements from Vec and move ownership to the new instance
        let new = Cluster::from_clusters(self.clusters.swap_remove(tmp_j),
                                         self.clusters.swap_remove(tmp_i));

        new_clusters.push(new);
        self.clusters = new_clusters;
    }

    /// get distance between clusters
    fn get_cluster_dist(&self, c1: &Cluster, c2: &Cluster) -> f64 {
        let mut dists: Vec<f64> = vec![];
        for i in &c1.nodes {
            for j in &c2.nodes {
                dists.push(self.get_node_dist(*i, *j));
            }
        }
        return self.fold_dist_vec(dists);
    }

    /// get cluster distance from node distances
    fn fold_dist_vec(&self, dists: Vec<f64>) -> f64 {
        match self.method {
            ClusterDistance::Single => {
                return dists.iter().fold(f64::MAX, |a, b| a.min(*b));
            },
            ClusterDistance::Complete => {
                return dists.iter().fold(f64::MIN, |a, b| a.max(*b));
            },
            ClusterDistance::Average => {
                return dists.iter().fold(0., |a, b| a + b) / (dists.len() as f64);
            },
        }
    }

    /// get distance matrix
    fn get_dist_matrix(&self, data: &DMat<f64>) -> DMat<f64> {
        // column corresponding to 0 to nthnodes
        // row corresponding to 1 to nth nodes
        return DMat::from_fn(data.nrows() - 1, data.nrows() - 1,
                             |i, j| if i >= j {
                                euc_dist(&data.row_slice(i + 1, 0, data.ncols()),
                                         &data.row_slice(j, 0, data.ncols()))}
                                else { 0. });
    }

    /// get distance between nodes using distance matrix
    fn get_node_dist(&self, i: usize, j: usize) -> f64 {
        match i > j {
            true => *self.dist_mat.index((i - 1, j)),
            false => *self.dist_mat.index((j - 1, i))
        }
    }
}

struct Cluster {
    nodes: Vec<usize>,
    children: Vec<Cluster>
}

impl Cluster {

    fn from_nodes(nodes: Vec<usize>) -> Cluster {
        Cluster {
            nodes: nodes,
            children: vec![]
        }
    }

    /// create a cluster merging 2 clusters
    fn from_clusters(left: Cluster, right: Cluster) -> Cluster {
        let mut id = vec![];
        for i in &left.nodes {
            id.push(*i);
        }
        for j in &right.nodes {
            id.push(*j);
        }
        Cluster {
            nodes: id,
            children: vec![left, right]
        }
    }
}

