extern crate csv;
extern crate brasswheels;

use brasswheels::io::read_csv_f64;
use brasswheels::hclust::{HClust, ClusterDistance};

fn main() {
    // cargo build --example hclust
    // ./target/debug/examples/hclust

    let data = "名前,算数,理科,国語,英語,社会
田中,89,90,67,46,50
佐藤,57,70,80,85,90
鈴木,80,90,35,40,50
本田,40,60,50,45,55
川端,78,85,45,55,60
吉野,55,65,80,75,85
斉藤,90,85,88,92,95";

    // http://burntsushi.net/rustdoc/csv/
    let mut reader = csv::Reader::from_string(data).has_headers(true);
    // http://nalgebra.org/doc/nalgebra/struct.DMat.html
    let dx = read_csv_f64(&mut reader);

    println!("最近隣法");
    let mut hclust = HClust::new(ClusterDistance::Single);
    hclust.fit(&dx);

    println!("最遠隣法");
    let mut hclust = HClust::new(ClusterDistance::Complete);
    hclust.fit(&dx);

    println!("群平均法");
    let mut hclust = HClust::new(ClusterDistance::Average);
    hclust.fit(&dx);
}