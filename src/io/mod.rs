use std::f64;
use std::io;
use std::str;
use std::str::FromStr;
use std::vec::Vec;

use csv::Reader;
use nalgebra::DMat;

//http://stackoverflow.com/questions/25272392/wrong-number-of-type-arguments-expected-1-but-found-0

pub fn read_csv_f64<R: io::Read>(reader: &mut Reader<R>) -> DMat<f64> {
    // csv::Reder から f64 に変換できるカラムのみ読み込み

    let mut x:Vec<f64> = vec![];
    let mut nrows: usize = 0;

    for record in reader.byte_records().map(|r| r.unwrap()) {
        // f64 に変換できる列のみ読み込み
        for item in record.iter().map(|i| str::from_utf8(i).unwrap()) {
            match f64::from_str(item) {
                Ok(v) => x.push(v),
                Err(_) => {}
            };
        }
        nrows += 1;
    }
    let ncols = x.len() / nrows;

    // http://nalgebra.org/doc/nalgebra/struct.DMat.html
    return DMat::from_row_vec(nrows, ncols, &x);
}
