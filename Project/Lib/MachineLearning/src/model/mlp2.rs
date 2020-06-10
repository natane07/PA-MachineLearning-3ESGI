extern crate rand;

use rand::{Rng};
use std::slice::from_raw_parts;

pub struct MLP {
    npl: Vec<i64>,
    l: usize,
    w: Vec<Vec<Vec<f64>>>,
    deltas: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}


impl MLP {
    pub fn create(mut self, npl: Vec<i64>) -> MLP{
        let mut rand = rand::thread_rng();
        self.w.push(vec![]);
        for layer in 1..(self.l + 1) {
            self.w.push(vec![]);
            for i in 0..(npl[layer - 1] + 1) as usize{
                self.w[layer].push(vec![]);
                for _j in 0..(npl[layer] + 1){
                    self.w[layer][i].push(rand.gen_range(-1., 1.));
                }
            }
        }

        self.deltas.push(vec![]);
        for layer in 1..(self.l + 1) {
            self.deltas.push(vec![]);
            for _j in 0..(npl[layer] + 1) as usize{
                self.deltas[layer].push(0.0);
            }
        }

        for layer in 0..(self.l + 1) {
            self.x.push(vec![]);
            for j in 0..(npl[layer] + 1) as usize {
                if j == 0 {
                    self.x[layer].push(1.0);

                } else {
                    self.x[layer].push(0.0);
                }
            }
        }
        return self;
    }
}

pub fn _create_mlp(npl: Vec<i64>, npl_size: usize) -> MLP {
    let mut mlp = MLP {
        npl: npl.clone(),
        l: npl_size - 1,
        w: vec![],
        deltas: vec![],
        x: vec![],
    };
    let mlp = mlp.create(npl);
    return mlp;
}

#[no_mangle]
pub fn create_mlp(npl: *mut i64, npl_size: usize) {

    let npl = unsafe { from_raw_parts(npl, npl_size) };

    let mlp = _create_mlp(npl.to_vec(), npl_size);

    println!("mlp.npl:{:?}", mlp.npl);
    println!("mlp.deltas:{:?}", mlp.deltas);
    println!("mlp.l:{:?}", mlp.l);
    println!("mlp.w:{:?}", mlp.w);
    println!("mlp.x:{:?}", mlp.x);
}

pub fn _mlp_predict_common(mlp: &mut MLP, sample_imput: Vec<f64>, classification_mode: bool) -> Vec<f64> {
    for j in 1..(mlp.npl[0] + 1) as usize{
        mlp.x[0][j] = sample_imput[j - 1];
    }

    for layer in 1..(mlp.l + 1) {
        for j in 1..(mlp.npl[layer] + 1) as usize {
            let mut result: f64 = 0.0;
            for i in 0..(mlp.npl[layer - 1] + 1) as usize {
                result += mlp.w[layer][i][j] * mlp.x[layer - 1][i];
            }
            if layer != mlp.l || classification_mode {
                result = result.tanh();
            }
            mlp.x[layer][j] = result;
        }
    }

    return mlp.x[mlp.l].clone();
}

pub extern fn _mlp_regression(mlp: & mut MLP, sample_imput: Vec<f64>) -> Vec<f64> {
    return _mlp_predict_common(mlp, sample_imput, false);
}

#[no_mangle]
pub extern fn mlp_regression(mlp: & mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> Vec<f64> {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    return _mlp_regression(mlp, sample_imput.to_vec());
}

pub extern fn _mlp_classification(mlp: & mut MLP, sample_imput: Vec<f64>) -> Vec<f64> {
    return _mlp_predict_common(mlp, sample_imput, true);
}

#[no_mangle]
pub extern fn mlp_classification(mlp: & mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> Vec<f64> {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    return _mlp_classification(mlp, sample_imput.to_vec());
}
