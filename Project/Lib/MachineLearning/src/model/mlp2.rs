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
pub fn create_mlp(npl: *mut i64, npl_size: usize) -> *const MLP {

    let npl = unsafe { from_raw_parts(npl, npl_size) };

    let mlp = _create_mlp(npl.to_vec(), npl_size);
    Box::leak(Box::new(mlp))

    /*println!("mlp.npl:{:?}", mlp.npl);
    println!("mlp.deltas:{:?}", mlp.deltas);
    println!("mlp.l:{:?}", mlp.l);
    println!("mlp.w:{:?}", mlp.w);
    println!("mlp.x:{:?}", mlp.x);*/
}

#[no_mangle]
pub fn test(model_ptr: *const MLP) {
    let mlp;
    unsafe {
        mlp = model_ptr.as_ref().unwrap();
    }
    // println!("FONCTION DE TEST");
    // println!("mlp.npl:{:?}", mlp.npl);
    // println!("mlp.deltas:{:?}", mlp.deltas);
    // println!("mlp.l:{:?}", mlp.l);
    // println!("mlp.w:{:?}", mlp.w);
    // println!("mlp.x:{:?}", mlp.x);
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

pub extern fn _mlp_regression(mlp: &mut MLP, sample_imput: Vec<f64>) -> Vec<f64> {
    return _mlp_predict_common(mlp, sample_imput, false);
}

#[no_mangle]
pub extern fn mlp_regression(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> *const f64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };;
    let test = _mlp_regression(mlp, sample_imput.to_vec());
    println!("{:?}", test);
    return _mlp_regression(mlp, sample_imput.to_vec()).as_ptr();
}

pub extern fn _mlp_classification(mlp: &mut MLP, sample_imput: Vec<f64>) -> Vec<f64> {
    return _mlp_predict_common(mlp, sample_imput, true);
}

#[no_mangle]
pub extern fn mlp_classification(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> *const f64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };
    let test = _mlp_classification(mlp, sample_imput.to_vec());
    println!("{:?}", test);
    return _mlp_classification(mlp, sample_imput.to_vec()).as_ptr();
}

pub extern fn mlp_train_common(mlp: &mut MLP,
                               inputs: Vec<Vec<f64>>,
                               expected_outputs: Vec<Vec<f64>>,
                               iterations: usize,
                               alpha: f64,
                               classification_mode: bool) {

    let mut rand = rand::thread_rng();
    for iterator in 0..iterations {
        let mut k: usize = rand.gen_range(0, inputs.len() - 1) as usize;
        _mlp_predict_common(mlp, inputs[k].clone(), classification_mode);
        for j in 1..(mlp.npl[mlp.l] + 1) as usize{
            mlp.deltas[mlp.l][j] = mlp.x[mlp.l][j] - expected_outputs[k][j - 1];
            if classification_mode {
                mlp.deltas[mlp.l][j] *= (1 - (mlp.x[mlp.l][j] * mlp.x[mlp.l][j]) as i32) as f64;
            }
        }
    }

    for layer in (mlp.l + 1)..2 {
        for i in 1..(mlp.npl[layer - 1] + 1) as usize {
            let mut result: f64 = 0.0;
            for j in 1..(mlp.npl[layer] + 1) as usize{
                result += mlp.w[layer][i][j] * mlp.deltas[layer][j];
            }
            result *= (1 - (mlp.x[layer - 1][i] * mlp.x[layer - 1][i]) as i32) as f64;
            mlp.deltas[layer - 1][i] = result;
        }
    }

    for layer in 1..(mlp.l + 1) {
        for i in 0..(mlp.npl[layer - 1] + 1) as usize{
            for j in 1..(mlp.npl[layer] + 1) as usize{
                mlp.w[layer][i][j] -= alpha * mlp.x[layer - 1][i] * mlp.deltas[layer][j];
            }
        }
    }
}

pub extern fn _mlp_train_classification(mlp: &mut MLP,
                                         inputs: Vec<Vec<f64>>,
                                         expected_outputs: Vec<Vec<f64>>,
                                         iterations: usize,
                                         alpha: f64) {
    mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha,true);
}


/*
pub extern fn _mlp_train_regression(mlp: &mut MLP,
                                        inputs: Vec<f64>,
                                        expected_outputs: Vec<f64>,
                                        iterations: usize,
                                        alpha: f64) {
    mlp_train_common(mlp: &mut MLP,
                     inputs: Vec<f64>,
                     expected_outputs: Vec<f64>,
                     iterations: usize,
                     alpha: f64,
                     false);
}
*/
#[no_mangle]
pub extern fn mlp_train_classification(mlp: *mut MLP,
                                       nb_row: usize,
                                       inputs: *mut f64,
                                       imput_size: usize,
                                       expected_outputs: *mut f64,
                                       expected_outputs_size: usize,
                                       iterations: usize,
                                       alpha: f64) {

    let inputs = unsafe { from_raw_parts(inputs, imput_size) };
    let outputs = unsafe { from_raw_parts(expected_outputs,expected_outputs_size ) };

    let inputs = convert_slice_to_2d_vec(inputs.to_vec(), (imput_size / nb_row) as usize);
    let outputs = convert_slice_to_2d_vec(outputs.to_vec(), (expected_outputs_size / nb_row) as usize);

    let mlp = unsafe { mlp.as_mut().unwrap() };
    println!("mlp_train_classification DEBUT");
    println!("mlp.npl:{:?}", mlp.npl);
    println!("mlp.deltas:{:?}", mlp.deltas);
    println!("mlp.l:{:?}", mlp.l);
    println!("mlp.w:{:?}", mlp.w);
    println!("mlp.x:{:?}", mlp.x);
    _mlp_train_classification(mlp, inputs, outputs, iterations, alpha);
    println!("mlp.npl:{:?}", mlp.npl);
    println!("mlp.deltas:{:?}", mlp.deltas);
    println!("mlp.l:{:?}", mlp.l);
    println!("mlp.w:{:?}", mlp.w);
    println!("mlp.x:{:?}", mlp.x);
    println!("mlp_train_classification FIN");

}

pub fn convert_slice_to_2d_vec(vec: Vec<f64>, arr_size_x: usize) -> Vec<Vec<f64>>{
    let (left, mut right) = vec.split_at(arr_size_x);
    let mut array: Vec<Vec<f64>> = Vec::new();
    array.push(left.to_vec());
    while right.len() != 0 {
        let (left2, right_split) = right.split_at(arr_size_x);
        array.push(left2.to_vec());
        right = right_split;
    }
    println!("List to 2D vec {:?}", array);
    return array;
}