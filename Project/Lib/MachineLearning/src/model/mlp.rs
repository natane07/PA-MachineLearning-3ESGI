extern crate rand;

use rand::{Rng, thread_rng};
use std::marker::PhantomData;

struct MLP<T: ?Sized> {
    phantom: PhantomData<T>,
    npl: Vec<i64>,
    L: usize,
    w: Vec<Vec<Vec<f64>>>,
    deltas: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

/**

    Model creation

*/


impl MLP<f64> {
    pub fn create(&mut self, &npl: &Vec<i64>, npl_size: usize) {
        self.npl = npl;
        self.L = npl_size - 1;
        self.w/*: Vec<Vec<Vec<f64>>>*/ = vec![vec![vec![]]];
        let mut rand = rand::thread_rng();
        // self.w.push(0.0); // à voir
        for layer in 1..(self.L + 1) {
            // self.w.push(); // A voir
            for i in 0..(self.npl[layer - 1] + 1) as usize {
                // self.w[layer].push([]);
                for j in O..(self.npl[layer] + 1) as usize {
                    self.w[layer][i].push(rand.gen_range(-1., 1.));
                }
            }
        }

        self.deltas/*: Vec<Vec<f64>>*/ = vec![vec![]];
        for layer in 1..(self.L + 1) {
            // self.deltas.push(); // A voir
            for j in 0..(self.npl[layer] + 1) as usize{
                self.deltas[layer].push(0.0);
            }
        }

        self.x/*: Vec<Vec<f64>>*/ = vec![vec![]];
        for layer in 0..(self.L + 1) {
            // self.x.push(); // A voir
            for j in 0..(self.npl[layer] + 1) as usize {
                self.x[layer].push(1.0);
            }
        }
    }
}

impl Default for MLP<f64> {
    fn default() -> Self {
        Self {
            phantom: Default::default(),
            npl: Default::default(),
            L: Default::default(),
            w: Default::default(),
            deltas: Default::default(),
            x: Default::default(),
        }
    }
}

pub fn create_mlp(npl: &Vec<i64>, npl_size: usize) {
    let mut mlp = MLP { ..Default::default() };
    return mlp.create(npl, npl_size);
}

/**

Predictions functions

*/

pub fn mlp_predict_common<'a>(mlp: &'a mut MLP<f64>, inputs: &'a [f64], classification_mode: bool) -> &'a [Vec<f64>] {
    for j in 1..(mlp.npl[0] + 1) as usize{
        mlp.x[0][j] = inputs[j - 1];
    }

    for layer in 1..(mlp.L + 1) {
        for j in 1..(mlp.npl[layer] + 1) as usize {
            let mut result: f64 = 0.0;
            for i in 0..(mlp.npl[layer - 1] + 1) as usize {
                result += mlp.w[layer][i][j] * mlp.x[layer - 1][i];
            }
            if layer != mlp.L || classification_mode {
                result = result.tanh();
            }
            mlp.x[layer][j] = result;
        }
    }

    mlp.x[mlp.L].remove(0);
    return mlp.x[mlp.L];
}

pub extern fn mlp_regression(mlp: MLP<f64>, inputs: &[f64]) -> &[Vec<f64>] {
    return mlp_predict_common(*mlp, inputs, false);
}

pub extern fn mlp_classification(mlp: MLP<f64>, inputs: &[f64]) -> &[Vec<f64>] {
    return mlp_predict_common(*mlp, inputs, true);
}

/**

Training functions

*/

pub extern fn mlp_train_common(mlp: &mut MLP<f64>,
                               inputs: &[Vec<f64>],
                               expected_outputs: &[Vec<f64>],
                               iterations: usize,
                               alpha: f64,
                               classification_mode: bool) {
    let mut rand = rand::thread_rng();
    for iterator in iterations {
        let mut k: usize = rand.gen_range(-1., 1.) as usize;
        mlp_predict_common(&mut mlp, &inputs[k], classification_mode); // pb ici
        for j in 1..(mlp.npl[mlp.L] + 1) as usize {
            mlp.deltas[mlp.L][j] = mlp.x[mlp.L][j] - expected_outputs[k][j - 1];
            if classification_mode {
                mlp.deltas[mlp.L][j] *= (1 - (mlp.x[mlp.L][j] * mlp.x[mlp.L][j]) as i32) as f64;
            }
        }
    }

    for layer in (mlp.L + 1)..2 {
        for i in 1..(mlp.npl[layer - 1] + 1) {
            let mut result: f64 = 0.0;
            for j in 1..(mlp.npl[layer] + 1) {
                result += mlp.w[layer][i][j] * mlp.deltas[layer][j];
            }
            result *= (1 - mlp.x[layer - 1][i] * mlp.x[layer - 1][i]) as f64;
            mlp.deltas[layer - 1][i] = result
        }
    }

    for layer in 1..(mlp.L + 1) {
        for i in 0..(mlp.npl[layer - 1] + 1) {
            for j in 1..(mlp.npl[layer] + 1) {
                mlp.w[layer][i][j] -= alpha * mlp.x[layer - 1][i] * mlp.deltas[layer][j];
            }
        }
    }
}

pub fn mlp_train_classification(mlp: MLP<f64>,
                                inputs: &[Vec<f64>],
                                expected_outputs: &[Vec<f64>],
                                iterations: usize,
                                alpha: f64) {
    return mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha, true);
}

pub fn mlp_train_regression(mlp: MLP<f64>,
                            inputs: &[Vec<f64>],
                            expected_outputs: &[Vec<f64>],
                            iterations: usize,
                            alpha: f64) {
    return mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha, false);
}