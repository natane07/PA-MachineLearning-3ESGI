extern crate rand;
use rand::{Rng, thread_rng};

struct MLP<T: ?Sized> {
    unused_var: T,
    npl:  [i32],
    L: i32,
    w: Vec<Vec<Vec<f64>>>,
    deltas: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

/**

    Model creation

*/

impl MLP<f64>{
    pub fn create(&mut self, npl: &[i32], npl_size: i32) {
        self.npl = *npl;
        self.L = npl_size - 1;
        self.w/*: Vec<Vec<Vec<f64>>>*/ = vec![vec![vec![]]];
        let mut rand = rand::thread_rng();
        self.w.push(None);
        for layer in 1..(self.L + 1) {
            self.w.push(None);
            for i in npl[layer - 1] + 1 {
                self.w[layer].push([]);
                for j in npl[layer] + 1 {
                    self.w[layer][i].append(rand.gen_range(-1., 1.));
                }
            }
        }

        self.deltas/*: Vec<Vec<f64>>*/ = vec![vec![]];
        for layer in 1..(self.L + 1) {
            self.deltas.push(None);
            for j in npl[layer] + 1 {
                self.deltas[layer].push(0.0);
            }
        }

        self.x/*: Vec<Vec<f64>>*/ = vec![vec![]];
        for layer in self.L + 1 {
            self.x.push(None);
            for j in npl[layer] + 1 {
                self.x[layer].push(1.0);
            }
        }
    }
}

pub fn create_mlp(npl: &[i32], npl_size: i32) {
    let mut mlp = MLP;
    return mlp.create(npl, npl_size);
}

/**

Predictions functions

*/

pub fn mlp_predict_common(mlp: MLP<f64>, inputs: &[f64], classification_mode: bool) -> &[Vec<f64>] {
    for j in 1..(mlp.npl[0] + 1) {
        mlp.x[0][j] = inputs[j - 1];
    }

    for layer in 1..(mlp.L + 1) {
        for j in 1..(mlp.npl[layer] + 1) {
            let mut result = 0.0;
            for i in 0..(mlp.npl[layer - 1] + 1) {
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
    return mlp_predict_common(mlp, inputs, false);
}

pub extern fn mlp_classification(mlp: MLP<f64>, inputs: &[f64]) -> &[Vec<f64>] {
    return mlp_predict_common(mlp, inputs, true);
}

/**

Training functions

*/


pub extern fn mlp_train_common(mlp: MLP<f64>,
                               inputs: &[Vec<f64>],
                               expected_outputs: &[Vec<f64>],
                               iterations: i32,
                               alpha: f64,
                               classification_mode: bool) {
    let mut rand = rand::thread_rng();
    for iterator in iterations {
        let mut k = rand.gen_range(-1., 1.);
        mlp_predict_common(*mlp, inputs[k], classification_mode); // pb ici
        for j in 1..(mlp.npl[mlp.L] + 1) {
            mlp.deltas[mlp.L][j] = mlp.x[mlp.L][j] - expected_outputs[k][j - 1];
            if classification_mode {
                mlp.deltas[mlp.L][j] *= 1 - mlp.x[mlp.L][j] * mlp.x[mlp.L][j];
            }
        }
    }

    for layer in (mlp.L + 1)..2 {
        for i in 1..(mlp.npl[layer - 1] + 1) {
            let mut result = 0;
            for j in 1..(mlp.npl[layer] + 1) {
                result += mlp.w[layer][i][j] * mlp.deltas[layer][j];
            }
            result *= 1 - mlp.x[layer - 1][i] * mlp.x[layer - 1][i];
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
                                iterations: i32,
                                alpha: f64) {
    return mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha, true);
}

pub fn mlp_train_regression(mlp: MLP<f64>,
                            inputs: &[Vec<f64>],
                            expected_outputs: &[Vec<f64>],
                            iterations: i32,
                            alpha: f64) {
    return mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha, false);
}