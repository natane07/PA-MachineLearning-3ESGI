extern crate rand;

use rand::{Rng};
use std::slice::from_raw_parts;
use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::fs::{File, read_to_string};
use std::os::raw::c_char;
use std::ffi::CStr;
use std::str::from_utf8;
use std::io::Read;

// Structure pour la Serialiation et désérialisation
#[derive(Serialize, Deserialize)]
pub struct MLP_serialize {
    npl: Vec<i64>,
    l: usize,
    w: Vec<Vec<Vec<f64>>>,
    deltas: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}

pub struct MLP {
    npl: Vec<i64>, // neurone perd layers (réseau de neurone)
    l: usize, // indice de la derniere couche du reseau
    w: Vec<Vec<Vec<f64>>>, // poids
    deltas: Vec<Vec<f64>>,
    x: Vec<Vec<f64>>,
}


impl MLP {
    // Création et initialisiation du mlp
    pub fn create(mut self, npl: Vec<i64>) -> MLP{
        let mut rand = rand::thread_rng();
        // Couche 0
        self.w.push(vec![]);
        // Pour chaque couche, on stock les poids
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
        // Pour chaque couche, on stock les deltas (autant que le nombre de neurone par couche + biais)
        for layer in 1..(self.l + 1) {
            self.deltas.push(vec![]);
            for _j in 0..(npl[layer] + 1) as usize{
                self.deltas[layer].push(0.0);
            }
        }

        // Pour chaque couche, on stock les x (autant que le nombre de neurone par couche + biais)
        for layer in 0..(self.l + 1) {
            self.x.push(vec![]);
            for j in 0..(npl[layer] + 1) as usize {
                if j == 0 {
                    // biais
                    self.x[layer].push(1.0);

                } else {
                    self.x[layer].push(0.0);
                }
            }
        }
        return self;
    }
}

// Création et initialisiation du mlp
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

// Création et initialisiation du mlp (coté C)
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

// Fonction pour sérialiser notre stucture MLP (coté C)
#[no_mangle]
pub fn serialized_mlp(model_ptr: *const MLP) {
    let mlp_python;
    unsafe {
        mlp_python = model_ptr.as_ref().unwrap();
    }

    let mlp = MLP_serialize {
        npl: mlp_python.npl.clone(),
        l: mlp_python.l,
        w: mlp_python.w.clone(),
        deltas: mlp_python.deltas.clone(),
        x: mlp_python.x.clone(),
    };

    let j = serde_json::to_string(&mlp).unwrap();
    let mut file = File::create("mlp.json").expect("Unable to create");
    serde_json::to_writer(file, &j);
}

// Fonction pour désérialiser notre stucture MLP (coté C)
#[no_mangle]
pub fn deserialized_mlp() -> *const MLP_serialize{
    let contents = read_to_string("mlp.json").expect("Unable to open");
    let deserialized: MLP_serialize = serde_json::from_str(&contents).unwrap();
    println!("MLP DATA: {:?}", deserialized.npl);
    Box::leak(Box::new(deserialized))
}

// Prediction
pub fn _mlp_predict_common(mlp: &mut MLP, sample_imput: Vec<f64>, classification_mode: bool) -> Vec<f64> {
    // Copie des input dans la couche d'entre du reseau
    for j in 1..(mlp.npl[0] + 1) as usize{
        mlp.x[0][j] = sample_imput[j - 1];
    }

    // Application de la formule pour la prediction
    for layer in 1..(mlp.l + 1) {
        // Iteration sur chaque neurone
        for j in 1..(mlp.npl[layer] + 1) as usize {
            // Somme pondérée des poids (w) * x
            let mut result: f64 = 0.0;
            for i in 0..(mlp.npl[layer - 1] + 1) as usize {
                result += mlp.w[layer][i][j] * mlp.x[layer - 1][i];
            }
            // Tangente hyperbolique (tanh) pour de la classification
            if layer != mlp.l || classification_mode {
                result = result.tanh();
            }
            // Stocké dans x(j,l)
            mlp.x[layer][j] = result;
        }
    }

    return mlp.x[mlp.l].clone();
}

// Prédiction pour la regression
pub extern fn _mlp_regression(mlp: &mut MLP, sample_imput: Vec<f64>) -> Vec<f64> {
    return _mlp_predict_common(mlp, sample_imput, false);
}

// Prédiction pour la regression (cote C)
#[no_mangle]
pub extern fn mlp_regression(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> *const f64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };;
    let test = _mlp_regression(mlp, sample_imput.to_vec());
    println!("{:?}", test);
    return _mlp_regression(mlp, sample_imput.to_vec()).as_ptr();
}

// Récupérer la valeur max de notre prediction pour la régression (coté C)
#[no_mangle]
pub extern fn mlp_regression_max_value(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> f64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };
    let mut result = _mlp_regression(mlp, sample_imput.to_vec());
    let mut max_index:i64 = 0;
    let mut max_value :f64 = result[1];
    for i in (1..result.len()){
        if result[i] > max_value {
            max_value = result[i];
            max_index = i as i64 - 1;
        }
    }
    return max_value;
}

// Prédiction pour la classification
pub extern fn _mlp_classification(mlp: &mut MLP, sample_imput: Vec<f64>) -> Vec<f64> {
    return _mlp_predict_common(mlp, sample_imput, true);
}

// Prédiction pour la classification (coté C)
#[no_mangle]
pub extern fn mlp_classification(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> *const f64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };
    let test = _mlp_classification(mlp, sample_imput.to_vec());
    println!("{:?}", test);
    return _mlp_classification(mlp, sample_imput.to_vec()).as_ptr();
}

// Récupérer le arg_max de notre prediction (coté C)
#[no_mangle]
pub extern fn mlp_classification_image(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> i64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };
    let mut result = _mlp_classification(mlp, sample_imput.to_vec());
    let mut max_index:i64 = 0;
    let mut max_value :f64 = result[1];
    for i in (1..result.len()){
        if result[i] > max_value {
            max_value = result[i];
            max_index = i as i64 - 1;
        }
    }
    return max_index;
}

// Récupérer la valeur max de notre prediction pour la classification (coté C)
#[no_mangle]
pub extern fn mlp_classification_max_value(mlp: *mut MLP, sample_imput: *mut f64, sample_imput_size: usize) -> f64 {
    let sample_imput = unsafe { from_raw_parts(sample_imput, sample_imput_size) };
    let mlp = unsafe { mlp.as_mut().unwrap() };
    let mut result = _mlp_classification(mlp, sample_imput.to_vec());
    let mut max_index:i64 = 0;
    let mut max_value :f64 = result[1];
    for i in (1..result.len()){
        if result[i] > max_value {
            max_value = result[i];
            max_index = i as i64 - 1;
        }
    }
    return max_value;
}

// Entrainement avec stochastic gradient descent
pub extern fn mlp_train_common(mlp: &mut MLP,
                               inputs: Vec<Vec<f64>>,
                               expected_outputs: Vec<Vec<f64>>,
                               iterations: usize,
                               alpha: f64, // learning rate
                               classification_mode: bool) {
    let mut rand = rand::thread_rng();
    // Boucle selon le nombre d'iteration
    for iterator in 0..iterations {
        // On prend un exemple au hasard
        let mut k: usize = rand.gen_range(0, inputs.len()) as usize;
        // Calcul des sorties de la derniere couche en propagent via le predict
        _mlp_predict_common(mlp, inputs[k].clone(), classification_mode);
        // Pour chaque neurone de la derniere couche (sauf biais)
        for j in 1..(mlp.npl[mlp.l] + 1) as usize{
            // Calcul des deltas pour chaque neuronne de la derniere couche
            mlp.deltas[mlp.l][j] = mlp.x[mlp.l][j] - expected_outputs[k][j - 1];
            if classification_mode {
                mlp.deltas[mlp.l][j] *= (1f64 - (mlp.x[mlp.l][j] * mlp.x[mlp.l][j]));
            }
        }

        // Calcul des deltas des couches precedentes
        for layer in -(mlp.l as i64)..-1 {
            let layer = -layer as usize;
            for i in 1..(mlp.npl[layer - 1] + 1) as usize {
                // Calcul de la somme pondérée
                let mut result: f64 = 0.0;
                for j in 1..(mlp.npl[layer] + 1) as usize{
                    result += mlp.w[layer][i][j] * mlp.deltas[layer][j];
                }
                result *= (1f64 - (mlp.x[layer - 1][i] * mlp.x[layer - 1][i]));
                mlp.deltas[layer - 1][i] = result;
            }
        }

        // Mise à jour des w
        for layer in 1..(mlp.l + 1) {
            for i in 0..(mlp.npl[layer - 1] + 1) as usize{
                for j in 1..(mlp.npl[layer] + 1) as usize{
                    mlp.w[layer][i][j] -= alpha * mlp.x[layer - 1][i] * mlp.deltas[layer][j];
                }
            }
        }
    }
}

// Entrainement pour la classification
pub extern fn _mlp_train_classification(mlp: &mut MLP,
                                        inputs: Vec<Vec<f64>>,
                                        expected_outputs: Vec<Vec<f64>>,
                                        iterations: usize,
                                        alpha: f64) {
    mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha,true);
}

// Entrainement pour la classification (pour le C)
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

// Entrainement pour la regression
pub extern fn _mlp_train_regression(mlp: &mut MLP,
                                    inputs: Vec<Vec<f64>>,
                                    expected_outputs: Vec<Vec<f64>>,
                                    iterations: usize,
                                    alpha: f64) {
    mlp_train_common(mlp, inputs, expected_outputs, iterations, alpha,false);
}

// Entrainement pour la regression (pour le C)
#[no_mangle]
pub extern fn mlp_train_regression(mlp: *mut MLP,
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
    println!("mlp_train_regression DEBUT");
    println!("mlp.npl:{:?}", mlp.npl);
    println!("mlp.deltas:{:?}", mlp.deltas);
    println!("mlp.l:{:?}", mlp.l);
    println!("mlp.w:{:?}", mlp.w);
    println!("mlp.x:{:?}", mlp.x);
    _mlp_train_regression(mlp, inputs, outputs, iterations, alpha);
    println!("mlp.npl:{:?}", mlp.npl);
    println!("mlp.deltas:{:?}", mlp.deltas);
    println!("mlp.l:{:?}", mlp.l);
    println!("mlp.w:{:?}", mlp.w);
    println!("mlp.x:{:?}", mlp.x);
    println!("mlp_train_regression FIN");
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