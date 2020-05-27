extern crate ndarray;
extern crate rand;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use rand::{Rng, thread_rng};
use nalgebra::*;

/**
    Création d'un modele lineaire
*/
#[no_mangle]
pub extern fn create_linear_model(nb_inputs: i64) -> *const f64 {
    let mut model: Vec<f64> = Vec::new();
    let mut rand = rand::thread_rng();
    for _i in 0..(nb_inputs + 1) {
        let k: f64 = rand.gen_range(-1., 1.);
        model.push(k);
    }
    println!("model: {:?}", model);

    let mut slice = model.into_boxed_slice(); // To Remove Excess Capacity
    let ptr = slice.as_mut_ptr();
    Box::leak(slice); // To prevent memory from being reclaimed
    ptr
}

/**
    Prédiction linéaire pour un probleme de régression (python)
*/
#[no_mangle]
pub extern fn predict_linear_regression(model: *mut f64, inputs: *const f64, inputs_size: usize) -> f64 {
    unsafe {
        let slice_model = from_raw_parts(model, inputs_size + 1);
        let slice_inputs = from_raw_parts(inputs, inputs_size);
        return predict_linear_regression_(&slice_model, slice_inputs, inputs_size);
    }
}

/**
    Prédiction linéaire pour un probleme de régression
*/
pub fn predict_linear_regression_(slice_model: &[f64], slice_inputs: &[f64], inputs_size: usize) -> f64{
    let mut sum = slice_model[0];
    for i in 0..inputs_size {
        sum += slice_model[i + 1] * slice_inputs[i];
    }
    return sum;
}

/**
    Prédiction linéaire pour un probleme de classification (python)
*/
#[no_mangle]
pub extern fn predict_linear_classification(model: *mut f64, inputs: *const f64, inputs_size: usize) -> f64 {
    let sum = predict_linear_regression(model, inputs, inputs_size);
    if sum >= 0.0 { return 1.0; } else { return -1.0 }
}

/**
    Prédiction linéaire pour un probleme de classification
*/
pub fn predict_linear_classification_(slice_model: &[f64], slice_inputs: &[f64], inputs_size: usize) -> f64{
    let sum = predict_linear_regression_(slice_model, slice_inputs, inputs_size);
    if sum >= 0.0 { return 1.0; } else { return -1.0 }
}

/**
    Entrainement du modéle avec la règle de Rosenblatt
*/
#[no_mangle]
pub extern fn train_linear_model_classification_python(model_ptr: *mut f64,
                                                       dataset_inputs_ptr: *mut f64,
                                                       dataset_expected_outputs_ptr: *mut f64,
                                                       dataset_samples_count: usize,
                                                       dataset_sample_features_count: usize,
                                                       iterations_count: usize,
                                                       alpha: f64) {
    let model;
    let dataset_inputs;
    let dataset_expected_outputs;

    unsafe {
        model = from_raw_parts_mut(model_ptr, dataset_sample_features_count + 1);
        dataset_inputs = from_raw_parts(dataset_inputs_ptr, dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr, dataset_samples_count);
    }
    let mut rng = thread_rng();

    for _ in 0..iterations_count {
        let k = rng.gen_range(0, dataset_samples_count);
        let index_k = k * dataset_sample_features_count;

        let inputs_k = &dataset_inputs[index_k..(index_k + dataset_sample_features_count)];
        let output_k = dataset_expected_outputs[k];

        let predicted_output_k = predict_linear_classification_(&model, inputs_k, dataset_sample_features_count);

        let semi_grad = alpha * (output_k - predicted_output_k);

        for i in 0..dataset_sample_features_count {
            model[i + 1] += semi_grad * inputs_k[i];
        }
        model[0] += semi_grad * 1.0;
    }
}
/**
    Entrainement modéle linéaire simple de régression
*/
#[no_mangle]
pub extern fn train_linear_model_regression_python(model_ptr: *mut f64,
                               dataset_inputs_ptr: *mut f64,
                               dataset_expected_outputs_ptr: *mut f64,
                               dataset_samples_count: usize,
                               dataset_sample_features_count: usize) {
    let model;
    let dataset_inputs;
    let dataset_expected_outputs;

    // Création des slice
    unsafe {
        model = from_raw_parts_mut(model_ptr, dataset_sample_features_count + 1);
        dataset_inputs = from_raw_parts(dataset_inputs_ptr, dataset_samples_count * dataset_sample_features_count);
        dataset_expected_outputs = from_raw_parts(dataset_expected_outputs_ptr, dataset_samples_count);
    }
    // Transformation des slice en Vec
    let dataset_inputs_vec = dataset_inputs.to_vec();
    let dataset_expected_outputs_vec = dataset_expected_outputs.to_vec();

    // Création des matrice
    let matrix_x = DMatrix::from_vec(dataset_samples_count, dataset_sample_features_count, dataset_inputs_vec);
    let matrix_y = DMatrix::from_vec(dataset_samples_count, 1, dataset_expected_outputs_vec);

    // Calcul pseudo inverse
    let matrix_w = (((matrix_x.transpose() * matrix_x.clone()).try_inverse()).unwrap() * matrix_x.transpose()) * matrix_y.clone();

    let matrix_vec = matrix_w.data.as_vec().to_vec();
    for i in 0..dataset_sample_features_count {
        model[i] = matrix_vec[0];
    }
}
