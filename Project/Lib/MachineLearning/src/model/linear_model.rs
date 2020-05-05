extern crate ndarray;
extern crate rand;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use nalgebra::*;


// #[no_mangle]  pour partager une fonction
/**
    Test
*/
pub fn test() {
    // Lineare Model classification
    let array_x3: Vec<Vec<f64>> = vec![vec![1., 1.], vec![2., 3.], vec![3., 3.]];
    let array_y3: Vec<f64> = vec![1., -1., -1.];
    test_linear_model_classification_python(array_x3.clone(), array_y3.clone());

    // Lineare Model Régression
    let array_x2: Vec<Vec<f64>> = vec![vec![1.], vec![2.]];
    let array_y2: Vec<f64> = vec![2., 3.];

    test_linear_model_regression_python(array_x2, array_y2);

}

/**
    Test lineare model regression in python
    @param array_x: Vec<Vec<f64>> Vector 2 dimension
    @param array_y: Vec<f64> Vector 1 dimension
*/
pub fn test_linear_model_regression_python(array_x: Vec<Vec<f64>>, array_y: Vec<f64>) {
    println!("MODELE LINEAIRE REGRESSION");

    // Entrainement du model
    let model_regression = train_regression(array_x.clone(), array_y);
    // println!("model_regression: {:?}", model_regression.clone());

    // Test du dataset
    for i in 0..(array_x.len()) {
        println!("{:?}", predict_linear_regression(model_regression.clone(), array_x[i].clone()));
    }
}

/**
    Test lineare model classification in python
    @param array_x: Vec<Vec<f64>> Vector 2 dimension
    @param array_y: Vec<f64> Vector 1 dimension
*/
pub fn test_linear_model_classification_python(array_x: Vec<Vec<f64>>, array_y: Vec<f64>) {
    println!("MODELE LINEAIRE CLASIFFICATION");

    // Initialisation des poids du modèle
    let mut model_classification2 = create_model(array_x.clone());

    // Lancement de l'entrainement du modèle
    train_rosenblatt_2(&mut model_classification2, array_x.clone(), array_y.clone(), 10000, 0.01);

    // Transformation du tableau 2D en 1D
    let array_x_transform: Vec<f64> = double_vec_in_one_vec(array_x.clone());
    // Transformation du tableau en matrice
    let array_x_matrix = Array::from_shape_vec((array_x.len(), array_x.first().unwrap().len()), array_x_transform);

    // Test sur le dataset
    for i in 0..(array_x_matrix.clone().unwrap().shape()[0]) {
        println!("{:?}", predict_linear_classification(&model_classification2, array_x_matrix.clone().unwrap().slice(s![i, ..])));
    }
}

/**
    Prédiction linéaire pour un probleme de classification
    @param &Array1<f64> random_array Tableau 1D de random
    @param ArrayView1<f64> array_x_k Tableau de visualisation 1D
    @return f64 1. ou -1.
*/
pub fn predict_linear_classification(model: &Array1<f64>, array_x_k: ArrayView1<f64>) -> f64 {
    let mut sum = model[0];
    for i in 0..(array_x_k.shape()[0]) {
        sum += model[i + 1] * array_x_k[i];
    }
    return if sum >= 0. { 1. } else { -1. };
}

/**
    Prédiction linéaire pour un probleme de régression
    @param &Array1<f64> random_array Tableau 1D de random
    @param ArrayView1<f64> array_x_k Tableau de visualisation 1D
    @return f64 somme
*/
pub fn predict_linear_regression(model: Vec<f64>, array_x_k: Vec<f64>) -> f64 {
    let mut sum = model[0];
    for i in 0..(array_x_k.len()) {
        sum += model[i] * array_x_k[i];
    }
    return sum;
}

/**
    Règle de Rosenblatt
    @param mutable Array1<f64> random_array Tableau 1D de random
    @param Array1<f64> array_x Tableau 2D des données x
    @param Array1<f64> array_y Tableau 2D des données y
    @param i64 nb_iter nombre d'itération
    @param f64 alpha le pas
*/
pub fn train_rosenblatt(model: &mut Array1<f64>, array_x: &Array2<f64>, array_y: &Array1<f64>, nb_iter: i64, alpha: f64) {
    for _it in 0..nb_iter {
        let k: usize = rand::thread_rng().gen_range(0, model.shape()[0]);
        let gxk: f64 = predict_linear_classification(model, array_x.slice(s![k, ..]));
        for i in 0..(array_x.shape()[1]) {
            model[i + 1] += alpha * (array_y[k] - gxk) * array_x[[k, i]];
        }
        model[0] += alpha * (array_y[k] - gxk);
    }
}

/**
    Initialisation des poids du modèle
    @param array_x: Vec<Vec<f64>> Tableau 2D des données x
    @return Array1<f64> Tableau 1D de random
*/
fn create_model(array_x: Vec<Vec<f64>>) -> Array1<f64> {
    let array_x_matrix = Array::from_shape_vec((array_x.len(), array_x.first().unwrap().len()), double_vec_in_one_vec(array_x.clone()));
    assert!(array_x_matrix.is_ok());
    return Array::random(array_x_matrix.unwrap().shape()[1] + 1, Uniform::new(0., 1.));
}

/**
    Règle de Rosenblatt
    @param model: &mut Array1<f64> random_array Tableau 1D de random
    @param array_x: Vec<Vec<f64>> Tableau 2D des données x
    @param array_y: Vec<f64> Tableau 1D des données y
    @param nb_iter: i64 nombre d'itération
    @param alpha: f64 le pas
*/
pub fn train_rosenblatt_2(model: &mut Array1<f64>, array_x: Vec<Vec<f64>>, array_y: Vec<f64>, nb_iter: i64, alpha: f64) {
    let array_x_transform: Vec<f64> = double_vec_in_one_vec(array_x.clone());
    let array_x_matrix = Array::from_shape_vec((array_x.len(), array_x.first().unwrap().len()), array_x_transform);
    let array_y_matrix = Array::from_shape_vec((1, array_y.len()), array_y.clone());
    assert!(array_x_matrix.is_ok());
    assert!(array_y_matrix.is_ok());

    for _it in 0..nb_iter {
        let k: usize = rand::thread_rng().gen_range(0, model.shape()[0]);
        let gxk: f64 = predict_linear_classification(model, array_x_matrix.clone().unwrap().slice(s![k, ..]));
        for i in 0..(array_x_matrix.clone().unwrap().shape()[1]) {
            model[i + 1] += alpha * (array_y.get(k).unwrap() - gxk) * array_x_matrix.clone().unwrap()[[k, i]];
        }
        model[0] += alpha * (array_y.get(k).unwrap() - gxk);
    }
}

/**
    Convertir un vecteur de vecteur en vecteur simple
    @param Vec<Vec<f64>> vecteur de vecteur
    @return Vec<f64> conversion du vecteur double en simple
*/
fn double_vec_in_one_vec(double_vec: Vec<Vec<f64>>) -> Vec<f64> {
    let mut array: Vec<f64> = Vec::new();
    for row in double_vec {
        for value in row {
            array.push(value);
        }
    }
    return array;
}

/**
    Entrainement modéle linéaire simple de régression
    @param Vec<Vec<f64>> array_x
    @param Vec<f64> array_y
    @return Vec<f64>
*/
fn train_regression(array_x: Vec<Vec<f64>>, array_y: Vec<f64>) -> Vec<f64> {
    let array_x_transform: Vec<f64> = double_vec_in_one_vec(array_x.clone());

    let matrix_x = DMatrix::from_row_slice(array_x.len(), array_x.first().unwrap().len(), &array_x_transform);

    let matrix_y = DMatrix::from_row_slice(array_y.len(), 1, &array_y);

    let matrix_w = (((matrix_x.transpose() * matrix_x.clone()).try_inverse()).unwrap() * matrix_x.transpose()) * matrix_y.clone();

    return matrix_w.data.as_vec().to_vec();
}
