pub fn test() {
    
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