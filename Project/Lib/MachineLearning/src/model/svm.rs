use osqp::{CscMatrix, Problem, Status, Settings};

pub struct SVM {
    p: Vec<Vec<f64>>, // Matrice ligne/colonne
    q: Vec<f64>, // Autant d'element que d'exemple
    l: Vec<f64>,
    u: Vec<f64>,
    a: Vec<Vec<f64>>
}

pub fn _create_svm(x_size: usize) -> SVM {
    let mut svm = SVM {
        p: Vec::with_capacity(x_size), // Remplire avec autant d'exemple
        q: Vec::with_capacity(x_size),
        l: Vec::with_capacity(x_size + 1),
        u: Vec::with_capacity(x_size + 1),
        a: Vec::with_capacity(x_size + 1)
    };
    return svm;
}

fn _predict_svm(x: &[f64], w: &[f64]) -> f64 {
    let mut sum: f64 = w[0];
    for i in 0..x.len() {
        sum += x[i] * w[i + 1];
    }
    return sum.signum();
}
fn _train_svm(svm: &mut SVM, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> Vec<f64>{

    for i in 0.. x_train.len() {
        // Remplie la matrice, autant d'exemple que de colonne
        svm.p.push(Vec::with_capacity(x_train.len()));
        svm.q.push(-1.0);

        if i == 0 {
            // 0 <= alpha <= 0
            svm.a.push(vec!());
            svm.l.push(0.0);
            svm.u.push(0.0);
        }
        // 0 <= alpha <= +Infinie
        svm.a.push(vec!());
        svm.l.push(0.0);
        svm.u.push(f64::INFINITY);

        // Pour chaque exemple
        for j in 0..x_train.len() {
            if i == 0 {
                svm.a[0].push(y_train[j]);
            }

            svm.a[i + 1].push(if i == j { 1.0 } else { 0.0 });
            let mut sum = 0.0;
            // Produit scalaire des vecteurs Xi * Transposer(Xj)
            for k in 0..x_train[i].len() {
                sum += x_train[i][k] * x_train[j][k]
            }
            // Ajout des elements dans la colonne
            svm.p[i].push(y_train[i] * y_train[j] * sum);
        }
    }

    // Conversion en slice
    let p: Vec<&[f64]> = (svm.p.iter().map(|v | v.as_slice()).collect());
    let q: &[f64] = svm.q.as_slice();
    let l: &[f64] = svm.l.as_slice();
    let u: &[f64] = svm.u.as_slice();
    let a: Vec<&[f64]> = (svm.a.iter().map(|v | v.as_slice()).collect());

    // Création de la matrice
    let P: CscMatrix = CscMatrix::from(p).into_upper_tri();

    // Parametrage des settings pour le solver
    let settings = Settings::default().verbose(false);

    // Creation du probleme quadratique
    let mut prob: Problem = Problem::new(P, q, a, l, u, &settings).expect("Failed to setup");

    // Resolution du probleme quadratique (récupere les alphas)
    let result: Status = prob.solve();

    println!("{:?}", result.x().expect("Failed to solve problem"));

    let alphas: &[f64] = result.x().unwrap();

    let mut w: Vec<f64> = Vec::new();

    for i in 0..x_train[0].len() {
        w.push(0.0);
        for n in 0..alphas.len() {
            w[i] += alphas[n] * y_train[n] * x_train[n][i]
        }
    }

    // Calcul du biais
    let mut max: f64 = std::f64::MIN;
    let mut arg_max: usize = 0;

    // Recuperer l'alpha max
    for n in 0..alphas.len() {
        if alphas[n].abs() > max {
            arg_max = n;
            max = alphas[n].abs();
        }
    }

    // Calcul du produit scalaire
    let mut sum = 0.0;
    for i in 0..x_train[arg_max].len() {
        sum += w[i] * x_train[arg_max][i];
    }

    // Ajout du vecteur support
    w.insert(0, 1.0 / y_train[arg_max] - sum);

    return w;
}

// for k in 0..x_train.len() {
//     dbg! (_predict_svn(&x_train[k], w.as_slice()))
// }