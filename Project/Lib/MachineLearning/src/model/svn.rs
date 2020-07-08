use osqp::{CscMatrix, Problem, Status, Settings};

pub struct SVN {
    p: Vec<Vec<f64>>,
    q: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    a: Vec<Vec<f64>>
}

pub fn _create_svn(x_size: usize) -> SVN {
    let mut svn = SVN {
        p: Vec::with_capacity(x_size),
        q: Vec::with_capacity(x_size),
        l: Vec::with_capacity(x_size + 1),
        u: Vec::with_capacity(x_size + 1),
        a: Vec::with_capacity(x_size + 1)
    };
    return svn;
}

fn _predict_svn(x: &[f64], w: &[f64]) -> f64 {
    let mut sum: f64 = w[0];
    for i in 0..x.len() {
        sum += x[i] * w[i + 1];
    }
    return sum.signum();
}
fn _train_svn(svn: &mut SVN, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> Vec<f64>{

    for i in 0.. x_train.len() {
        svn.p.push(Vec::with_capacity(x_train.len()));
        svn.q.push(-1.0);
        if i == 0 {
            svn.a.push(vec!());
            svn.l.push(0.0);
            svn.u.push(0.0);
        }
        svn.a.push(vec!());
        svn.l.push(0.0);
        svn.u.push(f64::INFINITY);

        for j in 0..x_train.len() {
            if i == 0 {
                svn.a[0].push(y_train[j]);
            }

            svn.a[i + 1].push(if i == j { 1.0 } else { 0.0 });
            let mut sum = 0.0;
            for k in 0..x_train[i].len() {
                sum += x_train[i][k] * x_train[j][k]
            }
            svn.p[i].push(y_train[i] * y_train[j] * sum);
        }
    }

    let p: Vec<&[f64]> = (svn.p.iter().map(|v | v.as_slice()).collect());
    let q: &[f64] = svn.q.as_slice();
    let l: &[f64] = svn.l.as_slice();
    let u: &[f64] = svn.u.as_slice();
    let a: Vec<&[f64]> = (svn.a.iter().map(|v | v.as_slice()).collect());

    let P: CscMatrix = CscMatrix::from(p).into_upper_tri();

    let settings = Settings::default().verbose(false);

    let mut prob: Problem = Problem::new(P, q, a, l, u, &settings).expect("Failed to setup");

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

    // Calcul du produit scaliare
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