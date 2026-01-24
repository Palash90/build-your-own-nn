use crate::{
    Layer, Rng,
    linear::Linear,
    loss::{mse_loss, mse_loss_gradient},
    tensor::{Tensor, TensorError},
};

pub fn linear_regression(rng: &mut dyn Rng) -> Result<(), TensorError> {
    let mut linear = Linear::new(2, 1, rng);

    println!("Initial Weights:");
    println!("{}", linear.weight());

    let input = Tensor::new(
        vec![
            1.0, 1.0_f32, 2.0, 1.0_f32, 3.0, 1.0_f32, 4.0, 1.0_f32, 5.0, 1.0_f32,
        ],
        vec![5, 2],
    )?;

    println!("Input:");
    println!("{}", input);

    let output = linear.forward(&input).unwrap();
    println!("Initial Output:");
    println!("{}", output);

    let actual = Tensor::new(vec![5.6, 6.6, 9.5, 10.2, 14.0], vec![5, 1])?;

    let loss = mse_loss(&output, &actual)?;

    println!("Initial MSE Loss:");
    println!("{}", loss);

    println!();
    println!();

    let epochs = 8000;

    for _ in 0..epochs {
        let predicted = linear.forward(&input)?;

        let grad = mse_loss_gradient(&predicted, &actual)?;

        linear.backward(&grad, 0.01)?;
    }

    let output = linear.forward(&input)?;
    let loss = mse_loss(&output, &actual)?;

    println!("Final MSE Loss after {epochs} iterations:");
    println!("{}", loss);

    println!("Final weights");
    println!("{}", linear.weight());

    println!("Final Prediction");
    println!("{}", output);

    println!("Actual Output:");
    println!("{}", actual);

    Ok(())
}
