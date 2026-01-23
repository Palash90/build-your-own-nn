use crate::{Layer, Rng, activation::{Activation, ActivationType}, linear::Linear, loss::bce_sigmoid_delta, tensor::{Tensor, TensorError}};

pub fn or_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {

    let mut linear_layer = Linear::new(3, 1, rng);

    let mut activation_layer = Activation::new(ActivationType::Sigmoid);

    let input = Tensor::new(vec![0.0, 0.0, 1.0_f32, 0.0, 1.0, 1.0_f32, 1.0, 0.0, 1.0_f32, 1.0, 1.0, 1.0_f32], vec![4, 3])?;
    let actual = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1])?;

    let learning_rate = 0.1;

    println!("Input:");
    println!("{}", input);

    println!("Actual Output");
    println!("{}", actual);

    for _ in 0..10000 {
        let linear_output = linear_layer.forward(&input)?;
        let activation_output = activation_layer.forward(&linear_output)?;

        let delta = bce_sigmoid_delta(&activation_output, &actual)?;

        let _ = linear_layer.backward(&delta, learning_rate);
    }

    let model_output = linear_layer.forward(&input)?;
    let model_output = activation_layer.forward(&model_output)?;

    println!("Model Output after training");
    println!("{}", model_output);

    Ok(())
}   