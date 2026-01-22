use crate::{Rng, activation::{Activation, ActivationType}, linear::Linear, loss::bce_sigmoid_delta, tensor::{Tensor, TensorError}};

pub fn xor_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {

    let mut input_layer = Linear::new(3, 4, rng);
    let mut activation_layer = Activation::new(ActivationType::ReLU);

    // These two lines creates the new layer
    let mut hidden_layer = Linear::new(4, 1, rng);
    let mut hidden_activation = Activation::new(ActivationType::Sigmoid);

    let input = Tensor::new(vec![0.0, 0.0, 1.0_f32, 0.0, 1.0, 1.0_f32, 1.0, 0.0, 1.0_f32, 1.0, 1.0, 1.0_f32], vec![4, 3])?;

    // Notice the change in the actual output
    let actual = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1])?;

    let learning_rate = 0.001;

    println!("Input:");
    println!("{}", input);

    println!("Actual Output");
    println!("{}", actual);

    for _ in 0..500_000 {
        let linear_output = input_layer.forward(&input)?;
        let activation_output = activation_layer.forward(&linear_output)?;

        // Following two lines are stacking the new layer on top of the existing
        let hidden_output = hidden_layer.forward(&activation_output)?;
        let hidden_activation_output = hidden_activation.forward(&hidden_output)?;

        let delta = bce_sigmoid_delta(&hidden_activation_output, &actual)?;

        // Loss is also passed in reverse direction from output to input
        let hidden_backward = hidden_layer.backward(&delta, learning_rate)?;
        let activation_backward = activation_layer.backward(&hidden_backward)?;

        let _ = input_layer.backward(&activation_backward, learning_rate);

    }

    let model_output = input_layer.forward(&input)?;
    let model_output = activation_layer.forward(&model_output)?;

    // During prediction also, the input should pass through the stacked layers.
    let model_output = hidden_layer.forward(&model_output)?;
    let model_output = hidden_activation.forward(&model_output)?;

    println!("Model Output after training");
    println!("{}", model_output);

    Ok(())
}