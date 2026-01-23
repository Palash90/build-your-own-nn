use crate::{
    Rng,
    activation::{Activation, ActivationType},
    linear::Linear,
    loss::bce_sigmoid_delta,
    neural_network::NetworkBuilder,
    tensor::{Tensor, TensorError},
};

pub fn xor_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {
    let mut nn = NetworkBuilder::new()
        .add_layer(Box::new(Linear::new(3, 12, rng)))
        .add_layer(Box::new(Activation::new(ActivationType::ReLU)))
        .add_layer(Box::new(Linear::new(12, 1, rng)))
        .add_layer(Box::new(Activation::new(ActivationType::Sigmoid)))
        .loss_gradient(bce_sigmoid_delta)
        .build()
        .expect("Error building network");

    let input = Tensor::new(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![4, 3],
    )?;

    let actual = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1])?;

    println!("Input:\n{}", input);
    println!("Actual Output:\n{}", actual);

    println!("Training...");
    nn.fit(&input, &actual, 20_000, 0.01)?;

    let model_output = nn.forward(input)?;

    println!("Model Output after training:\n{}", model_output);

    Ok(())
}
