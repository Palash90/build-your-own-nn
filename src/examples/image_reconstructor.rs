use std::time::Instant;

use crate::image_utils::{read_pbm_for_nn, render_image};
use crate::neural_network::Network;
use crate::tensor::TensorError;
use crate::{
    Rng,
    activation::{Activation, ActivationType},
    linear::Linear,
    loss::bce_sigmoid_delta,
    neural_network::NetworkBuilder,
    tensor::Tensor,
};

pub fn reconstruct_image(
    source: &str,
    size: usize,
    rng: &mut dyn Rng,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h, x_data, y_data) = read_pbm_for_nn(source);

    // Without Normalization, the gradient becomes zero. So, we make the data between 0 and 1
    let normalized_x_train: Vec<f32> = x_data
        .chunks(2)
        .flat_map(|coord| vec![coord[0] / h as f32, coord[1] / w as f32])
        .collect();

    let x_train = Tensor::new(normalized_x_train.clone(), vec![h * w, 2])?;
    let y_train = Tensor::new(y_data, vec![h * w, 1])?;

    let hl = 50; // Hidden layer size
    let mut nn = NetworkBuilder::new()
        .add_layer(Box::new(Linear::new(2, hl, rng)))
        .add_layer(Box::new(Activation::new(ActivationType::Tanh))) // For Image reconstruction tasks, Tanh is a better solution
        .add_layer(Box::new(Linear::new(hl, hl, rng)))
        .add_layer(Box::new(Activation::new(ActivationType::Tanh)))
        .add_layer(Box::new(Linear::new(hl, 2 * hl, rng))) // Expansion layer
        .add_layer(Box::new(Activation::new(ActivationType::Tanh)))
        .add_layer(Box::new(Linear::new(2 * hl, hl, rng))) // Contraction layer
        .add_layer(Box::new(Activation::new(ActivationType::Tanh)))
        .add_layer(Box::new(Linear::new(hl, hl / 2, rng)))
        .add_layer(Box::new(Activation::new(ActivationType::Tanh)))
        .add_layer(Box::new(Linear::new(hl / 2, 1, rng)))
        .add_layer(Box::new(Activation::new(ActivationType::Sigmoid))) // Final Sigmoid for pixel intensity
        .loss_gradient(bce_sigmoid_delta)
        .build()
        .map_err(|e| e.to_string())?;

    let total_epochs = 1000;
    let learning_rate = 0.01;

    // To perform back of the envelop calculation on how much time is required
    let mut last_checkpoint = Instant::now();

    for epoch in 0..total_epochs {
        nn.fit(&x_train, &y_train, 500, learning_rate)?;

        if epoch % 5 == 0 {
            println!("Reconstruction at epoch {epoch}");

            println!("Original Image:");
            // We use the original data for comparison
            render_image(w, h, &y_train.data());

            println!("Network Drawing after epoch {}:", epoch * 1000);
            draw_save_network_image(w, &mut nn, &format!("output/reconstruction{epoch}.pbm"))?;

            // Trace time
            let duration = last_checkpoint.elapsed();
            println!("\n==============================");
            println!("Epoch: {}", epoch);
            println!("Time since last checkpoint: {:.2?}", duration);
            println!("==============================");
            // Reset the timer for the next block
            last_checkpoint = Instant::now();

            // Let the CPU breath, otherwise thermal breakdown is possible
            std::thread::sleep(std::time::Duration::from_millis(2000));
        }
    }
    println!("Original Image:");
    // We use the original data for comparison
    render_image(w, h, &y_train.data());
    println!("Final Image Reconstruction");
    draw_save_network_image(size, &mut nn, "output/final.pbm")?;

    Ok(())
}

fn draw_save_network_image(size: usize, nn: &mut Network, dest: &str) -> Result<(), TensorError> {
    let mut dest_coords = Vec::with_capacity(size * size * 2);
    for r in 0..size {
        for c in 0..size {
            // Normalization here too
            dest_coords.push(r as f32 / size as f32);
            dest_coords.push(c as f32 / size as f32);
        }
    }

    let x_dest = Tensor::new(dest_coords, vec![size * size, 2])?;
    let prediction = nn.forward(x_dest)?;

    render_image(size, size, prediction.data());

    // Save the result to a file
    crate::image_utils::save_as_pbm(dest, size, size, prediction.data());
    println!("Saved reconstructed image to {}", dest);

    Ok(())
}
