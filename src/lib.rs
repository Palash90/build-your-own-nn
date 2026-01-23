use crate::tensor::{Tensor, TensorError};

pub mod activation;
pub mod examples;
pub mod image_generator;
pub mod image_utils;
pub mod linear;
pub mod loss;
pub mod neural_network;
pub mod tensor;

pub trait Rng {
    fn next_u32(&mut self) -> i32;
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (i32::MAX as f32)
    }
}

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError>;
    fn backward(&mut self, output_error: &Tensor, learning_rate: f32) -> Result<Tensor, TensorError>;
}
