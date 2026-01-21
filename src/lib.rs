pub mod linear;
pub mod tensor;
pub mod loss;
pub mod image_generator;
pub mod activation;
pub mod image_util;
pub mod examples;

pub trait Rng {
    fn next_u32(&mut self) -> i32;
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (i32::MAX as f32)
    }
}
