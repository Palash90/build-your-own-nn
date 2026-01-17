pub mod linear;
pub mod tensor;

pub trait Rng {
    fn next_u32(&mut self) -> i32;
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (i32::MAX as f32)
    }
}
