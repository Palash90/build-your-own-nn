use crate::Rng;
use crate::tensor::Tensor;
use crate::tensor::TensorError;
use std::vec;

pub struct Linear {
    weight: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &mut dyn Rng) -> Self {
        let weights = (0..in_features * out_features)
            .map(|_| rng.next_f32())
            .collect();
        
        let weight = Tensor::new(weights, vec![in_features, out_features]).unwrap();

        Linear { weight }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, TensorError> {
        input.matmul(&self.weight)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}
