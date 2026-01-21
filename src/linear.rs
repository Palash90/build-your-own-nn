use crate::Rng;
use crate::tensor::Tensor;
use crate::tensor::TensorError;
use std::vec;

pub struct Linear {
    weight: Tensor,
    input: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &mut dyn Rng) -> Self {
        let weights = (0..in_features * out_features)
            .map(|_| rng.next_f32())
            .collect();

        let weight = Tensor::new(weights, vec![in_features, out_features]).unwrap();

        let empty = Tensor::empty();

        Linear {
            weight,
            input: empty,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {
        // We store a copy of the input because the backward pass needs it
        // to calculate the gradient: dL/dW = input.T * output_error
        self.input = Tensor::new(input.data().to_vec(), input.shape().to_vec())?;

        input.matmul(&self.weight)
    }

    pub fn backward(
        &mut self,
        output_error: &Tensor,
        learning_rate: f32,
    ) -> Result<Tensor, TensorError> {
        let weight_t = self.weight.transpose()?;
        let input_error = output_error.matmul(&weight_t)?;

        let input_t = self.input.transpose()?;
        let weights_grad = input_t.matmul(output_error)?;

        let weight_step = weights_grad.scale(&learning_rate)?;
        self.weight = self.weight.sub(&weight_step)?;

        Ok(input_error)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn set_weight(&mut self, t: Tensor){
        self.weight = t;
    }
}
