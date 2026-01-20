use crate::tensor::Tensor;
use crate::tensor::TensorError;

pub enum ActivationType {
    ReLU,
    Sigmoid,
}
pub struct Activation {
    input: Tensor,
    t: ActivationType,
}

impl Activation {
    pub fn new(t: ActivationType) -> Self {
        Activation {
            input: Tensor::empty(),
            t,
        }
    }
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {
        self.input = Tensor::new(input.data().to_vec(), input.shape().to_vec())?;

        match self.t {
            ActivationType::ReLU => input.relu(),
            ActivationType::Sigmoid => {
                let neg_x = input.scale(&-1.0)?;
                let denominator = Tensor::one(input.shape().to_vec())?.add(&neg_x.exp()?)?;

                Tensor::one(input.shape().to_vec())?.div(&denominator)
            }
        }
    }

    pub fn backward(&self, output_error: &Tensor) -> Result<Tensor, TensorError> {
        match self.t {
            ActivationType::ReLU => {
                let mask = self.input.relu_prime()?;
                output_error.mul(&mask)
            }

            ActivationType::Sigmoid => {
                let neg_input = self.input.scale(&-1.0)?;
                let denominator =
                    Tensor::one(self.input.shape().to_vec())?.add(&neg_input.exp()?)?;
                let a = Tensor::one(self.input.shape().to_vec())?.div(&denominator)?;

                let one = Tensor::one(a.shape().to_vec())?;
                let sigmoid_prime = a.mul(&one.sub(&a)?)?;

                output_error.mul(&sigmoid_prime)
            }
        }
    }
}
