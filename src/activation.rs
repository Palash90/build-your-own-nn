use crate::Layer;
use crate::tensor::Tensor;
use crate::tensor::TensorError;

pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
}
pub struct Activation {
    input: Tensor,
    t: ActivationType,
}

impl Layer for Activation {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, TensorError> {
        self.input = Tensor::new(input.data().to_vec(), input.shape().to_vec())?;

        match self.t {
            ActivationType::ReLU => input.relu(),
            ActivationType::Sigmoid => {
                let neg_x = input.scale(&-1.0)?;
                let denominator = Tensor::one(input.shape().to_vec())?.add(&neg_x.exp()?)?;

                Tensor::one(input.shape().to_vec())?.div(&denominator)
            }
            ActivationType::Tanh => {
                // Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                let exp_x = input.exp()?;
                let exp_neg_x = input.scale(&-1.0)?.exp()?;

                let numerator = exp_x.sub(&exp_neg_x)?;
                let denominator = exp_x.add(&exp_neg_x)?;

                numerator.div(&denominator)
            }
        }
    }

    fn backward(&mut self, output_error: &Tensor, _: f32) -> Result<Tensor, TensorError> {
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
            ActivationType::Tanh => {
                // Derivative: 1 - tanh^2(x)
                let exp_x = self.input.exp()?;
                let exp_neg_x = self.input.scale(&-1.0)?.exp()?;
                let tanh_x = exp_x.sub(&exp_neg_x)?.div(&exp_x.add(&exp_neg_x)?)?;

                let one = Tensor::one(tanh_x.shape().to_vec())?;
                let tanh_sq = tanh_x.mul(&tanh_x)?;
                let tanh_prime = one.sub(&tanh_sq)?;

                output_error.mul(&tanh_prime)
            }
        }
    }
}

impl Activation {
    pub fn new(t: ActivationType) -> Self {
        Activation {
            input: Tensor::empty(),
            t,
        }
    }
}
