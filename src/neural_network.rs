use crate::Layer;
use crate::tensor::{Tensor, TensorError};

/// Type alias for the loss gradient function pointer
type LossGradFn = fn(&Tensor, &Tensor) -> Result<Tensor, TensorError>;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    loss_grad_fn: LossGradFn,
}

impl Network {
    /// Passes the input through all layers sequentially
    pub fn forward(&mut self, input: Tensor) -> Result<Tensor, TensorError> {
        if self.layers.is_empty() {
            return Ok(input);
        }

        let mut current_output = input;

        for layer in &mut self.layers {
            // Each layer processes the output of the previous layer
            current_output = layer.forward(&current_output)?;
        }

        Ok(current_output)
    }

    /// The training loop: Forward, Loss Gradient, and Backpropagation
    pub fn fit(
        &mut self,
        x_train: &Tensor,
        y_train: &Tensor,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<(), TensorError> {
        for epoch in 0..epochs {
            // Following is the forward pass
            let input = Tensor::new(x_train.data().to_vec(), x_train.shape().to_vec())?;
            let output = self.forward(input)?;

            // Loss gradient
            let mut gradient = (self.loss_grad_fn)(&output, y_train)?;

            // Passing the gradient backward from output to input
            for layer in self.layers.iter_mut().rev() {
                gradient = layer.backward(&gradient, learning_rate)?;
            }
        }
        Ok(())
    }
}

/// Builder pattern for cleaner Network initialization
pub struct NetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
    loss_grad: Option<LossGradFn>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            loss_grad: None,
        }
    }

    /// Adds a layer to the network stack
    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Injects the loss gradient function from loss.rs
    pub fn loss_gradient(mut self, f: LossGradFn) -> Self {
        self.loss_grad = Some(f);
        self
    }

    pub fn build(self) -> Result<Network, String> {
        let loss_grad_fn = self.loss_grad.ok_or("Loss gradient function is required")?;
        
        Ok(Network {
            layers: self.layers,
            loss_grad_fn,
        })
    }
}