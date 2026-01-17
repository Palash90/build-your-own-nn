use crate::tensor::Tensor;
use crate::tensor::TensorError;

pub fn l1_loss(predicted: &Tensor, actual: &Tensor) -> Result<Tensor, TensorError> {
    if predicted.shape() != actual.shape() {
        return Err(TensorError::ShapeMismatch);
    }

    let n = predicted.shape().iter().product::<usize>() as f32;

    let diff = predicted.sub(actual)?.abs()?;
    diff.sum(None)?.scale(&(1.0 / n))
}

pub fn mse_loss(predicted: &Tensor, actual: &Tensor) -> Result<Tensor, TensorError> {
    if predicted.shape() != actual.shape() {
        return Err(TensorError::ShapeMismatch);
    }

    let n = predicted.shape().iter().product::<usize>() as f32;

    predicted
        .sub(actual)?
        .powf(2.0)?
        .sum(None)?
        .scale(&(1.0 / n))
}
