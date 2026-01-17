#[cfg(test)]
mod tests {
    use build_your_own_nn::{loss::{l1_loss, mse_loss}, tensor::{Tensor, TensorError}};

    fn create_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).unwrap()
    }

    #[test]
    fn test_l1_loss_correctness() {
        // L1 = sum(|predicted - actual|)
        // |2.0 - 1.0| + |4.0 - 5.0| = 1.0 + 1.0 = 2.0
        let pred = create_tensor(vec![2.0, 4.0], vec![2, 1]);
        let actual = create_tensor(vec![1.0, 5.0], vec![2, 1]);
        
        let loss = l1_loss(&pred, &actual).unwrap();
        
        assert_eq!(loss.data()[0], 1.0);
    }

    #[test]
    fn test_mse_loss_correctness() {
        // MSE = 1/n * sum((predicted - actual)^2)
        // (1/2) * ((2.0 - 1.0)^2 + (4.0 - 6.0)^2)
        // (1/2) * (1.0 + 4.0) = 2.5
        let pred = create_tensor(vec![2.0, 4.0], vec![2, 1]);
        let actual = create_tensor(vec![1.0, 6.0], vec![2, 1]);
        
        let loss = mse_loss(&pred, &actual).unwrap();
        
        assert_eq!(loss.data()[0], 2.5);
    }

    #[test]
    fn test_loss_shape_mismatch() {
        let pred = create_tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let actual = create_tensor(vec![1.0, 2.0], vec![2, 1]);
        
        let l1_result = l1_loss(&pred, &actual);
        let mse_result = mse_loss(&pred, &actual);

        assert!(matches!(l1_result, Err(TensorError::ShapeMismatch)));
        assert!(matches!(mse_result, Err(TensorError::ShapeMismatch)));
    }

    #[test]
    fn test_zero_loss() {
        let pred = create_tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let actual = create_tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
        
        let loss = mse_loss(&pred, &actual).unwrap();
        assert_eq!(loss.data()[0], 0.0);
    }
}