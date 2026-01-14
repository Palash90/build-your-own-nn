use build_your_own_nn::tensor::{Tensor, TensorError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_shape_creation() {
        let result = Tensor::new(vec![1.0], vec![2, 2]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TensorError::InconsistentData);
    }

    #[test]
    fn test_tensor_operations() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

        let c = a.add(&b)?;
        assert_eq!(c.data(), &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(c.shape(), &[2, 2]);

        let d = a.sub(&b)?;
        assert_eq!(d.data(), &[-4.0, -4.0, -4.0, -4.0]);

        let e = a.mul(&b)?;
        assert_eq!(e.data(), &[5.0, 12.0, 21.0, 32.0]);

        Ok(())
    }

    #[test]
    fn test_tensor_display_2d() -> Result<(), TensorError> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let output = format!("{}", tensor);

        assert!(output.contains("1.0000"));
        assert!(output.contains("4.0000"));
        assert!(output.contains('|'));
        Ok(())
    }

    #[test]
    fn test_tensor_display_1d() -> Result<(), TensorError> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;

        let output = format!("{}", tensor);

        assert!(output.contains("[1.0, 2.0, 3.0]"));
        Ok(())
    }

    #[test]
    fn test_tensor_display_alignment() -> Result<(), TensorError> {
        let tensor = Tensor::new(vec![1.23456, 2.0, 100.1, 0.00001], vec![2, 2])?;
        let output = format!("{}", tensor);

        assert!(output.contains("1.2346"));
        assert!(output.contains("0.0000"));
        Ok(())
    }

    #[test]
    fn test_transpose_square() -> Result<(), TensorError> {
        // 1.0, 2.0

        // 3.0, 4.0

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

        let swapped = a.transpose()?;

        // Should become:

        // 1.0, 3.0

        // 2.0, 4.0

        assert_eq!(swapped.data(), &[1.0, 3.0, 2.0, 4.0]);

        assert_eq!(swapped.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_transpose_rectangular() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let swapped = a.transpose()?;

        assert_eq!(swapped.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(swapped.shape(), &[3, 2]);
        Ok(())
    }

    #[test]
    fn test_transpose_1d() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6])?;
        let swapped = a.transpose()?;

        assert_eq!(swapped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(swapped.shape(), &[6]);
        Ok(())
    }    
}
