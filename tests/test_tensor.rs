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

    #[test]
    fn test_matmul_naive() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        let c = a.matmul_naive(&b)?;
        
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(c.shape(), &[2, 2]);

        let d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let e = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?;
        let f = d.matmul_naive(&e)?;
        assert_eq!(f.data(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(f.shape(), &[2, 2]);

        let g = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1])?;
        let h = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3])?;
        let i = g.matmul_naive(&h)?;
        assert_eq!(i.data(), &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);;
        assert_eq!(i.shape(), &[3, 3]);

        let j = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3])?;
        let k = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])?;
        let l = j.matmul_naive(&k)?;
        assert_eq!(l.data(), &[32.0]);
        assert_eq!(l.shape(), &[1, 1]);

        let m = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let n = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
        let o = m.matmul_naive(&n)?;
        assert_eq!(o.data(), &[32.0]);
        assert_eq!(o.shape(), &[1]);

        let p = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let q = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3])?;
        let r = q.matmul_naive(&p)?;
        assert_eq!(r.data(), &[32.0]);
        assert_eq!(r.shape(), &[1]);

        let s = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let t = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])?;
        let u = s.matmul_naive(&t)?;
        assert_eq!(u.data(), &[32.0]);
        assert_eq!(u.shape(), &[1]);

        Ok(())
    }
}
