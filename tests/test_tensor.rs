use build_your_own_nn::tensor::{Tensor, TensorError};

#[cfg(test)]
mod tests {
    use super::*;

    // The happy path test
    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data.clone(), shape.clone()).unwrap();

        assert_eq!(tensor.data(), &data);
        assert_eq!(tensor.shape(), &shape);
    }

    // Test the invariant; ensure data and shape are consistent with each other
    #[test]
    fn test_invalid_shape_creation() {
        let result = Tensor::new(vec![1.0], vec![2, 2]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TensorError::InconsistentData);
    }

    // To test our self imposed restriction to allow only up to 2D
    // When we'll allow more dimensions, this test should be removed
    #[test]
    fn test_rank_limits() {
        // We currently don't support 3D tensors (Rank 3)
        let result = Tensor::new(vec![1.0; 8], vec![2, 2, 2]);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TensorError::InvalidRank);
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
    fn test_matmul() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        let c = a.matmul_naive(&b)?;
        let c1 = a.matmul(&b)?;
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), c1.data());
        assert_eq!(c.shape(), c1.shape());

        let d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let e = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?;
        let f = d.matmul_naive(&e)?;
        let f1 = d.matmul(&e)?;
        assert_eq!(f.data(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(f.shape(), &[2, 2]);
        assert_eq!(f.data(), f1.data());
        assert_eq!(f.shape(), f1.shape());

        let g = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1])?;
        let h = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3])?;
        let i = g.matmul_naive(&h)?;
        let i1 = g.matmul(&h)?;
        assert_eq!(
            i.data(),
            &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]
        );
        assert_eq!(i.shape(), &[3, 3]);
        assert_eq!(i.data(), i1.data());
        assert_eq!(i.shape(), i1.shape());

        let j = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3])?;
        let k = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])?;
        let l = j.matmul_naive(&k)?;
        let l1 = j.matmul(&k)?;
        assert_eq!(l.data(), &[32.0]);
        assert_eq!(l.shape(), &[1, 1]);
        assert_eq!(l.data(), l1.data());
        assert_eq!(l.shape(), l1.shape());

        let m = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let n = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
        let o = m.matmul_naive(&n)?;
        let o1 = m.matmul(&n)?;
        assert_eq!(o.data(), &[32.0]);
        assert_eq!(o.shape(), &[1]);
        assert_eq!(o.data(), o1.data());
        assert_eq!(o.shape(), o1.shape());

        let p = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let q = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3])?;
        let r = q.matmul_naive(&p)?;
        let r1 = q.matmul(&p)?;
        assert_eq!(r.data(), &[32.0]);
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.data(), r1.data());
        assert_eq!(r.shape(), r1.shape());

        let s = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let t = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])?;
        let u = s.matmul_naive(&t)?;
        let u1 = s.matmul(&t)?;
        assert_eq!(u.data(), &[32.0]);
        assert_eq!(u.shape(), &[1]);
        assert_eq!(u.data(), u1.data());
        assert_eq!(u.shape(), u1.shape());

        Ok(())
    }

    fn setup_matrix_for_reduction() -> Tensor {
        let data = vec![
            1000.0, 2000.0, 3000.0, 1200.0, 1800.0, 2000.0, 1500.0, 2500.0, 2200.0,
        ];
        Tensor::new(data, vec![3, 3]).expect("Failed to create matrix for reduction tests")
    }

    #[test]
    fn test_reduce_sum_global() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(None).unwrap();

        assert_eq!(res.data(), &[17200.0]);
        assert_eq!(res.shape(), &[1]);
    }

    #[test]
    fn test_reduce_sum_axis_0_brand_total() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(Some(0)).unwrap();

        assert_eq!(res.data(), &[3700.0, 6300.0, 7200.0]);
        assert_eq!(res.shape(), &[3]);
    }

    #[test]
    fn test_reduce_sum_axis_1_monthly_total() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(Some(1)).unwrap();

        assert_eq!(res.data(), &[6000.0, 5000.0, 6200.0]);
        assert_eq!(res.shape(), &[3]);
    }

    #[test]
    fn test_reduce_sum_1d_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let res = tensor.sum(Some(0)).unwrap();

        assert_eq!(res.data(), &[6.0]);
        assert_eq!(res.shape(), &[1]);
    }

    #[test]
    fn test_reduce_sum_invalid_axis() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(Some(2));

        assert_eq!(res.err(), Some(TensorError::InvalidRank));
    }

    #[test]
    fn test_reduce_sum_empty() {
        let tensor = Tensor::new(vec![], vec![0]).unwrap();
        let res = tensor.sum(Some(2));

        assert_eq!(res.err(), Some(TensorError::InvalidRank));
    }
}
