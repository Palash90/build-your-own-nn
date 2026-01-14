use build_your_own_nn::tensor::Tensor;

#[cfg(test)]
#[test]
pub fn test_tensor_operations() {
    use std::vec;

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = a.add(&b);
    assert_eq!(c.data(), &[6.0, 8.0, 10.0, 12.0]);
    assert_eq!(c.shape(), &[2, 2]);

    let d = a.sub(&b);
    assert_eq!(d.data(), &[-4.0, -4.0, -4.0, -4.0]);
    assert_eq!(d.shape(), &[2, 2]);

    let e = a.mul(&b);
    assert_eq!(e.data(), &[5.0, 12.0, 21.0, 32.0]);
    assert_eq!(e.shape(), &[2, 2]);
}

#[test]
fn test_tensor_display_2d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let output = format!("{}", tensor);

    println!("{}", output);

    assert!(output.contains("|  1.0000,   2.0000|"));
    assert!(output.contains("|  3.0000,   4.0000|"));
}

#[test]
fn test_tensor_display_alignment() {
    let tensor = Tensor::new(vec![1.23456, 2.0, 100.1, 0.00001], vec![2, 2]);

    let output = format!("{}", tensor);

    assert!(output.contains("  1.2346"));
    assert!(output.contains("  0.0000"));
}

#[test]
fn test_tensor_display_1d() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

    let output = format!("{}", tensor);
    assert!(output.contains("[1.0, 2.0, 3.0]"));
}
