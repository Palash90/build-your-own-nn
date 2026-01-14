use build_your_own_nn::tensor::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    )?;

    let c = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4])?;

    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
    Ok(())
}