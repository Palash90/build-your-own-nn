use build_your_own_nn::Rng;
use build_your_own_nn::linear::Linear;
use build_your_own_nn::loss::{l1_loss, mse_loss};
use build_your_own_nn::tensor::{Tensor, TensorError};

struct SimpleRng {
    state: u64,
}

impl Rng for SimpleRng {
    fn next_u32(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32 as i32
    }
}

fn main() -> Result<(), TensorError> {
    let mut rng = SimpleRng { state: 73 };

    let linear = Linear::new(2, 1, &mut rng);

    println!("Weights:");
    println!("{}", linear.weight());

    let input = Tensor::new(vec![1.0, 1.0_f32, 2.0, 1.0_f32, 3.0, 1.0_f32, 4.0, 1.0_f32, 5.0, 1.0_f32], vec![5, 2])?;

    println!("Input:");
    println!("{}", input);

    let output = linear.forward(&input).unwrap();
    println!("Output:");
    println!("{}", output);


    let actual = Tensor::new(vec![5.6, 6.6, 9.5, 10.2, 14.0], vec![5, 1])?;

    println!("Actual:");
    println!("{}", actual);

    let l1_loss = l1_loss(&output, &actual)?;
    let mse_loss = mse_loss(&output, &actual)?;

    println!("L1 Loss:");
    println!("{}", l1_loss);

    println!("MSE Loss:");
    println!("{}", mse_loss);

    Ok(())
}
