use build_your_own_nn::Rng;
use build_your_own_nn::examples::image_reconstructor;
use build_your_own_nn::examples::linear_regression;
use build_your_own_nn::examples::linear_regression_animated;
use build_your_own_nn::examples::neural_network_or;
use build_your_own_nn::examples::neural_network_or_animated;
use build_your_own_nn::examples::neural_network_xor;
use build_your_own_nn::examples::neural_network_xor_animated;
use build_your_own_nn::tensor::TensorError;
use std::env;
use std::io::{self, Write};

struct SimpleRng {
    state: u64,
}

impl Rng for SimpleRng {
    fn next_u32(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32 as i32
    }
}

fn get_user_choice(length: usize) -> usize {
    print!("Enter choice (1-{length}): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    input.trim().parse().unwrap_or(0)
}

fn run_user_io(rng: &mut dyn Rng) -> Result<(), TensorError> {
    let options = vec![
        "Simple Linear Regression (Trend Fitting)",
        "Animated Linear Regression",
        "Neural Network for OR Gate Approximation",
        "OR Gate (Decision Surface Slice)",
        "Neural Network for XOR Gate Approximation",
        "Animated XOR Decision Boundary",
        "Image Reconstructor",
        "Exit",
    ];

    loop {
        print!("\x1b[2J\x1b[H");
        println!("=== Neural Network Menu ===");
        options
            .iter()
            .enumerate()
            .for_each(|(index, op)| println!("{}. {}", index + 1, op));
        println!("==========================================");

        match get_user_choice(options.len()) {
            1 => {
                linear_regression::linear_regression(rng)?;
            }
            2 => {
                linear_regression_animated::linear_regression(rng)?;
            }
            3 => {
                neural_network_or::or_neural_network(rng)?;
            }
            4 => {
                neural_network_or_animated::or_neural_network(rng)?;
            }
            5 => {
                neural_network_xor::xor_neural_network(rng)?;
            }
            6 => {
                neural_network_xor_animated::xor_neural_network(rng)?;
            }
            7 => {
                image_reconstructor::reconstruct_image("assets/5_5.pbm", 20, rng);
            }
            8 => {
                println!("Goodbye!");
                break;
            }
            _ => {
                println!("Invalid choice, try again.");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }

        let mut pause = String::new();
        io::stdin().read_line(&mut pause).unwrap();
    }

    Ok(())
}

fn main() {
    let mut rng = SimpleRng { state: 73 };

    let args: Vec<String> = env::args().collect();
    let source = &args[1];

    match image_reconstructor::reconstruct_image(&source, 200, &mut rng) {
        Ok(_) => println!("Done"),
        err => println!("{:?}", err),
    }
}
