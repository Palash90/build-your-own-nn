use build_your_own_nn::Rng;
use build_your_own_nn::examples::image_reconstructor;
use build_your_own_nn::examples::linear_regression;
use build_your_own_nn::examples::linear_regression_animated;
use build_your_own_nn::examples::neural_network_not_animated;
use build_your_own_nn::examples::neural_network_logic;
use build_your_own_nn::examples::neural_network_logic::Gate;
use build_your_own_nn::examples::neural_network_logic_animated;
use build_your_own_nn::examples::neural_network_logic_animated::AnimatedGate;
use build_your_own_nn::examples::neural_network_xor;
use build_your_own_nn::examples::neural_network_xor_animated;
use build_your_own_nn::tensor::TensorError;
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
        "Simple Linear Regression",
        "Animated Linear Regression",
        "NOT Gate Approximation",
        "AND Gate Approximation",
        "OR Gate Approximation",
        "NAND Gate Approximation",
        "NOR Gate Approximation",
        "Animated AND Gate Decision Surface",
        "Animated OR Gate Decision Surface",
        "Animated NAND Gate Decision Surface",
        "Animated NOR Gate Decision Surface",
        "XOR Gate Approximation",
        "Animated XOR Decision Boundary",
        "Animated XNOR Decision Boundary",
        "Image Reconstructor",
        "Exit",
    ];

    loop {
        print!("\x1b[2J\x1b[H");
        println!("=== Neural Network Demonstrations ===");
        options
            .iter()
            .enumerate()
            .for_each(|(index, op)| println!("{}. {}", index + 1, op));
        println!("==========================================");

        match get_user_choice(options.len()) {
            1 => linear_regression::linear_regression(rng)?,
            2 => linear_regression_animated::linear_regression(rng)?,
            3 => neural_network_not_animated::not_neural_network(rng)?,

            // Static Binary Gates
            4 => neural_network_logic::demonstrate_logic(rng, Gate::AND)?,
            5 => neural_network_logic::demonstrate_logic(rng, Gate::OR)?,
            6 => neural_network_logic::demonstrate_logic(rng, Gate::NAND)?,
            7 => neural_network_logic::demonstrate_logic(rng, Gate::NOR)?,

            // Animated Gate
            8 => neural_network_logic_animated::demonstrate_logic(rng, AnimatedGate::AND)?,
            9 => neural_network_logic_animated::demonstrate_logic(rng, AnimatedGate::OR)?,
            10 => neural_network_logic_animated::demonstrate_logic(rng, AnimatedGate::NAND)?,
            11 => neural_network_logic_animated::demonstrate_logic(rng, AnimatedGate::NOR)?,

            // XOR Logic (Requires hidden layers)
            12 => neural_network_xor::xor_neural_network(rng)?,
            13 => neural_network_xor_animated::xor_neural_network(rng, false)?,
            14 => neural_network_xor_animated::xor_neural_network(rng, true)?,

            15 => match image_reconstructor::reconstruct_image("assets/spiral_25.pbm", 150, rng) {
                Ok(_) => println!("Done"),
                Err(err) => println!("Error: {:?}", err),
            },
            16 | _ => {
                println!("Goodbye!");
                break;
            }
        }

        print!("\x1b[?25h");
        println!("\nDemonstration Completed! Press Enter to continue...");
        let mut pause = String::new();
        io::stdin().read_line(&mut pause).unwrap();
    }
    Ok(())
}

fn main() {
    let mut rng = SimpleRng { state: 73 };
    run_user_io(&mut rng);
}
