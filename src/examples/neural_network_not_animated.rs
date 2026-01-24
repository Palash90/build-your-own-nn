use std::{thread, time::Duration};

use crate::{Layer, Rng, activation::{Activation, ActivationType}, image_utils::{PlotColor, Trace, render_plot}, linear::Linear, loss::bce_sigmoid_delta, tensor::{Tensor, TensorError}};

pub fn not_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {
    // 2 inputs: (X-coordinate and Bias) -> 1 output
    let mut linear_layer = Linear::new(2, 1, rng);
    
    // Initial weights: a negative weight for w1 will help the NOT logic
    linear_layer.set_weight(Tensor::new(vec![-1.0, 5.0], vec![2, 1])?);
    let mut activation_layer = Activation::new(ActivationType::Sigmoid);

    // Input: [X, Bias]
    // Point 1: 5.0 (Low)  -> Should be 1.0 (True)
    // Point 2: 15.0 (High) -> Should be 0.0 (False)
    let input = Tensor::new(
        vec![
            5.0, 1.0, 
            15.0, 1.0
        ],
        vec![2, 2],
    )?;
    
    let actual = Tensor::new(vec![1.0, 0.0], vec![2, 1])?;

    let learning_rate = 0.02;
    let bounds = Some((0.0, 20.0, 0.0, 20.0));

    print!("\x1b[?25l"); // Hide cursor

    for epoch in 0..500 {
        let linear_output = linear_layer.forward(&input)?;
        let activation_output = activation_layer.forward(&linear_output)?;

        if epoch % 15 == 0 {
            print!("\x1b[2J\x1b[1;1H"); // Clear screen
            let mut traces = Vec::new();
            let w = linear_layer.weight().data();
            let w1 = w[0]; // Weight for X
            let b = w[1];  // Bias

            let mut cyan_x = Vec::new();
            let mut cyan_y = Vec::new();
            let mut magenta_x = Vec::new();
            let mut magenta_y = Vec::new();

            // Fill background based on X-axis only
            for gx in (0..=20).step_by(2) {
                for gy in (0..=20).step_by(2) {
                    let x = gx as f32;
                    let decision = w1 * x + b; // Decision logic
                    if decision > 0.0 {
                        cyan_x.push(x);
                        cyan_y.push(gy as f32);
                    } else {
                        magenta_x.push(x);
                        magenta_y.push(gy as f32);
                    }
                }
            }

            traces.push(Trace { name: "Predict 1".into(), x: cyan_x, y: cyan_y, color: PlotColor::Cyan, is_line: false });
            traces.push(Trace { name: "Predict 0".into(), x: magenta_x, y: magenta_y, color: PlotColor::Magenta, is_line: false });

            // Boundary Line: Since it's 1D, the boundary is a vertical line where w1*x + b = 0
            let boundary_x = -b / w1;
            if boundary_x >= 0.0 && boundary_x <= 20.0 {
                traces.push(Trace {
                    name: format!("Boundary (x={:.1})", boundary_x),
                    x: vec![boundary_x, boundary_x],
                    y: vec![0.0, 20.0],
                    color: PlotColor::Yellow,
                    is_line: true,
                });
            }

            // Target Points
            let x_coords = [5.0, 15.0];
            let targets = actual.data();
            for i in 0..2 {
                let color = if targets[i] > 0.5 { PlotColor::Green } else { PlotColor::Red };
                traces.push(Trace {
                    name: format!("P{}", i),
                    x: vec![x_coords[i]],
                    y: vec![10.0], // Center on Y axis
                    color,
                    is_line: false,
                });
            }

            render_plot(&traces, 70, 25, bounds, format!("NOT Gate (Epoch {})", epoch));
            thread::sleep(Duration::from_millis(40));
        }

        let delta = bce_sigmoid_delta(&activation_output, &actual)?;
        let _ = linear_layer.backward(&delta, learning_rate)?;
    }

    print!("\x1b[?25h"); // Show cursor
    Ok(())
}