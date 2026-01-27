use crate::{
    Layer, Rng,
    activation::{Activation, ActivationType},
    image_utils::{PlotColor, Trace, render_dual_plots},
    linear::Linear,
    loss::bce_sigmoid_delta,
    tensor::{Tensor, TensorError},
};
use std::{thread, time::Duration};

/// Generates the Two Moons dataset
pub fn generate_two_moons(samples: usize) -> (Tensor, Tensor) {
    let mut inputs = Vec::with_capacity(samples * 3); // x, y, bias
    let mut targets = Vec::with_capacity(samples);

    for i in 0..samples {
        let n = i as f32 / (samples as f32 / 2.0);
        if i < samples / 2 {
            // Top Moon
            let ra = n * std::f32::consts::PI;
            let x = ra.cos();
            let y = ra.sin();
            inputs.extend_from_slice(&[x, y, 1.0]);
            targets.push(0.0);
        } else {
            // Bottom Moon
            let ra = (n - 1.0) * std::f32::consts::PI;
            let x = 1.0 - ra.cos();
            let y = 0.5 - ra.sin();
            inputs.extend_from_slice(&[x, y, 1.0]);
            targets.push(1.0);
        }
    }

    // Convert to Tensors (using your existing Tensor implementation)
    let x_tensor = Tensor::new(inputs, vec![samples, 3]).unwrap();
    let y_tensor = Tensor::new(targets, vec![samples, 1]).unwrap();
    (x_tensor, y_tensor)
}

pub fn two_moons_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {
    // 1. Setup Architecture: 3 Inputs (x, y, bias) -> 9 Hidden -> 1 Output
    let mut l1 = Linear::new(3, 9, rng);
    let mut a1 = Activation::new(ActivationType::Sigmoid);
    let mut l2 = Linear::new(9, 1, rng);
    let mut a2 = Activation::new(ActivationType::Sigmoid);

    // 2. Generate Data
    let (input, actual) = generate_two_moons(100);
    let learning_rate = 0.08;

    // Bounds adjusted for Two Moons coordinates
    let bounds = Some((-1.5, 2.5, -1.0, 1.5));

    for epoch in 0..100_000 {
        // Forward & Backward pass
        let z1 = l1.forward(&input)?;
        let h1 = a1.forward(&z1)?;
        let z2 = l2.forward(&h1)?;
        let pred = a2.forward(&z2)?;

        let d_z2 = bce_sigmoid_delta(&pred, &actual)?;
        let d_h1 = l2.backward(&d_z2, learning_rate)?;
        let d_z1 = a1.backward(&d_h1, learning_rate)?;
        let _ = l1.backward(&d_z1, learning_rate)?;

        if epoch % 500 == 0 {
            let mut traces = Vec::new();
            let (mut cx, mut cy, mut mx, mut my) = (vec![], vec![], vec![], vec![]);

            // 3. Generate Decision Boundary "Heatmap"
            for gx in 0..=30 {
                for gy in 0..=20 {
                    let x = -1.5 + (gx as f32 / 30.0) * 4.0;
                    let y = -1.0 + (gy as f32 / 20.0) * 2.5;

                    let test_in = Tensor::new(vec![x, y, 1.0], vec![1, 3])?;
                    let p_out = a2.forward(&l2.forward(&a1.forward(&l1.forward(&test_in)?)?)?)?;

                    if p_out.data()[0] > 0.5 {
                        cx.push(x);
                        cy.push(y);
                    } else {
                        mx.push(x);
                        my.push(y);
                    }
                }
            }

            // Add Heatmap Traces
            traces.push(Trace {
                name: "Class 1 Area".into(),
                x: cx,
                y: cy,
                color: PlotColor::Cyan,
                is_line: false,
                hide_axes: false,
            });
            traces.push(Trace {
                name: "Class 0 Area".into(),
                x: mx,
                y: my,
                color: PlotColor::Magenta,
                is_line: false,
                hide_axes: false,
            });

            // 4. Add Actual Data Points
            for i in 0..actual.data().len() {
                let color = if actual.data()[i] > 0.5 {
                    PlotColor::Green
                } else {
                    PlotColor::Red
                };
                traces.push(Trace {
                    name: "".into(),
                    x: vec![input.data()[i * 3]],
                    y: vec![input.data()[i * 3 + 1]],
                    color,
                    is_line: false,
                    hide_axes: false,
                });
            }

            render_dual_plots(
                &visualize_topology(l1.weight(), l2.weight(), -1.0, 1.5), // Note: Update visualize_topology for new layer sizes!
                &traces,
                100,
                30,
                bounds,
                format!("Two Moons Training - Epoch {}", epoch),
            );

            let weight_display = format_weights_side_by_side(l1.weight(), l2.weight());
            println!("{}", weight_display);

            thread::sleep(Duration::from_millis(50));
        }
    }
    Ok(())
}

pub fn visualize_topology(
    l1_weights: &Tensor, 
    l2_weights: &Tensor, 
    y_min: f32, 
    y_max: f32
) -> Vec<Trace> {
    let mut traces = Vec::new();

    let input_count = l1_weights.shape()[0];
    let hidden_count = l1_weights.shape()[1];
    let output_count = l2_weights.shape()[1];
    let layers = [input_count, hidden_count, output_count];

    let mut node_coords = Vec::new();
    
    let height = y_max - y_min;
    let padding = height * 0.1;
    let effective_min = y_min + padding;
    let effective_max = y_max - padding;
    let effective_height = effective_max - effective_min;

    for (l_idx, &count) in layers.iter().enumerate() {
        let mut layer_nodes = Vec::new();
        let x = -1.0 + (l_idx as f32 / (layers.len() - 1) as f32) * 3.0; 
        
        for i in 0..count {
            let y = if count > 1 {
                if l_idx == 0 {
                    // SHRINK LOGIC FOR INPUT LAYER
                    // We take the center point and spread nodes by only 40% of the total height
                    let center = y_min + (height / 2.0);
                    let shrink_factor = 0.4; 
                    let start = center - (effective_height * shrink_factor / 2.0);
                    start + (i as f32 / (count - 1) as f32) * (effective_height * shrink_factor)
                } else {
                    // FULL SPREAD FOR HIDDEN LAYER
                    effective_min + (i as f32 / (count - 1) as f32) * effective_height
                }
            } else {
                y_min + (height / 2.0) // Output layer
            };
            layer_nodes.push((x, y));
        }
        node_coords.push(layer_nodes);
    }

    // Drawing logic for lines (Input -> Hidden)
    let w1 = l1_weights.data();
    for i in 0..input_count {
        for j in 0..hidden_count {
            let weight = w1[i * hidden_count + j];
            traces.push(Trace {
                name: "".into(),
                x: vec![node_coords[0][i].0, node_coords[1][j].0],
                y: vec![node_coords[0][i].1, node_coords[1][j].1],
                color: if weight > 0.0 { PlotColor::Green } else { PlotColor::Red },
                is_line: true,
                hide_axes: true,
            });
        }
    }

    // Drawing logic for lines (Hidden -> Output)
    let w2 = l2_weights.data();
    for i in 0..hidden_count {
        for j in 0..output_count {
            let weight = w2[i * output_count + j];
            traces.push(Trace {
                name: "".into(),
                x: vec![node_coords[1][i].0, node_coords[2][j].0],
                y: vec![node_coords[1][i].1, node_coords[2][j].1],
                color: if weight > 0.0 { PlotColor::Cyan } else { PlotColor::Magenta },
                is_line: true,
                hide_axes: true,
            });
        }
    }

    // Drawing Nodes
    for (l_idx, layer) in node_coords.iter().enumerate() {
        for (i, &(x, y)) in layer.iter().enumerate() {
            let color = match l_idx {
                0 => PlotColor::Magenta, 
                1 => PlotColor::Blue,    
                _ => PlotColor::White,   
            };
            traces.push(Trace {
                name: format!("L{}N{}", l_idx, i),
                x: vec![x],
                y: vec![y],
                color,
                is_line: false,
                hide_axes: true,
            });
        }
    }
    traces
}


pub fn format_weights_side_by_side(l1_weights: &Tensor, l2_weights: &Tensor) -> String {
    let mut output = String::new();

    let w1 = l1_weights.data();
    let s1 = l1_weights.shape();
    let w2 = l2_weights.data();
    let s2 = l2_weights.shape();

    let max_rows = s1[0].max(s2[0]);

    // Import or use the ANSI codes from your PlotColor enum
    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let reset = "\x1b[0m";

    output.push_str("\n    L1 Weights (Input -> Hidden)         L2 Weights (Hidden -> Output)\n");
    output.push_str("    ----------------------------         -----------------------------\n");

    for r in 0..max_rows {
        let mut row_str = String::from("    ");

        // Format L1 Row
        if r < s1[0] {
            row_str.push_str("[ ");
            for c in 0..s1[1] {
                let val = w1[r * s1[1] + c];
                let color = if val >= 0.0 { green } else { red };
                row_str.push_str(&format!("{}{:>7.3}{} ", color, val, reset));
            }
            row_str.push_str("]");
        } else {
            // Adjust spacing for ANSI codes (which don't take up visual space)
            row_str.push_str(&" ".repeat(s1[1] * 8 + 3));
        }

        row_str.push_str("         ");

        // Format L2 Row
        if r < s2[0] {
            row_str.push_str("[ ");
            for c in 0..s2[1] {
                let val = w2[r * s2[1] + c];
                let color = if val >= 0.0 { green } else { red };
                row_str.push_str(&format!("{}{:>7.3}{} ", color, val, reset));
            }
            row_str.push_str("]");
        }

        output.push_str(&row_str);
        output.push('\n');
    }

    output
}
