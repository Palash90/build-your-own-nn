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
    // 1. Architecture: 3 -> 9 -> 6 -> 1
    let mut l1 = Linear::new(3, 9, rng);
    let mut a1 = Activation::new(ActivationType::Sigmoid);
    let mut l2 = Linear::new(9, 6, rng);
    let mut a2 = Activation::new(ActivationType::Sigmoid);
    let mut l3 = Linear::new(6, 1, rng);
    let mut a3 = Activation::new(ActivationType::Sigmoid);

    let (input, actual) = generate_two_moons(100);
    let learning_rate = 0.08;
    let bounds = Some((-1.5, 2.5, -1.0, 1.5));

    for epoch in 0..100_000 {
        // Forward Pass
        let h1 = a1.forward(&l1.forward(&input)?)?;
        let h2 = a2.forward(&l2.forward(&h1)?)?;
        let pred = a3.forward(&l3.forward(&h2)?)?;

        // Backward Pass
        let d_z3 = bce_sigmoid_delta(&pred, &actual)?;
        let d_h2 = l3.backward(&d_z3, learning_rate)?;
        let d_z2 = a2.backward(&d_h2, learning_rate)?;
        let d_h1 = l2.backward(&d_z2, learning_rate)?;
        let d_z1 = a1.backward(&d_h1, learning_rate)?;
        let _ = l1.backward(&d_z1, learning_rate)?;

        if epoch % 500 == 0 {
            let mut traces = Vec::new();
            let (mut cx, mut cy, mut mx, mut my) = (vec![], vec![], vec![], vec![]);

            // 2. Heatmap Generation
            for gx in 0..=30 {
                for gy in 0..=20 {
                    let x = -1.5 + (gx as f32 / 30.0) * 4.0;
                    let y = -1.0 + (gy as f32 / 20.0) * 2.5;
                    let test_in = Tensor::new(vec![x, y, 1.0], vec![1, 3])?;
                    let p = a3.forward(&l3.forward(&a2.forward(&l2.forward(&a1.forward(&l1.forward(&test_in)?)?)?)?)?)?;

                    if p.data()[0] > 0.5 { cx.push(x); cy.push(y); } 
                    else { mx.push(x); my.push(y); }
                }
            }

            // Heatmap Traces
            traces.push(Trace { name: "Class 1 Area".into(), x: cx, y: cy, color: PlotColor::Cyan, is_line: false, hide_axes: false });
            traces.push(Trace { name: "Class 0 Area".into(), x: mx, y: my, color: PlotColor::Magenta, is_line: false, hide_axes: false });

            // 3. Data Points
            for i in 0..actual.data().len() {
                let color = if actual.data()[i] > 0.5 { PlotColor::Green } else { PlotColor::Red };
                traces.push(Trace {
                    name: "".into(), x: vec![input.data()[i * 3]], y: vec![input.data()[i * 3 + 1]],
                    color, is_line: false, hide_axes: false,
                });
            }

            // Render with dynamic heights
            render_dual_plots(
                &visualize_topology_dynamic(l1.weight(), l2.weight(), l3.weight(), -1.0, 1.5),
                &traces, 100, 30, bounds,
                format!("Two Moons Training - Epoch {}", epoch),
            );

            println!("{}", format_3_layer_weights(l1.weight(), l2.weight(), l3.weight()));
            thread::sleep(Duration::from_millis(50));
        }
    }
    Ok(())
}


pub fn visualize_topology_dynamic(l1: &Tensor, l2: &Tensor, l3: &Tensor, y_min: f32, y_max: f32) -> Vec<Trace> {
    let mut traces = Vec::new();
    let layer_sizes = [l1.shape()[0], l1.shape()[1], l2.shape()[1], l3.shape()[1]];
    
    // Define unique heights for each layer (0.0 to 1.0 scale of total available height)
    let layer_height_scales = [0.4, 1.0, 0.7, 0.2]; 
    let center_y = (y_min + y_max) / 2.0;
    let total_h = y_max - y_min;

    let mut node_coords = Vec::new();

    // 1. Calculate Coordinates with Variable Heights
    for (l_idx, &count) in layer_sizes.iter().enumerate() {
        let mut layer_nodes = Vec::new();
        let x = -1.0 + (l_idx as f32 / (layer_sizes.len() - 1) as f32) * 3.0;
        
        let current_h = total_h * layer_height_scales[l_idx];
        let start_y = center_y - (current_h / 2.0);

        for i in 0..count {
            let y = if count > 1 {
                start_y + (i as f32 / (count - 1) as f32) * current_h
            } else {
                center_y
            };
            layer_nodes.push((x, y));
        }
        node_coords.push(layer_nodes);
    }

    // 2. Weighted Connections (Lines)
    let weights = [l1, l2, l3];
    for (l_idx, w_tensor) in weights.iter().enumerate() {
        let (rows, cols) = (w_tensor.shape()[0], w_tensor.shape()[1]);
        let data = w_tensor.data();
        for i in 0..rows {
            for j in 0..cols {
                let val = data[i * cols + j];
                traces.push(Trace {
                    name: "".into(),
                    x: vec![node_coords[l_idx][i].0, node_coords[l_idx+1][j].0],
                    y: vec![node_coords[l_idx][i].1, node_coords[l_idx+1][j].1],
                    color: if val > 0.0 { PlotColor::Green } else { PlotColor::Red },
                    is_line: true, hide_axes: true,
                });
            }
        }
    }

    // 3. Nodes with Layer Colors
    for (l_idx, layer) in node_coords.iter().enumerate() {
        let color = match l_idx {
            0 => PlotColor::Magenta,
            1 => PlotColor::Blue,
            2 => PlotColor::Cyan,
            _ => PlotColor::White,
        };
        for &(x, y) in layer {
            traces.push(Trace {
                name: "".into(), x: vec![x], y: vec![y],
                color, is_line: false, hide_axes: true,
            });
        }
    }
    traces
}

pub fn format_3_layer_weights(l1: &Tensor, l2: &Tensor, l3: &Tensor) -> String {
    let mut out = String::from("\n      L1 (3->9)                                                               L2 (9->6)                                  L3 (6->1)\n");
    let max_rows = l1.shape()[0].max(l2.shape()[0]).max(l3.shape()[0]);
    let (g, r, res) = ("\x1b[32m", "\x1b[31m", "\x1b[0m");

    for row_idx in 0..max_rows {
        let mut line = String::from("  ");
        for w in [l1, l2, l3] {
            if row_idx < w.shape()[0] {
                line.push('[');
                for col_idx in 0..w.shape()[1] {
                    let val = w.data()[row_idx * w.shape()[1] + col_idx];
                    line.push_str(&format!("{}{:>6.2}{} ", if val >= 0.0 {g} else {r}, val, res));
                }
                line.push(']');
            } else {
                // Keep spacing consistent for missing rows
                line.push_str(&" ".repeat(w.shape()[1] * 7 + 2));
            }
            line.push_str("    ");
        }
        out.push_str(&line); out.push('\n');
    }
    out
}









