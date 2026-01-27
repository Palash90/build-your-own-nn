use crate::{
    Layer, Rng,
    activation::{Activation, ActivationType},
    image_utils::{PlotColor, Trace, render_dual_plots, render_plot},
    linear::Linear,
    loss::bce_sigmoid_delta,
    tensor::{Tensor, TensorError},
};
use std::time::Duration;
use std::{fmt::format, thread};

pub fn xor_neural_network(rng: &mut dyn Rng, xnor: bool) -> Result<(), TensorError> {
    let mut l1 = Linear::new(3, 3, rng);
    let mut a1 = Activation::new(ActivationType::Sigmoid);

    let weight_init = match xnor {
        true => vec![8.578, 4.589, -2.254, -5.2, 0.5, -6.0, 0.98, 0.45, -3.21],
        false => vec![0.578, 8.589, 1.254, -2.2, 4.0, 02.0, 0.98, 0.45, -2.21],
    };

    l1.set_weight(Tensor::new(weight_init, vec![3, 3])?);

    let mut l2 = Linear::new(3, 1, rng);
    let mut a2 = Activation::new(ActivationType::Sigmoid);

    let input = Tensor::new(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![4, 3],
    )?;

    let actual = if xnor {
        Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1])?
    } else {
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![4, 1])?
    };

    let learning_rate = 0.5;
    let bounds = Some((0.0, 1.0, 0.0, 1.0));

    for epoch in 0..25000 {
        let z1 = l1.forward(&input)?;
        let h1 = a1.forward(&z1)?;
        let z2 = l2.forward(&h1)?;
        let pred = a2.forward(&z2)?;

        let d_z2 = bce_sigmoid_delta(&pred, &actual)?;
        let d_h1 = l2.backward(&d_z2, learning_rate)?;
        let d_z1 = a1.backward(&d_h1, learning_rate)?;
        let _ = l1.backward(&d_z1, learning_rate)?;

        if epoch % 100 == 0 {
            let mut traces = Vec::new();
            let mut cyan_x = Vec::new();
            let mut cyan_y = Vec::new();
            let mut magenta_x = Vec::new();
            let mut magenta_y = Vec::new();

            for gx in 0..=20 {
                for gy in 0..=20 {
                    let x = gx as f32 / 20.0;
                    let y = gy as f32 / 20.0;

                    let test_in = Tensor::new(vec![x, y, 1.0], vec![1, 3])?;
                    let p_z1 = l1.forward(&test_in)?;
                    let p_h1 = a1.forward(&p_z1)?;
                    let p_z2 = l2.forward(&p_h1)?;
                    let p_out = a2.forward(&p_z2)?;

                    if p_out.data()[0] > 0.5 {
                        cyan_x.push(x);
                        cyan_y.push(y);
                    } else {
                        magenta_x.push(x);
                        magenta_y.push(y);
                    }
                }
            }

            traces.push(Trace {
                name: format!("Epoch {}", epoch),
                x: vec![],
                y: vec![],
                color: PlotColor::Yellow,
                is_line: false,
                hide_axes: false,
            });
            traces.push(Trace {
                name: "Predict 1".into(),
                x: cyan_x,
                y: cyan_y,
                color: PlotColor::Cyan,
                is_line: false,
                hide_axes: false,
            });
            traces.push(Trace {
                name: "Predict 0".into(),
                x: magenta_x,
                y: magenta_y,
                color: PlotColor::Magenta,
                is_line: false,
                hide_axes: false,
            });

            let x_pts = [0.0, 0.0, 1.0, 1.0];
            let y_pts = [0.0, 1.0, 0.0, 1.0];
            for i in 0..4 {
                let color = if actual.data()[i] > 0.5 {
                    PlotColor::Green
                } else {
                    PlotColor::Red
                };
                traces.push(Trace {
                    name: format!("Point {}", i),
                    x: vec![x_pts[i]],
                    y: vec![y_pts[i]],
                    color,
                    is_line: false,
                    hide_axes: false,
                });
            }

            let topology_traces = visualize_topology(l1.weight(), l2.weight());

            render_dual_plots(
                &topology_traces,
                &traces,
                70,
                25,
                bounds,
                format!(
                    "{} Training - Epoch {}",
                    if xnor { "XNOR" } else { "XOR" },
                    epoch
                ),
            );

            let weight_display = format_weights_side_by_side(l1.weight(), l2.weight());
            println!("{}", weight_display);

            thread::sleep(Duration::from_millis(10));
        }
    }

    Ok(())
}

pub fn visualize_topology(l1_weights: &Tensor, l2_weights: &Tensor) -> Vec<Trace> {
    let mut traces = Vec::new();
    let layers = [3, 3, 1]; // Input (including bias), Hidden, Output
    let mut node_coords = Vec::new();

    // 1. Calculate and store coordinates first
    for (l_idx, &count) in layers.iter().enumerate() {
        let mut layer_nodes = Vec::new();
        let x = l_idx as f32 / (layers.len() - 1) as f32;
        for i in 0..count {
            let y = if count > 1 {
                i as f32 / (count - 1) as f32
            } else {
                0.5
            };
            layer_nodes.push((x, y));
        }
        node_coords.push(layer_nodes);
    }

    // 2. DRAW WEIGHTS FIRST (so nodes appear on top)
    let get_weight_type = |w: f32| {
        let abs_w = w.abs();
        if abs_w > 4.0 {
            "heavy"
        } else if abs_w > 1.5 {
            "medium"
        } else {
            "light"
        }
    };

    let w1 = l1_weights.data();
    for i in 0..3 {
        for j in 0..3 {
            let weight = w1[i * 3 + j];
            traces.push(Trace {
                name: get_weight_type(weight).into(),
                x: vec![node_coords[0][i].0, node_coords[1][j].0],
                y: vec![node_coords[0][i].1, node_coords[1][j].1],
                color: if weight > 0.0 {
                    PlotColor::Green
                } else {
                    PlotColor::Red
                },
                is_line: true,
                hide_axes: true,
            });
        }
    }

    let w2 = l2_weights.data();
    for i in 0..3 {
        let weight = w2[i];
        traces.push(Trace {
            name: get_weight_type(weight).into(),
            x: vec![node_coords[1][i].0, node_coords[2][0].0],
            y: vec![node_coords[1][i].1, node_coords[2][0].1],
            color: if weight > 0.0 {
                PlotColor::Cyan
            } else {
                PlotColor::Magenta
            },
            is_line: true,
            hide_axes: true,
        });
    }

    // 3. DRAW NODES LAST
    for (l_idx, layer) in node_coords.iter().enumerate() {
        for (i, &(x, y)) in layer.iter().enumerate() {
            let color = match l_idx {
                0 => PlotColor::Magenta, // Input Nodes
                1 => PlotColor::Blue,    // Hidden Nodes (Sigmoid)
                _ => PlotColor::White,   // Output Node
            };

            traces.push(Trace {
                name: format!("L{}I{}", l_idx, i),
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

/// Formats two tensors (matrices) side-by-side as strings for the terminal
pub fn format_weights_side_by_side(l1_weights: &Tensor, l2_weights: &Tensor) -> String {
    let mut output = String::new();

    // Get raw data and shapes
    let w1 = l1_weights.data();
    let s1 = l1_weights.shape(); // e.g., [3, 3]
    let w2 = l2_weights.data();
    let s2 = l2_weights.shape(); // e.g., [3, 1]

    let max_rows = s1[0].max(s2[0]);

    output.push_str("\n    L1 Weights (Input -> Hidden)         L2 Weights (Hidden -> Output)\n");
    output.push_str("    ----------------------------         -----------------------------\n");

    for r in 0..max_rows {
        // Format L1 Row
        let mut row_str = String::from("    ");
        if r < s1[0] {
            row_str.push_str("[ ");
            for c in 0..s1[1] {
                let val = w1[r * s1[1] + c];
                row_str.push_str(&format!("{:>7.3} ", val));
            }
            row_str.push_str("]");
        } else {
            row_str.push_str(&" ".repeat(s1[1] * 8 + 3));
        }

        // Spacer between matrices
        row_str.push_str("         ");

        // Format L2 Row
        if r < s2[0] {
            row_str.push_str("[ ");
            for c in 0..s2[1] {
                let val = w2[r * s2[1] + c];
                row_str.push_str(&format!("{:>7.3} ", val));
            }
            row_str.push_str("]");
        }

        output.push_str(&row_str);
        output.push('\n');
    }

    output
}
