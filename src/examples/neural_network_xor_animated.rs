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
                hide_axes:false
            });
            traces.push(Trace {
                name: "Predict 1".into(),
                x: cyan_x,
                y: cyan_y,
                color: PlotColor::Cyan,
                is_line: false,
                hide_axes:false
            });
            traces.push(Trace {
                name: "Predict 0".into(),
                x: magenta_x,
                y: magenta_y,
                color: PlotColor::Magenta,
                is_line: false,
                hide_axes:false
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
                    hide_axes:false
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
            thread::sleep(Duration::from_millis(10));
        }
    }

    Ok(())
}

pub fn visualize_topology(
    l1_weights: &Tensor, 
    l2_weights: &Tensor
) -> Vec<Trace> {
    let mut traces = Vec::new();
    let layers = [3, 3, 1];
    let mut node_coords = Vec::new();

    for (l_idx, &count) in layers.iter().enumerate() {
        let mut layer_nodes = Vec::new();
        let x = l_idx as f32 / (layers.len() - 1) as f32;
        for i in 0..count {
            let y = if count > 1 { i as f32 / (count - 1) as f32 } else { 0.5 };
            layer_nodes.push((x, y));
            
            // Logic for input point colors
            let color = if l_idx == 0 {
                if i < 2 { PlotColor::Green } else { PlotColor::White } // Features vs Bias
            } else if l_idx == 1 {
                PlotColor::Yellow // Hidden Layer
            } else {
                PlotColor::Cyan // Output Layer
            };

            let name = if l_idx == 0 && i < 2 { format!("Input {}", i) } 
                       else if l_idx == 0 { "Bias".into() } 
                       else { "".into() };

            traces.push(Trace {
                name,
                x: vec![x],
                y: vec![y],
                color,
                is_line: false,
                hide_axes:true
            });
        }
        node_coords.push(layer_nodes);
    }

    let get_weight_type = |w: f32| {
        let abs_w = w.abs();
        if abs_w > 4.0 { "heavy" } else if abs_w > 1.5 { "medium" } else { "light" }
    };

    // L1 -> L2 Weights
    let w1 = l1_weights.data();
    for i in 0..3 { 
        for j in 0..3 { 
            let weight = w1[i * 3 + j];
            traces.push(Trace {
                name: get_weight_type(weight).into(),
                x: vec![node_coords[0][i].0, node_coords[1][j].0],
                y: vec![node_coords[0][i].1, node_coords[1][j].1],
                color: if weight > 0.0 { PlotColor::Cyan } else { PlotColor::Magenta },
                is_line: true,
                hide_axes:true
            });
        }
    }

    // L2 -> Output Weights
    let w2 = l2_weights.data();
    for i in 0..3 { 
        let weight = w2[i];
        traces.push(Trace {
            name: get_weight_type(weight).into(),
            x: vec![node_coords[1][i].0, node_coords[2][0].0],
            y: vec![node_coords[1][i].1, node_coords[2][0].1],
            color: if weight > 0.0 { PlotColor::Cyan } else { PlotColor::Magenta },
            is_line: true,
            hide_axes:true
        });
    }
    traces
}