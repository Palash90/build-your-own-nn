use crate::{
    Rng,
    activation::{Activation, ActivationType},
    image_util::{PlotColor, Trace, render_plot},
    linear::Linear,
    loss::bce_sigmoid_delta,
    tensor::{Tensor, TensorError},
};
use std::thread;
use std::time::Duration;

pub fn or_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {
    let mut linear_layer = Linear::new(3, 1, rng);
    linear_layer.set_weight(Tensor::new(vec![1.0, 20.0, 2.0], vec![3, 1])?);
    let mut activation_layer = Activation::new(ActivationType::Sigmoid);

    let input = Tensor::new(
        vec![
            5.0, 5.0, 1.0, 5.0, 15.0, 1.0, 15.0, 5.0, 1.0, 15.0, 15.0, 1.0,
        ],
        vec![4, 3],
    )?;
    let actual = Tensor::new(vec![0.0, 1.0, 1.0, 1.0], vec![4, 1])?;

    let learning_rate = 0.015;
    let bounds = Some((0.0, 20.0, 0.0, 20.0));

    for epoch in 0..5000 {
        let linear_output = linear_layer.forward(&input)?;
        let activation_output = activation_layer.forward(&linear_output)?;

        if epoch % 15 == 0 {
            let mut traces = Vec::new();
            let w = linear_layer.weight().data();
            let w1 = w[0];
            let w2 = w[1];
            let b = w[2];

            let mut cyan_x = Vec::new();
            let mut cyan_y = Vec::new();
            let mut magenta_x = Vec::new();
            let mut magenta_y = Vec::new();

            for gx in (0..=20).step_by(2) {
                for gy in (0..=20).step_by(2) {
                    let x = gx as f32;
                    let y = gy as f32;

                    let decision = w1 * x + w2 * y + b;
                    if decision > 0.0 {
                        cyan_x.push(x);
                        cyan_y.push(y);
                    } else {
                        magenta_x.push(x);
                        magenta_y.push(y);
                    }
                }
            }

            traces.push(Trace {
                name: "Predict 1".to_string(),
                x: cyan_x,
                y: cyan_y,
                color: PlotColor::Cyan,
                is_line: false,
            });

            traces.push(Trace {
                name: "Predict 0".to_string(),
                x: magenta_x,
                y: magenta_y,
                color: PlotColor::Magenta,
                is_line: false,
            });

            let mut x_line = Vec::new();
            let mut y_line = Vec::new();
            for i in 0..=20 {
                let x = i as f32;
                let y = -(w1 * x + b) / w2;
                if y >= -2.0 && y <= 22.0 {
                    x_line.push(x);
                    y_line.push(y);
                }
            }
            traces.push(Trace {
                name: format!("Boundary (Epoch {})", epoch),
                x: x_line,
                y: y_line,
                color: PlotColor::Yellow,
                is_line: true,
            });

            let x_coords = [5.0, 5.0, 15.0, 15.0];
            let y_coords = [5.0, 15.0, 5.0, 15.0];
            let targets = actual.data();

            for i in 0..4 {
                let goal_color = if targets[i] > 0.5 {
                    PlotColor::Green
                } else {
                    PlotColor::Red
                };
                traces.push(Trace {
                    name: format!("P{}", i),
                    x: vec![x_coords[i]],
                    y: vec![y_coords[i]],
                    color: goal_color,
                    is_line: false,
                });
            }

            render_plot(&traces, 70, 25, bounds);
            thread::sleep(Duration::from_millis(40));
        }

        let delta = bce_sigmoid_delta(&activation_output, &actual)?;
        let _ = linear_layer.backward(&delta, learning_rate)?;
    }
    Ok(())
}
