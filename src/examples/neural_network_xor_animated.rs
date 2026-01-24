use crate::{
    Layer, Rng,
    activation::{Activation, ActivationType},
    image_utils::{PlotColor, Trace, render_plot},
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
            });
            traces.push(Trace {
                name: "Predict 1".into(),
                x: cyan_x,
                y: cyan_y,
                color: PlotColor::Cyan,
                is_line: false,
            });
            traces.push(Trace {
                name: "Predict 0".into(),
                x: magenta_x,
                y: magenta_y,
                color: PlotColor::Magenta,
                is_line: false,
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
                });
            }

            render_plot(
                &traces,
                70,
                25,
                bounds,
                format!("{} Gate Approximation", if xnor { "XNOR" } else { "XOR" }),
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    Ok(())
}
