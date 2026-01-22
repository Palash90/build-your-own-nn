use crate::{
    Rng,
    activation::{Activation, ActivationType},
    image_utils::{PlotColor, Trace, render_plot},
    linear::Linear,
    loss::bce_sigmoid_delta,
    tensor::{Tensor, TensorError},
};
use std::thread;
use std::time::Duration;

pub fn xor_neural_network(rng: &mut dyn Rng) -> Result<(), TensorError> {
    let mut l1 = Linear::new(3, 3, rng);
    let mut a1 = Activation::new(ActivationType::Sigmoid);
    
    let mut l2 = Linear::new(3, 1, rng);
    let mut a2 = Activation::new(ActivationType::Sigmoid);

    let input = Tensor::new(
        vec![
            0.0, 0.0, 1.0, 
            0.0, 1.0, 1.0, 
            1.0, 0.0, 1.0, 
            1.0, 1.0, 1.0,
        ],
        vec![4, 3],
    )?;
    
    let actual = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1])?;

    let learning_rate = 0.5; 
    let bounds = Some((0.0, 1.0, 0.0, 1.0));

    for epoch in 0..25000 {
        let z1 = l1.forward(&input)?;
        let h1 = a1.forward(&z1)?;
        let z2 = l2.forward(&h1)?;
        let pred = a2.forward(&z2)?;

        let d_z2 = bce_sigmoid_delta(&pred, &actual)?; 
        let d_h1 = l2.backward(&d_z2, learning_rate)?;
        let d_z1 = a1.backward(&d_h1)?;
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
                        cyan_x.push(x); cyan_y.push(y);
                    } else {
                        magenta_x.push(x); magenta_y.push(y);
                    }
                }
            }

            traces.push(Trace { name: "Predict 1".into(), x: cyan_x, y: cyan_y, color: PlotColor::Cyan, is_line: false });
            traces.push(Trace { name: "Predict 0".into(), x: magenta_x, y: magenta_y, color: PlotColor::Magenta, is_line: false });

            let x_pts = [0.0, 0.0, 1.0, 1.0];
            let y_pts = [0.0, 1.0, 0.0, 1.0];
            for i in 0..4 {
                let color = if actual.data()[i] > 0.5 { PlotColor::Green } else { PlotColor::Red };
                traces.push(Trace {
                    name: format!("Point {}", i),
                    x: vec![x_pts[i]],
                    y: vec![y_pts[i]],
                    color,
                    is_line: false,
                });
            }

            render_plot(&traces, 70, 25, bounds, String::from("XOR Gate Approximation"));
            thread::sleep(Duration::from_millis(15));
        }
    }

    Ok(())
}