use std::{thread, time::Duration};

use crate::{Layer, Rng, image_utils::{PlotColor, Trace, render_plot}, linear::Linear, loss::{mse_loss, mse_loss_gradient}, tensor::{Tensor, TensorError}};

pub fn linear_regression(rng: &mut dyn Rng) -> Result<(), TensorError> {
    let mut linear = Linear::new(2, 1, rng);

    let far_weights = Tensor::new(vec![-2.0, 45.0], vec![2, 1])?; 
    linear.set_weight(far_weights); 

    let num_points = 40;
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    let mut input_vec = Vec::new();

    for i in 0..num_points {
        let x = 2.0 + (i as f32 * 0.4); 
        let noise = (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 10.0;
        let y = 1.8 * x + 10.0 + noise;

        x_train.push(x);
        y_train.push(y);
        input_vec.push(x);
        input_vec.push(1.0);
    }
    let input = Tensor::new(input_vec, vec![num_points, 2])?;
    let actual = Tensor::new(y_train.clone(), vec![num_points, 1])?;

    let mut x_line = Vec::new();
    let mut line_input_vec = Vec::new();
    for i in 0..50 {
        let x = i as f32 * 0.408;
        x_line.push(x);
        line_input_vec.push(x);
        line_input_vec.push(1.0);
    }
    let line_input = Tensor::new(line_input_vec, vec![50, 2])?;

    let epochs = 10_000;
    let bounds = Some((0.0, 20.0, 0.0, 50.0)); 
    for epoch in 0..epochs {
        let predicted = linear.forward(&input)?;
        let loss_val = mse_loss(&predicted, &actual)?.data()[0];

        let grad = mse_loss_gradient(&predicted, &actual)?;
        linear.backward(&grad, 0.0005)?;

        if epoch % 10 == 0 {
            let line_pred = linear.forward(&line_input)?;

            let trace_actual = Trace {
                name: "Actual Data".to_string(),
                x: x_train.clone(),
                y: y_train.clone(),
                color: PlotColor::Blue,
                is_line: false,
            };

            let trace_pred = Trace {
                name: format!("Prediction at Epoch {} | Loss: {:.2}", epoch, loss_val),
                x: x_line.clone(),
                y: line_pred.data().to_vec(),
                color: PlotColor::Red,
                is_line: true,
            };

            render_plot(&[trace_actual, trace_pred], 100, 35, bounds, String::from("Linear Regression"));
            thread::sleep(Duration::from_millis(3));
        }
        print!("\x1b[2J\x1b[1;1H");
    }

    Ok(())
}