use std::fs::File;
use std::io::{Write, BufWriter};

pub fn draw_image() -> std::io::Result<()> {
    let size: i32 = 200;
    let file = File::create("spiral.pbm")?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "P1")?;
    writeln!(writer, "{} {}", size, size)?;

    for y in 0..size {
        let mut row = String::with_capacity(size as usize * 2);
        for x in 0..size {
            // Normalize coordinates to be between -1.0 and 1.0
            let nx = (x as f32 / size as f32) * 2.0 - 1.0;
            let ny = (y as f32 / size as f32) * 2.0 - 1.0;

            // Convert to Polar Coordinates (r, theta)
            let r = (nx * nx + ny * ny).sqrt();
            let theta = ny.atan2(nx);

            // The Spiral Logic: 
            // We check if the angle matches the radius in a way that creates a "wrap"
            // Adding 'r * 10.0' creates the tight turns.
            let spiral_val = (theta + r * 10.0).sin();

            if r < 0.9 && spiral_val > 0.0 {
                row.push_str("1 "); // Part of the spiral arm
            } else {
                row.push_str("0 "); // Empty space
            }
        }
        writeln!(writer, "{}", row)?;
    }
    println!("Spiral generated: spiral.pbm");
    Ok(())
}