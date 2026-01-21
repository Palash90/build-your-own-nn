use std::f32;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

pub fn draw_pbm(source: &str) {
    let file = File::open(source).expect("File not found");
    let reader = BufReader::new(file);

    // 1. Parse all tokens (skipping comments)
    let tokens: Vec<String> = reader
        .lines()
        .flatten()
        .filter(|l| !l.trim().starts_with('#'))
        .flat_map(|l| l.split_whitespace().map(String::from).collect::<Vec<_>>())
        .collect();

    if tokens.len() < 3 {
        return;
    }

    // 2. Extract dimensions from header
    let w: usize = tokens[1].parse().unwrap();
    let h: usize = tokens[2].parse().unwrap();
    let data = &tokens[3..];

    // 3. Render using Braille 2x4 blocks
    for y in (0..h).step_by(4) {
        let mut row = String::new();
        for x in (0..w).step_by(2) {
            let mut byte = 0u8;
            // The Braille dot mapping (standard bit positions)
            let dots = [
                (0, 0, 0x01),
                (0, 1, 0x02),
                (0, 2, 0x04),
                (1, 0, 0x08),
                (1, 1, 0x10),
                (1, 2, 0x20),
                (0, 3, 0x40),
                (1, 3, 0x80),
            ];

            for (dx, dy, mask) in dots {
                let (px, py) = (x + dx, y + dy);
                // Bounds check ensures any resolution works
                if px < w && py < h {
                    if data.get(py * w + px).map_or(false, |v| v == "1") {
                        byte |= mask;
                    }
                }
            }
            row.push(std::char::from_u32(0x2800 + byte as u32).unwrap());
        }
        println!("{}", row);
    }
}

pub struct Trace {
    pub name: String,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub color: PlotColor,
    pub is_line: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PlotColor {
    Red,
    Blue,
    Green,
    Cyan,
    Magenta,
    Yellow,
    White,
    Reset,
}

impl PlotColor {
    pub fn to_ansi(&self) -> &'static str {
        match self {
            PlotColor::Red     => "\x1b[31m",
            PlotColor::Blue    => "\x1b[34m",
            PlotColor::Green   => "\x1b[32m",
            PlotColor::Cyan    => "\x1b[36m",
            PlotColor::Magenta => "\x1b[35m",
            PlotColor::Yellow  => "\x1b[33m",
            PlotColor::White   => "\x1b[37m",
            PlotColor::Reset   => "\x1b[0m",
        }
    }
}

pub fn render_plot(
    traces: &[Trace], 
    width: usize, 
    height: usize, 
    fixed_bounds: Option<(f32, f32, f32, f32)>
) {
    // Determine bounds: use provided fixed bounds or calculate from data
    let (min_x, max_x, min_y, max_y) = match fixed_bounds {
        Some(bounds) => bounds,
        None => get_bounds(traces),
    };

    // 1. DYNAMIC MARGINS & CONFIG
    let margin_l = 10;
    let margin_b = 2;
    let plot_w = width - margin_l - 2;
    let plot_h = height - margin_b - 2;
    
    let y_tick_count = 5; 
    let x_tick_count = 4;

    let mut grid = vec![vec![" ".to_string(); width]; height];

    // 2. DRAW Y-AXIS AND INTERMEDIATE TICKS
    for i in 0..=y_tick_count {
        let t = i as f32 / y_tick_count as f32;
        // Map 0.0-1.0 to the vertical plot area (inverted for terminal rows)
        let py = map_val(t, 0.0, 1.0, plot_h as f32, 0.0) as usize;
        let val = map_val(t, 0.0, 1.0, min_y, max_y);

        // Draw the tick mark on the axis
        grid[py][margin_l] = "┼".to_string();

        // Format and place the Y label
        let label = format!("{:>9.1}", val);
        for (idx, c) in label.chars().enumerate() {
            if idx < margin_l {
                grid[py][idx] = c.to_string();
            }
        }
    }

    // 3. DRAW X-AXIS AND INTERMEDIATE TICKS
    for i in 0..=x_tick_count {
        let t = i as f32 / x_tick_count as f32;
        // Map 0.0-1.0 to the horizontal plot area
        let px = map_val(t, 0.0, 1.0, 0.0, plot_w as f32) as usize + margin_l + 1;
        let val = map_val(t, 0.0, 1.0, min_x, max_x);

        if px < width {
            // Draw the tick mark on the axis
            grid[plot_h][px] = "┴".to_string();

            // Format and place the X label below the axis
            let label = format!("{:.1}", val);
            for (idx, c) in label.chars().enumerate() {
                if px + idx < width {
                    grid[plot_h + 1][px + idx] = c.to_string();
                }
            }
        }
    }

    // 4. FILL REMAINING BORDERS
    for y in 0..plot_h {
        if grid[y][margin_l] == " " { grid[y][margin_l] = "│".to_string(); }
    }
    for x in margin_l + 1..width {
        if grid[plot_h][x] == " " { grid[plot_h][x] = "─".to_string(); }
    }
    grid[plot_h][margin_l] = "└".to_string();

    // 5. PLOT DATA
    for trace in traces {
        let color_code = trace.color.to_ansi(); 
        for i in 0..trace.x.len() {
            let px = map_val(trace.x[i], min_x, max_x, 0.0, plot_w as f32) as usize + margin_l + 1;
            let py = map_val(trace.y[i], min_y, max_y, plot_h as f32 - 1.0, 0.0) as usize;

            // Only draw if within the plot frame
            if py < plot_h && px > margin_l && px < width {
                if trace.is_line && i > 0 {
                    let prev_px = map_val(trace.x[i - 1], min_x, max_x, 0.0, plot_w as f32) as usize + margin_l + 1;
                    let prev_py = map_val(trace.y[i - 1], min_y, max_y, plot_h as f32 - 1.0, 0.0) as usize;
                    draw_line(&mut grid, prev_px, prev_py, px, py, color_code);
                }
                grid[py][px] = format!("{}●\x1b[0m", color_code);
            }
        }
    }

    // 6. ATOMIC BUFFER PRINT
    let mut buffer = String::new();
    // \x1b[2J = clear screen, \x1b[H = cursor top-left, \x1b[?25l = hide cursor
    buffer.push_str("\x1b[2J\x1b[H\x1b[?25l");
    for row in grid {
        buffer.push_str(&row.concat());
        buffer.push('\n');
    }

    // Legend
    buffer.push('\n');
    for t in traces {
        buffer.push_str(&format!(
            "{} {} {} \x1b[0m  ",
            t.color.to_ansi(),
            if t.is_line { "──" } else { "●" },
            t.name
        ));
    }
    print!("{}", buffer);
    println!("\x1b[?25h"); // Show cursor
}

fn draw_line(grid: &mut Vec<Vec<String>>, x0: usize, y0: usize, x1: usize, y1: usize, color: &str) {
    let steps = (x1 as i32 - x0 as i32)
        .abs()
        .max((y1 as i32 - y0 as i32).abs());
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 as f32 + (x1 as i32 - x0 as i32) as f32 * t) as usize;
        let y = (y0 as f32 + (y1 as i32 - y0 as i32) as f32 * t) as usize;
        if y < grid.len() && x < grid[0].len() {
            grid[y][x] = format!("{}·\x1b[0m", color);
        }
    }
}

fn get_bounds(traces: &[Trace]) -> (f32, f32, f32, f32) {
    let all_x: Vec<f32> = traces.iter().flat_map(|t| t.x.iter()).cloned().collect();
    let all_y: Vec<f32> = traces.iter().flat_map(|t| t.y.iter()).cloned().collect();
    (
        *all_x
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        *all_x
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        *all_y
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        *all_y
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
    )
}

fn map_val(val: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    if (in_max - in_min).abs() < 1e-6 {
        return out_min;
    }
    (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
}
