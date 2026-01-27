use std::{f32, fs::File, io::{BufWriter, Write}};

pub fn read_pbm_for_nn(path: &str) -> (usize, usize, Vec<f32>, Vec<f32>) {
    let content = std::fs::read_to_string(path).expect("Read failed");
    let mut tokens = content.split_whitespace();

    assert_eq!(tokens.next().unwrap(), "P1");
    let w: usize = tokens.next().unwrap().parse().unwrap();
    let h: usize = tokens.next().unwrap().parse().unwrap();

    let mut x_coords = Vec::with_capacity(w * h * 2);
    let mut y_values = Vec::with_capacity(w * h);

    for i in 0..(w * h) {
        // Input: [Row, Col]
        x_coords.push((i / w) as f32); 
        x_coords.push((i % w) as f32);
        // Target: [Pixel]
        y_values.push(tokens.next().unwrap().parse().unwrap());
    }

    (w, h, x_coords, y_values)
}

pub fn render_image(w: usize, h: usize, data: &[f32]) {
    let threshold = 0.8;

    for y in (0..h).step_by(4) {
        let mut row = String::new();
        for x in (0..w).step_by(2) {
            let mut byte = 0u8;
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
                if px < w && py < h {
                    if data[py * w + px] >= threshold {
                        byte |= mask;
                    }
                }
            }
            row.push(std::char::from_u32(0x2800 + byte as u32).unwrap());
        }
        println!("{}", row);
    }
}

pub fn draw_pbm(source: &str) {
    let content = std::fs::read_to_string(source).expect("Read failed");
    let mut tokens = content.split_whitespace().filter(|t| !t.starts_with('#'));

    let _magic = tokens.next(); // Skip "P1"
    let w: usize = tokens.next().unwrap().parse().unwrap();
    let h: usize = tokens.next().unwrap().parse().unwrap();

    // Convert ASCII "0"/"1" into actual 0 and 1 integers
    let data: Vec<f32> = tokens.map(|t| t.parse::<f32>().unwrap()).collect();

    render_image(w, h, &data);
}

pub fn save_as_pbm(path: &str, w: usize, h: usize, data: &[f32]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write the PBM Header
    // P1 means plain text black and white
    writeln!(writer, "P1")?;
    writeln!(writer, "{} {}", w, h)?;

    let threshold = 0.5;

    for (i, &pixel) in data.iter().enumerate() {
        // Convert float to "0" or "1" based on threshold
        let val = if pixel >= threshold { "1" } else { "0" };
        write!(writer, "{}", val)?;

        // Add a newline every 'w' pixels or spaces between values to keep it readable
        if (i + 1) % w == 0 {
            writeln!(writer)?;
        } else {
            write!(writer, " ")?;
        }
    }

    writer.flush()?;
    Ok(())
}

pub struct Trace {
    pub name: String,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub color: PlotColor,
    pub is_line: bool,
    pub hide_axes: bool,
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
            PlotColor::Red => "\x1b[31m",
            PlotColor::Blue => "\x1b[34m",
            PlotColor::Green => "\x1b[32m",
            PlotColor::Cyan => "\x1b[36m",
            PlotColor::Magenta => "\x1b[35m",
            PlotColor::Yellow => "\x1b[33m",
            PlotColor::White => "\x1b[37m",
            PlotColor::Reset => "\x1b[0m",
        }
    }
}

// In image_utils.rs

/// THE CORE LOGIC: Extracted so it can be reused without printing
fn create_plot_grid(
    traces: &[Trace],
    width: usize,
    height: usize,
    fixed_bounds: Option<(f32, f32, f32, f32)>,
) -> Vec<Vec<String>> {
    let (min_x, max_x, min_y, max_y) = match fixed_bounds {
        Some(bounds) => bounds,
        None => get_bounds(traces),
    };

    let margin_l = 10;
    let margin_b = 2;
    let plot_w = width - margin_l - 2;
    let plot_h = height - margin_b - 2;

    let y_tick_count = 5;
    let x_tick_count = 4;

    let mut grid = vec![vec![" ".to_string(); width]; height];

let hide_all_axes = traces.iter().any(|t| t.hide_axes);

if !hide_all_axes {

    for i in 0..=y_tick_count {
        let t = i as f32 / y_tick_count as f32;
        let py = map_val(t, 0.0, 1.0, plot_h as f32, 0.0) as usize;
        let val = map_val(t, 0.0, 1.0, min_y, max_y);
        grid[py][margin_l] = "┼".to_string();
        let label = format!("{:>9.1}", val);
        for (idx, c) in label.chars().enumerate() {
            if idx < margin_l { grid[py][idx] = c.to_string(); }
        }
    }

    for i in 0..=x_tick_count {
        let t = i as f32 / x_tick_count as f32;
        let px = map_val(t, 0.0, 1.0, 0.0, plot_w as f32) as usize + margin_l + 1;
        let val = map_val(t, 0.0, 1.0, min_x, max_x);
        if px < width {
            grid[plot_h][px] = "┴".to_string();
            let label = format!("{:.1}", val);
            for (idx, c) in label.chars().enumerate() {
                if px + idx < width { grid[plot_h + 1][px + idx] = c.to_string(); }
            }
        }
    }

    for y in 0..plot_h { if grid[y][margin_l] == " " { grid[y][margin_l] = "│".to_string(); } }
    for x in margin_l + 1..width { if grid[plot_h][x] == " " { grid[plot_h][x] = "─".to_string(); } }
    grid[plot_h][margin_l] = "└".to_string();
}

    for trace in traces {
        let color_code = trace.color.to_ansi();
        for i in 0..trace.x.len() {
            let px = map_val(trace.x[i], min_x, max_x, 0.0, plot_w as f32) as usize + margin_l + 1;
            let py = map_val(trace.y[i], min_y, max_y, plot_h as f32 - 1.0, 0.0) as usize;
            if py < plot_h && px > margin_l && px < width {
                if trace.is_line && i > 0 {
                    let prev_px = map_val(trace.x[i - 1], min_x, max_x, 0.0, plot_w as f32) as usize + margin_l + 1;
                    let prev_py = map_val(trace.y[i - 1], min_y, max_y, plot_h as f32 - 1.0, 0.0) as usize;
                    draw_line(&mut grid, prev_px, prev_py, px, py, color_code, &trace.name);
                }
                grid[py][px] = format!("{}●\x1b[0m", color_code);
            }
        }
    }
    grid
}

/// RENDER PLOT (UNCHANGED SIGNATURE): Safe for use elsewhere
pub fn render_plot(
    traces: &[Trace],
    width: usize,
    height: usize,
    fixed_bounds: Option<(f32, f32, f32, f32)>,
    title: String,
) {
    let grid = create_plot_grid(traces, width, height, fixed_bounds);
    
    let mut buffer = String::new();
    buffer.push_str("\x1b[2J\x1b[H\x1b[?25l");
    buffer.push_str("\n\n");
    let title_len = title.len();
    if title_len < width {
        let padding = (width - title_len) / 2;
        buffer.push_str(&" ".repeat(padding));
    }
    buffer.push_str(&format!("\x1b[1;36m{}\x1b[0m\n\n", title.to_uppercase()));

    for row in grid {
        buffer.push_str(&row.concat());
        buffer.push('\n');
    }

    buffer.push('\n');
    for t in traces {
        buffer.push_str(&format!("{} {} {} \x1b[0m  ", t.color.to_ansi(), if t.is_line { "──" } else { "●" }, t.name));
    }
    print!("{}", buffer);
    println!("\x1b[?25h");
}

pub fn render_dual_plots(
    traces_left: &[Trace],
    traces_right: &[Trace],
    width: usize,
    height: usize,
    bounds: Option<(f32, f32, f32, f32)>,
    title: String,
) {
    let grid_l = create_plot_grid(traces_left, width, height, bounds);
    let grid_r = create_plot_grid(traces_right, width, height, bounds);

    let mut buffer = String::new();
    buffer.push_str("\x1b[2J\x1b[H\x1b[?25l");

    let total_w = (width * 2) + 4;
    buffer.push_str(&format!("\n\x1b[1;36m{:^width$}\x1b[0m\n\n", title.to_uppercase(), width = total_w));

    for y in 0..height {
        buffer.push_str(&grid_l[y].concat());
        buffer.push_str("    "); 
        buffer.push_str(&grid_r[y].concat());
        buffer.push('\n');
    }

    buffer.push('\n');
    
    let mut seen_names = std::collections::HashSet::new();
    for t in traces_left.iter().chain(traces_right.iter()) {
        let is_metadata = matches!(t.name.as_str(), "heavy" | "medium" | "light") 
                          || t.name.is_empty()
                          || t.name.contains("Point"); // Hide the 4 XOR points to save space
        
        if !is_metadata && seen_names.insert(&t.name) {
            buffer.push_str(&format!(
                "{} {} {} \x1b[0m  ", 
                t.color.to_ansi(), 
                if t.is_line { "──" } else { "●" }, 
                t.name
            ));
        }
    }

    print!("{}", buffer);
    println!("\x1b[?25h");
}

fn draw_line(grid: &mut Vec<Vec<String>>, x0: usize, y0: usize, x1: usize, y1: usize, color: &str, weight_type: &str) {
    let steps = (x1 as i32 - x0 as i32)
        .abs()
        .max((y1 as i32 - y0 as i32).abs());
    
    // Visual Hierarchy using only dots and ANSI styles
    let (dot_char, style) = match weight_type {
        "heavy"  => ("·", "\x1b[1m"), // Bold dot
        "medium" => ("·", ""),        // Normal dot
        "light"  => ("·", "\x1b[2m"), // Dim/Faint dot
        _        => ("·", ""),
    };

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 as f32 + (x1 as i32 - x0 as i32) as f32 * t) as usize;
        let y = (y0 as f32 + (y1 as i32 - y0 as i32) as f32 * t) as usize;
        
        if y < grid.len() && x < grid[0].len() {
            grid[y][x] = format!("{}{}{}\x1b[0m", style, color, dot_char);
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

