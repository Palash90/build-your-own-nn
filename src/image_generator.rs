use std::{fs::File, io::{BufRead, BufReader}};

pub fn draw_image(source: &str)  {
    let file = File::open(source).expect("File not found");
    let reader = BufReader::new(file);

    // 1. Parse all tokens (skipping comments)
    let tokens: Vec<String> = reader.lines()
        .flatten()
        .filter(|l| !l.trim().starts_with('#'))
        .flat_map(|l| l.split_whitespace().map(String::from).collect::<Vec<_>>())
        .collect();

    if tokens.len() < 3 { return; }
    
    // 2. Extract dimensions from header
    let w: usize = tokens[1].parse().unwrap();
    let h: usize = tokens[2].parse().unwrap();
    let data = &tokens[3..];

    println!("Image: {} ({}x{})", source, w, h);

    // 3. Render using Braille 2x4 blocks
    for y in (0..h).step_by(4) {
        let mut row = String::new();
        for x in (0..w).step_by(2) {
            let mut byte = 0u8;
            // The Braille dot mapping (standard bit positions)
            let dots = [
                (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04),
                (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20),
                (0, 3, 0x40), (1, 3, 0x80)
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