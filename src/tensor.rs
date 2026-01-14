pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // As we are dealing with 2D tensors max, we can simply return the debug format for 1D tensors
        if self.shape.len() != 2 {
            return write!(f, "{:?}", self.data);
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        for row in 0..rows {
            write!(f, "  |")?;
            for col in 0..cols {
                let index = row * cols + col;
                write!(f, "{:>8.4}", self.data[index])?;

                if col < cols - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "|")?;
        }
        Ok(())
    }
}

impl Tensor {
    pub fn _element_wise_op(&self, other: &Tensor, op: fn(f32, f32) -> f32) -> Tensor {
        if self.shape != other.shape {
            panic!("Shapes do not match for element-wise operation");
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();

        Tensor::new(data, self.shape.clone())
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        if shape.len() == 0 || shape.len() > 2 {
            panic!("Only 1D and 2D tensors are supported");
        }

        if data.len() != shape.iter().product::<usize>() {
            panic!("Data length does not match shape");
        }
        Tensor { data, shape }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self._element_wise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self._element_wise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self._element_wise_op(other, |a, b| a * b)
    }
}
