use std::error::Error;

#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeMismatch,
    InvalidRank,
    InconsistentData,
}

impl Error for TensorError {}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch => {
                write!(f, "Tensor shapes do not match for the operation.")
            }
            TensorError::InvalidRank => write!(f, "Tensor rank is invalid (must be 1D or 2D)."),
            TensorError::InconsistentData => write!(f, "Data length does not match tensor shape."),
        }
    }
}

#[derive(Debug, PartialEq)]
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
    fn _element_wise_op(
        &self,
        other: &Tensor,
        op: fn(f32, f32) -> f32,
    ) -> Result<Tensor, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();

        Tensor::new(data, self.shape.clone())
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, TensorError> {
        if shape.len() == 0 || shape.len() > 2 {
            return Err(TensorError::InvalidRank);
        }

        if data.len() != shape.iter().product::<usize>() {
            return Err(TensorError::InconsistentData);
        }
        Ok(Tensor { data, shape })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a * b)
    }

    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        if self.shape.len() != 1 && self.shape.len() != 2 {
            return Err(TensorError::InvalidRank);
        }

        if self.shape.len() == 1 {
            return Tensor::new(self.data.clone(), self.shape.clone());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0.0; self.data.len()];

        for row in 0..rows {
            for col in 0..cols {
                transposed_data[col * rows + row] = self.data[row * cols + col];
            }
        }

        Tensor::new(transposed_data, vec![cols, rows])
    }

    pub fn matmul_naive(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let (a_rows, a_cols) = match self.shape.len() {
            1 => (1, self.shape[0]),
            2 => (self.shape[0], self.shape[1]),
            _ => return Err(TensorError::InvalidRank),
        };

        let (b_rows, b_cols) = match other.shape.len() {
            1 => (other.shape[0], 1),
            2 => (other.shape[0], other.shape[1]),
            _ => return Err(TensorError::InvalidRank),
        };

        if a_cols != b_rows {
            return Err(TensorError::ShapeMismatch);
        }

        let mut result_data = vec![0.0; a_rows * b_cols];

        for i in 0..a_rows {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    result_data[i * b_cols + j] +=
                        self.data[i * a_cols + k] * other.data[k * b_cols + j];
                }
            }
        }

        let out_shape = match (self.shape.len(), other.shape.len()) {
            (1, 1) => vec![1],
            (1, 2) => vec![b_cols],
            (2, 1) => vec![a_rows],
            _ => vec![a_rows, b_cols],
        };

        Tensor::new(result_data, out_shape)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let (a_rows, a_cols) = match self.shape.len() {
            1 => (1, self.shape[0]),
            2 => (self.shape[0], self.shape[1]),
            _ => return Err(TensorError::InvalidRank),
        };

        let (b_rows, b_cols) = match other.shape.len() {
            1 => (other.shape[0], 1),
            2 => (other.shape[0], other.shape[1]),
            _ => return Err(TensorError::InvalidRank),
        };

        if a_cols != b_rows {
            return Err(TensorError::ShapeMismatch);
        }

        let mut data = vec![0.0; a_rows * b_cols];

        for i in 0..a_rows {
            let out_row_offset = i * b_cols;

            for k in 0..a_cols {
                let aik = self.data[i * a_cols + k];
                let rhs_row_offset = k * b_cols;
                let rhs_slice = &other.data[rhs_row_offset..rhs_row_offset + b_cols];
                let out_slice = &mut data[out_row_offset..out_row_offset + b_cols];

                for j in 0..b_cols {
                    out_slice[j] = out_slice[j] + aik * rhs_slice[j];
                }
            }
        }

        let out_shape = match (self.shape.len(), other.shape.len()) {
            (1, 1) => vec![1],
            (1, 2) => vec![b_cols],
            (2, 1) => vec![a_rows],
            _ => vec![a_rows, b_cols],
        };

        Ok(Tensor {
            data,
            shape: out_shape,
        })
    }

    pub fn sum(&self, axis: Option<usize>) -> Result<Tensor, TensorError> {
        match axis {
            None => {
                let sum: f32 = self.data.iter().sum();
                Tensor::new(vec![sum], vec![1])
            }

            Some(0) => {
                if self.shape.len() < 2 {
                    return self.sum(None);
                }
                let rows = self.shape[0];
                let cols = self.shape[1];
                let mut result_data = vec![0.0; cols];

                for r in 0..rows {
                    for c in 0..cols {
                        result_data[c] += self.data[r * cols + c];
                    }
                }
                Tensor::new(result_data, vec![cols])
            }

            Some(1) => {
                if self.shape.len() < 2 {
                    return self.sum(None);
                }
                let rows = self.shape[0];
                let cols = self.shape[1];
                let mut result_data = vec![0.0; rows];

                for r in 0..rows {
                    for c in 0..cols {
                        result_data[r] += self.data[r * cols + c];
                    }
                }
                Tensor::new(result_data, vec![rows])
            }

            _ => Err(TensorError::InvalidRank),
        }
    }
}
