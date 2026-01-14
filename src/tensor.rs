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
            TensorError::ShapeMismatch => write!(f, "Tensor shapes do not match for the operation."),
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

        for r in 0..rows {
            for c in 0..cols {
                transposed_data[c * rows + r] = self.data[r * cols + c];
            }
        }

        Tensor::new(transposed_data, vec![cols, rows])
    }
}
