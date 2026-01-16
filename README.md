# Prologue: Breaking the Black Box
Machine Learning often felt like a "black box" to me. Every tutorial I found introduced `NumPy` as a baseline requirement. Libraries like `scikit-learn`, `PyTorch`, `TensorFlow`, etc. are excellent for building quick prototypes as well as production-grade models. However, they heavily obscure the underlying mechanics. Hence, I decided to start learning this technology by building it from scratch. 

I have spent years trying to learn Rust. After experimenting with various methods (The Book, RBE, Rustlings, etc.) over the years, I found the missing link: the difficulty lay not in the language, but in the lack of a motivating end-goal.

This project began as a month-long deep dive into Linear Regression. However, my momentum gradually slowed and after numerous iterations, I have finally reached a milestone where I can document this journey. As of today, the project is still evolving.

To be clear: This project is not meant to replace PyTorch, TensorFlow, or ndarray. It is a 'toy' engine by design. Its purpose is to replace the 'I don't know' in your head when you call torch.matmul() with a clear, mental model of memory buffers and cache lines. We aren't building for production; we are building for mastery.


# Prerequisites
This guide is designed as a self-contained journey. We do not assume a prior mathematical background. Instead, we derive the necessary mathematical principles as they arise.

While this guide does not assume mastery in Rust, a basic understanding of the following concepts will make the progression smoother:
- **Rust Fundamentals:** The use of `structs`, `enums`, and basic pattern matching.
- **The Memory Model:** A conceptual understanding of Ownership, Borrowing, and Slicing.
- **The Module System:** Familiarity with how Rust organizes code across files.
- **The Toolchain:** You’ll need `rustc` and `cargo` installed and ready.

>**NOTE**
>In keeping with our philosophy of Radical Transparency, we will not rely on external linear algebra crates like ndarray. Our only dependency is the Rust Standard Library.

# Project Philosophy
This guide is designed with a specific philosophy in mind: **Radical Transparency**. We do not start with frameworks or pre-built third-party libraries. We start with a blank file and a single `fn main()`. From there, we will incrementally build the mathematical and structural architecture required to perform complex tasks.

## The Roadmap

This is the roadmap I wish I had two years ago. Whether you are a Rustacean curious about AI or an ML practitioner looking to understand systems-level implementation, this journey is for you.

- **The Blank Canvas:** Initializing the environment and establishing the foundational data structures.
- **The Mathematical Engine:** Implementing tensors, gradients, and backpropagation from scratch.
- **Building the Network:** Constructing layers and activation functions.
- **The Visual Goal:** Training our library to interpret and reconstruct images, proving that 'magic' is just linear algebra and high-dimensional calculus written in a language with strict safety guarantees.

![Image Reconstruction]()

And that’s where the story begins...

# The Tensor
To build a neural network from scratch, we need to construct the fundamental building blocks first. In the world of Machine Learning, that building block would be a **Tensor**. In simple terms, a tensor is a collection of numbers organized in a grid.

## Journey from Scalar to Tensor
To understand the data structure we are building, we need to develop an intuition first. Let's start building it from scratch as well.

- **Scalar:** We are familiar with this and use it every time we perform arithmetic operations: a single number (e.g. 5). 
    In the world of tensors, we would define them as a tensor of rank 0.
    In programming, this would be a single numeric variable: `x=5`
- **Vector:** When we arrange a collection of numbers, we get a `Vector`.
    In the world of tensors, we would define them as a tensor of rank 1.
    In programming, this would be an array or `Vec` of numeric variables: `a = [1, 2, 3]`
- **Matrix:** When we arrange multiple vectors in an array, we get a matrix.
    In the world of tensors, we would define them as a tensor of rank 2.
    In programming, this would be an array of arrays (or `Vec` of `Vec`s): `a = [[1, 2], [3, 4]]`
- **Tensor:** When we arrange multiple matrices in an array or `Vec`, we get higher rank tensors. This would be beyond our scope in this guide and we will keep things simple by restricting ourselves to _2D_ tensors only.

Here is a visual representation of the concept:

$$
\begin{array}{ccc}
\mathbf{Scalar} & \mathbf{Vector} & \mathbf{Matrix} \\
\color{#E74C3C}{1} & \left[\begin{matrix} \color{cyan}1 \\ \color{cyan}2 \end{matrix} \right] & \left[\begin{matrix} \color{magenta}{1} & \color{magenta}{2} \\\ \color{magenta}{3} & \color{magenta}{4} \end{matrix} \right]
\end{array}
$$

## Matrix Notation and Indexing

When we want to refer to an element inside the matrix, we need a notation to identify a specific element.

A matrix $A$ with $m$ rows and $n$ columns is referred to as an $m \times n$ matrix. We denote an individual element within that matrix using subscripts:

$$
A_{i,j}
$$

Where:

- _i_ is the row index (1 ≤ _i_ ≤ _m_)
- _j_ is the column index (1 ≤ _j_ ≤ _n_)

In code, we usually achieve this by indexing into the array:

```rust
a = [[1, 2], [3, 4]];
println!("{}", a[0][0]); // Output: 1
```

>**NOTE**
>Mathematical notation and programming differ in how they index a collection of numbers. Mathematics typically uses 1-based indexing, whereas programming uses 0-based indexing.

## Implementation: Memory Buffers and Layout
With the mathematical background, now we'll design and implement the `Tensor`. Let's first kick off the project and then we'll add elements to it. We'll use the default `cargo new` command for this:

```shell
$ cargo new build_your_own_nn
    Creating binary (application) `build_your_own_nn` package
note: see more `Cargo.toml` keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

```
That's it. Nothing else. Let's begin translating our design into code.

We need a way to store multiple data points and we should be able to index the data structure to access or modify the data inside.

An array matches our requirements and is super fast. However, in Rust, arrays can't grow or shrink dynamically at run time. To maintain flexibility, we'll use `Vec` instead. A basic implementation of our `Tensor` can work well with `Vec<Vec<f32>>`. However, there are two problems in that approach.

1. **Indirection (Pointer Chasing):** A `Vec` of `Vec`s is a very performance-intensive structure. Each inner `Vec` is a separate heap allocation. Accessing elements requires jumping to different memory locations. 

$$
\begin{array}{c|l}
\text{Outer Index} & \text{Pointer to Inner Vec} \\\\ \hline
0 & \color{#3498DB}{\rightarrow [v_{0,0}, v_{0,1}, v_{0,2}]} \\\\
1 & \color{#E74C3C}{\rightarrow [v_{1,0}, v_{1,1}, v_{1,2}]} \\\\
2 & \color{#2ECC71}{\rightarrow [v_{2,0}, v_{2,1}, v_{2,2}]} \\\\
\end{array}
$$

2. **Rigidity:** `Vec` of `Vec` would permanently limit our application to a 2D matrix and later, if we want to support higher dimension tensors, we would have to change our code.

To avoid these problems, we'll use two `Vec`s instead. One will hold the data in a flat _1D_ structure and the other will hold the _shape_ definition like this:

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

These two fields should not be accessible directly, we need to define accessors for them, we'll also use the `TensorError` enum for error handling.

Let's write these definitions first in a new file `tensor.rs`. Later, we'll implement them one by one.

```rust
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


impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Tensor, TensorError> {
        todo!()
    }

    pub fn data(&self) -> &[f32] {
        todo!()
    }

    pub fn shape(&self) -> &[usize] {
        todo!()
    }
}
```

Once the definitions are written, we should expose the `struct` publicly. To do that, we create another file `lib.rs` and write the following line in it:

```rust
pub mod tensor;
```

Now we have defined our data structure, required functions and methods. Let's write a few tests now.

We put all the tests outside `src` directory; in a separate directory named `tests`.

```rust
use build_your_own_nn::tensor::Tensor;
use build_your_own_nn::tensor::TensorError;

#[cfg(test)]
#[test]
fn test_invalid_shape_creation() {
    let result = Tensor::new(vec![1.0], vec![2, 2]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TensorError::InconsistentData);
}
```

If we try to run the tests now, it will break. We need to first complete the implementations.

```rust
impl Tensor {
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
}
```

Now, if we run the tests, we can see the tests passing.

```text
~/git/build-your-own-nn$ cargo test
   Compiling build-your-own-nn v0.1.0 (/home/palash/git/build-your-own-nn)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.30s
     Running unittests src/lib.rs (target/debug/deps/build_your_own_nn-8e7fc48103748a00)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/main.rs (target/debug/deps/build_your_own_nn-fb385501dec7dedb)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/test_tensor.rs (target/debug/deps/test_tensor-25b5f99a2a90f9bb)

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests build_your_own_nn

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

>**NOTE**
>We will be using standard Rust module system throughout.

Currently the directory structure should look like the following:

```text
src
├── lib.rs
├── main.rs
└── tensor.rs
tests
└── test_tensor.rs
Cargo.toml
```

## Display: Pretty Printing Matrix
The definition and implementation of the tensor is now clear. But how can we intuitively inspect the data if we need to. Looking at the data directly from `Vec` isn't very intuitive.

Let's first try to understand the problem and then we'll fix it. We rewrite the `main` function to inspect the data inside the tensor:

```rust
use build_your_own_nn::tensor::Tensor;
use build_your_own_nn::tensor::TensorError;

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    println!("Tensor data: {:?} {:?}", a.data(), a.shape()); // Output: Tensor data: [1.0, 2.0, 3.0, 4.0] [2, 2]

    Ok(())
}

```

As you can see, the output is a linear array of data. It does not preserve the dimensionality of the tensor. To fix this linear display of tensors and a nice matrix-like format, we'll implement the `Display` trait for our `Tensor` struct, such that any time we want to display the tensor, it will show in a nice formatted way.

The shape `Vec` will help us here. First we define what do the elements map to and here we decide the rules:

1. If the length of `shape` is 1, it is a _vector_, we can simply return the default debug formatted data.
1. If the length of `shape` is 2, it is a _matrix_, the first element of the `shape` vector defines number of rows and the second element defines number of columns. By the way, this convention of defining matrix order is known as **Row-major**.
1. We won't go beyond _2D_
1. For each row we'll pick out elements matching column length indexing $(\mathbf{row} \times \mathbf{cols}) + \mathbf{col}$

Let's take an example,

$$\begin{bmatrix} \color{cyan}1_{0} & \color{magenta}2_{1} & \color{#2ECC71}3_{2} & \color{purple}4_{3} \end{bmatrix} \implies \begin{bmatrix} \color{cyan}1_{(0)} & \color{magenta}2_{(1)} \\\ \color{#2ECC71}3_{(2)} & \color{purple}4_{(3)} \end{bmatrix}$$

Here, we have a `Vec` of length 4 with 2 rows and 2 columns. The first row is formed by the elements at index 0 and index 1 and the second row is formed by the elements at index 2 and index 3.

Let's implement these rules for our tensor now.

First we add the tests as per our desirable matrix look:

```rust
#[test]
fn test_tensor_display_2d() -> Result<(), TensorError> {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    let output = format!("{}", tensor);

    println!("{}", output);

    assert!(output.contains("|  1.0000,   2.0000|"));
    assert!(output.contains("|  3.0000,   4.0000|"));
}

#[test]
fn test_tensor_display_alignment() -> Result<(), TensorError> {
    let tensor = Tensor::new(vec![1.23456, 2.0, 100.1, 0.00001], vec![2, 2])?;

    let output = format!("{}", tensor);

    assert!(output.contains("  1.2346"));
    assert!(output.contains("  0.0000"));
}

#[test]
fn test_tensor_display_1d() -> Result<(), TensorError> {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;

    let output = format!("{}", tensor);
    assert!(output.contains("[1.0, 2.0, 3.0]"));
}
```

And then we implement the `Display` trait for our `Tensor`, matching the rules to make the tests pass.

```rust
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
```

Let's rewrite the `main` function and look at our tensor getting displayed:

```rust
use build_your_own_nn::tensor::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

    let b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    )?;

    println!("{}", a);

    println!("{}", b);
    Ok(())
}
```

```text
  |  1.0000,   2.0000|
  |  3.0000,   4.0000|

  |  1.0000,   2.0000,   3.0000|
  |  4.0000,   5.0000,   6.0000|
  |  7.0000,   8.0000,   9.0000|

```

**Challenge to the readers:** I encourage the readers to implement their own formatting. I chose this formatting because I like it, you don't have to stick to this.


# Basic Tensor Arithmetic
We have defined our tensor and established its notation. Now let's see how we operate on them.

For tensors of any size or rank, we define the following operations:

## Element Wise Addition
Element wise addition is only defined for two matrices of the same shape. If $A$ and $B$ are both $m \times n$, then $C=A+B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ + B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} \color{cyan}{1} & \color{magenta}2 \\\ \color{#D4A017}3 & \color{#2ECC71}4 \end{bmatrix} + \begin{bmatrix} \color{cyan}5 & \color{magenta}6 \\\ \color{#D4A017}7 & \color{#2ECC71}8 \end{bmatrix} = \begin{bmatrix} \color{cyan}6 & \color{magenta}8 \\\ \color{#D4A017}10 & \color{#2ECC71}12 \end{bmatrix} $$


## Element Wise Subtraction
Element wise subtraction is only defined for two matrices of the same shape. If $A$ and $B$ are both $m \times n$, then $C=A-B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ - B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} \color{cyan}{5} & \color{magenta}6 \\\ \color{#D4A017}7 & \color{#2ECC71}7 \end{bmatrix} - \begin{bmatrix} \color{cyan}1 & \color{magenta}2 \\\ \color{#D4A017}3 & \color{#2ECC71}4 \end{bmatrix} = \begin{bmatrix} \color{cyan}4 & \color{magenta}4\\\ \color{#D4A017}4 & \color{#2ECC71}4 \end{bmatrix} $$

## Element Wise Multiplication
Element wise multiplication (a.k.a. _Hadamard Product_) is only defined for two matrices of the same shape. If $A$ and $B$ are both $m \times n$, then $C=A \odot B$ is calculated as:

$$
C_{i,j}​=A_{i,j}​ \odot B_{i,j}​
$$

Let's take an example,

$$ \begin{bmatrix} \color{cyan}{1} & \color{magenta}2 \\\ \color{#D4A017}3 & \color{#2ECC71}4 \end{bmatrix} \odot \begin{bmatrix} \color{cyan}5 & \color{magenta}6 \\\ \color{#D4A017}7 & \color{#2ECC71}8 \end{bmatrix} = \begin{bmatrix} \color{cyan}5 & \color{magenta}12\\\ \color{#D4A017}21 & \color{#2ECC71}32 \end{bmatrix} $$

Now that we have the mathematical blueprint, let's translate these concepts into Rust code.

## Implementation
We should expose methods for `add`, `sub` and `mul`. We'll first add these method definitions into our existing tensor `impl` block.

```rust
    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        todo!()
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        todo!()
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        todo!()
    }
```

Now we'll write tests for these methods:

```rust
#[test]
pub fn test_tensor_operations() -> Result<(), TensorError>  {
    use std::vec;

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    let c = a.add(&b)?;
    assert_eq!(c.data(), &[6.0, 8.0, 10.0, 12.0]);
    assert_eq!(c.shape(), &[2, 2]);

    let d = a.sub(&b)?;
    assert_eq!(d.data(), &[-4.0, -4.0, -4.0, -4.0]);
    assert_eq!(d.shape(), &[2, 2]);

    let e = a.mul(&b)?;
    assert_eq!(e.data(), &[5.0, 12.0, 21.0, 32.0]);
    assert_eq!(e.shape(), &[2, 2]);
}
```

Now we'll implement these operations. All the implementations so far operate on the data element wise and the shape of those two tensors must match. So, we will add a common method inside the `impl` block and use it to unify all the element wise logic using function pointers:

```rust
    pub fn _element_wise_op(
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
	
	
	pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self._element_wise_op(other, |a, b| a * b)
    }
```

# Linear Transformations and Aggregations
In the previous operations, we treated matrices like rigid containers—adding or multiplying elements that lived in the exact same "neighborhood." However, to build a neural network, we need to support a few _2D_ operations as well. To perform these, we need to move around a little.

The following are a few operations we are going to describe, write tests for and implement in our `Tensor`.

## Transpose
One of the most fundamental transformations in linear algebra involves changing the very orientation of the data. This is known as **Transpose**. In a transposition operation, the rows of the matrix become columns and the columns become rows.

$$
(A^T​)_{i,j}=A_{j,i}​
$$

Let's take a few examples:

### Vector Transpose

$$
\begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix} \xrightarrow{transpose} \begin{bmatrix} 1 \\\ 2 \\\ 3 \\\ 4 \end{bmatrix}
$$

### Square Matrix Transpose

$$
\begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix} \xrightarrow{transpose} \begin{bmatrix} 1 & 3 \\\ 2 & 4 \end{bmatrix}
$$

### Rectangular Matrix Transpose
$$
\begin{bmatrix} 1 & 2 \\\ 3 & 4 \\\ 5 & 6 \end{bmatrix} \xrightarrow{transpose} \begin{bmatrix} 1 & 3 & 5 \\\ 2 & 4 & 6 \end{bmatrix}
$$

>**NOTE**
>In the matrix transpose examples, take a note that the main diagonal elements ($A_{i,j}$ where $i=j$) stay in their positions and don't move. Additionally, in the case of rectangular matrix transposition the shape changes. 

For example, here transposition converts $(3 \times 2) \xrightarrow{} (2 \times 3)$.

### Implementation
With this mathematical background, we can now understand what transpose operation will transform the data. With that understanding, we'll first add these tests:

```rust
    #[test]
    fn test_transpose_square() -> Result<(), TensorError> {
        // 1.0, 2.0

        // 3.0, 4.0

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

        let swapped = a.transpose()?;

        // Should become:

        // 1.0, 3.0

        // 2.0, 4.0

        assert_eq!(swapped.data(), &[1.0, 3.0, 2.0, 4.0]);

        assert_eq!(swapped.shape(), &[2, 2]);
        Ok(())
    }

    

    #[test]
    fn test_transpose_rectangular() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let swapped = a.transpose()?;

        assert_eq!(swapped.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(swapped.shape(), &[3, 2]);
        Ok(())
    }

    #[test]
    fn test_transpose_1d() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6])?;
        let swapped = a.transpose()?;

        assert_eq!(swapped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(swapped.shape(), &[6]);
        Ok(())
    }
```

To implement transpose, we have to physically move our numbers into a new Vec. While some advanced libraries just change the "metadata" (using something called strides), we are going to actually rebuild the data. This keeps our memory "contiguous," which makes our other operations faster because the CPU can predict where the next number is.

**The Logic:**

1. Check the Rank: We only support transposing 1D or 2D tensors.

1. The 1D Shortcut: If it's a 1D vector, there's no "grid" to flip, so we just return a copy.

1. The 2D Re-map: We create a new Vec of the same size. Then, we use a nested loop to visit every "cell" of our grid.

>**NOTE**
>The Index Swap: In our original data, we find an element at $row * cols + col$. In our new data, the dimensions are swapped, so the position becomes $col * rows + row$.

```rust
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
```


## Dot Product
We have already seen how to multiply two matrices or vectors element wise. However, there is another multiplication operation we can perform on tensors, known as the **Dot Product**. It is slightly more involved, as it combines element wise multiplication and a reduction operation into a single step.

The dot product of two vectors $A$ and $B$ of length n is defined as:

$$
A \cdot B = \sum_{i=1}^{n} A_i B_i
$$

Let's take a few examples.

### Vector Vector Dot Product
Here is an example of a dot product between two vectors:

$$
\begin{bmatrix} \color{#2ECC71}{1} \\\ \color{cyan}{2} \\\ \color{magenta}{3} \\\ \color{#D4A017}{4} \end{bmatrix} \cdot \begin{bmatrix} \color{#2ECC71}1 \\\ \color{cyan}2 \\\ \color{magenta}3 \\\ \color{#D4A017}4 \end{bmatrix} = \color{#2ECC71}{(1 \times 1)} \color{white}+ \color{cyan}{(2 \times 2)} \color{white}+ \color{magenta}{(3 \times 3)} \color{white}+ \color{#D4A017}{(4 \times 4)}\color{white}=30
$$

### Matrix Vector Dot Product
In a Matrix-Vector dot product, we calculate the dot product of every row from the matrix with the single column of the vector.

To perform a dot product between a matrix $A$ and a vector $v$, the number of columns in the matrix must equal the number of elements (rows) in the vector.

If matrix $A$ has the shape $(m \times n)$ and vector $v$ has the shape $(n \times 1)$, the resulting vector w will have the shape $(m \times 1)$.

Matrix Vector dot product is defined as:

$$
C_{m,1} = A_{m, n}v_{n, 1}
$$

Let's take an example:

$$
\begin{bmatrix} \color{#2ECC71}{1} & \color{#2ECC71}{2} & \color{#2ECC71}{3} \\\ \color{#D4A017}{4} & \color{#D4A017}{5} & \color{#D4A017}{6} \end{bmatrix} \cdot \begin{bmatrix} \color{cyan}{7} \\\ \color{cyan}{8} \\\ \color{cyan}{9} \end{bmatrix} = \begin{bmatrix} \color{#2ECC71}{[1, 2, 3]} \cdot \color{cyan}{[7, 8, 9]} \\\ \color{#D4A017}{[4, 5, 6]} \cdot \color{cyan}{[7, 8, 9]} \end{bmatrix} = \begin{bmatrix} (\color{#2ECC71}{1} \times \color{cyan}{7} + \color{#2ECC71}{2} \times \color{cyan}{8} + \color{#2ECC71}{3} \times \color{cyan}{9}) \\\ (\color{#D4A017}{4} \times \color{cyan}{7} + \color{#D4A017}{5} \times \color{cyan}{8} + \color{#D4A017}{6} \times \color{cyan}{9})
\end{bmatrix} = \begin{bmatrix} 50 \\\ 122 \end{bmatrix}
$$

### Matrix Matrix Dot Product
In a Matrix-Matrix dot product (often simply called **Matrix Multiplication**), we don't just multiply corresponding "neighborhoods." Instead, we calculate the dot product of every row from the first matrix with every column of the second matrix.

To perform a dot product between matrix $A$ and matrix $B$, the number of columns in $A$ must equal the number of rows in $B$.

If $A$ is $(m \times n)$ and $B$ is $(n \times p)$, the resulting matrix $C$ will have the shape $(m \times p)$.

Matrix Multiplication is defined as:

$$
C_{m,p} = A_{m, n}B_{n, p}
$$

Let's take an example:

$$
\begin{bmatrix} \color{#2ECC71}1 & \color{#2ECC71}2 & \color{#2ECC71}3 \\\ \color{#D4A017}4 & \color{#D4A017}5 & \color{#D4A017}6 \end{bmatrix} \cdot \begin{bmatrix} \color{cyan}7 & \color{magenta}8 \\\ \color{cyan}9 & \color{magenta}10 \\\ \color{cyan}11 & \color{magenta}12 \end{bmatrix} = \begin{bmatrix} \color{#2ECC71}{[1, 2, 3]} \cdot \color{cyan}{[7, 9, 11]} & \color{#2ECC71}{[1, 2, 3]}\cdot \color{magenta}{[8, 10, 12]} \\\ \color{#D4A017}[4, 5, 6] \cdot \color{cyan}{[7, 9, 11]} & \color{#D4A017}[4, 5, 6] \cdot \color{magenta}{[8, 10, 12]} \\\ \end{bmatrix} = \begin{bmatrix} (\color{#2ECC71}{1} \times \color{cyan}{7} + \color{#2ECC71}{2} \times \color{cyan}{9} + \color{#2ECC71}{3} \times \color{cyan}{11}) & (\color{#2ECC71}{1} \times \color{magenta}{8} + \color{#2ECC71}{2} \times \color{magenta}{10} + \color{#2ECC71}{3} \times \color{magenta}{12}) \\\ (\color{#D4A017}{4} \times \color{cyan}{7} + \color{#D4A017}{5} \times \color{cyan}{9} + \color{#D4A017}{6} \times \color{cyan}{11}) & (\color{#D4A017}{4} \times \color{magenta}{8} + \color{#D4A017}{5} \times \color{magenta}{10} + \color{#D4A017}{6} \times \color{magenta}{12}) \end{bmatrix} = \begin{bmatrix} 58 & 64 \\\ 139 & 154 \end{bmatrix}
$$

### Implementation
Matrix multiplication is the ultimate workhorse in any neural network library and arguably the most complex operation too. In a single step of neural network with the most simple network architecture we can count matrix multiplication is used three times, element wise functional operations are called three times, addition/subtraction once and transpose twice. Don't worry if you did not understand this claim. We'll soon dive into this counting. For now, just understand Matrix Multiplication is the most frequent operation in a training cycle.

Unfortunately, by nature, matrix multiplication is an $O(n^3)$ operation. Tons of optimizations have been done over the decades on this operation both on Software front as well as Hardware front. Those optimization techniques are themselves worthy of their own book.

However, to make our tensor useful, we'll avoid the textbook naive implementation technique and will use a bit sophisticated technique with compiler optimizations. To understand the basics, we'll keep both the versions in our library.

First we'll write tests for matrix multiplications with correct assumptions and then we'll jump into both the implementations.

#### Tests for Matrix Multiplication
This test will capture many scenarios based on 1D, 2D matrix operations. We will add this to our existing tests:

```rust
    #[test]
    fn test_matmul() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        let c = a.matmul_naive(&b)?;
        let c1 = a.matmul(&b)?;  
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), c1.data());
        assert_eq!(c.shape(), c1.shape());

        let d = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let e = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?;
        let f = d.matmul_naive(&e)?;
        let f1 = d.matmul(&e)?;
        assert_eq!(f.data(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(f.shape(), &[2, 2]);
        assert_eq!(f.data(), f1.data());
        assert_eq!(f.shape(), f1.shape());

        let g = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1])?;
        let h = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3])?;
        let i = g.matmul_naive(&h)?;
        let i1 = g.matmul(&h)?;
        assert_eq!(i.data(), &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);
        assert_eq!(i.shape(), &[3, 3]);
        assert_eq!(i.data(), i1.data());
        assert_eq!(i.shape(), i1.shape());

        let j = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3])?;
        let k = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])?;
        let l = j.matmul_naive(&k)?;
        let l1 = j.matmul(&k)?;
        assert_eq!(l.data(), &[32.0]);
        assert_eq!(l.shape(), &[1, 1]);
        assert_eq!(l.data(), l1.data());
        assert_eq!(l.shape(), l1.shape());

        let m = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let n = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
        let o = m.matmul_naive(&n)?;
        let o1 = m.matmul(&n)?;
        assert_eq!(o.data(), &[32.0]);
        assert_eq!(o.shape(), &[1]);
        assert_eq!(o.data(), o1.data());
        assert_eq!(o.shape(), o1.shape());

        let p = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let q = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3])?;
        let r = q.matmul_naive(&p)?;
        let r1 = q.matmul(&p)?;
        assert_eq!(r.data(), &[32.0]);
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.data(), r1.data());
        assert_eq!(r.shape(), r1.shape());

        let s = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
        let t = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1])?;
        let u = s.matmul_naive(&t)?;
        let u1 = s.matmul(&t)?;
        assert_eq!(u.data(), &[32.0]);
        assert_eq!(u.shape(), &[1]);
        assert_eq!(u.data(), u1.data());
        assert_eq!(u.shape(), u1.shape());

        Ok(())
    }
```

#### The Naive Implementation (IJK)

>**TIP** 
>We will not use this function, this is here for reference and validation purposes. You may skip to the [next section](#the-optimized-implementation-ikj) if you want to.

In a standard textbook, you learn to calculate one cell of the result matrix at a time by taking the dot product of a row from $A$ and a column from $B$. In code, it looks like this:

```rust
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
```

This exactly mirrors the math. We perform shape normalizations and then directly go into a three-level nested for loop to calculate each cell of the resulting matrix (or vector).

Let's use our previous example. To find just the first element (top-left) of the result:

$$
A = \begin{bmatrix} \color{#2ECC71}1 & \color{#2ECC71}2 & \color{#2ECC71}3 \\\ \color{#D4A017}4 & \color{#D4A017}5 & \color{#D4A017}6 \end{bmatrix},
B = \begin{bmatrix} \color{cyan}7 & \color{magenta}8 \\\ \color{cyan}9 & \color{magenta}10 \\\ \color{cyan}11 & \color{magenta}12 \end{bmatrix}
$$

##### Calculation of $C_{0,0}$​ (Top Left)

$$
\begin{array}{}
\begin{array}{c|c|c|c|c}
C_{0,0} & i=0  & j=0 & k=0 & 0 + (\color{#2ECC71}A_{0,0}​ \color{white}\times \color{cyan}B_{0,0} \color{white})​= 0 + (\color{#2ECC71}1 \times \color{cyan}7\color{white}) = 7 \\
\hline
C_{0,0} & i=0  & j=0 & k=1 & 7 + (\color{#2ECC71}A_{0,1}​ \color{white}\times \color{cyan}B_{1,0}\color{white}) ​= 7+(\color{#2ECC71}2 \times \color{cyan}9\color{white}) = 25 \\
\hline
C_{0,0} & i=0  & j=0 & k=2 & 25 + (\color{#2ECC71}A_{0,2}​ \color{white}\times \color{cyan}B_{2,0}\color{white}) ​= (\color{#2ECC71}3 \times \color{cyan}11\color{white}) = 58 \\
\end{array}
\implies
\begin{bmatrix}
\mathbf{\color{lightgray}58} & 0 \\\
0 & 0
\end{bmatrix}
\end{array}
$$

##### Calculation of $C_{0,1}$​ (Top Right)

$$
\begin{array}{}
\begin{array}{c|c|c|c|c}
C_{0,1} & i=0  & j=1 & k=0 & 0 +(\color{#2ECC71}A_{0,0}​ \color{white}\times \color{magenta}B_{0,1} \color{white})​= 0 +(\color{#2ECC71}1 \times \color{magenta}8) \color{white}= 8 \\
\hline
C_{0,1} & i=0  & j=1 & k=1 & 8 + (\color{#2ECC71}A_{0,1}​ \color{white}\times \color{magenta}B_{1,1}\color{white}) ​= 8+(\color{#2ECC71}2 \times \color{magenta}10\color{white}) = 28 \\
\hline
C_{0,1} & i=0  & j=1 & k=2 & 28 + (\color{#2ECC71}A_{0,2}​ \color{white}\times \color{magenta}B_{1,2}\color{white}) ​= 28+(\color{#2ECC71}3 \times \color{magenta}12\color{white}) = 64\\
\end{array}
\implies
\begin{bmatrix}
58 & \mathbf{\color{lightgray}64} \\\
0 & 0
\end{bmatrix}
\end{array}
$$


##### Calculation of $C_{1,0}$​ (Bottom Left)

$$
\begin{array}{}
\begin{array}{c|c|c|c|c}
C_{1,0} & i=1  & j=0 & k=0 & 0+(\color{#D4A017}A_{1,0}​ \color{white}\times \color{cyan}B_{0,0} \color{white})​=0+ (\color{#D4A017}4 \times \color{cyan}7\color{white}) = 28 \\
\hline
C_{1,0} & i=1  & j=0 & k=1 & 28 + (\color{#D4A017}A_{1,1}​ \color{white}\times \color{cyan}B_{1,0}\color{white}) ​= 28+(\color{#D4A017}5 \times \color{cyan}9\color{white}) = 73 \\
\hline
C_{1,0} & i=1  & j=0 & k=2 & 73 + (\color{#D4A017}A_{1,2}​ \color{white}\times \color{cyan}B_{2,0}\color{white}) ​= 73+(\color{#D4A017}6 \times \color{cyan}11\color{white}) = 139 \\
\end{array}
\implies
\begin{bmatrix}
58 & 64 \\\
\mathbf{\color{lightgray}139} & 0
\end{bmatrix}
\end{array}
$$

##### Calculation of $C_{1,1}$​ (Bottom Right)

$$
\begin{array}{}
\begin{array}{c|c|c|c|c}
C_{1,1} & i=1  & j=1 & k=0 & 0+(\color{#D4A017}A_{1,0}​ \color{white}\times \color{magenta}B_{0,1} \color{white})​= 0+ (\color{#D4A017}4 \times \color{magenta}8\color{white}) = 32 \\
\hline
C_{1,1} & i=1  & j=1 & k=1 & 32 + (\color{#D4A017}A_{1,1}​ \color{white}\times \color{magenta}B_{1,1}\color{white}) ​= 32+(\color{#D4A017}5 \times \color{magenta}10\color{white}) = 82 \\
\hline
C_{1,1} & i=1  & j=1 & k=2 & 73 + (\color{#D4A017}A_{1,2}​ \color{white}\times \color{magenta}B_{2,0}\color{white}) ​= 73+(\color{#D4A017}6 \times \color{magenta}12\color{white}) = 154 \\
\end{array}
\implies
\begin{bmatrix}
58 & 64 \\\
139 & \mathbf{\color{lightgray}154}
\end{bmatrix}
\end{array}
$$

#### The Optimized Implementation (IKJ)
We have seen the naive implementation and how the math unfolds. While the naive version is mathematically intuitive, it is a nightmare to work with for the following reasons:
1. In the standard implementation, to calculate one element, the CPU has to jump across different rows of Matrix $B$ (`other.data[k * b_cols + j]`). Because memory itself is a one-dimensional array, jumping between rows means the CPU has to constantly fetch new data from the slow RAM into its fast Cache.
1. Modern CPU cores use SIMD (Single Instruction, Multiple Data) to perform the same operation on multiple values simultaneously as long as the operations can be performed independently of each other. The naive implementation is sequential. So, it cannot leverage the parallel processing power of the CPU.

To avoid these two problems, we can re-arrange the multiplication code a little bit and it will boost performance significantly. 

```rust
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
```

Instead of the standard $i \xrightarrow{} j \xrightarrow{} k$ loop order, if we switch to $i \xrightarrow{} k \xrightarrow{} j$ we can unlock two major hardware optimizations:

1. **Improved Cache Locality:** In the IKJ order, the innermost loop moves across index `j`. This means we are reading `other.data` and writing to data in a straight, continuous line. The CPU can predict this "streaming" access and pre-fetch the data into the cache avoiding the RAM fetch.

1. **Autovectorization (SIMD) Potential:** Because we are operating on contiguous slices of memory in the inner loop, the Rust compiler can easily apply SIMD. It can load 4 or 8 values from Matrix $B$, multiply them by the constant `aik` in one go, and add them to the result slice in a single CPU instruction.

We have an intuition how it will work under the hood but we also need to make sure that the mathematics involved is intact and we end up in same result. Let's verify the mathematics in this case to ensure we are not missing any crucial point:

$$
A = \begin{bmatrix} \color{#2ECC71}1 & \color{#2ECC71}2 & \color{#2ECC71}3 \\\ \color{#D4A017}4 & \color{#D4A017}5 & \color{#D4A017}6 \end{bmatrix}, 
B = \begin{bmatrix} \color{cyan}7 & \color{magenta}8 \\\ \color{cyan}9 & \color{magenta}10 \\\ \color{cyan}11 & \color{magenta}12 \end{bmatrix}
$$

##### Processing Row $i = 0$ (First row of A)
We work on the first row of the result $C$. The inner loop $j$ updates the entire row slice at once.
        
$$
\begin{array}{}
\begin{array}{c|c|c|c|}
C_{row 0} & k = 0 & C_{row 0} + (A_{0,0} \times B_{row0}) & [0, 0] + \color{#2ECC71}1 \color{white}\times [\color{cyan}7, \color{magenta}8\color{white}] = [7, 8] \\
\hline
C_{row 0} & k = 1 & C_{row 0} + (A_{0,1} \times B_{row1}) & [7, 8] + \color{#2ECC71}2 \color{white}\times [\color{cyan}9, \color{magenta}10\color{white}] = [7+18, 8+20] = [25, 28] \\
\hline
C_{row 0} & k = 2 & C_{row 0} + (A_{0,2} \times B_{row2}) & [25, 28] + \color{#2ECC71}3 \times [\color{cyan}11, \color{magenta}12\color{white}] = [25+33, 28+36] = \mathbf{[58, 64]}
\end{array}
\implies
\begin{bmatrix}
\mathbf{\color{lightgray}{58}} & \mathbf{\color{lightgray}{64}} \\\
0 & 0
\end{bmatrix}
\end{array}
$$

##### Processing Row $i = 1$ (Second row of A)
We move to the second row of our result $C$.

$$
\begin{array}{}
\begin{array}{c|c|c|c}
C_{row 1} & k = 0 & C_{row 1} + (A_{1,0} \times B_{row0}) & [0, 0] + \color{#D4A017}4 \times [\color{cyan}7, \color{magenta}8\color{white}] = [28, 32] \\
\hline
C_{row 1} & k = 1 & C_{row 1} + (A_{1,1} \times B_{row1}) & [28, 32] + \color{#D4A017}5 \times [\color{cyan}9, \color{magenta}10\color{white}] = [28+45, 32+50] = [73, 82] \\
\hline
C_{row 1} & k = 2 & C_{row 1} + (A_{1,2} \times B_{row2}) & [73, 82] + \color{#D4A017}6 \times [\color{cyan}11, \color{magenta}12]\color{white} = [73+66, 82+72] = \mathbf{[139, 154]} \\
\end{array}
\implies
\begin{bmatrix}
58 & 64 \\\
\mathbf{\color{lightgray}139} & \mathbf{\color{lightgray}154}
\end{bmatrix}
\end{array}
$$
 
#### Full Implementation
Here is the full implementation of the optimized method:

```rust
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

        Ok(Tensor { data, shape: out_shape })
 }
 ```

 Now we have both the methods, let's run some accuracy check and performance check:

 ```text
$ target/release/build-your-own-nn 
Input Tensor A:
  |  1.0000,   2.0000,   3.0000|
  |  4.0000,   5.0000,   6.0000|

Input Tensor B:
  |  7.0000,   8.0000|
  |  9.0000,  10.0000|
  | 11.0000,  12.0000|


Matrix Multiplication using naive method:
  | 58.0000,  64.0000|
  |139.0000, 154.0000|

Time taken (naive): 61.729µs

Matrix Multiplication using optimized method:
  | 58.0000,  64.0000|
  |139.0000, 154.0000|

Time taken (optimized): 12.845µs
 ```
We can clearly see that the optimized method performs the task faster than the naive method, that too for very small input tensors. If we increase the number of rows and columns in both the input matrices, we'll see a much larger benefit of using the optimized method:

```text
$ target/release/build-your-own-nn 
Input Tensor A Dimensions:
[50, 60]
Input Tensor B Dimensions:
[60, 40]

Matrix Multiplication using naive method:
Time taken (naive): 396.712µs

Matrix Multiplication using optimized method:
Time taken (optimized): 26.23µs

Results match!
```
Here is how both the methods performed the calculations:

```text
Matrix Multiplication using naive method:
    Processing row 0
        Processing column 0
            Multiplying A[0,0] = 1 with B[0,0] = 7
            Multiplying A[0,1] = 2 with B[1,0] = 9
            Multiplying A[0,2] = 3 with B[2,0] = 11
        Completed row 0, column 0, data now [58.0, 0.0, 0.0, 0.0]
        
        Processing column 1
            Multiplying A[0,0] = 1 with B[0,1] = 8
            Multiplying A[0,1] = 2 with B[1,1] = 10
            Multiplying A[0,2] = 3 with B[2,1] = 12
        Completed row 0, column 1, data now [58.0, 64.0, 0.0, 0.0]
    
    Completed row 0, data now [58.0, 64.0, 0.0, 0.0]
    
    Processing row 1
        Processing column 0
            Multiplying A[1,0] = 4 with B[0,0] = 7
            Multiplying A[1,1] = 5 with B[1,0] = 9
            Multiplying A[1,2] = 6 with B[2,0] = 11
        Completed row 1, column 0, data now [58.0, 64.0, 139.0, 0.0]
        
        Processing column 1
            Multiplying A[1,0] = 4 with B[0,1] = 8
            Multiplying A[1,1] = 5 with B[1,1] = 10
            Multiplying A[1,2] = 6 with B[2,1] = 12
        Completed row 1, column 1, data now [58.0, 64.0, 139.0, 154.0]
    
    Completed row 1, data now [58.0, 64.0, 139.0, 154.0]

Final Result:
    | 58.0000,  64.0000|
    |139.0000, 154.0000|

Matrix Multiplication using optimized method:
    Processing row 0 of A
        Multiplying A[0,0] = 1 with row 0 of B
            Adding 1 * 7 to output position (0,0)
            Adding 1 * 8 to output position (0,1)
        Completed processing A[0,0], output row now [7.0, 8.0]
        
        Multiplying A[0,1] = 2 with row 1 of B
            Adding 2 * 9 to output position (0,0)
            Adding 2 * 10 to output position (0,1)
        Completed processing A[0,1], output row now [25.0, 28.0]
        
        Multiplying A[0,2] = 3 with row 2 of B
            Adding 3 * 11 to output position (0,0)
            Adding 3 * 12 to output position (0,1)
        Completed processing A[0,2], output row now [58.0, 64.0]
    
    Completed row 0 of A, data now [58.0, 64.0, 0.0, 0.0]
    
    Processing row 1 of A
        Multiplying A[1,0] = 4 with row 0 of B
            Adding 4 * 7 to output position (1,0)
            Adding 4 * 8 to output position (1,1)
        Completed processing A[1,0], output row now [28.0, 32.0]
        
        Multiplying A[1,1] = 5 with row 1 of B
            Adding 5 * 9 to output position (1,0)
            Adding 5 * 10 to output position (1,1)
        Completed processing A[1,1], output row now [73.0, 82.0]
        
        Multiplying A[1,2] = 6 with row 2 of B
            Adding 6 * 11 to output position (1,0)
            Adding 6 * 12 to output position (1,1)
        Completed processing A[1,2], output row now [139.0, 154.0]
    
    Completed row 1 of A, data now [58.0, 64.0, 139.0, 154.0]

Final Result:
  | 58.0000,  64.0000|
  |139.0000, 154.0000|

```
>**NOTE** 
>We use raw loops here for educational clarity, though Rust iterators can offer similar or better performance via bounds-check elimination. If we switch to `chunk`, we can even squeeze some more performance.


## Reduction
A matrix or a vector gives us information about individual elements, but at times we need an aggregation of those individual elements.

Let's look at an example of a matrix which represents sales records of cars in the last three months:

$$
\begin{array}{c|ccc}
\mathbf {} & \mathbf{Maruti} & \mathbf{Hyundai} & \mathbf{Toyota} \\
\hline
Oct  & 1000 & 2000 & 3000 \\
Nov  & 1200 & 1800 & 2000 \\
Dec  & 1500 & 2500 & 2200 \\
\end{array}
$$

This individual representation is great for individual sales of a particular brand in a particular month.

However, if we need to know how many cars were sold in October or how many Maruti cars were sold in the last three months, we need to reduce all the row-wise or column-wise entries into a single number. This operation is known as **Reduction**.

Using reduction we can represent this:

$$
\begin{array}{c|ccc|c}
{} & \mathbf{Maruti} & \mathbf{Hyundai} & \mathbf{Toyota} & \mathbf{Monthly\ Total} \\
\hline
Oct  & 1000 & 2000 & 3000 & 6000 \\
Nov  & 1200 & 1800 & 2000 & 5000 \\
Dec  & 1500 & 2500 & 2200 & 6200 \\
\hline
Brand\ Total  & 3700 & 6300 & 7200 & \\
\end{array}
$$

The 'Brand Total' is a column wise (later represented as Axis 0 sum) reduction and the 'Monthly Total' is a row wise (later represented as Axis 1 sum) reduction.

If we sum across the rows first and then do another sum of the resulting vector, it will result in the grand sum (the bottom right corner '17200'). This sums up every element in the whole matrix into a single scalar value.

$$
\begin{array}{c|ccc|c}
\mathbf {} & \mathbf{Maruti} & \mathbf{Hyundai} & \mathbf{Toyota} & \mathbf{Monthly\ Total} \\
\hline
Oct  & 1000 & 2000 & 3000 & 6000 \\
Nov  & 1200 & 1800 & 2000 & 5000 \\
Dec  & 1500 & 2500 & 2200 & 6200 \\
\hline
\mathbf{Brand\ Total}  & 3700 & 6300 & 7200 & \mathbf{\color{green}17200} \\
\end{array}
$$

### Implementation
We are almost coming to an end to our tensor journey. The only remaining tensor operation we'll implement is a sum reducer.

Following our mathematical definitions, let's start defining our method first. We should be able to sum across rows or columns or reduce the whole tensor into a single scalar value. We would need the axis on which to sum but for a global sum, we don't have anything to pass. We will use `Option` type for the `axis` parameter and we will return a tensor object.

Let's put the definition in a function in the existing tensor `impl`

```rust
pub fn sum(&self, axis: Option<usize>) -> Result<Tensor, TensorError> {
    todo!()
}
```

We now have the definition ready, let's start writing a few tests. The following set of tests should cover most of the cases.

```rust
fn setup_matrix_for_reduction() -> Tensor {
        let data = vec![
            1000.0, 2000.0, 3000.0, 1200.0, 1800.0, 2000.0, 1500.0, 2500.0, 2200.0,
        ];
        Tensor::new(data, vec![3, 3]).expect("Failed to create matrix for reduction tests")
    }

    #[test]
    fn test_reduce_sum_global() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(None).unwrap();

        assert_eq!(res.data(), &[17200.0]);
        assert_eq!(res.shape(), &[1]);
    }

    #[test]
    fn test_reduce_sum_axis_0_brand_total() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(Some(0)).unwrap();

        assert_eq!(res.data(), &[3700.0, 6300.0, 7200.0]);
        assert_eq!(res.shape(), &[3]);
    }

    #[test]
    fn test_reduce_sum_axis_1_monthly_total() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(Some(1)).unwrap();

        assert_eq!(res.data(), &[6000.0, 5000.0, 6200.0]);
        assert_eq!(res.shape(), &[3]);
    }

    #[test]
    fn test_reduce_sum_1d_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let res = tensor.sum(Some(0)).unwrap();

        assert_eq!(res.data(), &[6.0]);
        assert_eq!(res.shape(), &[1]);
    }

    #[test]
    fn test_reduce_sum_invalid_axis() {
        let tensor = setup_matrix_for_reduction();
        let res = tensor.sum(Some(2));

        assert_eq!(res.err(), Some(TensorError::InvalidRank));
    }

    #[test]
    fn test_reduce_sum_empty() {
        let tensor = Tensor::new(vec![], vec![0]).unwrap();
        let res = tensor.sum(Some(2));

        assert_eq!(res.err(), Some(TensorError::InvalidRank));
    }
```

Now let's complete the implementation of reduction operation and run the tests

```rust
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
```

That's all the mathematics that we care for now and all the implementations are completed. Next we'll be able to dive deep into our first ML algorithm which we'll use to train a model to learn from data.

