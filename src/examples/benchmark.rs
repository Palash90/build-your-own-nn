use std::time::Instant;

use crate::tensor::Tensor;

pub fn run_benchmark() {
    let size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

    println!();

    for s in size {
        let a_data = vec![1.0; s * s];
        let b_data = vec![2.0; s * s];

        let tensor_a = Tensor::new(a_data, vec![s, s]).unwrap();
        let tensor_b = Tensor::new(b_data, vec![s, s]).unwrap();

        println!("Benchmarking {}x{} Matrix Multiplication...", s, s);

        // 2. Benchmark Naive Method
        let start_naive = Instant::now();
        let _res_naive = tensor_a.matmul_naive(&tensor_b).expect("Naive failed");
        let duration_naive = start_naive.elapsed();
        println!("Time taken (naive):     {:?}", duration_naive);

        // 3. Benchmark Optimized Method
        let start_opt = Instant::now();
        let _res_opt = tensor_a.matmul(&tensor_b).expect("Optimized failed");
        let duration_opt = start_opt.elapsed();
        println!("Time taken (optimized): {:?}", duration_opt);

        assert_eq!(_res_naive, _res_opt);

        println!("Results match!");

        // 4. Calculate Speedup
        let speedup = duration_naive.as_secs_f64() / duration_opt.as_secs_f64();
        println!("Speedup factor:         {:.2}x faster", speedup);
        println!();
    }
}
