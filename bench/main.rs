#![feature(test)]

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, black_box, Bencher};
use std::time::Duration;
use nanorand::{Rng, WyRand};
use crate::matmul_cpu::matmul_gpu::matmul_gpu;

#[path = "../cpu/src/mod.rs"]
mod matmul_cpu;

#[path = "../cpu/src/mod.rs"]
mod matmul_gpu;


const MEASUREMENT_SECS: u64 = 10;

fn mul_bench(ben: &mut Bencher, n: usize) {

    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; n];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; n];
    wyrand.fill(&mut rhs);
    // let mut out = vec![0.0; n * n];
    ben.iter(|| {
        black_box(matmul_gpu(&lhs, &rhs));
    })
}


fn mul_run(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("mul");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    // group.warm_up_time(Duration::from_secs(0));

    let ns = [100*100, 200*200, 500*500, 1000*1000];
    for n in ns.iter() {
        group.bench_with_input(BenchmarkId::new("cuda", n), n, |ben, &n| mul_bench(ben, n));
        // group.bench_with_input(BenchmarkId::new("na", n), n, |ben, &n| na_mul_bench(ben, n));
        // group.bench_with_input(BenchmarkId::new("mut_linear", n), n, |ben, &n| mut_linear_mul(ben, n));
    }
    group.finish();
}

criterion_group!(benches, mul_run);
criterion_main!(benches);



