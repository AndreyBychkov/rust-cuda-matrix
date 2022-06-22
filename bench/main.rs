#![feature(test)]

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, black_box, Bencher};
use std::time::Duration;
use nanorand::{Rng, WyRand};
use cust::prelude::*;
use std::error::Error;
use std::time::Instant;


static PTX: &str = include_str!("../resources/kernels.ptx");

const MEASUREMENT_SECS: u64 = 10;

fn mul_bench(ben: &mut Bencher, n: usize) {
    let n2 = n * n;
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; n2];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; n2];
    wyrand.fill(&mut rhs);
    let mut out = vec![0.0f32; n2];


    let _ctx = cust::quick_init().unwrap();
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();


    let func = module.get_function("matmul").unwrap();
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

    let grid_size = (n2 as u32 + block_size - 1) / block_size;

    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );

    stream.synchronize().unwrap();

    ben.iter(|| {
        black_box(unsafe {
            let lhs_gpu = lhs.as_slice().as_dbuf().unwrap();
            let rhs_gpu = rhs.as_slice().as_dbuf().unwrap();

            let out_buf = DeviceBuffer::<f32>::uninitialized(n2).unwrap();
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lhs_gpu.as_device_ptr(),
                    lhs_gpu.len(),
                    rhs_gpu.as_device_ptr(),
                    rhs_gpu.len(),
                    out_buf.as_device_ptr(),
                    n
                )
            ).unwrap();
            stream.synchronize().unwrap();
            out_buf.copy_to(&mut out).unwrap();
            stream.synchronize().unwrap();
        })
    })
}


fn mul_run(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("mul");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    let ns = [1024, 1024 * 2, 1024 * 4, 1024 * 8, 1024 * 16];
    for n in ns.iter() {
        group.bench_with_input(BenchmarkId::new("cuda_full", n), n, |ben, &n| mul_bench(ben, n));
    }
    group.finish();
}

criterion_group!(benches, mul_run);
criterion_main!(benches);



