use cuda_builder::CudaBuilder;

fn main() {
    let builder = CudaBuilder::new("gpu")
        .copy_to("resources/kernels.ptx")
        .release(true);
    println!("BUILDER RELEASE: {}", builder.release);
    builder.build().unwrap();
}