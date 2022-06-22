use cuda_builder::CudaBuilder;

fn main() {
    println!("Hello, building!");
    CudaBuilder::new("../gpu")
        .release(true)
        .copy_to("../resources/kernels.ptx")
        .build()
        .unwrap();
}