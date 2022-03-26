use cuda_builder::CudaBuilder;

fn main() {
    println!("Hello, world!");
    CudaBuilder::new("gpu/add_gpu")
        .copy_to("resources/add.ptx")
        .build()
        .unwrap();
}
