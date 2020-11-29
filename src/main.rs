use onnx_experiment::{F32Tensor, I32Tensor, ModelBuilder};

fn main() {
    let mut model = ModelBuilder::new();
    let a = model.add_input(F32Tensor::new(vec![1., 20., 30., 40.], vec![2, 2]));
    let b = model.add_input(F32Tensor::new(vec![10., 20., 3., 40.], vec![2, 2]));
    let label = model.add_i32_input(I32Tensor::new(vec![2, 2], vec![2]));

    let weights = model.add_initializer(F32Tensor::new(vec![2., 3., 4., 5.], vec![2, 2]));

    let add_res = model.add(&a, &b);
    let matmul_res = model.matmul(&add_res, &weights);
    let loss = model.cross_entropy(&matmul_res, &label);

    for _i in 0..50 {
        let w = model.train_get_loss(&loss);
        println!("{:?}", w);
    }

    // let output = model.add_operation(re);
    //
    // let l1_op = create_cross_entropy_op(output, label, "new_out");
    // let new_out = model.add_operation(l1_op);
    //
    // model.add_output(new_out.clone());
}
