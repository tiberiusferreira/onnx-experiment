use onnx_experiment::{
    create_add_op, create_cross_entropy_op, ConcreteF32Tensor, ConcreteI32Tensor, ModelBuilder,
};

fn main() {
    let mut model = ModelBuilder::new();
    let a = model.add_input(ConcreteF32Tensor {
        name: "left".to_string(),
        data: vec![1., 2., 3., 4.],
        shape: vec![2, 2],
    });
    let b = model.add_input(ConcreteF32Tensor {
        name: "right".to_string(),
        data: vec![1., 2., 3., 4.],
        shape: vec![2, 2],
    });
    let label = model.add_i32_input(ConcreteI32Tensor {
        name: "labels".to_string(),
        data: vec![1, 2],
        shape: vec![2],
    });

    // let re = create_add_op(a.clone(), b, "out");
    // let output = model.add_operation(re);
    //
    // let l1_op = create_cross_entropy_op(output, label, "new_out");
    // let new_out = model.add_operation(l1_op);
    //
    // model.add_output(new_out.clone());

    model.serialize_to_file();
}
