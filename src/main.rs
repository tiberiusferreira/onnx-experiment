use onnx_experiment::{ConcreteF32Tensor, ConcreteI32Tensor, ModelBuilder};

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
    let _label = model.add_i32_input(ConcreteI32Tensor {
        name: "labels".to_string(),
        data: vec![1, 2],
        shape: vec![2],
    });

    let re = model.add(&a, &b);
    let w = model.get_val_of(&re);
    println!("{:?}", w);
    // let output = model.add_operation(re);
    //
    // let l1_op = create_cross_entropy_op(output, label, "new_out");
    // let new_out = model.add_operation(l1_op);
    //
    // model.add_output(new_out.clone());

    model.serialize_to_file();
}
