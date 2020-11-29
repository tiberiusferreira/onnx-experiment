use mnist::{Mnist, MnistBuilder, NormalizedMnist};
use onnx_experiment::{F32Tensor, I32Tensor, ModelBuilder, PlaceholderF32Tensor};

/// Example:
///   input     weight    output
/// [128x20] x [20x30] = [128x30]
pub fn dense_layer(
    model: &mut ModelBuilder,
    input: &PlaceholderF32Tensor,
    out_last_dim: i64,
) -> PlaceholderF32Tensor {
    let in_shape = input.shape[1];
    let number_el_weights = in_shape * out_last_dim;
    let weights = (0..number_el_weights)
        .into_iter()
        .map(|_| 0.001 as f32)
        .collect();
    let weights = model.add_initializer(F32Tensor::new(weights, vec![in_shape, out_last_dim]));
    model.matmul(&input, &weights)
}

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let NormalizedMnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(100)
        .test_set_length(100)
        .finalize()
        .normalize();
    let mut model = ModelBuilder::new();
    let img = model.add_input(F32Tensor::new(trn_img[0..784].to_vec(), vec![1, 784]));
    println!("{:?}", I32Tensor::new(vec![trn_lbl[0] as i32], vec![1]));
    let label = model.add_i32_input(I32Tensor::new(vec![trn_lbl[0] as i32], vec![1]));

    // let a = model.add_input(F32Tensor::new(vec![1., 20., 30., 40.], vec![2, 2]));
    // let a = model.add_input(F32Tensor::new(vec![1., 20., 30., 40.], vec![2, 2]));
    // let b = model.add_input(F32Tensor::new(vec![10., 20., 3., 40.], vec![2, 2]));
    // let label = model.add_i32_input(I32Tensor::new(vec![2, 2], vec![2]));

    // let add_res = model.add(&a, &b);
    let matmul_res = dense_layer(&mut model, &img, 10);
    let sigmoid_res = model.sigmoid(&matmul_res);
    // let matmul_res = model.matmul(&add_res, &weights);
    println!("{:?}", sigmoid_res);
    let loss = model.cross_entropy(&sigmoid_res, &label);

    for _i in 0..20000 {
        let w = model.train_get_loss(&loss);
        println!("{:?}", w);
        println!("{:?}", model.get_val_of_f32(&sigmoid_res));
    }
}
