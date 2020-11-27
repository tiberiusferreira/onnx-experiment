use crate::onnx_proto_structs::NodeProto;
use crate::{ModelBuilder, PlaceholderF32Tensor};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

fn create_binary_op(onnx_op_type_name: &str, inputs: Vec<&str>) -> (NodeProto, String) {
    let rand_string: String = thread_rng().sample_iter(&Alphanumeric).take(10).collect();
    (
        NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![rand_string.clone()],
            name: "".to_string(),
            op_type: onnx_op_type_name.to_string(),
            domain: "".to_string(),
            attribute: vec![],
            doc_string: "".to_string(),
        },
        rand_string,
    )
}

impl ModelBuilder {
    pub fn add(
        &mut self,
        left: &PlaceholderF32Tensor,
        right: &PlaceholderF32Tensor,
    ) -> PlaceholderF32Tensor {
        assert_eq!(
            left.shape, right.shape,
            "Shapes don't match for add: left: {:?} right: {:?}",
            left.shape, right.shape
        );
        let (op, output_name) = create_binary_op("Add", vec![&left.name, &right.name]);
        self.graph_mut().node.push(op);
        PlaceholderF32Tensor {
            name: output_name,
            shape: left.shape.clone(),
        }
    }
}
