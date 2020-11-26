use crate::onnx_proto_structs::NodeProto;
use crate::{ModelBuilder, PlaceholderF32Tensor};

impl ModelBuilder {
    pub fn add(
        &mut self,
        left: &PlaceholderF32Tensor,
        right: &PlaceholderF32Tensor,
    ) -> PlaceholderF32Tensor {
        let output_name = left.name.clone() + &right.name;
        let operation = NodeProto {
            input: vec![left.name.clone(), right.name.clone()],
            output: vec![output_name.clone()],
            name: "".to_string(),
            op_type: "Add".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            doc_string: "".to_string(),
        };
        self.model.graph.as_mut().unwrap().node.push(operation);
        PlaceholderF32Tensor {
            name: output_name,
            shape: left.shape.clone(),
        }
    }
}
