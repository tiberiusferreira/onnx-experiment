use crate::onnx_proto_structs::tensor_shape_proto::Dimension;
use crate::onnx_proto_structs::type_proto::{Tensor, Value};
use crate::onnx_proto_structs::{
    AttributeProto, NodeProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use crate::{ModelBuilder, PlaceholderF32Tensor, PlaceholderI32Tensor};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

fn create_binary_op(
    onnx_op_type_name: &str,
    inputs: Vec<&str>,
    attributes: Vec<AttributeProto>,
) -> (NodeProto, String) {
    let rand_string: String = thread_rng().sample_iter(&Alphanumeric).take(10).collect();
    (
        NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![rand_string.clone()],
            name: "".to_string(),
            op_type: onnx_op_type_name.to_string(),
            domain: "".to_string(),
            attribute: attributes,
            doc_string: "".to_string(),
        },
        rand_string,
    )
}

fn create_unary_op(
    onnx_op_type_name: &str,
    input: &str,
    attributes: Vec<AttributeProto>,
) -> (NodeProto, String) {
    let rand_string: String = thread_rng().sample_iter(&Alphanumeric).take(10).collect();
    (
        NodeProto {
            input: vec![input.to_string()],
            output: vec![rand_string.clone()],
            name: "".to_string(),
            op_type: onnx_op_type_name.to_string(),
            domain: "".to_string(),
            attribute: attributes,
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
        let (op, output_name) = create_binary_op("Add", vec![&left.name, &right.name], vec![]);
        self.graph_mut().node.push(op);
        PlaceholderF32Tensor {
            name: output_name,
            shape: left.shape.clone(),
        }
    }

    pub fn sigmoid(&mut self, input: &PlaceholderF32Tensor) -> PlaceholderF32Tensor {
        let (op, output_name) = create_unary_op("Sigmoid", &input.name, vec![]);
        self.graph_mut().node.push(op);
        PlaceholderF32Tensor {
            name: output_name,
            shape: input.shape.clone(),
        }
    }

    pub fn matmul(
        &mut self,
        left: &PlaceholderF32Tensor,
        right: &PlaceholderF32Tensor,
    ) -> PlaceholderF32Tensor {
        // Lets keep things simple for now, for now we only accept
        // [NxAxB] x [BxC] = [NxAxC]
        let error = format!(
            "Cant multiply shapes: {:?} and {:?}",
            left.shape, right.shape
        );
        assert!(left.shape.len() == 3 || left.shape.len() == 2, "{}", &error);
        assert_eq!(right.shape.len(), 2, "{}", &error);
        // Get last 2 dim of left and right
        let last_two_dim_left = &left.shape.as_slice()[left.shape.len() - 2..];
        let last_two_dim_right = &right.shape.as_slice()[right.shape.len() - 2..];
        assert_eq!(
            last_two_dim_left[1], last_two_dim_right[0],
            "Shapes don't match for matmul: left: {:?} right: {:?}",
            left.shape, right.shape
        );
        let (op, output_name) = create_binary_op("MatMul", vec![&left.name, &right.name], vec![]);
        self.graph_mut().node.push(op);
        let mut out_shape = left.shape.clone();
        let out_shape_len = out_shape.len();
        out_shape[out_shape_len - 1] = last_two_dim_right[1];
        out_shape[out_shape_len - 2] = last_two_dim_left[0];
        PlaceholderF32Tensor {
            name: output_name,
            shape: out_shape,
        }
    }

    pub fn cross_entropy(
        &mut self,
        input: &PlaceholderF32Tensor,
        target: &PlaceholderI32Tensor,
    ) -> PlaceholderF32Tensor {
        assert!(
            input.shape.len() == 2 && target.shape.len() == 1 && input.shape[0] == target.shape[0],
            "Shapes don't match for cross_entropy: Input tensor needs to be of rank 2 and shape \
            [N, C] and Target of shape [N], was: Input: {:?} Target: {:?}",
            input.shape,
            target.shape
        );
        let (op, output_name) = create_binary_op(
            "SoftmaxCrossEntropyLoss",
            vec![&input.name, &target.name],
            vec![string_attr("reduction", "mean")],
        );
        self.graph_mut().node.push(op);
        PlaceholderF32Tensor {
            name: output_name,
            shape: vec![],
        }
    }

    pub fn grad_in_train(
        &mut self,
        input: &PlaceholderF32Tensor,
        deriv_wrt: &PlaceholderF32Tensor,
    ) -> PlaceholderF32Tensor {
        let dims: Vec<Dimension> = input.shape.iter().map(|&e| e.into()).collect();
        let output = ValueInfoProto {
            name: "random_str".to_string(),
            r#type: Some(TypeProto {
                denotation: "".to_string(),
                value: Some(Value::TensorType(Tensor {
                    elem_type: 1,
                    shape: Some(TensorShapeProto { dim: dims }),
                })),
            }),
            doc_string: "".to_string(),
        };
        let operation = NodeProto {
            input: vec![input.name.clone()],
            output: vec![output.name.clone()],
            name: "".to_string(),
            op_type: "Gradient".to_string(),
            domain: "ai.onnx.preview.training".to_string(),
            attribute: vec![
                string_vec_attr("xs", vec![&input.name]),
                string_vec_attr("y", vec![&deriv_wrt.name]),
            ],
            doc_string: "".to_string(),
        };
        self.model
            .training_info
            .first_mut()
            .unwrap()
            .algorithm
            .as_mut()
            .unwrap()
            .node
            .push(operation);
        PlaceholderF32Tensor {
            name: "random_str".to_string(),
            shape: input.shape.clone(),
        }
    }

    pub fn add_in_train(
        &mut self,
        left: &PlaceholderF32Tensor,
        right: &PlaceholderF32Tensor,
    ) -> PlaceholderF32Tensor {
        assert_eq!(
            left.shape, right.shape,
            "Shapes don't match for add: left: {:?} right: {:?}",
            left.shape, right.shape
        );
        let (op, output_name) = create_binary_op("Add", vec![&left.name, &right.name], vec![]);
        self.model
            .training_info
            .first_mut()
            .unwrap()
            .algorithm
            .as_mut()
            .unwrap()
            .node
            .push(op);
        PlaceholderF32Tensor {
            name: output_name,
            shape: left.shape.clone(),
        }
    }
}

pub fn string_vec_attr(name: &str, val: Vec<&str>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ref_attr_name: "".to_string(),
        doc_string: "".to_string(),
        r#type: 8,
        f: 0.0,
        i: 0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        floats: vec![],
        ints: vec![],
        strings: val.iter().map(|s| s.to_string().into_bytes()).collect(),
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
    }
}

pub fn string_attr(name: &str, val: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ref_attr_name: "".to_string(),
        doc_string: "".to_string(),
        r#type: 3,
        f: 0.0,
        i: 0,
        s: val.to_string().into_bytes(),
        t: None,
        g: None,
        sparse_tensor: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
    }
}

#[allow(unused)]
pub fn int_attr(name: &str, val: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ref_attr_name: "".to_string(),
        doc_string: "".to_string(),
        r#type: 2,
        f: 0.0,
        i: val,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
    }
}

#[allow(unused)]
pub fn int_vec_attr(name: &str, val: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ref_attr_name: "".to_string(),
        doc_string: "".to_string(),
        r#type: 7,
        f: 0.0,
        i: 0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        floats: vec![],
        ints: val,
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
    }
}
