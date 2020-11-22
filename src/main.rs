mod onnx_structs;
use crate::onnx_structs::tensor_shape_proto::Dimension;
use crate::onnx_structs::type_proto::{Tensor, Value};
use onnx_structs::*;
use prost::Message;
use std::io::Write;

pub struct ConcreteTensor {
    data: Vec<f32>,
    shape: Vec<i64>,
}

pub struct ModelBuilder {
    model: onnx_structs::ModelProto,
}

impl From<i64> for tensor_shape_proto::Dimension {
    fn from(input: i64) -> Self {
        tensor_shape_proto::Dimension {
            denotation: "".to_string(),
            value: Some(tensor_shape_proto::dimension::Value::DimValue(input)),
        }
    }
}

impl From<tensor_shape_proto::Dimension> for i64 {
    fn from(input: tensor_shape_proto::Dimension) -> Self {
        match input.value.as_ref().unwrap() {
            tensor_shape_proto::dimension::Value::DimValue(val) => *val,
            tensor_shape_proto::dimension::Value::DimParam(_) => {
                panic!()
            }
        }
    }
}

pub trait TensorDescriptor {
    fn name(&self) -> String;
    fn shape(&self) -> Vec<i64>;
}

impl TensorDescriptor for ValueInfoProto {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn shape(&self) -> Vec<i64> {
        match self.r#type.as_ref().unwrap().value.as_ref().unwrap() {
            Value::TensorType(tensor) => tensor
                .shape
                .as_ref()
                .unwrap()
                .dim
                .iter()
                .map(|dim| match dim.value.as_ref().unwrap() {
                    onnx_structs::tensor_shape_proto::dimension::Value::DimValue(val) => *val,
                    onnx_structs::tensor_shape_proto::dimension::Value::DimParam(_) => {
                        panic!()
                    }
                })
                .collect(),
            _ => {
                panic!()
            }
        }
    }
}

impl ModelBuilder {
    pub fn new() -> Self {
        let graph = onnx_structs::GraphProto {
            node: vec![],
            name: "rust-graph".to_string(),
            initializer: vec![],
            sparse_initializer: vec![],
            doc_string: "".to_string(),
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
        };
        ModelBuilder {
            model: ModelProto {
                ir_version: 7,
                opset_import: vec![OperatorSetIdProto {
                    domain: "".to_string(),
                    version: 13,
                }],
                producer_name: "rust".to_string(),
                producer_version: "0.1.0".to_string(),
                domain: "".to_string(),
                model_version: 0,
                doc_string: "".to_string(),
                graph: Some(graph),
                metadata_props: vec![],
                training_info: vec![],
            },
        }
    }
    pub fn add_input(&mut self, input: ConcreteTensor, name: &str) -> onnx_structs::ValueInfoProto {
        let dims: Vec<Dimension> = input
            .shape
            .iter()
            .map(|dim| Dimension {
                denotation: "".to_string(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
            })
            .collect();
        self.model
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .push(TensorProto {
                dims: input.shape,
                data_type: 1, // float
                segment: None,
                float_data: input.data,
                int32_data: vec![],
                string_data: vec![],
                int64_data: vec![],
                name: name.to_string(),
                doc_string: "".to_string(),
                raw_data: vec![],
                external_data: vec![],
                data_location: 0,
                double_data: vec![],
                uint64_data: vec![],
            });
        let input = onnx_structs::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_structs::TypeProto {
                denotation: "".to_string(),
                value: Some(onnx_structs::type_proto::Value::TensorType(
                    onnx_structs::type_proto::Tensor {
                        elem_type: 1,
                        shape: Some(TensorShapeProto { dim: dims }),
                    },
                )),
            }),
            doc_string: "".to_string(),
        };
        self.model.graph.as_mut().unwrap().input.push(input.clone());
        input
    }

    pub fn add_node(&mut self, operation: OperationSingleOutput) -> ValueInfoProto {
        self.model
            .graph
            .as_mut()
            .unwrap()
            .value_info
            .push(operation.output_description.clone());
        self.model
            .graph
            .as_mut()
            .unwrap()
            .node
            .push(operation.operation);
        operation.output_description
    }

    pub fn add_output(&mut self, node: ValueInfoProto) {
        self.model.graph.as_mut().unwrap().output.push(node);
    }

    pub fn serialize_to_file(&self) {
        let mut buf: Vec<u8> = Vec::new();
        buf.reserve(self.model.encoded_len());
        self.model.encode(&mut buf).unwrap();
        let mut out_file = std::fs::File::create("my_model.onnx").unwrap();
        out_file.write_all(&buf).unwrap();
    }
}

fn main() {
    let mut model = ModelBuilder::new();
    let a = model.add_input(
        ConcreteTensor {
            data: vec![1., 2., 3., 4.],
            shape: vec![2, 2],
        },
        "left",
    );
    let b = model.add_input(
        ConcreteTensor {
            data: vec![1., 2., 3., 4.],
            shape: vec![2, 2],
        },
        "right",
    );

    let re = create_add(a, b, "out");
    let output = model.add_node(re);
    model.add_output(output);

    model.serialize_to_file();
}

pub struct OperationSingleOutput {
    operation: NodeProto,
    output_description: ValueInfoProto,
}

pub fn create_add<T: TensorDescriptor>(left: T, right: T, output: &str) -> OperationSingleOutput {
    let dims: Vec<Dimension> = left.shape().iter().map(|&e| e.into()).collect();
    let output = ValueInfoProto {
        name: output.to_string(),
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
        input: vec![left.name(), right.name()],
        output: vec![output.name()],
        name: "".to_string(),
        op_type: "Add".to_string(),
        domain: "".to_string(),
        attribute: vec![],
        doc_string: "".to_string(),
    };
    OperationSingleOutput {
        operation,
        output_description: output,
    }
}
