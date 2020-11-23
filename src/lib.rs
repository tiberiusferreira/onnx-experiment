use std::io::Write;
mod onnx_structs;
use crate::onnx_structs::tensor_shape_proto::Dimension;
use crate::onnx_structs::type_proto::{Tensor, Value};
use onnx_structs::*;
use prost::Message;

#[derive(Debug, Clone)]
pub struct ConcreteF32Tensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct ConcreteI32Tensor {
    pub name: String,
    pub data: Vec<i32>,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct PlaceholderF32Tensor {
    pub name: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct PlaceholderI32Tensor {
    pub name: String,
    pub shape: Vec<i64>,
}

impl From<TensorProto> for ConcreteF32Tensor {
    fn from(ten: TensorProto) -> Self {
        ConcreteF32Tensor {
            name: ten.name,
            data: ten.float_data,
            shape: ten.dims,
        }
    }
}

impl From<TensorProto> for ConcreteI32Tensor {
    fn from(ten: TensorProto) -> Self {
        ConcreteI32Tensor {
            name: ten.name,
            data: ten.int32_data,
            shape: ten.dims,
        }
    }
}

impl From<ConcreteF32Tensor> for TensorProto {
    fn from(ten: ConcreteF32Tensor) -> Self {
        TensorProto {
            dims: ten.shape,
            data_type: 1, // float
            segment: None,
            float_data: ten.data,
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
            name: ten.name,
            doc_string: "".to_string(),
            raw_data: vec![],
            external_data: vec![],
            data_location: 0,
            double_data: vec![],
            uint64_data: vec![],
        }
    }
}

impl From<ConcreteI32Tensor> for TensorProto {
    fn from(ten: ConcreteI32Tensor) -> Self {
        TensorProto {
            dims: ten.shape,
            data_type: 1, // float
            segment: None,
            float_data: vec![],
            int32_data: ten.data,
            string_data: vec![],
            int64_data: vec![],
            name: ten.name,
            doc_string: "".to_string(),
            raw_data: vec![],
            external_data: vec![],
            data_location: 0,
            double_data: vec![],
            uint64_data: vec![],
        }
    }
}

impl From<ConcreteF32Tensor> for ValueInfoProto {
    fn from(ten: ConcreteF32Tensor) -> Self {
        let dims: Vec<Dimension> = ten
            .shape
            .iter()
            .map(|dim| Dimension {
                denotation: "".to_string(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
            })
            .collect();
        ValueInfoProto {
            name: ten.name,
            r#type: Some(onnx_structs::TypeProto {
                denotation: "".to_string(),
                value: Some(onnx_structs::type_proto::Value::TensorType(
                    onnx_structs::type_proto::Tensor {
                        elem_type: 1, //float
                        shape: Some(TensorShapeProto { dim: dims }),
                    },
                )),
            }),
            doc_string: "".to_string(),
        }
    }
}

impl From<ConcreteI32Tensor> for ValueInfoProto {
    fn from(ten: ConcreteI32Tensor) -> Self {
        let dims: Vec<Dimension> = ten
            .shape
            .iter()
            .map(|dim| Dimension {
                denotation: "".to_string(),
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
            })
            .collect();
        ValueInfoProto {
            name: ten.name,
            r#type: Some(onnx_structs::TypeProto {
                denotation: "".to_string(),
                value: Some(onnx_structs::type_proto::Value::TensorType(
                    onnx_structs::type_proto::Tensor {
                        elem_type: 6, // i32
                        shape: Some(TensorShapeProto { dim: dims }),
                    },
                )),
            }),
            doc_string: "".to_string(),
        }
    }
}

impl From<ValueInfoProto> for PlaceholderF32Tensor {
    fn from(val: ValueInfoProto) -> Self {
        let a = val.r#type.unwrap().value.unwrap();
        let b = match a {
            Value::TensorType(val) => {
                if val.elem_type != 1 {
                    panic!("ValueInfoProto was not of type f32.")
                }
                val.shape.unwrap()
            }
            Value::SequenceType(_) => {
                panic!()
            }
            Value::MapType(_) => {
                panic!()
            }
        };
        let c: Vec<i64> = b
            .dim
            .iter()
            .map(|e| match e.value.as_ref().unwrap() {
                onnx_structs::tensor_shape_proto::dimension::Value::DimValue(dim) => *dim,
                onnx_structs::tensor_shape_proto::dimension::Value::DimParam(_) => {
                    panic!()
                }
            })
            .collect();
        PlaceholderF32Tensor {
            name: val.name,
            shape: c,
        }
    }
}

impl From<ValueInfoProto> for PlaceholderI32Tensor {
    fn from(val: ValueInfoProto) -> Self {
        let a = val.r#type.unwrap().value.unwrap();
        let b = match a {
            Value::TensorType(val) => {
                if val.elem_type != 6 {
                    panic!("ValueInfoProto was not of type i32.")
                }
                val.shape.unwrap()
            }
            Value::SequenceType(_) => {
                panic!()
            }
            Value::MapType(_) => {
                panic!()
            }
        };
        let c: Vec<i64> = b
            .dim
            .iter()
            .map(|e| match e.value.as_ref().unwrap() {
                onnx_structs::tensor_shape_proto::dimension::Value::DimValue(dim) => *dim,
                onnx_structs::tensor_shape_proto::dimension::Value::DimParam(_) => {
                    panic!()
                }
            })
            .collect();
        PlaceholderI32Tensor {
            name: val.name,
            shape: c,
        }
    }
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
                opset_import: vec![
                    OperatorSetIdProto {
                        domain: "".to_string(),
                        version: 12,
                    },
                    OperatorSetIdProto {
                        domain: "ai.onnx.preview.training".to_string(),
                        version: 1,
                    },
                ],
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

    fn add_initializer(&mut self, value: TensorProto) {
        self.model.graph.as_mut().unwrap().initializer.push(value);
    }

    pub fn add_input(&mut self, input: ConcreteF32Tensor) -> PlaceholderF32Tensor {
        self.add_initializer(TensorProto::from(input.clone()));
        let place_holder = onnx_structs::ValueInfoProto::from(input);
        self.model
            .graph
            .as_mut()
            .unwrap()
            .input
            .push(place_holder.clone());
        PlaceholderF32Tensor::from(place_holder)
    }

    pub fn add_i32_input(&mut self, input: ConcreteI32Tensor) -> PlaceholderI32Tensor {
        self.add_initializer(TensorProto::from(input.clone()));
        let place_holder = onnx_structs::ValueInfoProto::from(input);
        self.model
            .graph
            .as_mut()
            .unwrap()
            .input
            .push(place_holder.clone());
        PlaceholderI32Tensor::from(place_holder)
    }

    pub fn add_operation(&mut self, operation: OperationSingleOutput) -> ValueInfoProto {
        // self.model
        //     .graph
        //     .as_mut()
        //     .unwrap()
        //     .value_info
        //     .push(operation.output_description.clone());
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

pub struct OperationSingleOutput {
    operation: NodeProto,
    output_description: ValueInfoProto,
}

pub fn create_add_op<T: TensorDescriptor>(
    left: T,
    right: T,
    output: &str,
) -> OperationSingleOutput {
    // let dims: Vec<Dimension> = left.shape().iter().map(|&e| e.into()).collect();
    // let output = ValueInfoProto {
    //     name: output.to_string(),
    //     r#type: Some(TypeProto {
    //         denotation: "".to_string(),
    //         value: Some(Value::TensorType(Tensor {
    //             elem_type: 1,
    //             shape: Some(TensorShapeProto { dim: dims }),
    //         })),
    //     }),
    //     doc_string: "".to_string(),
    // };
    // let operation = NodeProto {
    //     input: vec![left.name(), right.name()],
    //     output: vec![output.name()],
    //     name: "".to_string(),
    //     op_type: "Add".to_string(),
    //     domain: "".to_string(),
    //     attribute: vec![],
    //     doc_string: "".to_string(),
    // };
    // OperationSingleOutput {
    //     operation,
    //     output_description: output,
    // }
    unimplemented!()
}

pub fn create_cross_entropy_op<T: TensorDescriptor>(
    left: T,
    right: T,
    output: &str,
) -> OperationSingleOutput {
    let output = ValueInfoProto {
        name: output.to_string(),
        r#type: Some(TypeProto {
            denotation: "".to_string(),
            value: Some(Value::TensorType(Tensor {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![Dimension::from(1)],
                }),
            })),
        }),
        doc_string: "".to_string(),
    };
    let operation = NodeProto {
        input: vec![left.name(), right.name()],
        output: vec![output.name()],
        name: "".to_string(),
        op_type: "SoftmaxCrossEntropyLoss".to_string(),
        domain: "".to_string(),
        attribute: vec![string_attr("reduction", "mean")],
        doc_string: "".to_string(),
    };
    OperationSingleOutput {
        operation,
        output_description: output,
    }
}

pub fn create_grad_op<T: TensorDescriptor>(
    input: T,
    deriv_wrt: T,
    deriv_name: &str,
) -> OperationSingleOutput {
    let dims: Vec<Dimension> = input.shape().iter().map(|&e| e.into()).collect();
    let output = ValueInfoProto {
        name: deriv_name.to_string(),
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
        input: vec![input.name()],
        output: vec![output.name()],
        name: "".to_string(),
        op_type: "Gradient".to_string(),
        domain: "ai.onnx.preview.training".to_string(),
        attribute: vec![
            string_vec_attr("xs", vec![&input.name()]),
            string_vec_attr("y", vec![&deriv_wrt.name()]),
        ],
        doc_string: "".to_string(),
    };
    OperationSingleOutput {
        operation,
        output_description: output,
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
