mod onnx_proto_structs;
use crate::onnx_proto_structs::tensor_shape_proto::Dimension;
use crate::onnx_proto_structs::type_proto::{Tensor, Value};
use onnx_proto_structs::*;
mod onnx_runner;
mod ops;
use crate::onnx_runner::OnnxRunner;
use serde::{Deserialize, Serialize};
mod onnx_to_internal_types_conversions;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

/// A generic Tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConcreteTensor {
    ConcreteF32Tensor(F32Tensor),
    ConcreteI32Tensor(I32Tensor),
}

/// A Concrete F32 Tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F32Tensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

fn validate_tensor<T>(data: &Vec<T>, shape: &Vec<i64>) {
    let number_els_shape = shape.iter().fold(1, |acc, curr| acc * curr);
    assert_eq!(
        number_els_shape,
        data.len() as i64,
        "Shape: {:?} is not compatible with data of length: {}",
        shape,
        data.len()
    );
}
impl F32Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        validate_tensor(&data, &shape);
        let rand_string: String = thread_rng().sample_iter(&Alphanumeric).take(10).collect();
        Self {
            name: rand_string,
            data,
            shape,
        }
    }
}

/// A Concrete I32 Tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct I32Tensor {
    pub name: String,
    pub data: Vec<i32>,
    pub shape: Vec<i64>,
}

impl I32Tensor {
    pub fn new(data: Vec<i32>, shape: Vec<i64>) -> Self {
        let rand_string: String = thread_rng().sample_iter(&Alphanumeric).take(10).collect();
        validate_tensor(&data, &shape);
        Self {
            name: rand_string,
            data,
            shape,
        }
    }
}

/// Placeholder Tensors are tensors representations inside a ONNX graph, they don't hold any actual
/// data  
pub struct PlaceholderF32Tensor {
    pub name: String,
    pub shape: Vec<i64>,
}

/// Placeholder Tensors are tensors representations inside a ONNX graph, they don't hold any actual
/// data
#[derive(Debug, Clone)]
pub struct PlaceholderI32Tensor {
    pub name: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCollection {
    f32_tensors: Vec<F32Tensor>,
    i32_tensors: Vec<I32Tensor>,
}

pub struct ModelBuilder {
    model: onnx_proto_structs::ModelProto,
    model_inputs: TensorCollection,
}

impl From<i64> for tensor_shape_proto::Dimension {
    fn from(input: i64) -> Self {
        tensor_shape_proto::Dimension {
            denotation: "".to_string(),
            value: Some(tensor_shape_proto::dimension::Value::DimValue(input)),
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
                    onnx_proto_structs::tensor_shape_proto::dimension::Value::DimValue(val) => *val,
                    onnx_proto_structs::tensor_shape_proto::dimension::Value::DimParam(_) => {
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
        let graph = onnx_proto_structs::GraphProto {
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
            model_inputs: TensorCollection {
                f32_tensors: vec![],
                i32_tensors: vec![],
            },
        }
    }

    pub fn add_input(&mut self, input: F32Tensor) -> PlaceholderF32Tensor {
        self.model_inputs.f32_tensors.push(input.clone());
        let place_holder = onnx_proto_structs::ValueInfoProto::from(input);
        self.graph_mut().input.push(place_holder.clone());
        PlaceholderF32Tensor::from(&place_holder)
    }

    fn graph(&self) -> &onnx_proto_structs::GraphProto {
        self.model
            .graph
            .as_ref()
            .expect("Tried to get graph but there was none!")
    }

    fn graph_mut(&mut self) -> &mut onnx_proto_structs::GraphProto {
        self.model
            .graph
            .as_mut()
            .expect("Tried to get graph but there was none!")
    }

    pub fn get_val_of_f32(&mut self, placeholder: &PlaceholderF32Tensor) -> F32Tensor {
        let output = ValueInfoProto::from(placeholder.clone());
        assert_eq!(self.graph().output.len(), 0);
        self.graph_mut().output.push(output);
        let mut runner = onnx_runner::PythonOnnxRunner::eval(&self.model, &self.model_inputs);
        self.graph_mut().output.clear();
        assert_eq!(runner.f32_tensors.len(), 1);
        runner.f32_tensors.pop().unwrap()
    }

    pub fn add_i32_input(&mut self, input: I32Tensor) -> PlaceholderI32Tensor {
        self.model_inputs.i32_tensors.push(input.clone());
        let place_holder = onnx_proto_structs::ValueInfoProto::from(input);
        self.graph_mut().input.push(place_holder.clone());
        PlaceholderI32Tensor::from(&place_holder)
    }

    pub fn add_operation(&mut self, operation: OperationSingleOutput) -> ValueInfoProto {
        self.graph_mut().node.push(operation.operation);
        operation.output_description
    }

    pub fn add_output(&mut self, node: ValueInfoProto) {
        self.graph_mut().output.push(node);
    }
}

pub struct OperationSingleOutput {
    operation: NodeProto,
    output_description: ValueInfoProto,
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
