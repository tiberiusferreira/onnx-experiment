mod onnx_proto_structs;
use crate::onnx_proto_structs::type_proto::Value;
use onnx_proto_structs::*;
mod onnx_runner;
mod ops;
use crate::onnx_runner::{OnnxRunner, TrainingOutput};
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
#[derive(Debug, Clone)]
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
                training_info: vec![TrainingInfoProto {
                    initialization: None,
                    algorithm: Some(GraphProto {
                        node: vec![],
                        name: "".to_string(),
                        initializer: vec![],
                        sparse_initializer: vec![],
                        doc_string: "".to_string(),
                        input: vec![],
                        output: vec![],
                        value_info: vec![],
                        quantization_annotation: vec![],
                    }),
                    initialization_binding: vec![],
                    update_binding: vec![],
                }],
            },
            model_inputs: TensorCollection {
                f32_tensors: vec![],
                i32_tensors: vec![],
            },
        }
    }

    pub fn add_initializer(&mut self, initializer: F32Tensor) -> PlaceholderF32Tensor {
        let onnx_concrete = TensorProto::from(initializer);
        let placeholder = PlaceholderF32Tensor::from(&onnx_concrete);
        self.graph_mut().initializer.push(onnx_concrete);
        placeholder
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
        let output = ValueInfoProto::from(placeholder);
        assert_eq!(self.graph().output.len(), 0);
        self.graph_mut().output.push(output);
        let mut network_outputs =
            onnx_runner::PythonOnnxRunner::eval(&self.model, &self.model_inputs);
        self.graph_mut().output.clear();
        assert_eq!(network_outputs.f32_tensors.len(), 1);
        network_outputs.f32_tensors.pop().unwrap()
    }

    pub fn train_get_loss(&mut self, placeholder: &PlaceholderF32Tensor) -> f32 {
        let output = ValueInfoProto::from(placeholder);
        assert_eq!(self.graph().output.len(), 0);
        assert_eq!(
            placeholder.shape.len(),
            0,
            "Can only get train loss from scalar tensor"
        );
        self.graph_mut().output.push(output);
        let training_output = onnx_runner::PythonOnnxRunner::train(&self.model, &self.model_inputs);
        let loss = training_output.loss;
        self.update_initializers(training_output);
        self.graph_mut().output.clear();
        loss
    }

    pub fn update_initializers(&mut self, training_output: TrainingOutput) {
        for initializer in &mut self.graph_mut().initializer {
            let new_val = training_output
                .updated_initializers
                .get(&initializer.name)
                .expect("Couldn't get new value for initializer");
            initializer.float_data = new_val.clone();
        }
    }
    pub fn add_i32_input(&mut self, input: I32Tensor) -> PlaceholderI32Tensor {
        self.model_inputs.i32_tensors.push(input.clone());
        let place_holder = onnx_proto_structs::ValueInfoProto::from(input);
        self.graph_mut().input.push(place_holder.clone());
        PlaceholderI32Tensor::from(&place_holder)
    }

    // pub fn add_operation(&mut self, operation: OperationSingleOutput) -> ValueInfoProto {
    //     self.graph_mut().node.push(operation.operation);
    //     operation.output_description
    // }

    pub fn add_output(&mut self, node: ValueInfoProto) {
        self.graph_mut().output.push(node);
    }

    // pub fn add_assignment(&mut self, origin: &PlaceholderF32Tensor, dest: &PlaceholderF32Tensor) {
    //     self.model
    //         .training_info
    //         .first_mut()
    //         .unwrap()
    //         .update_binding
    //         .push(StringStringEntryProto {
    //             key: origin.name.clone(),
    //             value: dest.name.clone(),
    //         });
    // }
    //
    // pub fn train_set_output(&mut self, placeholder: &PlaceholderF32Tensor) {
    //     self.model
    //         .training_info
    //         .first_mut()
    //         .unwrap()
    //         .algorithm
    //         .as_mut()
    //         .unwrap()
    //         .output
    //         .push(ValueInfoProto::from(placeholder));
    // }
}
