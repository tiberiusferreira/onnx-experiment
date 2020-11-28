use crate::onnx_proto_structs::tensor_shape_proto::Dimension;
use crate::onnx_proto_structs::type_proto::Value;
use crate::onnx_proto_structs::{
    tensor_shape_proto, TensorProto, TensorShapeProto, ValueInfoProto,
};
use crate::{onnx_proto_structs, F32Tensor, I32Tensor, PlaceholderF32Tensor, PlaceholderI32Tensor};

/// Make sure we can turn a Tensor with data from the ONNX graph into a Concrete Tensor
impl From<TensorProto> for F32Tensor {
    fn from(ten: TensorProto) -> Self {
        F32Tensor {
            name: ten.name,
            data: ten.float_data,
            shape: ten.dims,
        }
    }
}

/// Make sure we can turn a Tensor with data from the ONNX graph into a Concrete Tensor
impl From<TensorProto> for I32Tensor {
    fn from(ten: TensorProto) -> Self {
        I32Tensor {
            name: ten.name,
            data: ten.int32_data,
            shape: ten.dims,
        }
    }
}

/// Make sure we can turn a Concrete Tensor into a ONNX graph Tensor with data
impl From<F32Tensor> for TensorProto {
    fn from(ten: F32Tensor) -> Self {
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

/// Make sure we can turn a Concrete Tensor into a ONNX graph Tensor with data
impl From<I32Tensor> for TensorProto {
    fn from(ten: I32Tensor) -> Self {
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

/// Convert our internal Tensor shape representation to the ONNX representation
fn our_shape_to_onnx(shape: Vec<i64>) -> Vec<Dimension> {
    shape
        .iter()
        .map(|dim| Dimension {
            denotation: "".to_string(),
            value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
        })
        .collect()
}

/// Convert ONNX Tensor shape representation to the our representation
fn onnx_shape_to_our(shape: &Vec<Dimension>) -> Vec<i64> {
    shape
        .iter()
        .map(
            |dim| match dim.value.as_ref().expect("ONNX dimension was empty!") {
                onnx_proto_structs::tensor_shape_proto::dimension::Value::DimValue(dim) => *dim,
                onnx_proto_structs::tensor_shape_proto::dimension::Value::DimParam(_) => {
                    panic!("ONNX shape dimension was dynamic!!")
                }
            },
        )
        .collect()
}

/// From our Concrete Tensors to ONNX Placeholders
impl From<F32Tensor> for ValueInfoProto {
    fn from(ten: F32Tensor) -> Self {
        let dims: Vec<Dimension> = our_shape_to_onnx(ten.shape);
        ValueInfoProto {
            name: ten.name,
            r#type: Some(onnx_proto_structs::TypeProto {
                denotation: "".to_string(),
                value: Some(onnx_proto_structs::type_proto::Value::TensorType(
                    onnx_proto_structs::type_proto::Tensor {
                        elem_type: 1, //float
                        shape: Some(TensorShapeProto { dim: dims }),
                    },
                )),
            }),
            doc_string: "".to_string(),
        }
    }
}

/// From our Concrete Tensors to ONNX Placeholders
impl From<I32Tensor> for ValueInfoProto {
    fn from(ten: I32Tensor) -> Self {
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
            r#type: Some(onnx_proto_structs::TypeProto {
                denotation: "".to_string(),
                value: Some(onnx_proto_structs::type_proto::Value::TensorType(
                    onnx_proto_structs::type_proto::Tensor {
                        elem_type: 6, // i32
                        shape: Some(TensorShapeProto { dim: dims }),
                    },
                )),
            }),
            doc_string: "".to_string(),
        }
    }
}

/// From ONNX Placeholders to our Placeholders
impl From<&ValueInfoProto> for PlaceholderF32Tensor {
    fn from(val: &ValueInfoProto) -> Self {
        let type_enum = val
            .r#type
            .as_ref()
            .expect("Placeholder with no type")
            .value
            .as_ref()
            .expect("Empty placeholder type!");
        let tensor_type_and_shape = match type_enum {
            Value::TensorType(tensor_type_and_shape) => {
                if tensor_type_and_shape.elem_type != 1 {
                    panic!("ValueInfoProto was not of type f32.")
                }
                tensor_type_and_shape
                    .shape
                    .as_ref()
                    .expect("Placeholder Tensor had no shape!")
            }
            _ => {
                panic!("Placeholder type was not a tensor!")
            }
        };
        PlaceholderF32Tensor {
            name: val.name.clone(),
            shape: onnx_shape_to_our(&tensor_type_and_shape.dim),
        }
    }
}

impl From<&ValueInfoProto> for PlaceholderI32Tensor {
    fn from(val: &ValueInfoProto) -> Self {
        let tensor_type_enum = val
            .r#type
            .as_ref()
            .expect("Placeholder with no type")
            .value
            .as_ref()
            .expect("Empty placeholder type!");
        let tensor_type_and_shape = match tensor_type_enum {
            Value::TensorType(val) => {
                if val.elem_type != 6 {
                    panic!("ValueInfoProto was not of type i32.")
                }
                val.shape.as_ref().unwrap()
            }
            _ => {
                panic!("Placeholder type was not a tensor!")
            }
        };
        PlaceholderI32Tensor {
            name: val.name.clone(),
            shape: onnx_shape_to_our(&tensor_type_and_shape.dim),
        }
    }
}

/// From ONNX Placeholders to our Placeholders
impl From<&PlaceholderF32Tensor> for ValueInfoProto {
    fn from(val: &PlaceholderF32Tensor) -> Self {
        let dims: Vec<Dimension> = our_shape_to_onnx(val.shape.clone());
        ValueInfoProto {
            name: val.name.clone(),
            r#type: Some(onnx_proto_structs::TypeProto {
                denotation: "".to_string(),
                value: Some(onnx_proto_structs::type_proto::Value::TensorType(
                    onnx_proto_structs::type_proto::Tensor {
                        elem_type: 1, // i32
                        shape: Some(TensorShapeProto { dim: dims }),
                    },
                )),
            }),
            doc_string: "".to_string(),
        }
    }
}

/// From ONNX Concrete to our Placeholder
impl From<&TensorProto> for PlaceholderF32Tensor {
    fn from(val: &TensorProto) -> Self {
        PlaceholderF32Tensor {
            name: val.name.clone(),
            shape: val.dims.clone(),
        }
    }
}
