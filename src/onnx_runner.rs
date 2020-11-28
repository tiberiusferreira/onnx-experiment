use crate::onnx_proto_structs::ModelProto;
use crate::{onnx_proto_structs, TensorCollection};
use prost::Message;
use std::io::Write;
use std::num::ParseFloatError;

pub struct PythonOnnxRunner {
    model: onnx_proto_structs::ModelProto,
    model_inputs: TensorCollection,
}

pub trait OnnxRunner {
    fn eval(
        model: &onnx_proto_structs::ModelProto,
        external_inputs: &TensorCollection,
    ) -> TensorCollection;
    fn train(model: &onnx_proto_structs::ModelProto, external_inputs: &TensorCollection) -> f32;
}

impl OnnxRunner for PythonOnnxRunner {
    fn eval(model: &ModelProto, external_inputs: &TensorCollection) -> TensorCollection {
        let runner = Self::new(model, external_inputs);

        let model_name = "my_model.onnx";
        runner.serialize_model(model_name);
        let mut model_absolute_path = std::env::current_dir().unwrap();
        model_absolute_path.push(model_name);

        let model_inputs_name = "my_model_inputs.json";
        runner.serialize_inputs(model_inputs_name);
        let mut inputs_absolute_path = std::env::current_dir().unwrap();
        inputs_absolute_path.push(model_inputs_name);

        let output = std::process::Command::new("./onnx_runner/eval_model.sh")
            .args(&[
                "--model-file",
                model_absolute_path.to_str().unwrap(),
                "--inputs-file",
                inputs_absolute_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("Error running model")
            .wait_with_output()
            .expect("Error running model");
        if !output.status.success() {
            panic!("Error running model");
        }
        let output_as_string = String::from_utf8_lossy(&output.stdout);
        let output: Result<TensorCollection, _> = serde_json::from_str(&output_as_string);
        match output {
            Ok(output) => output,
            Err(_e) => {
                panic!("Error getting network output: {}", output_as_string);
            }
        }
    }

    fn train(model: &ModelProto, external_inputs: &TensorCollection) -> f32 {
        let runner = Self::new(model, external_inputs);

        let model_name = "my_model.onnx";
        runner.serialize_model(model_name);
        let mut model_absolute_path = std::env::current_dir().unwrap();
        model_absolute_path.push(model_name);

        let model_inputs_name = "my_model_inputs.json";
        runner.serialize_inputs(model_inputs_name);
        let mut inputs_absolute_path = std::env::current_dir().unwrap();
        inputs_absolute_path.push(model_inputs_name);

        let output = std::process::Command::new("./onnx_runner/train_model.sh")
            .args(&[
                "--model-file",
                model_absolute_path.to_str().unwrap(),
                "--inputs-file",
                inputs_absolute_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("Error running model")
            .wait_with_output()
            .expect("Error running model");
        if !output.status.success() {
            panic!("Error running model");
        }
        let output_as_string = String::from_utf8_lossy(&output.stdout);
        match output_as_string.parse::<f32>() {
            Ok(loss) => loss,
            Err(e) => {
                panic!("Loss was not f32, was: {}", output_as_string);
            }
        }
    }
}
impl PythonOnnxRunner {
    pub fn new(model: &onnx_proto_structs::ModelProto, external_inputs: &TensorCollection) -> Self {
        Self {
            model: model.clone(),
            model_inputs: external_inputs.clone(),
        }
    }
    fn serialize_model(&self, filename: &str) {
        let mut buf: Vec<u8> = Vec::new();
        buf.reserve(self.model.encoded_len());
        self.model.encode(&mut buf).unwrap();
        let mut out_file = std::fs::File::create(filename).unwrap();

        out_file.write_all(&buf).unwrap();
    }

    fn serialize_inputs(&self, filename: &str) {
        let inputs = serde_json::to_string_pretty(&self.model_inputs).unwrap();
        let mut inputs_file = std::fs::File::create(filename).unwrap();
        inputs_file.write_all(inputs.as_bytes()).unwrap();
    }
}
