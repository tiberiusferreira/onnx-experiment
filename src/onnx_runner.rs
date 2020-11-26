use crate::onnx_proto_structs::ModelProto;
use crate::{onnx_proto_structs, ConcreteF32Tensor, ConcreteTensor, ExternalInputs};
use prost::Message;
use std::io::Write;

struct PythonOnnxRunner {
    model: onnx_proto_structs::ModelProto,
    external_inputs: ExternalInputs,
}

pub trait OnnxRunner {
    fn eval(
        &mut self,
        model: &onnx_proto_structs::ModelProto,
        external_inputs: &ExternalInputs,
    ) -> Vec<ConcreteTensor>;
}

/*
 {
        model.serialize_to_file();
        std::process::Command::new("./eval_model.sh")
            .spawn()
            .expect("Error running model")
            .wait()
            .unwrap();
        let file = std::fs::File::open("my_model_output.json").unwrap();
        let output: ConcreteF32Tensor = serde_json::from_reader(file).unwrap();
        output
    }
*/

impl OnnxRunner for PythonOnnxRunner {
    fn eval(
        &mut self,
        model: &ModelProto,
        external_inputs: &ExternalInputs,
    ) -> Vec<ConcreteTensor> {
        let runner = Self::new(model, external_inputs);
        runner.serialize_model("my_model.onnx");
        runner.serialize_inputs("my_model_inputs.json");
        std::process::Command::new("./onnx_runner/eval_model.sh")
            .spawn()
            .expect("Error running model")
            .wait()
            .unwrap();
        let file = std::fs::File::open("my_model_output.json").unwrap();
        let output: ConcreteF32Tensor = serde_json::from_reader(file).unwrap();
        vec![ConcreteTensor::ConcreteF32Tensor(output)]
    }
}
impl PythonOnnxRunner {
    pub fn new(model: &onnx_proto_structs::ModelProto, external_inputs: &ExternalInputs) -> Self {
        Self {
            model: model.clone(),
            external_inputs: external_inputs.clone(),
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
        let inputs = serde_json::to_string_pretty(&self.external_inputs).unwrap();
        let mut inputs_file = std::fs::File::create(filename).unwrap();
        inputs_file.write_all(inputs.as_bytes()).unwrap();
    }
}
