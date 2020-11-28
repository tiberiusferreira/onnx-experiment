# source /Users/tiberio/Documents/github/onnxruntime/venv/bin/activate
import sys
import torch
import json
import onnx
import argparse

from onnxruntime.training import ORTTrainer, ORTTrainerOptions, optim
from onnxruntime.training.optim import SGDConfig, AdamConfig

parser = argparse.ArgumentParser(description='ONNX runner parameters')
parser.add_argument('--model-file', metavar='model_file', type=str, nargs=1, required=True,
                    help='The path to the file containing the ONNX model in protobuf format (.onnx)')
parser.add_argument('--inputs-file', metavar='inputs_file', type=str, nargs=1, required=True,
                    help='The path to the file containing the inputs for the ONNX model in JSON format')
parser.add_argument('--output-file', metavar='output_file', type=str, nargs=1, required=False,
                    help='The path where the file with the model output should be put')
args = parser.parse_args()

model_file_path = args.model_file[0]
model_inputs_file = args.inputs_file[0]
output_file_path = None
if args.output_file:
    output_file_path = args.output_file[0]

with open(model_inputs_file) as f:
    data_inputs = json.load(f)

inputs_description = []
inputs_values = []
eval_inputs = {}

for single_input in data_inputs['f32_tensors']:
    inputs_description.append((single_input['name'], single_input['shape']))
    value = torch.tensor(single_input['data'], dtype=torch.float32).reshape(single_input['shape'])
    eval_inputs[single_input['name']] = value.numpy()
    inputs_values.append(value)

for single_input in data_inputs['i32_tensors']:
    inputs_description.append((single_input['name'], single_input['shape']))
    value = torch.tensor(single_input['data'], dtype=torch.int32).reshape(single_input['shape'])
    eval_inputs[single_input['name']] = value.numpy()  # single_input['data']
    inputs_values.append(value)

model = onnx.load_model(model_file_path)
if len(model.graph.output) != 1:
    print("During training the model must have exactly one output: the loss. Had: " + str(len(model.graph.output)))
    exit(1)

model_output = model.graph.output[0]

if model_output.type.tensor_type.elem_type != 1:
    print("Training graph output must be of type f32, was of element type: " +
          str(model_output.type.tensor_type.elem_type))
    exit(1)


if len(model_output.type.tensor_type.shape.dim) > 1 or \
        (len(model_output.type.tensor_type.shape.dim) == 1 and model_output.type.tensor_type.shape.dim[0].dim_value != 1):
    print("Training graph output must be a scalar with shape either: [] or [1] shape was: \n" +
          str(model_output.type.tensor_type.shape))
    exit(1)

model_desc = {
    "inputs": inputs_description,  # (name, shape)
    "outputs": [(model_output.name, [], True)],  # (name, shape, is_loss)
}

options = {'device': {'id': 'cpu'}}

trainer = ORTTrainer(model,
                     model_desc,
                     optim_config=SGDConfig(),
                     options=ORTTrainerOptions(options))

loss = trainer.train_step(inputs_values)

outputs_as_tensors = {
    "f32_tensors": [],
    "i32_tensors": []
}

if loss.dtype == torch.float32:
    output_as_tensor = {
        "name": "loss",
        "data": loss.numpy().flatten().tolist(),
        "shape": list(loss.shape)
    }
    outputs_as_tensors['f32_tensors'].append(output_as_tensor)
else:
    sys.stderr.write("Loss was not float! Was: " + str(loss.dtype))

sys.stdout.write(str(loss.numpy().flatten().tolist()[0]))