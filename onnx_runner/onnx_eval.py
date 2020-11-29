# source /Users/tiberio/Documents/github/onnxruntime/venv/bin/activate
import sys
import onnxruntime
import torch
import json

import argparse

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
    eval_inputs[single_input['name']] = value.numpy() # single_input['data']
    inputs_values.append(value)
sess = onnxruntime.InferenceSession(model_file_path)

names = []
for out in sess.get_outputs():
    names.append(out.name)
output = sess.run(names, input_feed=eval_inputs)

outputs_as_tensors = {
    "f32_tensors": [],
    "i32_tensors": []
}

for i, out in enumerate(sess.get_outputs()):
    if out.type == "tensor(float)":
        output_as_tensor = {
            "name": out.name,
            "data": output[i].flatten().tolist(),
            "shape": output[i].shape
        }
        outputs_as_tensors['f32_tensors'].append(output_as_tensor)
    else:
        print("Unexpected tensor of type:" + out.type)

if output_file_path:
    with open(output_file_path, 'w') as file:
        file.write(json.dumps(outputs_as_tensors))
else:
    sys.stdout.write(json.dumps(outputs_as_tensors))
