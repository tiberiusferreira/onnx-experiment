# source /Users/tiberio/Documents/github/onnxruntime/venv/bin/activate
import onnx
import onnxruntime
from onnxruntime.training import TrainingSession, TrainingParameters, ORTTrainer, ORTTrainerOptions
from onnxruntime.training.optim import SGDConfig
import torch
import json

import argparse
parser = argparse.ArgumentParser(description='Add some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='interger list')
parser.add_argument('--sum', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args.sum(args.integers))

# print("Loading Model")
model = onnx.load_model("my_model.onnx")

# print("Loading Model Inputs")
with open('my_model_inputs.json') as f:
    data_inputs = json.load(f)

inputs_descr = []
inputs_values = []
eval_inputs = {}
for single_input in data_inputs['external_f32_inputs']:
    inputs_descr.append((single_input['name'], single_input['shape']))
    value = torch.tensor(single_input['data'], dtype=torch.float32).reshape(single_input['shape'])
    eval_inputs[single_input['name']] = value.numpy() # single_input['data']
    inputs_values.append(value)

for single_input in data_inputs['external_i32_inputs']:
    inputs_descr.append((single_input['name'], single_input['shape']))
    value = torch.tensor(single_input['data'], dtype=torch.int32).reshape(single_input['shape'])
    eval_inputs[single_input['name']] = value.numpy() # single_input['data']
    inputs_values.append(value)

# model_desc = {
#     "inputs": inputs_descr,
#     "outputs": [('new_out', [], True)],
# }

sess = onnxruntime.InferenceSession("my_model.onnx")
out = sess.get_outputs()[0].name
names = [out]
output = sess.run(names, input_feed=eval_inputs)

output_as_tensor = {
    "name": out,
    "data": output[0].flatten().tolist(),
    "shape": output[0].shape
}

with open('my_model_output.json', 'w') as file:
    file.write(json.dumps(output_as_tensor))




# options = {'device': {'id': 'cpu'}}
# print("Loading Trainer")
# trainer = ORTTrainer(model, model_desc, optim_config=SGDConfig(), options=ORTTrainerOptions(options))
# outputs = trainer.train_step(inputs_values)
# print(outputs)
# for output in outputs:
#     print(output)





# sess = onnxruntime.InferenceSession("my_model.onnx")
# outs = sess.get_outputs()
# names = []
# for out in outs:
#     names.append(out.name)
# outputs = sess.run(names, input_feed={})
# print(outputs)



