# source /Users/tiberio/Documents/github/onnxruntime/venv/bin/activate
import onnx
from onnxruntime.training import TrainingSession, TrainingParameters, ORTTrainer, ORTTrainerOptions
from onnxruntime.training.optim import SGDConfig
import torch

print("Loading Model")
model = onnx.load_model("my_model.onnx")

model_desc = {
    "inputs": [
        ("left", [2, 2]),
        ("right", [2, 2])
    ],
    "outputs": [('new_out', [], True)],
}
options = {'device': {'id': 'cpu'}}
print("Loading Trainer")
trainer = ORTTrainer(model, model_desc, optim_config=SGDConfig(), options=ORTTrainerOptions(options))
left = torch.tensor([1., 2., 3., 4.], dtype=torch.float32).reshape([2, 2])
right =torch.tensor([1., 2., 3., 4.], dtype=torch.float32).reshape([2, 2])
print(left)
outputs = trainer.train_step(left, right)
print(outputs)
outputs = trainer.train_step(left, right)
print(outputs)
# for output in outputs:
#     print(output)





# sess = onnxruntime.InferenceSession("my_model.onnx")
# outs = sess.get_outputs()
# names = []
# for out in outs:
#     names.append(out.name)
# outputs = sess.run(names, input_feed={})
# print(outputs)



