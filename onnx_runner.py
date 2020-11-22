# source /Users/tiberio/Documents/github/onnxruntime/venv/bin/activate
import onnxruntime
import onnx

sess = onnxruntime.InferenceSession("my_model.onnx")
outs = sess.get_outputs()
names = []
for out in outs:
    names.append(out.name)
outputs = sess.run(names, input_feed={})
print(outputs)



