from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])

# Create second input (ValueInfoProto)
Pads = helper.make_tensor_value_info('Pads', TensorProto.INT64, [4])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Pad', # node name
    ['X', 'Pads'], # inputs
    ['Y'], # outputs
    mode='constant', # Attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    "test-model",
    [X, Pads],
    [Y],
    [helper.make_tensor('Pads', TensorProto.INT64, [4,], [0, 0, 1, 1,])],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def,
                              producer_name='onnx-example')

onnx.save_model(model_def, "model.onnx")

print('The ir_version in model: {}\n'.format(model_def.ir_version))
print('The producer_name in model: {}\n'.format(model_def.producer_name))
print('The graph in model:\n{}'.format(model_def.graph))
onnx.checker.check_model(model_def)
print('The model is checked!')