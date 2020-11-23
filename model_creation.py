from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
from onnx.backend.test.case.model import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN, ONNX_DOMAIN

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




add_node = onnx.helper.make_node('Add',
                                 ['a', 'b'], ['c'], name='my_add')
mul_node = onnx.helper.make_node('Mul',
                                 ['c', 'a'], ['d'], name='my_mul')
gradient_node = onnx.helper.make_node(
    'Gradient', ['a', 'b'],
    ['dd_da', 'dd_db'], name='my_gradient',
    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    xs=['a', 'b'], y='d')

a = np.array(1.0).astype(np.float32)
b = np.array(2.0).astype(np.float32)
c = a + b
# d = a * c = a * (a + b)
d = a * c
# dd / da = d(a*a+a*b) / da = 2 * a + b
dd_da = (2 * a + b).astype(np.float32)
# dd / db = d(a*a+a*b) / db = a
dd_db = a

graph = onnx.helper.make_graph(
    nodes=[add_node, mul_node, gradient_node],
    name='GradientOfTwoOperators',
    inputs=[
        onnx.helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT,
                                           []),
        onnx.helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT,
                                           [])],
    outputs=[
        onnx.helper.make_tensor_value_info('d', onnx.TensorProto.FLOAT,
                                           []),
        onnx.helper.make_tensor_value_info('dd_da',
                                           onnx.TensorProto.FLOAT, []),
        onnx.helper.make_tensor_value_info('dd_db',
                                           onnx.TensorProto.FLOAT, [])])

opsets = [
    onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
    onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
model = onnx.helper.make_model(graph,
                               producer_name='backend-test',
                               opset_imports=opsets)

onnx.save_model(model, "model_grad.onnx")

expect(model, inputs=[a, b], outputs=[d, dd_da, dd_db],
       name='test_gradient_of_add_and_mul')


