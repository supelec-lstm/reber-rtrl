import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import pytest
from network import *

@pytest.fixture
def graph1():
	input_node = InputNode()
	expected_output = InputNode()
	substraction_node = SubstractionNode(expected_output, input_node)
	cost_node = Norm2Node(substraction_node)
	nodes = [input_node, expected_output, substraction_node, cost_node]
	return Graph(nodes, [input_node], [input_node], [expected_output], cost_node, [])

@pytest.fixture
def network1():
	network = Network(1, 0, 1)
	network.W = np.ones((1, 2))
	return network

def test_propagate(network1):
	network = network1
	output = network.propagate(np.array([1]))
	assert np.allclose(output, np.array([0.73105858]))

	output = network.propagate(np.array([2]))
	assert np.allclose(output, np.array([0.93883465]))

def test_backpropagate(network1):
	network = network1
	network.propagate(np.array([1]))
	network.propagate_gradient()
	assert np.allclose(network.p, np.array([[[0], [0.19661193]]]))
	network.accumulate_gradient(2)
	expected_gradient1 = np.array([[0, -0.4989780520514972]])
	assert np.allclose(2*network.acc_dJdW, expected_gradient1)

	network.propagate(np.array([2]))
	network.propagate_gradient()
	assert np.allclose(network.p, np.array([[[0.04198042], [0.12613857]]]))
	network.accumulate_gradient(3)
	expected_gradient2 = np.array([[-0.17305715, -0.51998488]])
	print(2*network.acc_dJdW - expected_gradient1)
	assert np.allclose(2*network.acc_dJdW - expected_gradient1, expected_gradient2)

	network.update_weights(2)
	expected_W = np.array([[1.17305715, 2.01896294]])
	assert np.allclose(network.W, expected_W)