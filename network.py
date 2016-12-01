import numpy as np
import pickle

def activation_function(x):
	return 1 / (1+np.exp(-x))

def gradient_activation_function(x, y):
	return y * (1-y)

def cost_function(expected_y, y):
	return 1/2*np.sum(np.square(y - expected_y))

def gradient_cost_function(expected_y, y):
	return y - expected_y

class Network:
	def __init__(self, nb_input_units, nb_hidden_units, nb_output_units):
		self.nb_input_units = nb_input_units
		self.nb_hidden_units = nb_hidden_units
		self.nb_output_units = nb_output_units

		self.W = np.random.rand(self.nb_hidden_units+self.nb_output_units, \
			self.nb_input_units+self.nb_hidden_units+self.nb_output_units) - 1

		self.reset_memoization()

	def propagate(self, x):
		self.z = np.concatenate((self.y, x))
		self.s = np.dot(self.W, self.z)
		self.y = activation_function(self.s)
		return self.y[:self.nb_output_units]

	def propagate_gradient(self):
		# Update p = dydW
		self.p = np.tensordot(self.p, self.W[:,:-self.nb_input_units], axes=(2, 1))
		for i in range(min(self.p.shape[0], self.p.shape[2])):
			self.p[i,:,i] += self.z
		for k in range(self.p.shape[2]):
			self.p[:,:,k] *= gradient_activation_function(self.s[k], self.y[k])

	def accumulate_gradient(self, expected_y):
		# Compute dJdy
		e = gradient_cost_function(expected_y, self.y[:self.nb_output_units])
		e = np.concatenate((e, np.zeros(self.nb_hidden_units)))
		# Update acc_dJdW
		self.acc_dJdW += np.tensordot(self.p, e, axes=(2, 0))

	def update_weights(self, learning_rate):
		self.W -= learning_rate * self.acc_dJdW
		self.acc_dJdW = np.zeros(self.W.shape)

	def reset_memoization(self):
		self.y = np.zeros(self.nb_hidden_units+self.nb_output_units)
		self.z = None
		self.s = None
		self.p = np.zeros((self.nb_hidden_units+self.nb_output_units, \
			self.nb_input_units+self.nb_hidden_units+self.nb_output_units, \
			self.nb_hidden_units+self.nb_output_units))
		self.acc_dJdW = np.zeros(self.W.shape)

	def train_sequence(self, sequence, learning_rate):
		self.reset_memoization()
		for i, x in enumerate(sequence[:-1]):
			self.propagate(x)
			self.propagate_gradient()
			next_x = sequence[i+1]
			self.accumulate_gradient(next_x)
		# Better to update after each sequence (and not after each character)
		self.update_weights(learning_rate)