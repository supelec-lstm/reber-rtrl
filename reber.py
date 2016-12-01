import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../reber-grammar/')))

import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from network import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

nb_hidden_units = 12
train_path = '../reber-datasets/symmetrical_reber_train_2.4M.txt'
test_path = '../reber-datasets/symmetrical_reber_test_1M.txt'
automaton = symmetrical_reber.create_automaton(0.5)

"""nb_hidden_units = 2
train_path = '../reber-datasets/reber_train_2.4M.txt'
test_path = '../reber-datasets/reber_test_1M.txt'
automaton = reber.create_automaton()"""

def string_to_sequence(string):
	sequence = np.zeros((len(string), len(letters)))
	for i, letter in enumerate(string):
		sequence[i,letter_to_index[letter]] = 1
	return sequence

def train_reber(path, N):
	network = Network(7, nb_hidden_units, 7)
	f = open(path)
	learning_rate = 0.1
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		network.train_sequence(string_to_sequence(string), learning_rate)
	f.close()
	return network

def train_and_monitor(path, N_train):
	network = Network(7, nb_hidden_units, 7)
	t = []
	accuracies = []
	f = open(path)
	learning_rate = 0.1
	for _ in range(np.random.randint(0, 1000000)):
		next(f)
	for i in range(N_train):
		string = f.readline().strip()
		network.train_sequence(string_to_sequence(string), learning_rate)
		if i % 1000 == 0:
			print(i)
			t.append(i)
			accuracies.append(accuracy(network, test_path, 10))
	f.close()
	print(t, accuracies)
	plt.plot(t, accuracies)
	plt.xlabel("Nombre de chaînes")
	plt.ylabel("Précision (10000 chaînes testées)")
	plt.title("Apprentissage après chaque séquence")
	plt.show()
	return network

def predict_correctly(network, string, threshold):
	network.reset_memoization()
	cur_state = automaton.start
	for i, x in enumerate(string_to_sequence(string)[:-1]):
		y = network.propagate(x)
		cur_state = cur_state.next(string[i])
		predicted_transitions = {letters[j] for j, activated in enumerate(y > threshold) if activated}
		if set(predicted_transitions) != set(cur_state.transitions.keys()):
			return False
	return True

def predict_last_letter(network, string, threshold):
	network.reset_memoization()
	cur_state = automaton.start
	for i, x in enumerate(string_to_sequence(string)[:-2]):
		y = network.propagate(x)
		cur_state = cur_state.next(string[i])
		predicted_transitions = {letters[j] for j, activated in enumerate(y > threshold) if activated}
	return predicted_transitions == {string[-2]}

def accuracy(network, path, N):
	f = open(path)
	c = 0
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		if predict_correctly(network, string, 0.3):
			c += 1
	return c / N

def accuracy_last_letter(network, path, N):
	f = open(path)
	c = 0
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		if predict_last_letter(network, string, 0.3):
			c += 1
	return c / N

def test_reber(network):
	network.reset_memoization()
	print(prettify(network.propagate([1, 0, 0, 0, 0, 0, 0])))
	print(prettify(network.propagate([0, 1, 0, 0, 0, 0, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 1, 0, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 1, 0, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 0, 1, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 0, 1, 0])))

def prettify(proba):
	return ' & '.join(['{0:.2f}'.format(p) for p in proba]) + ' \\\\\n\\hline'
	#return ', '.join('(' + l + ', {0:.2f})'.format(p) for l, p in zip(letters, proba))

if __name__ == '__main__':
	for _ in range(1):
		network = train_and_monitor(train_path, 500000)
		# Save the network
		pickle.dump(network, open('reber.pickle', 'wb'))
		network = pickle.load(open('reber.pickle', 'rb'))
		#test_reber(network)
		print(accuracy(network, test_path, 10000))
		print()
		print(accuracy_last_letter(network, test_path, 10000))