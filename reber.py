from network import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

def string_to_sequence(string):
	sequence = np.zeros((len(string), len(letters)))
	for i, letter in enumerate(string):
		sequence[i,letter_to_index[letter]] = 1
	return sequence

def train_reber(path, N):
	network = Network(7, 12, 7)
	f = open(path)
	learning_rate = 0.1
	for i in range(N):
		if i % 1000 == 0:
			print(i)
		string = f.readline().strip()
		network.train_sequence(string_to_sequence(string), learning_rate)
	f.close()
	return network

def predict_correctly(network, string, threshold):
	network.reset_memoization()
	for i, x in enumerate(string_to_sequence(string)[:-1]):
		y = network.propagate(x)
		expected_index = letter_to_index[string[i+1]]
		if y[expected_index] < threshold:
			return False
	return True

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

def test_reber(network):
	network.reset_memoization()
	print(prettify(network.propagate([1, 0, 0, 0, 0, 0, 0])))
	print(prettify(network.propagate([0, 1, 0, 0, 0, 0, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 1, 0, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 1, 0, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 0, 1, 0])))
	print(prettify(network.propagate([0, 0, 0, 0, 0, 1, 0])))

def prettify(proba):
	return ', '.join('(' + l + ', {0:.2f})'.format(p) for l, p in zip(letters, proba))

if __name__ == '__main__':
	accuracies = []
	for _ in range(1):
		network = train_reber('../reber-datasets/reber_train_2.4M.txt', 50000)
		# Save the network
		pickle.dump(network, open('reber.pickle', 'wb'))
		#network = pickle.load(open('reber.pickle', 'rb'))
		#test_reber(network)
		accuracies.append(accuracy(network, '../reber-datasets/reber_test_1M.txt', 1000000))
	print(accuracies)