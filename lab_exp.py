EXP : 2,3,4,5,6,7,8,9,10

1. exp 2 create a simple nn classification

	import pandas as pd
	import numpy as np
	from keras.models import Sequential
	from keras.layers import Dense

	dataset = pd.read_csv('/content/diabetes.csv')
	print(dataset.head())
	dataset = dataset.apply(pd.to_numeric, errors='coerce')
	dataset = dataset.fillna(dataset.mean())
	dataset = dataset.values
	X = dataset[:, 0:8]
	y = dataset[:, 8]
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X, y, epochs=150, batch_size=10)
	_, accuracy = model.evaluate(X, y)
	print(f'Accuracy: {accuracy * 100:.2f}%')
	predictions = (model.predict(X) > 0.5).astype(int)
	for i in range(5):
		print(f'{X[i].tolist()} => Predicted: {predictions[i][0]}, Actual: {int(y[i])}')
		
2. exp 3 multi-layer neural network with diff activation function

	# Import necessary libraries
	import numpy as np
	import pandas as pd
	import tensorflow as tf
	from sklearn.model_selection import KFold
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense
	from scikeras.wrappers import KerasClassifier  # Use scikeras for KerasClassifier
	from sklearn.metrics import accuracy_score

	# Load the dataset using pandas to handle the header row
	data = pd.read_csv('/content/diabetes.csv')

	# Split into input (X) and output (y) variables
	X = data.iloc[:, 0:8].values  # Input features
	y = data.iloc[:, 8].values    # Output label

	# Define the keras model
	def create_model(activation='relu'):
		model = Sequential()
		model.add(Dense(12, input_shape=(8,), activation=activation))  # Use the passed activation
		model.add(Dense(8, activation=activation))
		model.add(Dense(1, activation='sigmoid'))
		# Compile the keras model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	# Create KerasClassifier
	model = KerasClassifier(model=create_model, epochs=150, batch_size=10, verbose=0)

	# K-Fold Cross Validation
	kfold = KFold(n_splits=5, shuffle=True, random_state=42)
	accuracies = []

	# Loop through each fold
	for train_index, test_index in kfold.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		# Fit the model
		model.fit(X_train, y_train)

		# Predict and evaluate
		y_pred = (model.predict(X_test) > 0.5).astype(int)
		accuracy = accuracy_score(y_test, y_pred)
		accuracies.append(accuracy)

	# Summarize results
	mean_accuracy = np.mean(accuracies)
	std_accuracy = np.std(accuracies)

	print(f"Mean Accuracy: {mean_accuracy:.4f}")
	print(f"Standard Deviation: {std_accuracy:.4f}")

3. EXP 4 improve performace of nn with hyper parameter tuning

	# Import necessary libraries
	import numpy as np
	import pandas as pd
	import tensorflow as tf
	from sklearn.model_selection import GridSearchCV
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense
	from scikeras.wrappers import KerasClassifier

	# Load the dataset using pandas to handle the header row
	data = pd.read_csv('/content/diabetes.csv')

	# Split into input (X) and output (y) variables
	X = data.iloc[:, 0:8].values  # Input features
	y = data.iloc[:, 8].values    # Output label

	# Define the keras model
	def create_model2(activation='relu'):
		model = Sequential()
		model.add(Dense(12, input_shape=(8,), activation=activation))  # Use the passed activation
		model.add(Dense(8, activation=activation))
		model.add(Dense(1, activation='sigmoid'))
		# Compile the keras model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	def perform_grid_search(X, y):
		model = KerasClassifier(model=create_model2, epochs=150, batch_size=10, verbose=0)
		param_grid = {'model__activation': ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']}
		grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
		return grid.fit(X, y)

	grid_result = perform_grid_search(X, y)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	  
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
		
4. exp 8 object detection using cnn

	import numpy as np
	import cv2
	import os
	import matplotlib.pyplot as plt
	import tensorflow as tf
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

	# Define the CNN model for object detection
	def create_model(input_shape):
		model = Sequential([
			Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
			MaxPooling2D(pool_size=(2, 2)),
			Conv2D(64, (3, 3), activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Conv2D(128, (3, 3), activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Flatten(),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(4, activation='sigmoid')  # Predicting x_min, y_min, x_max, y_max
		])
		model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
		return model

	# Load your dataset (dummy example)
	def load_data(data_dir):
		images = []
		labels = []
		
		# Iterate through images and load them
		for filename in os.listdir(data_dir):
			if filename.endswith('.jpg'):
				img_path = os.path.join(data_dir, filename)
				img = cv2.imread(img_path)
				img = cv2.resize(img, (224, 224))  # Resize to match model input
				images.append(img)

				# Dummy label for bounding box (x_min, y_min, x_max, y_max)
				label = [0.1, 0.1, 0.9, 0.9]  # Example bounding box; replace with actual labels
				labels.append(label)

		return np.array(images), np.array(labels)

	# Load data
	data_dir = 'path/to/your/dataset'  # Update this path
	X, y = load_data(data_dir)

	# Normalize images
	X = X / 255.0  # Normalize pixel values

	# Split into training and testing sets
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create the model
	model = create_model(input_shape=(224, 224, 3))

	# Train the model
	model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

	# Evaluate the model
	loss, accuracy = model.evaluate(X_test, y_test)
	print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

	# Example prediction
	def predict_and_draw_bounding_box(model, image):
		image_resized = cv2.resize(image, (224, 224)) / 255.0
		pred = model.predict(np.expand_dims(image_resized, axis=0))[0]

		# Rescale bounding box coordinates
		height, width, _ = image.shape
		x_min, y_min, x_max, y_max = pred * np.array([width, height, width, height])
		
		# Draw bounding box on the original image
		cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
		return image

	# Load a test image and make a prediction
	test_image = cv2.imread('path/to/test/image.jpg')  # Update with an actual image path
	result_image = predict_and_draw_bounding_box(model, test_image)

	# Display the result
	plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.show()




5. exp 9 sentiment analysis model using lstm
	import numpy as np
	import pandas as pd
	import re
	import tensorflow as tf
	from tensorflow.keras.preprocessing.text import Tokenizer
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import LSTM, Dense, Embedding
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import LabelEncoder
	data = {
		'text': ['I love this! ❤️', 'Baseball is great! ⚾', 'I am so happy! 😁', 'This is scary! 🔪'],
		'label': ['positive', 'positive', 'positive', 'negative']
	}
	df = pd.DataFrame(data)
	def clean_text(text):
		return re.sub(r'[^a-zA-Z0-9s❤️⚾😁🔪]', '', text)
	df['clean_text'] = df['text'].apply(clean_text)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(df['clean_text'])
	X = pad_sequences(tokenizer.texts_to_sequences(df['clean_text']), maxlen=10)
	y = LabelEncoder().fit_transform(df['label'])
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	model = Sequential([
		Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=10),
		LSTM(128, dropout=0.2, recurrent_dropout=0.2),
		Dense(1, activation='sigmoid')
	])
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
	accuracy = model.evaluate(X_test, y_test)[1]
	print(f'Test accuracy: {accuracy * 100:.2f}%')
	new_texts = ['I am so excited! 😁', 'This is terrible! 🔪']
	new_padded = pad_sequences(tokenizer.texts_to_sequences([clean_text(text) for text in new_texts]), maxlen=10)
	predictions = model.predict(new_padded)
	for text, prediction in zip(new_texts, predictions):
		sentiment = 'positive' if prediction > 0.5 else 'negative'
		print(f'Text: {text} | Sentiment: {sentiment}')
		
6. exp 10 lstm by tuning hyper parameter

	# Import necessary libraries
	import numpy as np
	import pandas as pd
	import re
	import tensorflow as tf
	from tensorflow.keras.preprocessing.text import Tokenizer
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
	from tensorflow.keras.optimizers import Adam
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import LabelEncoder

	# Sample dataset
	data = {
		'text': [
			'I love this! ❤️',
			'Baseball is great! ⚾',
			'I am so happy! 😁',
			'This is scary! 🔪'
		],
		'label': ['positive', 'positive', 'positive', 'negative']
	}
	df = pd.DataFrame(data)

	def clean_text(text):
		return re.sub(r'[^a-zA-Z0-9\s❤️⚾😁🔪]', '', text)

	df['clean_text'] = df['text'].apply(clean_text)

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(df['clean_text'])
	X = pad_sequences(tokenizer.texts_to_sequences(df['clean_text']), maxlen=10)

	y = LabelEncoder().fit_transform(df['label'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model = Sequential([
		Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=10),
		LSTM(256, dropout=0.3, recurrent_dropout=0.3),
		Dropout(0.3),
		Dense(1, activation='sigmoid')
	])

	model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

	model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))

	loss, accuracy = model.evaluate(X_test, y_test)
	print(f'Test accuracy: {accuracy*100:.2f}%')

	new_texts = ['I am so excited! 😁', 'This is terrible! 🔪']
	new_sequences = pad_sequences(tokenizer.texts_to_sequences(new_texts), maxlen=10)
	predictions = model.predict(new_sequences)

	# Display predictions
	for text, prediction in zip(new_texts, predictions):
		sentiment = 'positive' if prediction > 0.5 else 'negative'
		print(f'Text: {text} | Sentiment: {sentiment}')

7. exp 5 nn for image classification

	import tensorflow as tf
	from tensorflow.keras import layers, models
	from tensorflow.keras.datasets import mnist
	import numpy as np

	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	train_images = train_images.astype('float32') / 255.0
	test_images = test_images.astype('float32') / 255.0

	train_images = np.expand_dims(train_images, axis=-1)
	test_images = np.expand_dims(test_images, axis=-1)

	model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.Flatten(),
		layers.Dense(64, activation='relu'),
		layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	model.summary()

	history = model.fit(train_images, train_labels, epochs=5, batch_size=64,
						validation_split=0.1)

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print(f"Test accuracy: {test_acc}")

8. exp 6 tuning hyper parameter for cnn

	import numpy as np
	import tensorflow as tf
	from tensorflow.keras import layers, models
	from tensorflow.keras.datasets import mnist
	from sklearn.base import BaseEstimator, ClassifierMixin
	from sklearn.model_selection import RandomizedSearchCV
	from scipy.stats import randint

	# Define the CNN model
	def create_model(num_filters=32, kernel_size=(3, 3), pool_size=(2, 2), dense_units=64):
		model = models.Sequential([
			layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=(28, 28, 1)),
			layers.MaxPooling2D(pool_size),
			layers.Conv2D(2*num_filters, kernel_size, activation='relu'),
			layers.MaxPooling2D(pool_size),
			layers.Conv2D(2*num_filters, kernel_size, activation='relu'),
			layers.Flatten(),
			layers.Dense(dense_units, activation='relu'),
			layers.Dense(10, activation='softmax')
		])
		model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])
		return model

	# Create a custom KerasClassifier wrapper
	class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
		def __init__(self, build_fn=None, epochs=5, batch_size=32, **kwargs):
			self.build_fn = build_fn
			self.epochs = epochs
			self.batch_size = batch_size
			self.kwargs = kwargs

		def fit(self, X, y):
			self.model = self.build_fn(**self.kwargs)
			self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
			return self

		def predict(self, X):
			return np.argmax(self.model.predict(X), axis=-1)

		def score(self, X, y):
			loss, accuracy = self.model.evaluate(X, y, verbose=0)
			return accuracy

		def set_params(self, **params):
			for key, value in params.items():
				if key == 'build_fn':
					continue
				elif key in ['epochs', 'batch_size']:
					setattr(self, key, value)
				else:
					self.kwargs[key] = value
			return self

	# Load and preprocess the MNIST dataset
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.astype('float32') / 255.0
	test_images = test_images.astype('float32') / 255.0
	train_images = np.expand_dims(train_images, axis=-1)
	test_images = np.expand_dims(test_images, axis=-1)

	# Create the KerasClassifierWrapper based on the Keras model
	model = KerasClassifierWrapper(build_fn=create_model, epochs=3, batch_size=32)

	# Define hyperparameters for randomized search
	param_distributions = {
		'num_filters': [32, 64],
		'kernel_size': [(3, 3), (5, 5)],
		'pool_size': [(2, 2), (3, 3)],
		'dense_units': [64, 128]
	}

	# Create the RandomizedSearchCV object
	random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
									   n_iter=4, cv=2, verbose=2, n_jobs=1)  # Set n_jobs=1 if facing memory issues

	# Perform the randomized search
	random_search.fit(train_images, train_labels)

	# Get the best parameters and model
	best_params = random_search.best_params_
	best_model = random_search.best_estimator_

	# Evaluate the best model on the test set
	test_acc = best_model.score(test_images, test_labels)
	print(f"Best parameters: {best_params}")
	print(f"Test accuracy of the best model: {test_acc}")

9. exp 7 auto-encoder for dimensity reduction

	# exp7
	import tensorflow as tf
	from tensorflow.keras.datasets import fashion_mnist
	from tensorflow.keras import layers, models
	import numpy as np
	import matplotlib.pyplot as plt

	# Load the Fashion MNIST dataset and preprocess it
	(train_images, _), (test_images, _) = fashion_mnist.load_data()

	# Normalize the pixel values to range [0, 1]
	train_images = train_images.astype('float32') / 255.0
	test_images = test_images.astype('float32') / 255.0

	# Flatten the images for the autoencoder
	train_images_flat = train_images.reshape((len(train_images), np.prod(train_images.shape[1:])))
	test_images_flat = test_images.reshape((len(test_images), np.prod(test_images.shape[1:])))

	# Define the autoencoder model
	def build_autoencoder(encoding_dim):
		input_img = layers.Input(shape=(784,))

		# Encoder
		encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

		# Decoder
		decoded = layers.Dense(784, activation='sigmoid')(encoded)

		# Autoencoder
		autoencoder = models.Model(input_img, decoded)

		# Separate encoder model
		encoder = models.Model(input_img, encoded)

		# Separate decoder model
		encoded_input = layers.Input(shape=(encoding_dim,))
		decoder_layer = autoencoder.layers[-1]
		decoder = models.Model(encoded_input, decoder_layer(encoded_input))

		return autoencoder, encoder, decoder

	# Define the size of the encoding dimension (reduce from 784 to 32 dimensions)
	encoding_dim = 32

	# Build the autoencoder model
	autoencoder, encoder, decoder = build_autoencoder(encoding_dim)

	# Compile the autoencoder
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

	# Train the autoencoder
	history = autoencoder.fit(train_images_flat, train_images_flat,
							  epochs=10,
							  batch_size=256,
							  shuffle=True,
							  validation_data=(test_images_flat, test_images_flat))

	# Encode and decode the test images
	encoded_imgs = encoder.predict(test_images_flat)
	decoded_imgs = decoder.predict(encoded_imgs)

	# Display some test images and their reconstructed counterparts
	n = 3  # number of images to display
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# Original images
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(test_images[i])
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# Reconstructed images
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(decoded_imgs[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

10. exp 1 basic tensor-flow

	import tensorflow as tf
	# Example: Create a 3-dimensional tensor (tensor of rank 3)
	tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	print(tensor_3d)

	import tensorflow as tf
	r1 = tf.constant(1, tf.int32)
	print(r1) 

	import tensorflow as tf

	# Named my_scalar
	r2 = tf.constant(1, dtype=tf.int32, name="my_scalar")
	print(r2)

	import tensorflow as tf

	# Decimal data
	q1_decimal = tf.constant(1.12345, dtype=tf.float32)
	print(q1_decimal)

	# String data
	q1_string = tf.constant("JavaTpoint", dtype=tf.string)
	print(q1_string)

	import tensorflow as tf

	# Decimal data
	q1_decimal = tf.constant(1.12345, dtype=tf.float32)
	print(q1_decimal)

	# String data
	q1_string = tf.constant("JavaTpoint", dtype=tf.string)
	print(q1_string)

	import tensorflow as tf

	# Rank 2 matrix
	q2_matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
	print(q2_matrix)

	import tensorflow as tf

	# Corrected tensor with consistent row lengths
	m_shape = tf.constant([[11, 10], [13, 12], [15, 14]])

	# Get the shape of the tensor
	shape = tf.shape(m_shape)

	print("Tensor:", m_shape)
	print("Shape:", shape)

	import tensorflow as tf

	# Create a vector of 0
	vector_of_zeros = tf.zeros(10)
	print(vector_of_zeros)

	import tensorflow as tf

	# Create a constant tensor
	X = tf.constant([2])

	# Function to demonstrate the concept of evaluation
	@tf.function
	def evaluate_tensor(tensor):
		return tensor * 2

	# Call the function and print the result
	result = evaluate_tensor(X)
	print(result)

	import tensorflow as tf
	# Define variables
	var = tf.Variable(initial_value=5, dtype=tf.int32, name='var')
	var_init_1 = tf.Variable(initial_value=10, dtype=tf.int32, name='var_init_1')
	var_init_2 = tf.Variable(initial_value=15, dtype=tf.int32, name='var_init_2')

	# Print variable values directly
	print(var.numpy())         # Prints the value of 'var'
	print(var_init_1.numpy())  # Prints the value of 'var_init_1'
	print(var_init_2.numpy())  # Prints the value of 'var_init_2'
