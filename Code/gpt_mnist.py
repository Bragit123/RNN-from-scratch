import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding of labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define model
model = Sequential()
model.add(SimpleRNN(100, input_shape=(28, 28), activation='tanh'))
# model.add(LSTM(100, input_shape=(28, 28), activation='tanh'))  # Using LSTM
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
