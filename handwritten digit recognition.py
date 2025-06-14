# prompt: give a code for handwritten recognition using ai  give fast result and accurate result



import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Define the model
model = keras.Sequential([
  keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Conv2D(64, (3, 3), activation='relu'),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (you can adjust epochs and batch_size)
model.fit(x_train, y_train, epochs=5, batch_size=32) # Reduced epochs for faster execution

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy}")

def predict_digit(image_path):
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, -1)

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        return predicted_digit

    except FileNotFoundError:
        return "Image file not found."
    except Exception as e:
        return f"An error occurred: {e}"


# Example usage
image_path = "/content/Screenshot 2025-04-02 133543.png" # Replace with the path to your image
predicted_digit = predict_digit(image_path)
print(f"Predicted digit: {predicted_digit}")
