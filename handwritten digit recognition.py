import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess the MNIST dataset
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape and normalize
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(X_train, y_train,
                       epochs=20,
                       batch_size=64,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping])

    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

# Preprocess user-provided image
def preprocess_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found or couldn't be loaded")

    # Invert colors (MNIST has white digits on black background)
    img = cv2.bitwise_not(img)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize and reshape
    img = img.astype('float32') / 255
    img = np.reshape(img, (1, 28, 28, 1))

    return img


# Predict digit from image
def predict_digit(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display the image
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f"Predicted: {digit} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

    return digit, confidence

# Main function
def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Build model
    model = build_model()
    model.summary()

    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Example usage with user-provided image
    while True:
        image_path = input("\nEnter path to your handwritten digit image (or 'q' to quit): ")
        if image_path.lower() == 'q':
            break

        try:
            digit, confidence = predict_digit(model, image_path)
            print(f"Predicted digit: {digit} with confidence {confidence:.2f}")
        except Exception as e:
            print(f"Error: {e}")






if __name__ == "__main__":
    main()
