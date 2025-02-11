import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

y_train = train_data["label"].values
x_train = train_data.drop("label", axis=1).values

y_test = test_data["label"].values
x_test = test_data.drop("label", axis=1).values

x_train = x_train/255.0
x_test = x_test/255.0

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test),
                    batch_size=64)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Pick a random test image
index = np.random.randint(0, len(x_test))
image = x_test[index]

# Predict
prediction = model.predict(image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)

# Show the image and prediction
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_label}, Actual: {y_test[index]}")
plt.show()