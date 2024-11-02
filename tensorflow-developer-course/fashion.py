import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf._optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, _, logs=None):
        if logs['accuracy'] >= 0.8:
            self.model.stop_training = True

model.fit(
    train_images, train_labels, epochs=5, callbacks=[myCallback()]
)

model.evaluate(test_images, test_labels)