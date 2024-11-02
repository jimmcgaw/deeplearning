import json

import tensorflow as tf

from keras.src.models.sequential import Sequential
from keras.src.callbacks.history import History

sentences = []
labels = []

with open("./Sarcasm_Headlines_Dataset.json", "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        sentences.append(
            data['headline']
        )
        labels.append(
            data['is_sarcastic']
        )

TRAINING_SIZE = int(len(labels) * 0.85)

training_sentences = sentences[0:TRAINING_SIZE]
testing_sentences = sentences[TRAINING_SIZE:]

training_labels = labels[0:TRAINING_SIZE]
testing_labels = labels[TRAINING_SIZE:]

# ???????
VOCAB_SIZE = 1000
# ???????
MAX_LENGTH = 16

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_LENGTH
)

vectorize_layer.adapt(training_sentences)

train_sequences = vectorize_layer(training_sentences)
test_sequences = vectorize_layer(testing_sentences)

train_dataset_vectorized = tf.data.Dataset.from_tensor_slices(
    (train_sequences, training_labels)
)
test_dataset_vectorized = tf.data.Dataset.from_tensor_slices(
    (test_sequences, testing_labels)
)

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

train_dataset_final = (
    train_dataset_vectorized.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)
)

test_dataset_final = (
    test_dataset_vectorized.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE)
)

# ????????
EMBEDDING_DIM = 16

model: Sequential = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(
    model.summary()
)


history: History = model.fit(train_dataset_final, epochs=30, validation_data=test_dataset_final, verbose=2)

import matplotlib.pyplot as plt

def plot_graphs(history: History, string: str):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string], color="red")
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

breakpoint()