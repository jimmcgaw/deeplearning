import tensorflow as tf
from tensorflow.python.data.ops.map_op import _MapDataset
from tensorflow.python.data.ops.from_tensor_slices_op import _TensorSliceDataset

import tensorflow_datasets as tfds

imdb: dict
info: tfds.core.dataset_info.DatasetInfo
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# look at a single example
single_example = list(imdb['train'].take(2))[1]

review: tf.Tensor = single_example[0]
# 1 = positive review, 0 = negative
label: tf.Tensor = single_example[1]

print(review)
print(label)

train_data, test_data = imdb['train'], imdb['test']

train_reviews: _MapDataset = train_data.map(lambda review, _: review)
train_labels: _MapDataset = train_data.map(lambda _, label: label)

test_reviews: _MapDataset = test_data.map(lambda review, _: review)
test_labels: _MapDataset = test_data.map(lambda _, label: label)

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=10000)
vectorize_layer.adapt(train_reviews)

def padding_func(sequeneces):
    sequeneces = sequeneces.ragged_batch(batch_size=sequeneces.cardinality())
    sequeneces = sequeneces.get_single_element()

    padded_sequences = tf.keras.utils.pad_sequences(sequeneces.numpy(), maxlen=120, truncating='post', padding='pre')
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

    return padded_sequences

train_sequences: _TensorSliceDataset = train_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)
test_sequences: _TensorSliceDataset = test_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)

train_dataset_vectorized = tf.data.Dataset.zip(train_sequences, train_labels)
test_dataset_vectorized = tf.data.Dataset.zip(test_sequences, test_labels)

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

train_dataset_final = (train_dataset_vectorized.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE))
test_dataset_final = (test_dataset_vectorized.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(BATCH_SIZE))

# just setting these for the example model below
vocab_size = 10000
embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(120,)),
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # alternatively, insted of Flatten, we can use,
    # which average across the vector to flatten it out. Should make this faster.
    # tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_dataset_final, epochs=10, validation_data=test_dataset_final)

# check the structure of the model
embedding_layer = model.layers[0]  # isn't first layer input?
embedding_weights = embedding_layer.get_weights()[0]
print(embedding_weights.shape)  # shape = (vocab_size, embedding_dim)

import io

out_v = io.open("vecs.tsv", "w", encoding="utf-8")
out_m = io.open("meta.tsv", "w", encoding="utf-8")

vocabulary = vectorize_layer.get_vocabulary()

for word_num in range(1, len(vocabulary)):
    word_name = vocabulary[word_num]
    word_embedding = embedding_weights[word_num]
    out_m.write(word_name + "\n")
    out_v.write("\t".join([str(x) for x in word_embedding]) + "\n")

out_v.close()
out_m.close()