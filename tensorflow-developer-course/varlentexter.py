import tensorflow as tf

sentences = [
    "I love my dog",
    "I love my cat",
    "you love my dog!",
    "Do you think my dog is amazing?",
]

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(sentences)

vocabulary = vectorize_layer.get_vocabulary()

sequence = vectorize_layer("I love my dog")

for index, word in enumerate(vocabulary):
    print(index, word)

# shorter sentences are "post-padded" with zeroes.
sequences = vectorize_layer(sentences)
print(sequences)

# you can also "pre-pad" with zeroes.
sentences_dataset = tf.data.Dataset.from_tensor_slices(sentences)
sequences = sentences_dataset.map(vectorize_layer)

sequences_test= tf.keras.utils.pad_sequences(sequences, padding='post')
# breakpoint()
sequences_pre = tf.keras.utils.pad_sequences(sequences, padding='pre')

for sentence, sequence in zip(sentences, sequences):
    print(sentence, sequence)

for sentence, sequence in zip(sentences, sequences_pre):
    print(sentence)
    print(sequence)

# ragged padding
vectorize_layer = tf.keras.layers.TextVectorization(ragged=True)
vectorize_layer.adapt(sentences)
vocabulary = vectorize_layer.get_vocabulary()

ragged_sequences = vectorize_layer(sentences)
for index, word in enumerate(vocabulary):
    print(index, word)

print(ragged_sequences)

pre_padded_sequences = tf.keras.utils.pad_sequences(ragged_sequences.numpy())

print(pre_padded_sequences)

# unexpected words
test_data = [
    "i really love my dog",
    "my dog loves my manatee"
]

test_seq = vectorize_layer(test_data)
print(test_seq)