import tensorflow as tf

sentences = [
    "I love my dog",
    "You love my dog!",
    "I love my cat",
    "Do you love my cat?"
]

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(sentences)

# vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)
vocabulary = vectorize_layer.get_vocabulary()

print(vocabulary)