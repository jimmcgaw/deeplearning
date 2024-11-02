import tensorflow as tf

label_encoder = tf.keras.layers.StringLookup(max_tokens=1000, num_oov_indices=0)

breakpoint()
print('try')