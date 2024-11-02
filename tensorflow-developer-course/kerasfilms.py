import tensorflow as tf

import keras_nlp

import tensorflow_datasets as tfds

from tensorflow.python.data.ops.map_op import _MapDataset

imdb = tfds.load("imdb_reviews", as_supervised=True, data_dir="./data", download=True)

train_reviews: _MapDataset = imdb['train'].map(lambda review, label: review)
train_labels: _MapDataset = imdb['train'].map(lambda review, label: label)

test_reviews = imdb['test'].map(lambda review, label: review)
test_labels = imdb['test'].map(lambda review, label: label)

breakpoint()

keras_nlp.tokenizers.compute_word_piece_vocabulary(
    train_reviews,
    vocabulary_size=8000,
    reserved_tokens=["[PAD]", "[UNK]"],
    vocabulary_output_file="imdb_vocab_subwords.txt"
)

subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary="./imdb_vocam_subwords.txt"
)

sample_string = "Tensorflow, from basics to mastery"

tokenized_string = subword_tokenizer.tokenize(sample_string)

