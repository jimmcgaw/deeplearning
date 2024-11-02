import tensorflow as tf

import pathlib
import os

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(PARENT_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR, "val")

train_dataset_raw: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(1500, 1500),
    batch_size=128,
    label_mode="binary"
)

train_dataset = (train_dataset_raw.cache().shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE))

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(1500, 1500),
    batch_size=128,
    label_mode="binary"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(1500, 1500),
    batch_size=128,
    label_mode="binary"
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1500, 1500, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    # better choice for binary, numbers will be between 0 and 1, but above option also works
    # above is multiclass classification with 2 classes.
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.RMSProp(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=validation_dataset,
    verbose=2
)

# breakpoint()

# model.fit()