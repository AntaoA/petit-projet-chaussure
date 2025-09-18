import tensorflow as tf
import keras

data_dir = "data/Shoes"

dataset = keras.utils.image_dataset_from_directory(     # original size 1080x1440
    data_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=(135, 180),                              # resize to 135x180 (1/8 of original size)
    interpolation="bilinear",
    batch_size=32,
    shuffle=True,
    seed=42
)

print("Classes :", dataset.class_names) # type: ignore