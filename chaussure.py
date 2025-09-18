import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split


# -------------------------------
# 1. Import des données
# -------------------------------

data_dir = "data/Shoes"

dataset = keras.utils.image_dataset_from_directory(     # original size 1080x1440
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=(135, 180),                              # resize to 135x180 (1/8 of original size)
    interpolation="bilinear",
    shuffle=True,
    batch_size=1,
    seed=42
)
class_names = dataset.class_names # type: ignore
print("Classes :", class_names)

# -------------------------------
# 2. Stratified split train/test
# -------------------------------

images = []
labels = []
for img, label in dataset:
    images.append(img.numpy())
    labels.append(label.numpy())
    

images = np.array(images)
labels = np.array(labels)


print("Shape images :", images.shape)
print("Shape labels :", labels.shape)

train_img, test_img, train_lbl, test_lbl = train_test_split(
    images, labels, test_size=500, stratify=labels, random_state=42
)

# -------------------------------
# 3. Visualisation répartition des classes
# -------------------------------

# compter le nombre d'images par classe
unique_train, counts_train = np.unique(train_lbl, return_counts=True)
unique_test, counts_test = np.unique(test_lbl, return_counts=True)

train_counts = dict(zip([class_names[i] for i in unique_train], counts_train))
test_counts = dict(zip([class_names[i] for i in unique_test], counts_test))

train_total = sum(counts_train)
test_total = sum(counts_test)

print("\nRépartition TRAIN (avec %):")
for cls in class_names:
    n = train_counts.get(cls, 0)
    print(f"{cls:15s}: {n:4d} images ({100*n/train_total:.1f}%)")
print(f"Total train: {train_total} images")

print("\nRépartition TEST (avec %):")
for cls in class_names:
    n = test_counts.get(cls, 0)
    print(f"{cls:15s}: {n:4d} images ({100*n/test_total:.1f}%)")
print(f"Total test: {test_total} images")

# --- Graphique en barres --- 
x = np.arange(len(class_names))  # positions

plt.figure(figsize=(8,5))
plt.bar(x, counts_test, label="Test", color="salmon")
plt.bar(x, counts_train, bottom=counts_test, label="Train", color="skyblue")

plt.xticks(x, class_names, rotation=30, ha="right")
plt.ylabel("Nombre d'images")
plt.title("Répartition des classes dans les datasets Train et Test")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# -------------------------------
# 4. Modèles
# -------------------------------

# MLP
def build_mlp(input_shape=(135,180,3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="MLP")

# CNN
def build_cnn(input_shape=(135,180,3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)                                  # mieux que Flatten qui crée trop de paramètres (128 contre 33*45*128) avec le dense après
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="CNN")

