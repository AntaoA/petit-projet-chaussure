import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from pathlib import Path



# -------------------------------
# 1. Import des données
# -------------------------------

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

# -------------------------------
# 2. Comptage des images par classe
# -------------------------------

class_counts = {cls: 0 for cls in dataset.class_names} # type: ignore
for images, labels in dataset:
    for label in labels:
        class_index = tf.argmax(label).numpy()
        class_name = dataset.class_names[class_index] # type: ignore
        class_counts[class_name] += 1

print("\nRépartition par classes :")
total = sum(class_counts.values())
for cls, count in class_counts.items():
    print(f"{cls:15s} : {count} images ({100*count/total:.1f}%)")

print("\nTotal d’images :", total)

# -------------------------------
# 3. Camembert
# -------------------------------

plt.figure(figsize=(6,6))
plt.pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%", startangle=140)   # type: ignore
plt.title("Répartition relative des classes")
plt.show()
