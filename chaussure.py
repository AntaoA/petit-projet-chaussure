import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import os



import os
import random
import matplotlib.pyplot as plt
import keras

# -------------------------------
# 0. Affichage de quelques images par classe
# -------------------------------

data_dir = "data/Shoes"
nb_par_classe = 1   # une seule image par classe
compressions = [1, 8, 15, 24, 30, 36, 45, 60, 72, 90]

# Récupérer uniquement les dossiers (classes)
name_class = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])

plt.figure(figsize=(len(compressions)*3, len(name_class)*3))

for i, cls in enumerate(name_class):
    cls_dir = os.path.join(data_dir, cls)
    images = os.listdir(cls_dir)
    img_name = random.choice(images)
    img_path = os.path.join(cls_dir, img_name)

    # afficher la même image avec plusieurs tailles
    for j, rate in enumerate(compressions):
        size = (1440//rate, 1080//rate)
        img = keras.utils.load_img(img_path, target_size=size)
        plt.subplot(len(name_class), len(compressions), i*len(compressions) + j + 1)
        plt.imshow(img)
        if j == 0:
            plt.ylabel(cls, fontsize=14)
        plt.title(f"{size[0]}x{size[1]}", fontsize=10)
        plt.xticks([])
        plt.yticks([])

plt.suptitle("Compression progressive des images par classe", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# -------------------------------
# 1. Import des données
# -------------------------------
rate = 90
size = (1440//rate, 1080//rate)

dataset = keras.utils.image_dataset_from_directory(     # original size 1080x1440
    data_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=size,
    interpolation="bilinear",
    shuffle=True,
    batch_size=None,    # type: ignore
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

labels_int = np.argmax(labels, axis=1)  # convertir one-hot en int pour stratify

train_img, test_img, train_lbl, test_lbl = train_test_split(
    images, labels, test_size=500, stratify=labels_int, random_state=42
)

# normalisation des images
train_img = train_img.astype("float32") / 255.0
test_img  = test_img.astype("float32") / 255.0

# convertir one-hot en int pour stratify (pour la cross-validation) et répartition des classes
train_lbl_int = np.argmax(train_lbl, axis=1)
test_lbl_int  = np.argmax(test_lbl, axis=1)

# -------------------------------
# 3. Visualisation répartition des classes
# -------------------------------

# compter le nombre d'images par classe
unique_train, counts_train = np.unique(train_lbl_int, return_counts=True)
unique_test, counts_test = np.unique(test_lbl_int, return_counts=True)

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
input_shape = train_img.shape[1:]
# MLP
def build_mlp(input_shape=input_shape, num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="MLP")

# CNN
def build_cnn(input_shape=input_shape, num_classes=5):
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

# ResNet (pre-activation, bottle-neck design)
def residual_block(x, filters):
    shortcut = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters//4, (1,1), padding="same")(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters//4, (3,3), padding="same")(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (1,1), padding="same")(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    return x


def build_resnet(n_block=8, input_shape=input_shape, num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(n_block):
        x = residual_block(x, 32)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="ResNet_like")



# Transfert learning
from keras.applications import EfficientNetB1
def build_efficientnet(input_shape=input_shape, num_classes=5):
    base = EfficientNetB1(
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    base.trainable = False  # on gèle le backbone au début

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="EfficientNetB1_transfer")





# -------------------------------
# 5. Training
# -------------------------------

# callbacks : early stopping
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
]

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
histories = []
val_scores = []

for train_idx, val_idx in skfold.split(train_img, train_lbl_int):
    # Choisir le modèle à entraîner
    model = build_mlp()
    # model = build_cnn()
    # model = build_resnet()
    # model = build_efficientnet()

    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_img[train_idx], train_lbl[train_idx],
        validation_data=(train_img[val_idx], train_lbl[val_idx]),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1   # type: ignore
    )
    histories.append(history)
    val_scores.append(history.history["val_accuracy"][-1])  # type: ignore
