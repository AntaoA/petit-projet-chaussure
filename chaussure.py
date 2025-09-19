import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import os



# -------------------------------
# 1. Affichage de quelques images par classe
# -------------------------------

data_dir = "data/Shoes"
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
# 2. Import des données
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
# 3. Stratified split train/test
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
# 4. Visualisation répartition des classes
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
# 5. Modèles
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
# 6. Training
# -------------------------------

# callbacks : early stopping
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
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
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1   # type: ignore
    )
    histories.append(history)
    val_scores.append(history.history["val_accuracy"][-1])  # type: ignore

print(f"\nValidation accuracy (5-fold CV): {np.mean(val_scores):.3f} ± {np.std(val_scores):.3f}")

model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_img, train_lbl,
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1   # type: ignore
)

# Évaluation finale
test_loss, test_acc = model.evaluate(test_img, test_lbl, verbose=1) # type: ignore
print(f"Test accuracy: {test_acc:.3f}")



# -------------------------------
# 7. Métriques
# -------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report

# 1. Courbes d'apprentissage avec moyenne et écart-type

# Récupérer le nombre d'epochs réellement utilisées (early stopping)
min_epochs = min(len(h.history["loss"]) for h in histories)

# Sélectionner epochs communes
train_losses = np.array([h.history["loss"][:min_epochs] for h in histories])
val_losses   = np.array([h.history["val_loss"][:min_epochs] for h in histories])
train_accs   = np.array([h.history["accuracy"][:min_epochs] for h in histories])
val_accs     = np.array([h.history["val_accuracy"][:min_epochs] for h in histories])

# Moyennes et écarts-types
mean_train_loss, std_train_loss = train_losses.mean(axis=0), train_losses.std(axis=0)
mean_val_loss,   std_val_loss   = val_losses.mean(axis=0),   val_losses.std(axis=0)
mean_train_acc,  std_train_acc  = train_accs.mean(axis=0),  train_accs.std(axis=0)
mean_val_acc,    std_val_acc    = val_accs.mean(axis=0),    val_accs.std(axis=0)

epochs = range(1, min_epochs+1)

plt.figure(figsize=(12,5))

# Courbes Loss
plt.subplot(1,2,1)
plt.plot(epochs, mean_train_loss, label="train_loss", color="blue")
plt.fill_between(epochs, mean_train_loss-std_train_loss, mean_train_loss+std_train_loss, alpha=0.2, color="blue")
plt.plot(epochs, mean_val_loss, label="val_loss", color="orange")
plt.fill_between(epochs, mean_val_loss-std_val_loss, mean_val_loss+std_val_loss, alpha=0.2, color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("CV – Perte moyenne ± écart-type")

# Courbes Accuracy 
plt.subplot(1,2,2)
plt.plot(epochs, mean_train_acc, label="train_acc", color="green")
plt.fill_between(epochs, mean_train_acc-std_train_acc, mean_train_acc+std_train_acc, alpha=0.2, color="green")
plt.plot(epochs, mean_val_acc, label="val_acc", color="red")
plt.fill_between(epochs, mean_val_acc-std_val_acc, mean_val_acc+std_val_acc, alpha=0.2, color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CV – Accuracy moyenne ± écart-type")

plt.show()


# 2. Courbes d'apprentissage sur un seul fold

fold_to_plot = 0 
h = histories[fold_to_plot]

# Courbe Loss
plt.figure(figsize=(12,5))
nb_epochs = range(1, len(h.history["loss"])+1)
plt.subplot(1,2,1)
plt.plot(nb_epochs, h.history["loss"], label=f"Train fold {fold_to_plot+1}")
plt.plot(nb_epochs, h.history["val_loss"], label=f"Val fold {fold_to_plot+1}", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Courbes de perte par fold")
plt.legend()

# Courbe Accuracy
plt.subplot(1,2,2)
plt.plot(nb_epochs, h.history["accuracy"], label=f"Train fold {fold_to_plot+1}")
plt.plot(nb_epochs, h.history["val_accuracy"], label=f"Val fold {fold_to_plot+1}", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Courbes d'accuracy par fold")
plt.legend()
plt.show()


