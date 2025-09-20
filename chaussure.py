import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import os


seed = 42
random.seed(seed)

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
# rate mlp = 72
# rate cnn = 15
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
    seed=seed
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
    images, labels, test_size=500, stratify=labels_int, random_state=seed
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

    x = layers.Conv2D(16, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.GlobalAveragePooling2D()(x)                                   # mieux que Flatten qui crée trop de paramètres
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="CNN")

# ResNet (pre-activation, bottle-neck design)
def residual_block(x, filters, downsample=False):
    shortcut = x    
    strides = (2,2) if downsample else (1,1)
    
    x = layers.Conv2D(filters, (3,3), strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Si dimensions différentes, adapter le shortcut
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=strides, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x



def build_resnet(input_shape=input_shape, num_classes=5):
    inputs = keras.Input(shape=input_shape)

    
    # Couche initiale
    x = layers.Conv2D(64, (7,7), strides=(2,2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)
    
    # Stages ResNet
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs, name="ResNet")
    
# Transfert learning
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
def build_efficientnet(input_shape=input_shape, num_classes=5):
    base = EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )

    # Fine-tuning progressif
    base.trainable = True
    for layer in base.layers[:-20]:   # gèle toutes sauf les 20 dernières
        layer.trainable = False

    inputs = keras.Input(shape=input_shape)

    x = preprocess_input(inputs)

    # Backbone
    x = base(x, training=True)

    # Head
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="EfficientNetB0_transfer")

build_model = build_mlp

# -------------------------------
# 6. Training
# -------------------------------

# callbacks : early stopping
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
histories = []
val_scores = []
best_epochs = []

for train_idx, val_idx in skfold.split(train_img, train_lbl_int):
    model = build_model()

    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_img[train_idx], train_lbl[train_idx],
        validation_data=(train_img[val_idx], train_lbl[val_idx]),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1   # type: ignore
    )
    histories.append(history)
    val_scores.append(history.history["val_accuracy"][-1])  # type: ignore
    
    best_epoch = np.argmin(history.history["val_loss"]) + 1  # +1 car index 0-based
    best_epochs.append(best_epoch)

print(f"\nValidation accuracy (5-fold CV): {np.mean(val_scores):.3f} ± {np.std(val_scores):.3f}")
print(f"Meilleur epoch (5-fold CV): {np.mean(best_epochs)} ± {np.std(best_epochs)}")
print(f"Epoch choisi : {np.mean(best_epochs)}")




# Entraînement final sur tout le train set
model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_img, train_lbl,
    epochs=int(np.median(best_epochs)),
    batch_size=32,
    verbose=1   # type: ignore
)

# Évaluation finale
test_loss, test_acc = model.evaluate(test_img, test_lbl, verbose=1) # type: ignore
print(f"Test accuracy: {test_acc:.3f}")



# -------------------------------
# 7. Métriques
# -------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 1. Courbes d'apprentissage avec moyenne et écart-type

max_epochs = max(len(h.history["loss"]) for h in histories)

# Construire des arrays remplis avec NaN pour les folds plus courts
def pad_history(h, key, max_epochs):
    vals = np.array(h.history[key])
    if len(vals) < max_epochs:
        vals = np.concatenate([vals, [np.nan]*(max_epochs - len(vals))])
    return vals

# Appliquer pad_history
train_losses = np.array([pad_history(h, "loss", max_epochs) for h in histories])
val_losses   = np.array([pad_history(h, "val_loss", max_epochs) for h in histories])
train_accs   = np.array([pad_history(h, "accuracy", max_epochs) for h in histories])
val_accs     = np.array([pad_history(h, "val_accuracy", max_epochs) for h in histories])

# Moyennes et écarts-types (en ignorant les NaN)
mean_train_loss, std_train_loss = np.nanmean(train_losses, axis=0), np.nanstd(train_losses, axis=0)
mean_val_loss,   std_val_loss   = np.nanmean(val_losses,   axis=0), np.nanstd(val_losses,   axis=0)
mean_train_acc,  std_train_acc  = np.nanmean(train_accs,   axis=0), np.nanstd(train_accs,   axis=0)
mean_val_acc,    std_val_acc    = np.nanmean(val_accs,     axis=0), np.nanstd(val_accs,     axis=0)

epochs = range(1, max_epochs+1)

median_best = int(np.median(best_epochs))

plt.figure(figsize=(12,5))

# Courbes Loss
plt.subplot(1,2,1)
plt.plot(epochs, mean_train_loss, label="train_loss", color="blue")
plt.fill_between(epochs, mean_train_loss-std_train_loss, mean_train_loss+std_train_loss, alpha=0.2, color="blue")
plt.plot(epochs, mean_val_loss, label="val_loss", color="orange")
plt.fill_between(epochs, mean_val_loss-std_val_loss, mean_val_loss+std_val_loss, alpha=0.2, color="orange")

# Marquer la fin de chaque fold
for i, h in enumerate(histories):
    stop_epoch = len(h.history["loss"])
    plt.axvline(stop_epoch, color="grey", linestyle="--", alpha=0.3)
    if i == 0:
        plt.scatter(stop_epoch, h.history["val_loss"][-1], color="orange", s=30, zorder=5, label="end val_loss (par fold)")
        plt.scatter(stop_epoch, h.history["loss"][-1], color="blue", s=30, zorder=5, label="end train_loss (par fold)")
    plt.scatter(stop_epoch, h.history["val_loss"][-1], color="orange", s=30, zorder=5)
    plt.scatter(stop_epoch, h.history["loss"][-1], color="blue", s=30, zorder=5)

# Médiane des best_epochs
plt.axvline(median_best, color="black", linestyle="--", linewidth=2, label=f"median best={median_best}")

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

# Marquer la fin de chaque fold
for i, h in enumerate(histories):
    stop_epoch = len(h.history["accuracy"])
    plt.axvline(stop_epoch, color="grey", linestyle="--", alpha=0.3)
    if i == 0:
        plt.scatter(stop_epoch, h.history["val_accuracy"][-1], color="red", s=30, zorder=5, label="end val_acc (par fold)")
        plt.scatter(stop_epoch, h.history["accuracy"][-1], color="green", s=30, zorder=5, label="end train_acc (par fold)")
    plt.scatter(stop_epoch, h.history["val_accuracy"][-1], color="red", s=30, zorder=5)
    plt.scatter(stop_epoch, h.history["accuracy"][-1], color="green", s=30, zorder=5)

# Médiane des best_epochs
plt.axvline(median_best, color="black", linestyle="--", linewidth=2, label=f"median best={median_best}")

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


# 3. Courbes ROC et PR

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

n_classes = len(name_class)

y_pred = model.predict(test_img)
y_true = np.argmax(test_lbl, axis=1)
y_true_bin = label_binarize(y_true, classes=range(n_classes))   # Binariser labels



# ROC
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])   # type: ignore
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average (pondéré globalement)
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())   # type: ignore
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Weighted-average AUC
roc_auc["weighted"] = roc_auc_score(y_true_bin, y_pred, average="weighted")



# PR
precision, recall, ap = {}, {}, {}
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i]) # type: ignore
    mask = ~((precision[i] == 0) & (recall[i] == 0))
    precision[i] = precision[i][mask]
    recall[i] = recall[i][mask]
    ap[i] = average_precision_score(y_true_bin[:, i], y_pred[:, i]) # type: ignore


# Micro-average PR (pondérée)
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_pred.ravel()) # type: ignore
ap["micro"] = average_precision_score(y_true_bin, y_pred, average="micro")

# Weighted-average AP
ap["weighted"] = average_precision_score(y_true_bin, y_pred, average="weighted")

# Affichage des courbes ROC et PR
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- ROC subplot ---
ax = axes[0]
for i in range(n_classes):
    ax.plot(fpr[i], tpr[i], label=f"{name_class[i]} (AUC={roc_auc[i]:.2f})")
ax.plot(fpr["micro"], tpr["micro"], color="black", linestyle="--", linewidth=2,
        label=f"Micro-average (AUC={roc_auc['micro']:.2f}, W={roc_auc['weighted']:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("Courbes ROC")
ax.legend()

# --- PR subplot ---
ax = axes[1]
for i in range(n_classes):
    ax.plot(recall[i], precision[i], label=f"{name_class[i]} (AP={ap[i]:.2f})")
ax.plot(recall["micro"], precision["micro"], color="black", linestyle="--", linewidth=2,
        label=f"Micro-average (AP={ap['micro']:.2f}, W={ap['weighted']:.2f})")
ax.hlines(y=sum(y_true_bin.ravel())/len(y_true_bin.ravel()), xmin=0, xmax=1, colors="gray", linestyles=":", label="Baseline (prévalence)")  # type: ignore
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Courbes Precision-Recall")
ax.legend()

plt.tight_layout()
plt.show()

# 4. Matrice de confusion
y_pred_classes = np.argmax(y_pred, axis=1)


cm = confusion_matrix(y_true, y_pred_classes)
cm_norm = confusion_matrix(y_true, y_pred_classes, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
disp.plot(cmap=plt.cm.Blues)    # type: ignore
plt.title("Matrice de confusion")
plt.show()


# 5. Rapport de classification

def classification_report_extended(cm=cm):
    n_classes = cm.shape[0]

    header = f"{'class':<10}{'precision':<12}{'recall':<12}{'specificity':<14}{'f1-score':<12}{'BER':<10}{'MCC':<10}{'support':<10}"
    print(header)
    print("-" * len(header))

    precisions, recalls, specificities, f1s, bers, mccs, supports = [], [], [], [], [], [], []

    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        precision = TP / (TP + FP)                              # exactitude des prédictions positives
        recall = TP / (TP + FN)                                 # sensibilité, capacité à détecter qu'une image est dans la classe
        specificity = TN / (TN + FP)                            # capacité à détecter qu'une image n'est pas dans la classe
        f1 = (2 * precision * recall) / (precision + recall)    # équilibre précision/recall
        ber = 1 - 0.5 * (recall + specificity)                  # Balanced Error Rate: moyenne des erreurs positives et négatives (classes déséquilibrées)
        mcc_num = (TP * TN - FP * FN)                          
        mcc_den = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc = mcc_num / mcc_den                                 # Matthews Correlation Coefficient: corrélation entre les vraies et les prédites (-1 à 1)
        support = TP + FN

        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1s.append(f1)
        bers.append(ber)
        mccs.append(mcc)
        supports.append(support)

        print(f"{i:<10}{precision:<12.2f}{recall:<12.2f}{specificity:<14.2f}{f1:<12.2f}{ber:<10.2f}{mcc:<10.2f}{support:<10}")

    # Moyennes
    macro = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "specificity": np.mean(specificities),
        "f1": np.mean(f1s),
        "BER": np.mean(bers),
        "MCC": np.mean(mccs),
        "support": np.sum(supports)/n_classes
    }

    weighted = {
        "precision": np.average(precisions, weights=supports),
        "recall": np.average(recalls, weights=supports),
        "specificity": np.average(specificities, weights=supports),
        "f1": np.average(f1s, weights=supports),
        "BER": np.average(bers, weights=supports),
        "MCC": np.average(mccs, weights=supports),
        "support": np.sum(supports)
    }

    overall_acc = np.trace(cm) / np.sum(cm)

    print("-" * len(header))
    print(f"{'accuracy':<10}{'':<12}{'':<12}{'':<14}{'':<12}{'':<10}{'':<10}{overall_acc:.2f}")
    print(f"{'avg':<10}{macro['precision']:<12.2f}{macro['recall']:<12.2f}{macro['specificity']:<14.2f}{macro['f1']:<12.2f}{macro['BER']:<10.2f}{macro['MCC']:<10.2f}{int(macro['support']):<10}")
    print(f"{'w avg':<10}{weighted['precision']:<12.2f}{weighted['recall']:<12.2f}{weighted['specificity']:<14.2f}{weighted['f1']:<12.2f}{weighted['BER']:<10.2f}{weighted['MCC']:<10.2f}{int(weighted['support']):<10}")


# Rapport détaillé
print("\nRapport de classification :\n")
classification_report_extended()




