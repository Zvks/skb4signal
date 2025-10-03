from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

# Создадим коллекцию DataFrame
df_collection = []
states = ['Cracking','Ideal','Offset_Pulley','Wear']
for i in range(3):
    df_coord = []
    for folder_defect_name in os.listdir("Fourier"):
        df_defect = []
        folder_coordinate_name = os.listdir(f"Fourier\\{folder_defect_name}")[i]
        print(folder_coordinate_name)
        for filename in os.listdir(f"Fourier\\{folder_defect_name}\\{folder_coordinate_name}"):
            if filename.endswith(".csv"):
                df = pd.read_csv(f"Fourier\\{folder_defect_name}\\{folder_coordinate_name}\\{filename}")
                df_defect.append(df)
        df_coord.append(df_defect)
    df_collection.append(df_coord)

df_X = df_collection[0]

X = []
y = []

for class_idx, df_list in enumerate(df_X):
    for df in df_list:
        X.append(df.values)  # numpy array
        y.append(class_idx)  # или labels[class_idx]

X = np.array(X)
y = np.array(y)

# === Если датафреймы 2D (например, (n_timesteps, n_features)), нужно flatten или использовать Flatten ===
if len(X.shape) > 2:
    X = X.reshape(X.shape[0], -1)  # flatten

# === Кодируем метки ===
label_encoder = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(y, num_classes=4)

# === Построение модели ===
model = Sequential([
    Flatten(input_shape=(X.shape[1],)),  # если X.shape = (n_samples, flattened_features)
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')  # 4 класса
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Обучение ===
model.fit(X, y_encoded, epochs=50, batch_size=8, validation_split=0.2)

# === Построение графиков ===
plt.figure(figsize=(12, 4))

# === График потерь ===
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# === График точности ===
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# === Оценка ===
test_loss, test_acc = model.evaluate(X, y_encoded)
print(f"Точность: {test_acc:.2f}")