import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Démarrage du timer
start_time = time.time()

# Lire le chemin du modèle depuis les arguments
model_path = sys.argv[1]
print(f"[INFO] Chargement du modèle depuis {model_path}")

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

def preprocess(img, target_height=64, target_width=256):
    final_img = np.ones((target_height, target_width)) * 255
    img = img[:target_height, :target_width]
    x_offset = (target_width - img.shape[1]) // 2
    y_offset = (target_height - img.shape[0]) // 2
    final_img[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

def num_to_label(num_seq):
    return ''.join([alphabets[i] for i in num_seq if i != -1])

# Charger le modèle
model = load_model(model_path, compile=False)
model_lock = threading.Lock()

# Lire le fichier test
test = pd.read_csv('written_name_test_v2.csv')

# Traitement d'une image (prétraitement + prédiction)
def process_image(index):
    img_path = os.path.join('test_v2/test', test.loc[index, 'FILENAME'])
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    preprocessed = preprocess(image) / 255.0
    with model_lock:
        pred = model.predict(preprocessed.reshape(1, 256, 64, 1), verbose=0)
    decoded = tf.keras.backend.get_value(
        tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
    label = num_to_label(decoded[0])
    return image, label, index

# Nombre d’images à tester
n = 8

# Multithreading
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_image, i) for i in range(n)]
    for future in futures:
        results.append(future.result())

# Affichage
plt.figure(figsize=(16, 12))
for image, label, i in results:
    if image is not None:
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label, fontsize=12)
        plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Fin du timer
end_time = time.time()
print(f"[INFO] Temps total d'exécution : {end_time - start_time:.2f} secondes")
