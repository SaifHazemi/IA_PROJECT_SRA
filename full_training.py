import os
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization
from keras.optimizers import Adam
import time
import tqdm

# Silence les logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Timer global
start_time = time.time()

# Paramètres globaux
train_directory = 'train_v2/train/'
valid_directory = 'validation_v2/validation/'
test_directory = 'test_v2/test/'
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24
num_of_timestamps = 64

def preprocess(img, target_height=64, target_width=256):
    final_img = np.ones((target_height, target_width)) * 255
    img = img[:target_height, :target_width]
    x_offset = (target_width - img.shape[1]) // 2
    y_offset = (target_height - img.shape[0]) // 2
    final_img[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

def load_and_preprocess_image(img_path):
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = preprocess(img) / 255.0
            return img.reshape(256, 64, 1)
    return None

def prepare_data(data_frame, directory):
    image_paths = [os.path.join(directory, row['FILENAME']) for _, row in data_frame.iterrows()]
    results = []
    for path in tqdm.tqdm(image_paths, desc=f"Prétraitement ({directory})"):
        img = load_and_preprocess_image(path)
        if img is not None:
            results.append(img)
    return np.array(results)

def label_to_num(label):
    return np.array([alphabets.find(ch) for ch in label])

def num_to_label(num_seq):
    return ''.join([alphabets[i] for i in num_seq if i != -1])

# Lecture des fichiers CSV
train = pd.read_csv('written_name_train_v2.csv').sample(n=10000, random_state=1)
valid = pd.read_csv('written_name_validation_v2.csv').sample(n=10000, random_state=1)
test = pd.read_csv('written_name_test_v2.csv')

train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()
train = train[train['IDENTITY'].notna() & ~train['IDENTITY'].isin(['EMPTY', 'UNREADABLE'])]
valid = valid[valid['IDENTITY'].notna() & ~valid['IDENTITY'].isin(['EMPTY', 'UNREADABLE'])]

# Préparation des données
train_x = prepare_data(train, train_directory)
valid_x = prepare_data(valid, valid_directory)

train_y = np.ones([len(train), max_str_len]) * -1
valid_y = np.ones([len(valid), max_str_len]) * -1
for i in range(len(train)):
    encoded = label_to_num(train.iloc[i]['IDENTITY'])
    train_y[i, :len(encoded)] = encoded
for i in range(len(valid)):
    encoded = label_to_num(valid.iloc[i]['IDENTITY'])
    valid_y[i, :len(encoded)] = encoded

# Architecture du modèle
input_data = Input(shape=(256, 64, 1), name='input')
inner = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_data)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2))(inner)
inner = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2))(inner)
inner = Reshape(target_shape=(64, 1024))(inner)
inner = Bidirectional(LSTM(256, return_sequences=True))(inner)
inner = Bidirectional(LSTM(256, return_sequences=True))(inner)
inner = Dense(len(alphabets) + 1, kernel_initializer='he_normal')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='int32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred[:, 2:, :], input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001))

input_length_array = np.ones((len(train_x), 1)) * (num_of_timestamps - 2)
label_length_array = np.array([[len(label)] for label in train['IDENTITY']])

val_input_length = np.ones((len(valid_x), 1)) * (num_of_timestamps - 2)
val_label_length = np.array([[len(label)] for label in valid['IDENTITY']])

# Entraînement
model_final.fit(
    x=[train_x, train_y, input_length_array, label_length_array],
    y=np.zeros(len(train_x)),
    validation_data=([valid_x, valid_y, val_input_length, val_label_length], np.zeros(len(valid_x))),
    epochs=3,
    batch_size=16
)

# Utilisation pour prédiction
model_predict = Model(inputs=input_data, outputs=y_pred)

# Nombre d’images de test à afficher
n = 8
plt.figure(figsize=(16, 12))
for i in range(n):
    img_path = os.path.join(test_directory, test.loc[i, 'FILENAME'])
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    preprocessed = preprocess(image) / 255.0
    pred = model_predict.predict(preprocessed.reshape(1, 256, 64, 1), verbose=0)
    decoded = tf.keras.backend.get_value(
        tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
    label = num_to_label(decoded[0])
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(label, fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Temps total
end_time = time.time()
print(f"[INFO] Temps total d'exécution : {end_time - start_time:.2f} secondes")