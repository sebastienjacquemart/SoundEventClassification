import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, BatchNormalization, Bidirectional, LSTM, Reshape
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

import parameter
from utils import plot_history, get_f1, convert_to_binary
from data_generator import DataGenerator

# Init parameters
params = parameter.get_params()

# Specify the paths of the input and output folders
features_path = os.path.abspath(os.path.join(params['output_dir'], './features_seb_tuned/'))
train_events_path = os.path.abspath(os.path.join(features_path, './events_train.npy/'))
train_labels_path = os.path.abspath(os.path.join(features_path, './labels_train.npy/'))
test_events_path = os.path.abspath(os.path.join(features_path, './events_test.npy/'))
test_labels_path = os.path.abspath(os.path.join(features_path, './labels_test.npy/'))

max_event_length = int(params['avg_event_length'] / params['hop_length'])
mel_bins = params['nb_mel_bins'] * params['nb_channels']
input_shape = (max_event_length, mel_bins, 1)
nb_classes = params['nb_classes']

# PARAMS
models = [3]

# https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
def cnn0():
    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),#8
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Dropout(0.5),
        Reshape((23,-1)),
        Bidirectional(LSTM(256)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(nb_classes, activation='sigmoid')
    ])
    return model

# CONV-POOL-CONV-POOL
def cnn1():
    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape), #8
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Dropout(0.5),
        Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same'), #16
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Dropout(0.5),
        Reshape((11,-1)),
        Bidirectional(LSTM(64)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(120, activation='relu'),
        Dense(nb_classes, activation='sigmoid')
    ])
    return model

# CONV-POOL-CONV-POOL (increasing filters)
def cnn2():
    model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape), #32
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'), #64
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'), #128
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Reshape((5, -1)),
    Bidirectional(LSTM(256)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(nb_classes, activation='sigmoid')
    ])

    return model

# CONV-CONV-POOL-CONV-CONV-POOL
def cnn3(lstm_nodes=128, lrn_rate=1e-5):
    model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same',input_shape=input_shape), #16
    Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same'),#16
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),#32
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),#32
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Reshape((11, -1)),
    Bidirectional(LSTM(lstm_nodes)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64),
    Dense(nb_classes, activation='sigmoid')
    ])

    opt = Adam(lr=lrn_rate)
    model.compile(optimizer=opt,
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy', get_f1])
    model.summary()

    return model

# DATA GENERATORS
data_generator = DataGenerator(train_events_path, train_labels_path, test_events_path, test_labels_path, params['batch_size'])
validation_data = data_generator.get_validation_data()
test_data = data_generator.get_test_data()

# DATA STATISTICS
train_counts, test_counts = data_generator.get_statistics()

train_events = np.sum(train_counts)
test_events = np.sum(test_counts)
class_weights_train = train_counts / np.sum(train_counts)
class_weights = {0:class_weights_train[0], 1: class_weights_train[1], 2: class_weights_train[2], 3: class_weights_train[3]}
class_weights_test = test_counts / np.sum(test_counts)
print(class_weights_test)

# GRID SEARCH
param_grid = dict(lstm_nodes=[128,256], batch_size=[16, 32, 64]) #learn_rate=[1e-3, 1e-5, 1e-7]

model = KerasClassifier(build_fn=cnn3, epochs=10, batch_size=32, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=2)
grid_result = grid.fit(data_generator.get_train_data()[0], data_generator.get_train_data()[1])

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_.model

print(best_params)

'''
for m in models:
    checkpoint_path = os.path.abspath(os.path.join(params['output_dir'], './results/checkpoints{m}/'))
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_get_f1',
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_get_f1', patience=10)

    if m == 0:
        model = cnn0()
    elif m == 1:
        model = cnn1()
    elif m == 2:
        model = cnn2()
    elif m == 4:
        model = cnn3()
    history0 = model.fit(data_generator,
                         validation_data=validation_data,
                         epochs=params['nb_epochs'],
                         batch_size=params['batch_size'],
                         callbacks=[ckpt_callback])

    predictions = model.predict(test_data[0])
    predictions = convert_to_binary(predictions)
    print(f1_score(test_data[1], predictions, average=None))

    plot_history(history0, m)
    '''