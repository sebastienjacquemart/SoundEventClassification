import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils import spectro_augment
import parameter
params = parameter.get_params()

max_event_length = int(params['avg_event_length'] / params['hop_length'])
mel_bins = params['nb_mel_bins'] * params['nb_channels']
nb_classes = params['nb_classes']

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, train_events_file, train_labels_file, test_events_file, test_labels_file, batch_size=32, input_size=(max_event_length, mel_bins), shuffle=True, split=0.5):
        self.train_events_file = train_events_file
        self.train_labels_file = train_labels_file
        self.test_events_file = test_events_file
        self.test_labels_file = test_labels_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.split = split

        self.train_events = np.load(train_events_file, allow_pickle=True)
        self.train_labels = np.load(train_labels_file, allow_pickle=True)
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels, num_classes=nb_classes)
        self.test_events = np.load(test_events_file, allow_pickle=True)
        self.test_labels = np.load(test_labels_file, allow_pickle=True)
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels, num_classes=nb_classes)

        if params['augment'] == 0:
            self.train_events, self.train_labels = self._undersample_data(self.train_events, self.train_labels)
        else:
            self.train_events, self.train_labels = self.augment_data(self.train_events, self.train_labels)

        self.num_classes = len(np.unique(self.train_labels))

        self.train_indexes = np.arange(len(self.train_events))
        self.test_indexes = np.arange(len(self.test_events))

        self.validation_indexes, self.test_indexes = train_test_split(self.test_indexes, test_size=self.split, shuffle=self.shuffle)

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.train_indexes)
            np.random.shuffle(self.validation_indexes)
            np.random.shuffle(self.test_indexes)

    def __get_data(self, events, labels, indexes):
        X_batch = events[indexes]
        y_batch = labels[indexes]

        return X_batch, y_batch

    def __getitem__(self, index):
        indexes = self.train_indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(self.train_events, self.train_labels, indexes)
        return X, y

    def __len__(self):
        return int(np.ceil(len(self.train_indexes) / self.batch_size))

    def get_validation_data(self):
        X, y = self.__get_data(self.test_events, self.test_labels, self.validation_indexes)
        return X, y

    def get_test_data(self):
        X, y = self.__get_data(self.test_events, self.test_labels, self.test_indexes)
        return X,y

    def get_train_data(self):
        X, y = self.__get_data(self.train_events, self.train_labels, self.train_indexes)
        return X, y

    def get_statistics(self):
        train_counts = np.sum(self.train_labels, axis=0)
        test_counts = np.sum(self.test_labels, axis=0)

        return train_counts, test_counts

    def _undersample_data(self, events, labels):
        class_counts = np.sum(labels, axis=0)
        min_samples = np.min(class_counts)
        num_classes = labels.shape[1]

        undersampled_events = []
        undersampled_labels = []

        for class_idx in range(num_classes):
            class_indices = np.where(labels[:, class_idx] == 1)[0]
            selected_indices = np.random.choice(class_indices, size=int(min_samples), replace=False)
            undersampled_events.extend(events[selected_indices])
            undersampled_labels.extend(labels[selected_indices])

        undersampled_events = np.array(undersampled_events)
        undersampled_labels = np.array(undersampled_labels)

        # Shuffle the undersampled data
        shuffle_indices = np.arange(len(undersampled_labels))
        np.random.shuffle(shuffle_indices)
        undersampled_events = undersampled_events[shuffle_indices]
        undersampled_labels = undersampled_labels[shuffle_indices]

        return undersampled_events, undersampled_labels

    #https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
    def augment_data(self, events, labels):
        class_counts = np.sum(labels, axis=0)
        unique_classes = np.arange(len(class_counts))
        max_count = np.max(class_counts)

        # Calculate the number of instances needed for each class to match the max count
        instances_needed = max_count - class_counts
        instances_needed = [int(x) for x in instances_needed]

        augmented_X = []
        augmented_y = []

        for label, needed in zip(unique_classes, instances_needed):
            # Select the instances of the current class
            class_X = events[np.argmax(labels, axis=1) == label]
            class_y = labels[np.argmax(labels, axis=1) == label]

            # Augment the instances until reaching the desired count
            for _ in range(needed):
                random_idx = np.random.randint(len(class_X))
                mel = class_X[random_idx]
                augmented_mel = spectro_augment(mel)
                augmented_X.append(augmented_mel)
                augmented_y.append(class_y[random_idx])

        # Concatenate the augmented data with the original data
        augmented_X = np.concatenate([events, np.array(augmented_X)])
        augmented_y = np.concatenate([labels, np.array(augmented_y)])

        return augmented_X, augmented_y
