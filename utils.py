import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf

import parameter

params = parameter.get_params()

def _load_audio(audio_path):
    fs, audio = wav.read(audio_path)
    audio = audio[:, :params['nb_channels']] / 32768.0 + 1e-8
    if audio.shape[0] < params['max_audio_len_s'] * params['fs']:
        zero_pad = np.random.rand(params['max_audio_len_s'] * params['fs'] - audio.shape[0], audio.shape[1]) * 1e-8
        audio = np.vstack((audio, zero_pad))
    elif audio.shape[0] > params['max_audio_len_s'] * params['fs']:
        audio = audio[:params['max_audio_len_s'] * params['fs'], :]
    return audio, fs

def _spectrogram(audio_input):
    _nb_ch = audio_input.shape[1] # number of channels = 2
    nb_bins = params['nfft'] // 2
    spectra = np.zeros((int(params['avg_event_length']/params['hop_length']), nb_bins + 1, _nb_ch), dtype=complex)
    for ch_cnt in range(_nb_ch):
        stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=params['nfft'], hop_length=params['hop_length'],
                                        win_length=params['window_length'], window='hann')
        spectra[:, :, ch_cnt] = stft_ch[:, :int(params['avg_event_length']/params['hop_length'])].T
    return spectra

def _get_mel_spectrogram(linear_spectra):
    mel_feat = np.zeros((linear_spectra.shape[0], params['nb_mel_bins'], linear_spectra.shape[-1]))
    for ch_cnt in range(linear_spectra.shape[-1]):
        mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
        mel_spectra = np.dot(mag_spectra, params['mel_wts']) # same if I want to extract gammatone
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        mel_feat[:, :, ch_cnt] = log_mel_spectra
    mel_feat = mel_feat.reshape((linear_spectra.shape[0], params['nb_mel_bins'] * linear_spectra.shape[-1]))
    return mel_feat

def _normalize(mel_feat):
    spec_scaler = preprocessing.StandardScaler()
    spec_scaler.partial_fit(mel_feat)
    feat_file = spec_scaler.transform(mel_feat)

    return feat_file


def time_masking(spec, T=30, num_masks=1, replace_value=0.0):
    """
    Apply frequency masking to a spectrogram.

    Parameters:
        spec (np.ndarray): Input spectrogram (2D NumPy array).
        F (int): Maximum number of consecutive frequency bins to mask.
        num_masks (int): Number of masks to apply.
        replace_value (float): Value to replace masked elements.

    Returns:
        np.ndarray: Frequency-masked spectrogram (2D NumPy array).
    """
    masked_spec = np.copy(spec)

    for _ in range(num_masks):
        t = np.random.randint(1, spec.shape[0])
        if t - T >= 0:
            t0 = np.random.randint(t - T, t-1)
        else:
            t0 = np.random.randint(0, t)

        mask = np.concatenate((np.ones((t0, spec.shape[1])), np.zeros((t-t0, spec.shape[1])), np.ones((spec.shape[0] - t , spec.shape[1]))), axis=0)
        masked_spec *= mask.astype(np.float32)

    # Replace the masked elements with the specified replace_value
    masked_spec += (1 - mask).astype(np.float32) * replace_value

    return masked_spec
def frequency_masking(spec, F=40, num_masks=1, replace_value=0.0):
    """
    Apply time masking to a spectrogram.

    Parameters:
        spec (np.ndarray): Input spectrogram (2D NumPy array).
        T (int): Maximum number of consecutive time frames to mask.
        num_masks (int): Number of masks to apply.
        replace_value (float): Value to replace masked elements.

    Returns:
        np.ndarray: Time-masked spectrogram (2D NumPy array).
    """
    masked_spec = np.copy(spec)

    for _ in range(num_masks):
        f = np.random.randint(1, spec.shape[1])
        if f - F >= 0:
            f0 = np.random.randint(f - F, f)
        else:
            f0 = np.random.randint(0, f)

        mask = np.concatenate((np.ones((spec.shape[0], f0)), np.zeros((spec.shape[0], f-f0)), np.ones((spec.shape[0], spec.shape[1] - f))), axis=1)
        masked_spec *= mask.astype(np.float32)

    # Replace the masked elements with the specified replace_value
    masked_spec += (1 - mask).astype(np.float32) * replace_value

    return masked_spec

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    n_steps, n_mels = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = int(max_mask_pct * n_mels)
    #for _ in range(n_freq_masks):
    aug_spec = frequency_masking(aug_spec, F=freq_mask_param, replace_value=mask_value)

    time_mask_param = int(max_mask_pct * n_steps)
    #for _ in range(n_time_masks):
    aug_spec = time_masking(aug_spec, T=time_mask_param, replace_value=mask_value)

    #plot_mel(aug_spec)
    return aug_spec

def convert_to_binary(predictions):
    binary_indices = np.argmax(predictions, axis=1)
    binary_predictions = np.zeros_like(predictions)
    binary_predictions[np.arange(len(binary_indices)), binary_indices] = 1

    return binary_predictions
def plot_mel(spectrogram, sample_rate=params['fs'], hop_length=params['hop_length']):
    # Convert the spectrogram to decibels (dB)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    # Determine the time and frequency axes
    num_frames = spectrogram.shape[0]
    num_bins = spectrogram.shape[1]
    times = np.arange(num_frames) * hop_length / sample_rate
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=num_bins)

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the spectrogram
    im = ax.imshow(spectrogram_db.T, origin='lower', aspect='auto',
                   extent=[times[0], times[-1], frequencies[0], frequencies[-1]])

    # Set the colorbar
    cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.set_label('Magnitude (dB)')

    # Set the labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram')

    # Show the plot
    plt.show()

def plot_spec(spectrogram, sample_rate=params['fs'], hop_length=params['hop_length']):
    num_frames, num_bins, num_channels = spectrogram.shape

    # Convert the spectrogram to decibels (dB) for each channel
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # Determine the time and frequency axes
    times = np.arange(num_frames) * hop_length / sample_rate
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=num_bins)

    # Create subplots for each channel
    fig, axes = plt.subplots(num_channels, 1, figsize=(8, 6 * num_channels))

    # Iterate over each channel and plot the spectrogram
    for i in range(num_channels):
        ax = axes[i]
        channel_spectrogram_db = spectrogram_db[:, :, i]

        # Plot the spectrogram
        librosa.display.specshow(channel_spectrogram_db.T, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='linear', ax=ax)

        # Set the colorbar
        #cbar = plt.colorbar(format='%+2.0f dB', ax=ax)
        #cbar.set_label('Magnitude (dB)')

        # Set the labels and title for each channel
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram - Channel {i+1}')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_audio(y, sr=params['fs']):
    # Create the time axis
    t = np.arange(0, len(y)) / sr

    # Plot the audio signal
    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid()
    plt.show()

def plot_history(history, n):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_f1 = history.history['get_f1']
    val_f1 = history.history['val_get_f1']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plot_metric_or_loss(epochs, train_acc, val_acc, 'Accuracy')
    plt.subplot(1, 3, 2)
    plot_metric_or_loss(epochs, train_loss, val_loss, 'Loss')
    plt.subplot(1, 3, 3)
    plot_metric_or_loss(epochs, train_f1, val_f1, 'F1-score')
    plt.tight_layout()

    plt.savefig(os.path.abspath(os.path.join(params['output_dir'], f'./results/crnn{n}_.png')), dpi=300)


def plot_metric_or_loss(x, y1, y2, metric_or_loss):
    plt.plot(x, y1, 'b-', label=f'Training {metric_or_loss}')
    plt.plot(x, y2, 'r-', label=f'Validation {metric_or_loss}')
    plt.title(f'Training and Validation {metric_or_loss}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_or_loss)
    plt.legend()

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val