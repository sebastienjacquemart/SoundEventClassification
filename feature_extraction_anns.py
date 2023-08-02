import os
import csv
import numpy as np
import parameter
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import _load_audio, _spectrogram, _get_mel_spectrogram, _normalize, plot_mel, plot_spec, plot_audio


# Init parameters
params = parameter.get_params()

# Init mode
mode = params['mode_feat']

# Specify the paths of the input and output folders
metadata_path = os.path.abspath(os.path.join(params['data_dir'],params['metadata_txt_dir']))
audio_path = os.path.abspath(os.path.join(params['data_dir'],params['wav_dir']))

output_path = params['output_dir']

events_out_path = os.path.abspath(os.path.join(output_path, './features_' + params['name'] + '/'))
#events_out_path1 = os.path.abspath(os.path.join(output_path, './features_' + params['name'] + '_bettertest2/'))

if mode != 0:
    if not os.path.exists(events_out_path):
        os.makedirs(events_out_path)
    #if not os.path.exists(events_out_path1):
        #os.makedirs(events_out_path1)
metadata_out_path = output_path


# Initialize the metadata list
metadata = []

# Set the timestamp boundaries for assigning labels
label_25_timestamp = datetime(2022, 10, 26, 18, 44, 30)  # YYYY, MM, DD, HH, MM, SS
label_50_timestamp = datetime(2022, 10, 26, 20, 47, 00)
label_75_timestamp = datetime(2022, 10, 26, 22, 51, 00)

# Initialize variables for event length calculation
num_files = len(os.listdir(metadata_path))
total_event_length_s = 0
max_event_length_s = 0
total_events = 0

def define_label(ts):
    if ts < label_25_timestamp:
        label = 0
    elif label_25_timestamp <= ts < label_50_timestamp:
        label = 1
    elif label_50_timestamp <= ts < label_75_timestamp:
        label = 2
    else:
        label = 3

    return label
def divide_in_chunks(audio, start_time_original, end_time_original, label_original):
    event = []
    label = []

    event_length = end_time_original - start_time_original
    nb_avg_events = event_length // params['avg_event_length']
    for i in range(nb_avg_events + 1):
        start_time = start_time_original + i * params['avg_event_length']
        end_time = start_time + params['avg_event_length']
        if end_time <= end_time_original:
            event_audio = audio[start_time:end_time, :]
        else:
            end_time = end_time_original
            pad_event_audio = audio[start_time:end_time, :]
            pad_length = start_time + params['avg_event_length'] - end_time_original
            event_audio = np.pad(pad_event_audio, ((0, pad_length), (0, 0)), mode='constant')

        audio_spec = _spectrogram(event_audio)
        audio_mel = _get_mel_spectrogram(audio_spec)
        audio_mel = _normalize(audio_mel)

        event.append(audio_mel)
        label.append(label_original)

    return event, label

def split_filenames(txt_files):
    filenames = []
    filenames_train = []
    filenames_test = []

    for txt_file in os.listdir(txt_files):
        if txt_file.endswith('.txt'):
            filename = os.path.splitext(txt_file)[0]
            filenames.append(filename)

            if int(txt_file[4]) in [1,2,3,4,5]:
                filenames_train.append(filename)
            if int(txt_file[4]) in [6]:
                filenames_test.append(filename)

    return filenames, filenames_train, filenames_test

def metadata_extraction(filename):
    # Initialize statistics
    total_length = 0
    total_length_overlap = 0
    max_length = 0
    min_length = np.inf
    num_events = 0
    num_events_overlap = 0

    txt_file_path = os.path.join(metadata_path, filename + '.txt')
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        num_events += 1

        line = line.strip()
        segments = line.split('\t')
        start_time_s = float(segments[0])
        end_time_s = float(segments[1])
        if end_time_s > params['max_audio_len_s']:
            end_time_s = params['max_audio_len_s']

        # Update statistics
        cur_length = end_time_s - start_time_s
        total_length += cur_length
        if len(segments) == 3 and segments[2].upper() == 'O':
            num_events_overlap += 1
            total_length_overlap += cur_length
        max_length = max(max_length, cur_length)
        min_length = min(min_length, cur_length)

    return [num_events, num_events_overlap, total_length, total_length_overlap, max_length, min_length]
def feature_extraction1(audio, file_label):
    # Initialize events+labels
    file_events=[]
    file_labels=[]

    txt_file_path = os.path.join(metadata_path, filename + '.txt')
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        segments = line.split('\t')
        start_time_s = float(segments[0])
        end_time_s = float(segments[1])
        if end_time_s > params['max_audio_len_s']:
            end_time_s = params['max_audio_len_s']

        start_time_original = int(start_time_s * params['fs'])
        end_time_original = int(end_time_s * params['fs'])
        event, label = divide_in_chunks(audio, start_time_original, end_time_original, file_label)

        file_events.extend(event)
        file_labels.extend(label)

    return file_events, file_labels

def feature_extraction2(audio, file_label):
    file_segments, file_labels = divide_in_chunks(audio, 0, params['max_audio_len_s'] * params['fs'], file_label)

    return file_segments, file_labels

# MAIN PART OF THE CODE:
filenames, filenames_train, file_names_test = split_filenames(metadata_path)

train_events = []
test_events = []
train_labels = []
test_labels = []
train_events1 = []
test_events1 = []
train_labels1 = []
test_labels1 = []

for filename in tqdm(filenames):
    # Load the corresponding WAV file
    wav_file = filename + '.wav'
    wav_path = os.path.join(audio_path, wav_file)
    audio, _ = _load_audio(wav_path)

    # Determine the label based on the timestamp
    timestamp_str = filename.split('_')[1] + '_' + filename.split('_')[2]
    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    file_label = define_label(timestamp)
    #file_label = None

    if mode == 0 or mode == 3: # EXTRACT METADATA
        stats = metadata_extraction(filename)
        metadata.append([filename, file_label, stats[0], stats[1], round(stats[2], 6), round(stats[3], 6), round(stats[4], 6), round(stats[5], 6), filename in file_names_test])

    if mode == 1 or mode == 3: # FEATURE EXTRACTION 1: Extract ONLY events
        file_events, file_labels = feature_extraction1(audio, file_label)

        if filename in filenames_train:
            train_events.extend(file_events)
            train_labels.extend(file_labels)
        else:
            test_events.extend(file_events)
            test_labels.extend(file_labels)

    if mode == 2 or mode == 3: #FEATURE EXTRACTION 2: Extract events + background noise
        file_segments, file_labels2 = divide_in_chunks(audio, 0, params['max_audio_len_s'] * params['fs'], file_label)
        if filename in filenames_train:
            train_events1.extend(file_segments)
            train_labels1.extend(file_labels2)
        else:
            test_events1.extend(file_segments)
            test_labels1.extend(file_labels2)


# Convert the events and labels to numpy arrays
if mode == 1 or mode == 3:
    events_array_train = np.array(train_events)
    labels_array_train = np.array(train_labels)
    events_array_test = np.array(test_events)
    labels_array_test = np.array(test_labels)

    events_file_train = os.path.join(events_out_path, 'events_train.npy')
    labels_file_train = os.path.join(events_out_path, 'labels_train.npy')
    events_file_test = os.path.join(events_out_path, 'events_test.npy')
    labels_file_test = os.path.join(events_out_path, 'labels_test.npy')

    np.save(events_file_train, events_array_train)
    np.save(labels_file_train, labels_array_train)
    np.save(events_file_test, events_array_test)
    np.save(labels_file_test, labels_array_test)

if mode == 2 or mode == 3:
    events_array_train1 = np.array(train_events1)
    labels_array_train1 = np.array(train_labels1)
    events_array_test1 = np.array(test_events1)
    labels_array_test1 = np.array(test_labels1)

    events_file_train1 = os.path.join(events_out_path1, 'events_train1.npy')
    labels_file_train1 = os.path.join(events_out_path1, 'labels_train1.npy')
    events_file_test1 = os.path.join(events_out_path1, 'events_test1.npy')
    labels_file_test1 = os.path.join(events_out_path1, 'labels_test1.npy')

    np.save(events_file_train1, events_array_train1)
    np.save(labels_file_train1, labels_array_train1)
    np.save(events_file_test1, events_array_test1)
    np.save(labels_file_test1, labels_array_test1)

if mode == 1 or mode == 2 or mode == 3:
    print("Events and labels saved as numpy files.")

if mode == 0 or mode == 3:
    # Save the metadata as a CSV file
    #metadata.sort(key=lambda x: datetime.strptime(x[0].split('_')[0] + '_' + x[0].split('_')[1], '%Y%m%d_%H%M%S'))

    metadata_file = os.path.join(metadata_out_path, 'metadata_' + params['name'] + '_bettertest.csv')
    with open(metadata_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label', 'num_events', 'num_events_overlap','total_length', 'total_length_overlap', 'min_length', 'max_length', 'is_test?'])
        writer.writerows(metadata)

    print("Metadata file created.")
    os.rename(os.path.abspath(os.path.join(output_path, './statistics_' + params['name'] + '/')), os.path.abspath(os.path.join(output_path, './statistics_' + params['name'] + '_notUTD/')))
    print('Statistics not up to date!')

