# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
import librosa

def get_params():
    # ########### default parameters ##############
    params = dict(
        # DATA DIRECTORY
        data_dir = '/Users/sebastienjacquemart/Library/Mobile Documents/com~apple~CloudDocs/Documents_2/PRIVE/SCHOOL/MasterAI/Thesis/Code/data/Data_Sebastien/data_sebastien2/feature_prep',
        # METADATA DIRECTORIES
        metadata_dir='./metadata/',
        metadata_txt_dir='./txt/',
        metadata_csv_dir='./metadata_dev/',
        #metadata_csv_dir='./metadata_dev44k_128_O/',
        # AUDIO DIRECTORIES
        wav_dir='./foa_dev/',
        # OUTPUT DIRECTORY
        output_dir='./output/',
        name='seb',
        # FEATURE PARAMS
        fs=44100,
        window_length=1024,
        hop_length=128,
        nb_channels=2,
        nb_classes=4,

        max_audio_len_s=15,
        max_audio_len=5165,

        nb_mel_bins=128,
        # MODEL PARAMS
        batch_size=32,
        nb_epochs=100,
        augment=1,
        # CHICK STATISTICS
        avg_event_length=5888, # AFTER RUNNING STATISTICS

        # STATISTICS
        mode_stats=2,
        # FEATURES
        mode_feat=1
    )

    params['nfft'] = 2 ** (params['window_length'] - 1).bit_length()

    params['mel_wts'] = librosa.filters.mel(sr=params['fs'], n_fft=params['nfft'], n_mels=params['nb_mel_bins']).T

    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
