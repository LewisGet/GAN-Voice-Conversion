import os
import time

from speech_tools import *

import glob
from itertools import permutations

dataset = 'vcc2018'

sampling_rate = 22050

num_mcep = 36
frame_period = 5.0
n_frames = 128

data_dir = os.path.join('datasets', dataset)
exp_dir = os.path.join('experiments', dataset)

train_basic_dir = os.path.join(data_dir, 'vcc2018_training')

dirs = glob.glob(os.path.join(train_basic_dir, 'VCC*'))

for i in permutations(dirs, 2):
    train_A_dir = i[0]
    train_B_dir = i[1]

    print (train_A_dir, train_B_dir)

    exp_A_dir = os.path.join(exp_dir, os.path.basename(i[0]))
    exp_B_dir = os.path.join(exp_dir, os.path.basename(i[1]))

    os.makedirs(exp_A_dir, exist_ok=True)
    os.makedirs(exp_B_dir, exist_ok=True)

    print('Loading Wavs...')

    start_time = time.time()

    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    print('Extracting acoustic features...')

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate,
                                                                     frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs=wavs_B, fs=sampling_rate,
                                                                     frame_period=frame_period, coded_dim=num_mcep)

    print('Calculating F0 statistics...')

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

    print('Normalizing data...')

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_B_transposed)

    print('Saving data...')
    save_pickle(os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)),
                (coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A))
    save_pickle(os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)),
                (coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B))

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
        time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
