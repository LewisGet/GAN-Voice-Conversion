import os
import numpy as np

from models.cyclegan_vc2 import CycleGAN2
from speech_tools import load_pickle, sample_train_data_single

import random

np.random.seed(300)

dataset = 'vcc2018'
src_speaker = 'b'
trg_speaker = 'a'
model_name = 'cyclegan_vc2_two_step'

os.makedirs(os.path.join('experiments', dataset, model_name, 'checkpoints'), exist_ok=True)

log_dir = os.path.join('logs', '{}_{}'.format(dataset, model_name))
os.makedirs(log_dir, exist_ok=True)

data_dir = os.path.join('datasets', dataset)
exp_dir = os.path.join('experiments', dataset)

train_A_dir = os.path.join(data_dir, 'training', src_speaker)
train_B_dir = os.path.join(data_dir, 'training', trg_speaker)

exp_A_dir = os.path.join(exp_dir, src_speaker)
exp_B_dir = os.path.join(exp_dir, trg_speaker)

sampling_rate = 16000

num_mcep = 52
frame_period = 5.0
n_frames = 128

start_at = 35000

# Training parameters
num_iterations = 15100 + start_at
mini_batch_size = 5
generator_learning_rate = 0.0002
discriminator_learning_rate = 0.0001
lambda_cycle_min = 2
lambda_cycle_max = 15
lambda_identity_min = 3
lambda_identity_max = 5

print('Loading cached data...')
coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
    os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
    os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))

model = CycleGAN2(num_features=num_mcep, batch_size=mini_batch_size, log_dir=log_dir)
if start_at is not 0:
    model.load(os.path.join('experiments', dataset, model_name, 'checkpoints', 'cyclegan_vc2_two_step_' + str(start_at) + '.ckpt'))
iteration = start_at

while iteration <= num_iterations:
    dataset_B = sample_train_data_single(dataset=coded_sps_B_norm, n_frames=n_frames)
    n_samples = dataset_B.shape[0]

    for i in range(n_samples // mini_batch_size):
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size

        b_loss = model.train_discriminator_b_to_a(input_B=dataset_B[start: end], discriminator_learning_rate=discriminator_learning_rate)

        if iteration % 10 == 0:
            print (b_loss)

        if iteration % 2500 == 0:
            print('Checkpointing...')
            model.save(directory=os.path.join('experiments', dataset, model_name, 'checkpoints'),
                       filename='{}_{}.ckpt'.format(model_name, iteration))

        iteration += 1
