import os
import numpy as np

from models.cyclegan_vc2 import CycleGAN2
from speech_tools import load_pickle, sample_train_data

import random

import glob
from itertools import permutations

np.random.seed(300)

sampling_rate = 22050

num_mcep = 36
frame_period = 5.0
n_frames = 128

# if 0 will not load existed model 
start_at = 0
iteration = start_at

# Training parameters
num_iterations = 5100 + start_at
mini_batch_size = 1
generator_learning_rate = 0.0002
discriminator_learning_rate = 0.0001
lambda_cycle = 10
lambda_identity_min = 8
lambda_identity_max = 15
lambda_identity = random.randint(lambda_identity_min, lambda_identity_max)

dataset = 'vcc2018'
model_name = 'cyclegan_vc2_two_step'
os.makedirs(os.path.join('experiments', dataset, model_name, 'checkpoints'), exist_ok=True)
log_dir = os.path.join('logs', '{}_{}'.format(dataset, model_name))
os.makedirs(log_dir, exist_ok=True)

data_dir = os.path.join('datasets', dataset)
exp_dir = os.path.join('experiments', dataset)
train_basic_dir = os.path.join(data_dir, 'vcc2018_training')

dirs = glob.glob(os.path.join(train_basic_dir, 'VCC*'))

data_a = list()
data_b = list()

for i in permutations(dirs, 2):
    train_A_dir = i[0]
    train_B_dir = i[1]

    exp_A_dir = os.path.join(exp_dir, os.path.basename(i[0]))
    exp_B_dir = os.path.join(exp_dir, os.path.basename(i[1]))

    data_a.append(load_pickle(os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))[0])
    data_b.append(load_pickle(os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))[0])

model = CycleGAN2(num_features=num_mcep, batch_size=mini_batch_size, log_dir=log_dir)

if start_at is not 0:
    model.load(os.path.join('experiments', dataset, model_name, 'checkpoints', 'cyclegan_vc2_two_step_' + str(start_at) + '.ckpt'))

while iteration <= num_iterations:
    train_ab_index = random.randint(0, len(data_a) - 1)

    dataset_A, dataset_B = sample_train_data(
        dataset_A=data_a[train_ab_index],
        dataset_B=data_b[train_ab_index],
        n_frames=n_frames
    )

    n_samples = dataset_A.shape[0]

    for i in range(n_samples // mini_batch_size):
        #if iteration > 10000:
        #    lambda_identity = 0

        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size

        generator_loss, discriminator_loss = model.train(input_A=dataset_A[start:end], input_B=dataset_B[start:end],
                                                         lambda_cycle=lambda_cycle, lambda_identity=lambda_identity,
                                                         generator_learning_rate=generator_learning_rate,
                                                         discriminator_learning_rate=discriminator_learning_rate)

        if iteration % 10 == 0:
            print('Iteration: {:07d}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(iteration,
                                                                                                   generator_loss,
                                                                                                   discriminator_loss))
        if iteration % 2500 == 0:
            print('Checkpointing...')
            model.save(directory=os.path.join('experiments', dataset, model_name, 'checkpoints'),
                       filename='{}_{}.ckpt'.format(model_name, iteration))

        if iteration % 100 == 0:
            lambda_identity = random.randint(lambda_identity_min, lambda_identity_max)

        iteration += 1
