import os
import glob

from models.cyclegan_vc2 import CycleGAN2
from speech_tools import *

dataset = 'vcc2018'
src_speaker = 'b'
trg_speaker = 'a'
model_name = 'cyclegan_vc2_two_step'

data_dir = os.path.join('datasets', dataset)
exp_dir = os.path.join('experiments', dataset)
model_dir = os.path.join(exp_dir, model_name, 'checkpoints')
all_models = glob.glob(os.path.join(model_dir, model_name + "_*.ckpt*"))

def get_train_step_by_name(str):
    return int(str.split(os.path.join(model_dir, model_name + "_"))[1].split(".ckpt")[0])

last_train_model_step = max([get_train_step_by_name(i) for i in all_models])

eval_A_dir = os.path.join(data_dir, 'evaluation', src_speaker)
eval_B_dir = os.path.join(data_dir, 'reference', trg_speaker)
exp_A_dir = os.path.join(exp_dir, src_speaker)
exp_B_dir = os.path.join(exp_dir, trg_speaker)

validation_A_output_dir = os.path.join('experiments', dataset, model_name,
                                       'converted_{}_to_{}'.format(src_speaker, trg_speaker))
validation_B_output_dir = os.path.join('experiments', dataset, model_name,
                                       'converted_{}_to_{}'.format(trg_speaker, src_speaker))

os.makedirs(validation_A_output_dir, exist_ok=True)
os.makedirs(validation_B_output_dir, exist_ok=True)

sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128
model_step = last_train_model_step

print('Loading cached data...')
coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
    os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
    os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))

model = CycleGAN2(num_features=num_mcep, batch_size=1, mode='test')
model.load(
    filepath=os.path.join('experiments', dataset, model_name, 'checkpoints', '{}_{}.ckpt'.format(model_name, str(model_step))))

print('Generating Validation Data B from A...')
for file in glob.glob(eval_A_dir + '/*.wav'):
    wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
    wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                    mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
    coded_sp_transposed = coded_sp.T
    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
    coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='A2B')[0]
    if coded_sp_converted_norm.shape[1] > len(f0):
        coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
    coded_sp_converted = coded_sp_converted.T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                             frame_period=frame_period)
    librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed,
                             sampling_rate)

print('Generating Validation Data A from B...')
for file in glob.glob(eval_B_dir + '/*.wav'):
    wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
    wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_B, std_log_src=log_f0s_std_B,
                                    mean_log_target=log_f0s_mean_A, std_log_target=log_f0s_std_A)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
    coded_sp_transposed = coded_sp.T
    coded_sp_norm = (coded_sp_transposed - coded_sps_B_mean) / coded_sps_B_std
    coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='B2A')[0]
    if coded_sp_converted_norm.shape[1] > len(f0):
        coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
    coded_sp_converted = coded_sp_converted_norm * coded_sps_A_std + coded_sps_A_mean
    coded_sp_converted = coded_sp_converted.T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                             frame_period=frame_period)
    librosa.output.write_wav(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed,
                             sampling_rate)
