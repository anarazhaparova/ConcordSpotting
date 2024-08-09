import os
import sys
import random

import numpy as np
import soundfile as sf
import torch
from scipy.io import wavfile
from torchaudio import transforms

from settings import *


# class SoundFactory:
def load_audio(audio_path):
    try:
        sample_rate, waveform_np = wavfile.read(audio_path)
        waveform = get_waveform_from_np(waveform_np)

        return waveform, sample_rate
    except Exception as e:
        print(os.path.basename(audio_path))
        print(e)


def save_audio(waveform, audio_path):
    waveform_np = get_numpy_from_waveform(waveform)
    sf.write(audio_path, waveform_np, target_sample_rate)


def get_waveform_from_np(waveform_np):
    is_stereo = len(waveform_np.shape) == 2 and waveform_np.shape[1] == 2
    if is_stereo:
        waveform_np = np.mean(waveform_np, axis=1)

    # Преобразование в тензор PyTorch и нормализация
    waveform = torch.from_numpy(waveform_np).float()

    # Преобразование формы, если необходимо (например, добавление размерности канала)
    if waveform.dim() == 1:  # Если это монофонический сигнал
        waveform = waveform.unsqueeze(0)  # Добавление канала, чтобы получить [1, длина_сигнала]
    return waveform


def get_numpy_from_waveform(waveform):
    waveform_np = waveform.squeeze().numpy()

    # Проверка диапазона значений данных
    if waveform_np.min() < -1.0 or waveform_np.max() > 1.0:
        waveform_np = waveform_np / np.max(np.abs(waveform_np))

    waveform_np = waveform_np.astype(np.float32)
    return waveform_np


def resample_audio_to_target_sr(filepath):
    filename = os.path.basename(filepath)
    waveform, sample_rate = load_audio(filepath)
    sys.stdout.write(f"Файл {filename} частота: {sample_rate}, стерео: {waveform.size(0) > 1}")
    is_stereo = waveform.ndim > 1 and waveform.size(0) > 1
    if sample_rate != target_sample_rate:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    if is_stereo:  # Если аудио стерео
        print()
        print(waveform.shape)
        waveform = waveform.mean(dim=0, keepdim=True)
        print(waveform.shape)
    if sample_rate != target_sample_rate or is_stereo:
        save_audio(waveform, filepath)
        print(f"\n---> Преобразован к частоте дискретизации {target_sample_rate} Гц. Стерео {is_stereo}")
    else:
        sys.stdout.write("\r")
        sys.stdout.flush()
    return waveform, target_sample_rate


def get_max_sample_length(directory):
    max_length = 0
    this_file = ""
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            waveform, sample_rate = load_audio(filepath)
            if max_length < waveform.size(1):
                max_length = waveform.size(1)
                this_file = filename
    print(f"Максимальная длина в сэмплах: {max_length}, в секундах {max_length / target_sample_rate} у {this_file}")
    return max_length


def sampling_audio(filepath, num_samples, output_dir, max_segments=None):
    os.makedirs(output_dir, exist_ok=True)
    waveform, sample_rate = load_audio(filepath)
    total_samples = waveform.size()[1]
    num_segments = total_samples // num_samples

    for i in range(num_segments):
        if max_segments is not None and i > max_segments:
            break

        start_sample = i * num_samples
        end_sample = start_sample + num_samples if i < num_segments - 1 else total_samples
        segment = waveform[:, start_sample:end_sample]
        segment_filename = f"{filepath[:-4]}_segment_{i + 1}.wav"
        output_filepath = f"{output_dir}/{os.path.basename(segment_filename)}"
        if num_samples == segment.size(1):
            save_audio(segment, output_filepath)


def pad_or_trim(waveform, num_samples):
    if waveform.size(1) > num_samples:
        return waveform[:, :num_samples]  # обрезка
    elif waveform.size(1) < num_samples:
        padding = num_samples - waveform.size(1)
        return torch.nn.functional.pad(waveform, (0, padding))  # дополнение
    return waveform


def change_volume(waveform, volume):
    waveform_np = get_numpy_from_waveform(waveform)
    waveform_np *= volume
    waveform = get_waveform_from_np(waveform_np)
    return waveform


# Advice: max noise_level=0.01
def add_white_noise_audio(waveform, noise_level):
    # Добавление шума
    noise = torch.randn_like(waveform) * noise_level
    augmented_waveform = waveform + noise

    return augmented_waveform


# Advice: max shift_amount=0.01
def shift_audio(waveform, shift_amount):
    shift = int(random.uniform(shift_amount * -1, shift_amount) * target_sample_rate)
    augmented_waveform = torch.roll(waveform, shift)

    return augmented_waveform
