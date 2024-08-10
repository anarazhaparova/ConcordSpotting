from datetime import datetime

import pyaudio
import numpy as np
import torch

from new_spotting import transform, KeywordModel, n_mfcc
from settings import spotting_model, positive_path
from sound_factory import get_max_sample_length, save_audio, get_waveform_from_np

model = KeywordModel(input_size=n_mfcc)
model.load_state_dict(torch.load(spotting_model,
                                 weights_only=True))
model.eval()

CHUNK = 16000  # Размер чанка (1 секунда для частоты 16kHz)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TARGET_SAMPLE_SIZE = get_max_sample_length(positive_path)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

buffer = np.zeros(TARGET_SAMPLE_SIZE, dtype=np.float32)
buffer_offset = 0

while True:
    data = stream.read(CHUNK)
    new_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

    if buffer_offset + len(new_samples) < TARGET_SAMPLE_SIZE:
        buffer[buffer_offset:buffer_offset + len(new_samples)] = new_samples
        buffer_offset += len(new_samples)
    else:
        remaining_space = TARGET_SAMPLE_SIZE - buffer_offset
        buffer[buffer_offset:] = new_samples[:remaining_space]
        waveform = torch.tensor(buffer).unsqueeze(0)

        mfcc = transform(waveform)

        # Обработать аудио с использованием MFCC и модели
        mfcc = transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(mfcc)
            _, prediction = torch.max(output, dim=1)
            prediction = prediction.item()

        if prediction == 1:
            print("Keyword detected!")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = f"detected/detected_keyword_{timestamp}.wav"
            waveform = buffer / np.max(np.abs(buffer))
            waveform = get_waveform_from_np(waveform)
            save_audio(waveform, file_path)

        # Перезапуск буфера
        buffer_offset = len(new_samples) - remaining_space
        buffer[:buffer_offset] = new_samples[remaining_space:]
