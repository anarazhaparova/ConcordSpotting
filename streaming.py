import numpy as np
import pyaudio
import torch
from torchaudio.transforms import MelSpectrogram

from settings import spotting_model
from spotting import SimpleCNN

# Настройки PyAudio
chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 16000

model = SimpleCNN()
model.load_state_dict(torch.load(spotting_model,
                                 weights_only=True))
model.eval()

# Инициализация PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)


# Функция для обработки стрима
def process_stream():
    print("Listening for the keyword...")
    while True:
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        mel_spec = MelSpectrogram(sample_rate=rate, n_mels=64)(audio_tensor)
        output = model(mel_spec.unsqueeze(1))
        _, predicted = torch.max(output, 1)
        if predicted.item() == 1:
            print("Keyword detected!")
            # Здесь можно добавить действие при обнаружении ключевого слова


process_stream()

# Закрытие стрима
stream.stop_stream()
stream.close()
p.terminate()
