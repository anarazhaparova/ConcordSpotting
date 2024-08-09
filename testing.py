import os.path

import torch
from torchaudio.transforms import MelSpectrogram

from settings import spotting_model, target_sample_rate
from sound_factory import load_audio
from spotting import SimpleCNN


def test_on_real_input(audio_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(spotting_model, weights_only=True))
    model.eval()
    waveform, sr = load_audio(audio_path)

    mel_spec = MelSpectrogram(sample_rate=target_sample_rate, n_mels=64)(waveform)
    output = model(mel_spec.unsqueeze(1))
    _, predicted = torch.max(output, 1)

    print(f'Prediction on {audio_path}: {"Keyword detected" if predicted.item() == 1 else "No keyword detected"}')


def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def preprocess_audio(audio_path, mel_spec_transform):
    waveform, sample_rate = load_audio(audio_path)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_spec = mel_spec_transform(waveform)
    if mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(1)
    return mel_spec


def predict(model, mel_spec):
    with torch.no_grad():
        output = model(mel_spec)
        predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label


def test_keyword_spotting(test_audio_path):
    mel_spec_transform = MelSpectrogram(sample_rate=16000, n_mels=64)
    model = load_model(spotting_model)
    mel_spec = preprocess_audio(test_audio_path, mel_spec_transform)
    predicted_label = predict(model, mel_spec)
    print(
        f'Prediction on {os.path.basename(test_audio_path)}: '
        f'{"Keyword detected" if predicted_label == 1 else "No keyword detected"}')
    return predicted_label

for filename in os.listdir("test_files"):
    if filename.endswith('.wav'):
        filepath = os.path.join("test_files", filename)
        test_keyword_spotting(filepath)
