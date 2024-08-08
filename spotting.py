import os

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram

from settings import concord_path, background_path
from sound_factory import load_audio


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # Приводим к фиксированному размеру выхода
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


class KeywordSpottingDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.audio_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.wav')]
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=64)  # Замените sample_rate на нужное значение

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = None
        try:
            waveform, sample_rate = load_audio(audio_path)
        except Exception as e:
            print(audio_path)
            print(e)

        mel_spec1 = self.mel_spec(waveform)

        label1 = 0 if os.path.basename(audio_path).startswith("negative") else 1

        return mel_spec1, label1


def start_spotting(dataset_path):
    dataset = KeywordSpottingDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    print(f"Epoch stared")
    for epoch in range(num_epochs):
        data_count = len(dataloader)
        index = 0
        for mel_spec, label in dataloader:
            index += 1
            if index % 4 == 0:
                print("#", end="")
            optimizer.zero_grad()
            # Убедитесь, что данные имеют правильную размерность
            if mel_spec.dim() == 3:
                mel_spec = mel_spec.unsqueeze(1)  # Добавление размерности для каналов (если требуется)
            output = model(mel_spec)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Сохранение модели
    torch.save(model.state_dict(), "new_keyword_spotting_model.pth")
