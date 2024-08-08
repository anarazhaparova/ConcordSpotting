import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MFCC

from settings import target_sample_rate
from sound_factory import load_audio

# Определение трансформаций
n_mfcc = 13
n_fft = 400
hop_length = 160
n_mels = 23

transform = MFCC(
    sample_rate=target_sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': n_mels}
)


# Подготовка данных для датасета
class KeywordDataset(Dataset):
    def __init__(self, keyword_dir, background_dir, transform=None, max_length=81):
        self.keyword_dir = keyword_dir
        self.background_dir = background_dir
        self.transform = transform
        self.max_length = max_length
        self.keyword_files = [os.path.join(keyword_dir, f) for f in os.listdir(keyword_dir) if f.endswith('.wav')]
        self.background_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if
                                 f.endswith('.wav')]

    def __len__(self):
        return len(self.keyword_files) + len(self.background_files)

    def __getitem__(self, idx):
        if idx < len(self.keyword_files):
            file_path = self.keyword_files[idx]
            label = 1
        else:
            file_path = self.background_files[idx - len(self.keyword_files)]
            label = 0

        waveform, sr = load_audio(file_path)
        mfcc = self.transform(waveform)

        # Приведение MFCC к одной длине (дополнение или обрезка)
        if mfcc.shape[2] < self.max_length:
            pad_length = self.max_length - mfcc.shape[2]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_length))
        elif mfcc.shape[2] > self.max_length:
            mfcc = mfcc[:, :, :self.max_length]

        return mfcc.squeeze(0).transpose(0, 1), label


class KeywordModel(nn.Module):
    def __init__(self, input_size, hidden_dim=128, output_dim=2):
        super(KeywordModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Последний выходной вектор из LSTM
        x = self.fc1(x)
        return x


# Инициализация модели
model = KeywordModel(input_size=n_mfcc)  # Размерность после MFCC

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def start_spotting(keyword_dir, background_dir):
    # Создание датасета и загрузчика данных
    dataset = KeywordDataset(keyword_dir, background_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Обучение модели
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for waveforms, labels in dataloader:
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    # Сохранение модели
    torch.save(model.state_dict(), '/content/keyword_model.pth')


def detect_keyword(audio_path, target_length=81):
    model.eval()

    # Загрузка аудиофайла
    waveform, sample_rate = load_audio(audio_path)
    waveform = waveform.unsqueeze(0)  # Добавляем размерность для batch_size

    # Применение преобразования MFCC
    waveform = transform(waveform)

    # Проверяем размерность после MFCC
    print(f"MFCC Shape: {waveform.shape}")

    # Приведение MFCC к одной длине (дополнение или обрезка)
    current_length = waveform.shape[-1]

    if current_length < target_length:
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_length > target_length:
        waveform = waveform[:, :, :target_length]

    # Транспонирование для LSTM (sequence_length, input_size)
    waveform = waveform.squeeze(0).transpose(0, 1)  # (sequence_length, input_size)

    # Проверка размерности входа
    if waveform.size(-1) != n_mfcc:
        raise RuntimeError(f"Expected input size {n_mfcc}, but got {waveform.size(-1)}")

    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 1:
            print("Keyword detected!")
        else:
            print("No keyword detected.")
