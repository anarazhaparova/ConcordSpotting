import os
import sys
import time

import librosa
import psutil
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MFCC

from settings import target_sample_rate, spotting_model, test_files_path, test_data, positive_path, negative_path
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
    def __init__(self, keyword_dir, background_dir, transform=None, max_length=148):
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
    total = 50
    # Создание датасета и загрузчика данных
    dataset = KeywordDataset(keyword_dir, background_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Обучение модели
    num_epochs = 10
    print(f"Spotting stared")
    process = psutil.Process(os.getpid())
    data_count = len(dataloader)
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        index = 0
        running_loss = 0.0
        for waveforms, labels in dataloader:
            index += 1
            process_count = index / data_count * total
            current_time = time.time()
            elapsed_time = current_time - start_time
            sys.stdout.write("\r")
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            memory_info = process.memory_info().rss / (1024 * 1024)
            sys.stdout.write(
                f"Epoch {epoch + 1} --> Время: {int(minutes):02}:{int(seconds):02}, Память: {memory_info:.2f}MB")
            sys.stdout.write("  |")
            sys.stdout.write("#" * int(process_count))
            sys.stdout.write(" " * (total - int(process_count)))
            procent = 100 / total * process_count
            sys.stdout.write(f"| {procent:.2f}%")
            sys.stdout.flush()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        print(f'\nEpoch {epoch + 1}/{num_epochs} done, Loss: {average_loss:.4f}')

    # Сохранение модели
    torch.save(model.state_dict(), spotting_model)


def detect_keyword(audio_path, target_length=148):
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
    waveform = waveform.squeeze(1).transpose(1, 2)  # (sequence_length, input_size)

    # Проверка размерности входа
    if waveform.size(-1) != n_mfcc:
        raise RuntimeError(f"Expected input size {n_mfcc}, but got {waveform.size(-1)}")

    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def test_model(test_dataset, batch_size=32):
    model.eval()  # Переводим модель в режим тестирования
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    running_loss = 0.0

    result_data = [["Excepted background", 0, 0], ["Excepted keyword", 0, 0]]
    headers = ["Predict background", "Predict keyword"]
    # for label, filename in test_data:
    #     if filename.endswith('.wav'):
    #         filepath = os.path.join(path, filename)
    #         result = detect_keyword(filepath)
    #         old_val = result_data[label][result+1]
    #         old_val += 1
    #         result_data[label][result+1] = old_val
    #

    with torch.no_grad():  # Отключаем вычисление градиентов для тестирования
        for waveforms, labels in dataloader:
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            print(f"Probabilities: {probabilities}")

            for index, label in enumerate(labels):
                result = predicted[index]
                old_val = result_data[label][result + 1]
                old_val += 1
                result_data[label][result + 1] = old_val
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(tabulate(result_data, headers=headers, tablefmt="grid"))
    accuracy = correct / total * 100
    average_loss = running_loss / len(dataloader)

    print(f'Test Accuracy: {accuracy:.2f}%, Average Loss: {average_loss:.4f}')
    return accuracy, average_loss

def start_test():
    result_data = [["Excepted background", 0, 0], ["Excepted keyword", 0, 0]]
    path = "test_files"
    headers = ["Predict background", "Predict keyword"]
    for label, filename in test_data:
        if filename.endswith('.wav'):
            filepath = os.path.join(path, filename)
            result = detect_keyword(filepath)
            old_val = result_data[label][result + 1]
            old_val += 1
            result_data[label][result + 1] = old_val
    print(tabulate(result_data, headers=headers, tablefmt="grid"))


def plot_mfcc(mfcc, title='MFCC'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
