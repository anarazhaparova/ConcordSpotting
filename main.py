import argparse
import os
import random

from augmenting import augment_audio
from settings import *
from sound_factory import get_max_sample_length, resample_audio_to_target_sr, load_audio, sampling_audio
from sound_factory import pad_or_trim, save_audio, add_white_noise_audio, shift_audio
from spotting import start_spotting
from testing import test_keyword_spotting
from file_manipulating import merge_directories


def spotting():
    path = concord_path
    start_spotting(path)


def merge_pos_and_neg_dataset():
    merge_directories(positive_path, negative_path, concord_path)


def generate_akylai_dataset():
    path = keyword_path
    output_path = positive_path
    sample_path = negative_path
    os.makedirs(output_path, exist_ok=True)
    sample_files = [f for f in os.listdir(sample_path) if os.path.isfile(os.path.join(sample_path, f))]
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            filepath = os.path.join(path, filename)
            waveform, sample_rate = load_audio(filepath)
            save_audio(waveform, os.path.join(output_path, filename))

            for index in range(1, 10, 1):
                noise_level = index/1000
                wn_waveform = add_white_noise_audio(waveform, noise_level)
                save_audio(wn_waveform, os.path.join(output_path, f"white_noise_{noise_level}_" + filename))
                shifted_waveform = shift_audio(waveform, noise_level)
                save_audio(shifted_waveform, os.path.join(output_path, f"shifted_{noise_level}_" + filename))

            for index in range(400):
                random_file = random.choice(sample_files)
                augment_waveform = augment_audio(filepath, os.path.join(sample_path, random_file))
                save_audio(augment_waveform,
                           os.path.join(output_path, random_file.replace("incorrect_", "") + "_" + filename))


def generate_sample():
    path = samples_path
    output_path = negative_path
    num_samples = get_max_sample_length(keyword_path)
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            filepath = os.path.join(path, filename)
            sampling_audio(filepath, num_samples, output_path)


def prepare_audio():
    path = str(input("Введите директорию с аудио..."))
    num_samples = get_max_sample_length(keyword_path)
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            filepath = os.path.join(path, filename)
            waveform, sr = resample_audio_to_target_sr(filepath)
            waveform = pad_or_trim(waveform, num_samples)
            save_audio(waveform, filepath)


def check_audio():
    path = str(input("Введите директорию с аудио..."))
    num_samples = get_max_sample_length(keyword_path)
    has_at_least_one_error = False
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            filepath = os.path.join(path, filename)
            waveform, sr = load_audio(filepath)
            is_same_samples = num_samples == waveform.size(1)
            is_same_sample_rate = sr == target_sample_rate
            is_stereo = waveform.ndim > 1 and waveform.size(0) > 1

            if not is_same_samples or not is_same_sample_rate or is_stereo:
                has_at_least_one_error = True
                print(f"Файл {filename} не соответствует стандартам")
                if not is_same_samples:
                    print(f"--> Длинна аудио {waveform.size(1)}. Требуется: {num_samples}")
                if not is_same_sample_rate:
                    print(f"--> Частота дискретизации {sr}. Требуется: {target_sample_rate}")
                if is_stereo:
                    print(f"--> Аудио стерео. Требуется: моно")

    if not has_at_least_one_error:
        print(f"Все файлы в пути соответствуют стандартам")


def start_test():
    path = test_files_path
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            filepath = os.path.join(path, filename)
            test_keyword_spotting(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пример скрипта с командами")

    # Создаем субпарсеры для команд
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare_parser = subparsers.add_parser('prepare',
                                           help=f"Стандартизирует звук ({target_sample_rate} Hz, Моно-канальный и одной длинны)")
    check_parser = subparsers.add_parser('check', help=f"Проверяет звуки на стандарты")
    sampling_parser = subparsers.add_parser('sampling',
                                            help=f"Разделяет аудио дорожку на несколько файлов с равной длинной")
    gen_dataset_parser = subparsers.add_parser('gen_dataset',
                                               help=f"Генерирует дата сет ключевых слов")
    spotting_parser = subparsers.add_parser('spotting',
                                            help=f"Генерирует дата сет ключевых слов")
    test_parser = subparsers.add_parser('test',
                                        help=f"Тестрирует модель на входных данных")
    merge_parser = subparsers.add_parser('merge',
                                        help=f"Перемещает все датасеты в общую папку")

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_audio()

    if args.command == 'check':
        check_audio()

    if args.command == 'sampling':
        generate_sample()

    if args.command == 'gen_dataset':
        generate_akylai_dataset()

    if args.command == 'spotting':
        spotting()

    if args.command == 'test':
        start_test()

    if args.command == 'merge':
        merge_pos_and_neg_dataset()
