import os
import shutil


def copy_directory_contents(source_dir, destination_dir):
    """
    Копирует все файлы и поддиректории из source_dir в destination_dir.
    """
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(destination_dir, item)

        if os.path.isdir(source_path):
            # Копируем поддиректорию рекурсивно
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            # Копируем файл
            shutil.copy2(source_path, destination_path)


def copy_directory_contents_with_rename(source_dir, destination_dir, prefix):
    os.makedirs(destination_dir, exist_ok=True)
    file_count = 0
    for item in os.listdir(source_dir):
        if item.endswith('.wav'):
            source_path = os.path.join(source_dir, item)

            # Изменяем название файла или директории
            new_item_name = f"{prefix}_{item}"
            destination_path = os.path.join(destination_dir, new_item_name)

            if not os.path.isdir(source_path):
                shutil.copy2(source_path, destination_path)
                file_count += 1
    return file_count


def merge_directories(positive, negative, target_dir):
    """
    Копирует содержимое двух директорий dir1 и dir2 в целевую директорию target_dir.
    """
    # Создаем целевую директорию, если она не существует
    os.makedirs(target_dir, exist_ok=True)
    file_count = 0
    # Копируем содержимое первой директории
    file_count += copy_directory_contents_with_rename(positive, target_dir, "positive")

    # Копируем содержимое второй директории
    file_count += copy_directory_contents_with_rename(negative, target_dir, "negative")

    print(f"Общее количество файлов в {target_dir}: {file_count}")


def delete_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
