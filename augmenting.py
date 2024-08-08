from sound_factory import *


def augment_data(keyword_files, background_files, output_dir):
    keyword_files = [os.path.join(keyword_path, f) for f in os.listdir(keyword_files) if f.endswith('.wav')]
    background_files = [os.path.join(background_path, f) for f in os.listdir(background_files) if f.endswith('.wav')]
    os.makedirs(output_dir, exist_ok=True)
    for index, keyword_file in enumerate(keyword_files):
        print(f"\n({index + 1}/{len(keyword_files)}) {os.path.basename(keyword_file)} в процессе")
        sys.stdout.write(f"Всего {len(background_files)} выполнено 0")
        keyword_waveform, sr = load_audio(keyword_file)
        for background_index, background_file in enumerate(background_files):
            background_waveform, sr = load_audio(background_file)
            combined_waveform = keyword_waveform + background_waveform
            combined_waveform = combined_waveform / torch.max(torch.abs(combined_waveform))
            output_filepath = os.path.join(output_dir,
                                           os.path.basename(keyword_file) + "." + os.path.basename(background_file))
            save_audio(combined_waveform, output_filepath)
            sys.stdout.write(f"\rВсего {len(background_files)} выполнено {background_index + 1}")
            sys.stdout.flush()


def augment_audio(keyword_path1, background_path1):
    first_waveform, sr = load_audio(keyword_path1)
    second_waveform, sr = load_audio(background_path1)
    combined_waveform = first_waveform + second_waveform
    combined_waveform = combined_waveform / torch.max(torch.abs(combined_waveform))
    return combined_waveform