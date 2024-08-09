target_sample_rate = 16000

keyword_path = 'akylai_dataset'
background_path = 'background'
spotting_model = "new_keyword_spotting_model.pth"
concord_path = "concord"
test_files_path = "positive_test"
samples_path = "samples"

positive_path = "positive_dataset"
negative_path = "negative_dataset"

test_data = [
    (0, "not_detected28.wav"),
    (1, "detected43.wav"),
    (1, "detected13.wav"),
    (0, "not_detected21.wav"),
    (0, "not_detected25.wav"),
    (0, "not_detected36.wav"),
    (0, "not_detected_45.13.wav"),
    (1, "detected45.06.wav"),
    (1, "detected_45.17.wav"),
]