import torch

DATA_CONFIG = {
    'data_dir': r'C:\Users\24837\Desktop\Course design\data',
    'patch_size': (128, 128, 128),
    'modalities': ['flair', 't1', 't1ce', 't2'],
    'num_classes': 4,
}

TRAIN_CONFIG = {
    'batch_size': 1,
    'epochs': 50,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}