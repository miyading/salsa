import os
import pandas as pd
import torch
import torchaudio
from data.datasets.dataset_base_classes import DatasetBaseClass
from sacred import Ingredient
from utils.directories import get_dataset_dir
from sklearn.model_selection import train_test_split

# CLI:
# CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 with \
# train_on=musiccaps \
# loss_weight=1.0 \
# distill_weight=0.0 \
# load_parameters=mild-mountain-1 \
# load_last=best \
# data_loader.batch_size=32 \
# max_epochs=10

musiccaps = Ingredient('musiccaps')

@musiccaps.config
def config():
    folder_name = 'musiccaps'
    compress = True

@musiccaps.capture
def get_musiccaps(split, folder_name, compress):
    path = os.path.join(get_dataset_dir(), folder_name)
    ds = MusicCapsDataset(path, split=split)
    ds.compress = compress
    return ds


class MusicCapsDataset(DatasetBaseClass):
    @musiccaps.capture
    def __init__(self, folder_name, compress, split='train'):
        super().__init__()
        self.split = split
        self.compress = compress
        self.sample_rate = 32000
        self.audio_dir = os.path.join(get_dataset_dir(), folder_name, "audio")

        # Load full CSV
        self.df = pd.read_csv(os.path.join(get_dataset_dir(), folder_name, "musiccaps.csv"))

        # Apply split logic
        if split == 'train':
            self.df = self.df[self.df['is_balanced_subset'] == False]
        else:
            df_bal = self.df[self.df['is_balanced_subset'] == True]
            val_df, test_df = train_test_split(df_bal, test_size=0.2, random_state=42)
            self.df = val_df if split == 'val' else test_df

        # Build metadata
        self.paths = [
            os.path.join(self.audio_dir, f"{ytid}.wav") for ytid in self.df['ytid']
        ]
        self.captions = self.df['caption'].tolist()
        self.start_times = self.df['start_s'].tolist()
        self.end_times = self.df['end_s'].tolist()
        self.keywords = [''] * len(self.df)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        caption = self.captions[idx]
        start_sample = int(self.start_times[idx] * self.sample_rate)
        end_sample = int(self.end_times[idx] * self.sample_rate)

        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        waveform = waveform[:, start_sample:end_sample]

        return {
            'audio': waveform,
            'audio_length': torch.tensor([(end_sample - start_sample) / self.sample_rate]),
            'caption': caption,
            'keywords': self.keywords[idx],
            'path': path,
            'idx': idx + 3000000,
            'caption_hard': '',
            'html': ''
        }

    def __get_audio_paths__(self):
        return self.paths

    def __str__(self):
        return f'MusicCaps_{self.split}'
