from __future__ import print_function

import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os

REAL_DATASET = "Real Database"
FAKE_DATASET = "Merged Simulated Database"

BATCH_SIZE = 16

class RealDataset(Dataset):
    def __init__(self, real_dir):
        self.real_dir = real_dir
        self.real_signals = os.listdir(real_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        path_signal = os.path.join(self.real_dir, self.real_signals[idx])
        signal = loadmat(path_signal)['data']
        return signal



if __name__ == "__main__":
    glasses_on, glasses_off = get_img_lists('/home/dcor/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt')

    all_images = np.concatenate((glasses_on, glasses_off))

    dataset_original = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG)

    agent = ModelAgentColorCorrection(dataset_original)

    agent.train()


