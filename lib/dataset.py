import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class PDEDataset(Dataset):
    def __init__(
        self,
        path: str,
        device: str = "cpu",
    ):
        self.original_data = (
            torch.from_numpy(np.load(path)).type(torch.float32).to(device)
        )
        self.length = self.original_data.shape[0]

        self.sample_resolution = self.original_data.shape[-1]
        self.sample_timesteps = self.original_data.shape[-2]

        self.x_values = torch.tensor(
            np.linspace(0, 1, self.sample_resolution), dtype=torch.float32
        ).to(device)

        # for each sample add the x and t values
        self.data = self.original_data.unsqueeze(-1)

        self.x_values = self.x_values.expand(
            self.length, self.sample_timesteps, -1
        ).unsqueeze(-1)

        self.data = torch.cat([self.data, self.x_values], dim=-1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index, :-1], self.original_data[index, -1]


class PDEDatasetAll2All(Dataset):
    def __init__(
        self,
        path: str,
        device: str = "cpu",
        time_pairs: List[Tuple[int, int]] = None,
        augment_data: bool = False,
    ):
        super(PDEDatasetAll2All, self).__init__()

        self.augment_data = augment_data

        self.data = torch.tensor(np.load(path)).type(torch.float32).to(device)

        self.samples = self.data.shape[0]
        self.time_steps = self.data.shape[1]
        self.spacial_res = self.data.shape[2]

        if time_pairs is not None:
            self.time_pairs = time_pairs
        else:
            # Precompute all possible (t_initial, t_final) pairs within the specified range.
            self.time_pairs = [
                (i, j)
                for i in range(0, self.time_steps)
                for j in range(i, self.time_steps)
            ]

        self.len_times = len(self.time_pairs)

        self.total_samples = self.len_times * self.samples

        self.x_grid = torch.linspace(0, 1, self.spacial_res).to(device).reshape(1, -1)

        # Compute mean and std of the data
        self.mean = self.data.mean()
        self.std = self.data.std()

    def __len__(self):
        if self.augment_data:
            return 2 * self.total_samples
        return self.total_samples

    def __getitem__(self, index):
        invert = False

        if self.augment_data and index >= self.total_samples:
            invert = True
            index -= self.total_samples

        sample_idx = index // self.len_times
        time_pair_idx = index % self.len_times
        t_inp, t_out = self.time_pairs[time_pair_idx]
        time_delta = (t_out - t_inp) * 1.0 / (1 - self.time_steps)

        inputs = self.data[sample_idx, t_inp].reshape(1, self.spacial_res)

        target = self.data[sample_idx, t_out]

        if invert:
            inputs = torch.flip(inputs, [1]) * -1
            target = torch.flip(target, [0]) * -1

        # inputs = (inputs - self.mean) / self.std  # Normalize
        inputs_t = torch.ones_like(inputs).type_as(inputs) * time_delta
        inputs = torch.cat((inputs, self.x_grid, inputs_t), 0)  # Cat time to the input

        # outputs = (outputs - self.mean) / self.std  # Normalize

        # if invert:
        #     return -1 * abs(float(time_delta)), inputs.T, target

        return abs(float(time_delta)), inputs.T, target


if __name__ == "__main__":
    test = PDEDatasetAll2All("FNO-wave-equation/data/train_sol.npy", augment_data=True)

    print(test[1])
