from typing import Literal
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
import pickle
from lib.layers import LinOSSModel


torch.manual_seed(0)
np.random.seed(0)


BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "mps"


class WormsDataset(Dataset):
    def __init__(
        self,
        mode: Literal["train", "test"],
        device: str = "cpu",
    ):
        self.device = device
        self.mode = mode
        self.data = np.array(pickle.load(open("data/EigenWorms/data.pkl", "rb")))
        self.labels = np.array(pickle.load(open("data/EigenWorms/labels.pkl", "rb")))

        self.train_idx, self.test_idx = pickle.load(
            open("data/EigenWorms/original_idxs.pkl", "rb")
        )

        self.train_idx = np.array(self.train_idx)
        self.test_idx = np.array(self.test_idx)

    def __len__(self):
        return self.mode == "train" and len(self.train_idx) or len(self.test_idx)

    def __getitem__(self, index):
        if self.mode == "train":
            idx = self.train_idx[index]
        else:
            idx = self.test_idx[index]

        data = self.data[idx]
        label = self.labels[idx]

        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.float32, device=self.device)

        return data, label


training_data = WormsDataset("train", device=DEVICE)

# choose N_TRAIN samples randomly
val_data, train_data = torch.utils.data.random_split(training_data, [0.2, 0.8])


train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


learning_rate = 0.001
epochs = 5
step_size = 2
gamma = 0.75


model = LinOSSModel(
    input_dim=6, output_dim=5, hidden_dim=128, num_layers=2, classification=True
).to(DEVICE)


optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=step_size
)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = step_size, eta_min=1e-6)


model.train()

metrics = {
    "training_loss": [],
    "validation_loss": [],
    "lr": [],
}

progress_bar = tqdm.tqdm(range(epochs))
for epoch in progress_bar:
    train_loss = 0.0
    for input, target in train_data_loader:
        optimizer.zero_grad()
        prediction = model(input, 1).squeeze(-1)

        # calculate cross entropy loss
        loss = torch.nn.functional.cross_entropy(prediction, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_data_loader)

    # Compute validation loss
    validation_relative_l2 = 0.0
    for input, target in val_data_loader:
        with torch.no_grad():
            prediction = model(input).squeeze(-1)

        loss = torch.nn.functional.cross_entropy(prediction, target)
        validation_relative_l2 += loss.item()

    validation_relative_l2 /= len(val_data)

    metrics["training_loss"].append(train_loss)
    metrics["lr"].append(scheduler.get_last_lr())
    metrics["validation_loss"].append(validation_relative_l2)

    scheduler.step(validation_relative_l2)

    progress_bar.set_postfix(
        {
            "train_loss": train_loss,
            "val_loss": validation_relative_l2,
        }
    )


# save model to disk
torch.save(model.state_dict(), "models/LinOSS_model.pth")
