import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from lib.layers import LinOSSModel
from lib.dataset import PDEDataset
from lib.utils import relative_l2_error


torch.manual_seed(0)
np.random.seed(0)


N_TRAIN = 64  # number of training samples
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


training_data = PDEDataset("data/train_sol.npy", device=DEVICE)
# choose N_TRAIN samples randomly
val_data, train_data = torch.utils.data.random_split(
    training_data, [N_TRAIN, len(training_data) - N_TRAIN]
)


train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


learning_rate = 0.0001
epochs = 100
step_size = 2
gamma = 0.75


model = LinOSSModel(
    input_dim=2,
    output_dim=1,
    hidden_dim=32,
    num_layers=3,
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
        prediction = model(input).squeeze(-1)

        prediction = prediction[:, -1].squeeze(1)[..., 0]

        loss = relative_l2_error(prediction, target, dim=None)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_data_loader)

    # Compute validation loss
    validation_relative_l2 = 0.0
    for input, target in val_data_loader:
        with torch.no_grad():
            prediction = model(input).squeeze(-1)
            prediction = prediction[:, -1].squeeze(1)[..., 0]

        loss = torch.sum(relative_l2_error(prediction, target))
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
