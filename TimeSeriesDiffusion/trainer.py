import torch

import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader



def trainer(
    data_loader: DataLoader,
    optimizer: str,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    time_steps: int,
    loss: str,
    diffusion_model,
) -> float:
    """
    trainings procedure over a mini batch
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    diff = diffusion_model
    model = diff.model
    model.train()
    model.to(device)

    if optimizer == "adam":
        opti = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sdg":
        opti = optim.SGD(model.parameters(), lr=learning_rate)

    total_loss = 0.0

    with tqdm(data_loader) as train_bar:

        for x, _ in train_bar:

            x = x.to(device)

            x = x.unsqueeze(1)

            t = torch.randint(0, time_steps, (batch_size,), device=device).long()

            losses = diff.p_loss(model=model, x_start=x, t=t)

            opti.zero_grad()
            losses.backward()
            opti.step()

            total_loss += losses.item()

            train_bar.set_postfix({"Loss": total_loss})

    return total_loss
