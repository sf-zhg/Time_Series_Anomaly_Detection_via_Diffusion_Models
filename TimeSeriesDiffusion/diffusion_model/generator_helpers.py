import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def bundle_test_set(test_loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """
    concatenate all batches of test serieses for subsequent sampling
    """
    test_data = []
    test_labels = []

    for data, labels in test_loader:
        test_data.append(data)
        test_labels.append(labels)

    return torch.cat(test_data, dim=0), torch.cat(test_labels, dim=0)


def get_label_indices(test_labels: Tensor) -> Dict[int, list]:
    """
    extract indices of unique labels
    """
    unique_labels = torch.unique(test_labels)
    label_indices = {label.item(): [] for label in unique_labels}

    for i, label in enumerate(test_labels):
        label_indices[label.item()].append(i)

    return label_indices


def select_random_indices(
    label_indices: Dict[int, List[int]], num_samples: int
) -> List[int]:
    """
    select num_samples random serieses from each class by their indices in the
    dict of test serieses
    """
    selected_indices = []

    for label, indices in label_indices.items():
        selected_indices.extend(np.random.choice(indices, num_samples, replace=False))

    return selected_indices


def get_samples(
    test_data: DataLoader, test_labels: Tensor, selected_indices: List
) -> Tuple[Tensor, Tensor]:
    """
    get samples from dict according to a list of indices and return serieses
    and their label
    """
    sample_test_data = test_data[selected_indices]
    sample_test_labels = test_labels[selected_indices]

    return sample_test_data, sample_test_labels


def forward_diffusion_stepwise(
    x: Tensor, noise: Tensor, log_schedule: int, diffusion_model
) -> List:
    """
    add noise to original series at every log_schedule time step
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noisy_series_list = []
    noisy_series_list.append(x)

    for i in tqdm(
        range(0, diffusion_model.time_steps),
        desc="sampling loop time step",
        total=diffusion_model.time_steps,
    ):
        t = torch.full((1,), i, device=device)
        noisy_series_t = diffusion_model.q_sample(x_start=x, t=t, base_noise=noise)
        noisy_series_list.append(noisy_series_t)

    # for i in range(0, diffusion_model.time_steps):
    # if i % log_schedule == 0:
    #    t = torch.full((1,), i, device = device)
    #    noisy_series_t = diffusion_model.q_sample(x_start=x, t=t, base_noise=noise)
    #    noisy_series_list.append(noisy_series_t)

    return noisy_series_list


def backward_diffusion_stepwise(
    diffusion_model, log_schedule: int, x_last: Tensor
) -> List:
    """
    take maximally forward diffused series and substact noise log_schedule
    step wise until estimate of original series is reached
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_series_list = []
    gen_series_list.append(x_last)

    for i in reversed(range(0, diffusion_model.time_steps - 1)):
        t = torch.full((1,), i, device=device)
        t = t.repeat(x_last.size(0))
        gen_series_t = diffusion_model.p_sample(x=x_last, t=t)
        gen_series_list.append(gen_series_t)

    return gen_series_list


def save_to_csv(
    sample_generated_series: List,
    sample_noisy_series: List,
    gen_sample_file_path: str,
    noisy_sample_file_path: str,
):
    """
    get csv file from step wise generated series sample and their noisy
    counterpart
    """
    gen_size = sample_generated_series.shape[1]

    for i in range(gen_size):
        batch_generated_data = sample_generated_series[:, i, :]

        df_generated = pd.DataFrame(batch_generated_data)
        df_generated.to_csv(f"{gen_sample_file_path}_{i}.csv", index=False, header=True)

    noisy_size = sample_noisy_series.shape[1]
    for i in range(noisy_size):
        batch_noisy_data = sample_noisy_series[:, i, :]

        df_noisy = pd.DataFrame(batch_noisy_data)
        df_noisy.to_csv(f"{noisy_sample_file_path}_{i}.csv", index=False, header=True)


def plot_time_series(
    sample_generated_series: List,
    sample_noisy_series: List,
    epoch: int,
    label: int,
    csv_path: str,
):
    """
    Plot noisy series and the backward denoised ones.
    """
    num_timesteps_generated = sample_generated_series.shape[1]
    num_timesteps_noisy = sample_noisy_series.shape[1]

    for i in range(num_timesteps_generated):
        if i >= num_timesteps_generated:
            print(f"index {i} out of bounds for generated series. skipping...")
            continue

        y = label[i]

        fig, axs = plt.subplots(len(sample_generated_series), 1, figsize=(10, 5))

        for j, series in enumerate(sample_generated_series):
            axs[j].plot(series[i, :], label="Generated Series")
            axs[j].set_title(f"Time Step {len(sample_generated_series)-j}")
            axs[j].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(csv_path, f"gen_plot_epoch_{epoch}_class_{y}_sample_{i}.png")
        )
        plt.close()

    for i in range(num_timesteps_noisy):
        if i >= num_timesteps_noisy:
            print(f"index {i} out of bounds for noisy series. skipping...")
            continue

        y = label[i]

        fig, axs = plt.subplots(len(sample_noisy_series), 1, figsize=(10, 5))

        for j, series in enumerate(sample_noisy_series):
            axs[j].plot(series[i, :], label="Noisy Series")
            axs[j].set_title(f"Time Step {j}")
            axs[j].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(csv_path, f"noisy_plot_epock_{epoch}_class_{y}_sample_{i}.png")
        )
        plt.close()
