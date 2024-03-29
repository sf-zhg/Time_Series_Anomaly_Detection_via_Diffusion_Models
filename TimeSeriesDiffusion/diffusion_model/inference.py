import torch
import os

from TimeSeriesDiffusion.noise.get_noise import get_noise
from TimeSeriesDiffusion.diffusion_model import generator_helpers as gh
from TimeSeriesDiffusion.diffusion_model.inference_utils import (
    sample_wise_gen_and_ano_detect,
)
from TimeSeriesDiffusion.anomaly_detection import detect_anomalies
from torch.utils.data import DataLoader


def generative_inference(
    diffusion_model,
    test_loader: DataLoader,
    num_samples: int,
    csv_path: str,
    generated_file_name: str,
    noisy_file_name: str,
    device: str,
    noise: str,
    log_schedule: int,
    anno_threshold: float,
    epoch: int,
):
    """
    conduct ineference by adding noise and generating from that noisy samples.
    at the same time conduct anomaly detection as shit relies on the generated
    serieses anyway
    """
    print("Running generative inference for epoch:", epoch)

    if not os.path.exists("hello"):
        os.makedirs("hello")

    diff = diffusion_model

    diff.model.eval()
    # get sample data from test test. need to jump over some hoops though
    test_data, test_labels = gh.bundle_test_set(test_loader)

    print("Sample data loaded")
    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)

    label_indices = gh.get_label_indices(test_labels=test_labels)

    selected_indices = gh.select_random_indices(
        label_indices=label_indices, num_samples=num_samples
    )

    sample_test_data, sample_test_labels = gh.get_samples(
        test_data=test_data, test_labels=test_labels, selected_indices=selected_indices
    )

    # weg machen sind schon tensoren
    x_tensor = sample_test_data
    x_tensor = x_tensor.unsqueeze(1)
    y_tensor = sample_test_labels

    # shit now we need to enforce that num_samples * unique(label) is smaller
    # equal to batch_size or loop is necessary for memory limits

    base_noise = get_noise(x=x_tensor, noise=noise)

    noisy_series_list = gh.forward_diffusion_stepwise(
        x=x_tensor,
        noise=base_noise,
        log_schedule=log_schedule,
        diffusion_model=diffusion_model,
    )

    gen_series_list = gh.backward_diffusion_stepwise(
        diffusion_model=diffusion_model,
        log_schedule=log_schedule,
        x_last=noisy_series_list[-1],
    )

    gen_tensor = torch.stack(gen_series_list)

    noisy_tensor = torch.stack(noisy_series_list)

    gen_array = gen_tensor.squeeze().cpu().detach().numpy()
    noisy_array = noisy_tensor.squeeze().cpu().detach().numpy()

    gen_file_path = os.path.join(csv_path, generated_file_name)
    noisy_file_path = os.path.join(csv_path, noisy_file_name)
    gh.save_to_csv(gen_array, noisy_array, gen_file_path, noisy_file_path)

    gh.plot_time_series(
        sample_generated_series=gen_array,
        sample_noisy_series=noisy_array,
        epoch=epoch,
        label=y_tensor,
        csv_path=csv_path,
    )

    detect_anomalies(
        sample_generated_series=gen_array,
        sample_noisy_series=noisy_array,
        anno_threshold=anno_threshold,
        epoch=epoch,
        label=y_tensor,
        csv_path=csv_path,
    )

    """
    sample_size = gen_array.shape[0]    
    
    sample_wise_gen_and_ano_detect(
        sample_size=sample_size,
        gen_array=gen_array,
        noisy_array=noisy_array,
        label=y_tensor,
        csv_path=csv_path,
        epoch=epoch,
        generated_file_name=generated_file_name,
        noisy_file_name=noisy_file_name,
        anno_threshold=anno_threshold,
    )
    """
    return "hello there"
