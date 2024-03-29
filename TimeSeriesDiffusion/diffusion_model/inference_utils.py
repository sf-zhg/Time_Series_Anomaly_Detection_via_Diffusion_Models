from torch import Tensor
import os

from TimeSeriesDiffusion.diffusion_model import generator_helpers as gh
from TimeSeriesDiffusion.anomaly_detection import detect_anomalies


def sample_wise_gen_and_ano_detect(
    sample_size: int,
    gen_array: Tensor,
    noisy_array: Tensor,
    label: Tensor,
    csv_path: str,
    epoch: int,
    generated_file_name: str,
    noisy_file_name: str,
    anno_threshold: float,
) -> None:
    """
    for each series in time_steps / log_schedule save and plot the noisy and
    generated serieses. further apply the anomaly detection.
    """
    for i in range(sample_size):
        sample_generated_series = gen_array[i, :, :]
        sample_noisy_series = noisy_array[i, :, :]
        label = label[i]
        gen_sample_file_path = os.path.join(
            csv_path, f"{generated_file_name}_{epoch}_{label}_{i}.csv"
        )
        noisy_sample_file_path = os.path.join(
            csv_path, f"{noisy_file_name}_{epoch}_{label}_{i}.csv"
        )

        gh.save_to_csv(
            sample_generated_series,
            sample_noisy_series,
            gen_sample_file_path,
            noisy_sample_file_path,
        )

        gh.plot_time_series(
            sample_generated_series, sample_noisy_series, epoch, label, i, csv_path
        )

        detect_anomalies(
            sample_generated_series,
            sample_noisy_series,
            anno_threshold,
            epoch,
            label,
            i,
            csv_path,
        )
