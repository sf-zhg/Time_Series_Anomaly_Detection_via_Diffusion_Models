import os
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


def detect_anomalies(
    sample_generated_series: Tensor,
    sample_noisy_series: Tensor,
    anno_threshold: float,
    epoch: int,
    label: Tensor,
    csv_path: str,
):
    """
    take generated and original series and calculate difference. further takes
    the differences between the series and defines anomalies as deltas
    larger than a threshold. lastly, plots are created as well.
    """
    num_batches = sample_generated_series.shape[1]
    og_series = sample_noisy_series[0, :, :]
    gen_series = sample_generated_series[-1, :, :]

    for i in range(num_batches):
        y = label[i]

        og_series_s = og_series[i, :]
        gen_series_s = gen_series[i, :]

        delta_og_gen = abs(og_series_s - gen_series_s)
        anomaly_indices = np.where(delta_og_gen > anno_threshold)[0]

        fig, ax = plt.subplots()
        ax.plot(og_series_s, label="Original Series")
        ax.plot(gen_series_s, label="Generated Series")

        for idx in anomaly_indices:
            ax.axvline(x=idx, color="red", linestyle="--", label="Anomaly")

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Values")
        ax.set_title(f"Original and Generated Series (Sample {i})")
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                csv_path, f"anomaly_plot_Epoch_{epoch}_Label_{y}_sample{i}.png"
            )
        )
        plt.close()
