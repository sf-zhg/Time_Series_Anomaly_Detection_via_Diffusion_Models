import os.path

# from synth_data.sine_loader import SineWaveDataset
from TimeSeriesDiffusion.data.synth_data.sine_loader import SineWaveDataset

from typing import Tuple


def get_dataset(dataset: str, root: str) -> Tuple:
    """
    get the dataset of choice with this convenient function
    """
    # get train and test set
    if dataset == "sine_wave":
        train_file = os.path.join(root, "sine_wave_train_data.csv")
        test_file = os.path.join(root, "sine_wave_test_data.csv")
        args_train = {"csv_file": train_file}
        args_test = {"csv_file": test_file}
        train_dataset_class = SineWaveDataset(**args_train)
        test_data_class = SineWaveDataset(**args_test)
    else:
        raise ValueError(
            "dataset not existing, stop fucking around please {}".format(dataset)
        )

    return train_dataset_class, test_data_class
