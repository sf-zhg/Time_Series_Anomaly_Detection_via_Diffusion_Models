import numpy as np
import pandas as pd
import os
import random

import argparse


def get_sine_toy(
    amp: tuple,
    freq: tuple,
    phi: tuple,
    samp_rate: int,
    time_lower_lim: int,
    time_upper_lim: int,
    path: str,
    name_train: str,
    name_test: str,
    cardinality: int,
):
    """
    sample dataset
    """

    # get limits of tuples. there has to be a sexier way but i dont have enough shit left to google this shit
    amp_lower, amp_upper = amp
    freq_lower, freq_upper = freq
    phi_lower, phi_upper = phi

    # sample amplitudes, frew and phi for data
    amp_vec = np.array(
        [random.uniform(amp_lower, amp_upper) for _ in range(cardinality)]
    )
    freq_vec = np.array(
        [random.uniform(freq_lower, freq_upper) for _ in range(cardinality)]
    )
    phi_vec = np.array(
        [random.uniform(phi_lower, phi_upper) for _ in range(cardinality)]
    )

    time_points = np.arange(time_lower_lim, time_upper_lim, 1 / samp_rate)

    # make df to save. like who the fuck we thing we are to just save shit
    data_df = pd.DataFrame(time_points)

    # sine shit
    for i in range(cardinality):
        sine_wave = amp_vec[i] * np.sin(
            2 * np.pi * freq_vec[i] * time_points + phi_vec[i]
        )

        df = pd.DataFrame({f"y_{i}": sine_wave})

        # add shit to df
        data_df = pd.concat([data_df, df], axis=1)

    # do for sexyness and get as many folders as possible. make the machines work for you
    if not os.path.exists(path):
        os.makedirs(path)

    # actually save shit
    file_path = os.path.join(path, name_train)
    data_df.to_csv(file_path, index=False)

    # do same shit for train data
    # generate some regular sine waves to feel real comfy

    num_reg_sine_in_test = int(cardinality // 4)

    # sample amplitudes, frew and phi for data, limits can stay the same, actually amp doesnt matter as we normalize anyway
    amp_vec_test = np.array(
        [random.uniform(amp_lower, amp_upper) for _ in range(num_reg_sine_in_test)]
    )
    freq_vec_test = np.array(
        [random.uniform(freq_lower, freq_upper) for _ in range(num_reg_sine_in_test)]
    )
    phi_vec_test = np.array(
        [random.uniform(phi_lower, phi_upper) for _ in range(num_reg_sine_in_test)]
    )

    # make df to save. like who the fuck we thing we are to just save shit
    reg_test_data_df = pd.DataFrame(time_points)

    # sine shit
    for i in range(num_reg_sine_in_test):
        sine_wave = amp_vec_test[i] * np.sin(
            2 * np.pi * freq_vec_test[i] * time_points + phi_vec_test[i]
        )

        df = pd.DataFrame({f"y_{i}": sine_wave})

        reg_test_data_df = pd.concat([reg_test_data_df, df], axis=1)

    # generate unregular data, let the size be the same as the regular shit for simplicity and make this fucking shit work
    num_weird_shit_in_test = int(cardinality // 4)
    weird_test_data_df = pd.DataFrame(time_points)
    for i in range(num_weird_shit_in_test):
        sine_wave = amp_vec_test[i] * np.sin(
            2 * np.pi * freq_vec_test[i] * time_points + phi_vec_test[i]
        )

        # draw random numbers to add onto wave
        rand_lower = np.random.randint(-10, 0)
        rand_upper = np.random.randint(0, 10)

        # random number to put onto the time points at of series
        max_ts_changes = len(time_points) / 2
        min_ts_changes = len(time_points) / 6
        rand_tp = np.random.randint(min_ts_changes, max_ts_changes)

        # get changes for a random umber of points
        random_nums_vec = np.array(
            [random.randint(rand_lower, rand_upper) for _ in range(rand_tp)]
        )

        # add shit to random positions of series to make it irregular, weird, pretty in its own way
        rand_corruption = np.zeros(len(time_points))

        rand_position = np.random.choice(
            np.arange(0, len(time_points)), size=len(random_nums_vec), replace=False
        )

        # add shit together
        rand_corruption[rand_position] = random_nums_vec

        unreg_wave = sine_wave + rand_corruption

        df = pd.DataFrame({f"y_{i}": unreg_wave})

        # add shit to df
        weird_test_data_df = pd.concat([weird_test_data_df, df], axis=1)

    test_data_df = pd.concat([reg_test_data_df, weird_test_data_df.iloc[:, 1:]], axis=1)

    # actually save shit
    file_path = os.path.join(path, name_test)
    test_data_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="get sine waves and you will be happy about it"
    )
    parser.add_argument(
        "--amp",
        default=[0, 5],
        type=tuple,
        help="amplitude range as tuple. dont fuck this up",
    )
    parser.add_argument(
        "--freq", default=[0, 5], type=tuple, help="freqency range as tuple"
    )
    parser.add_argument(
        "--phi",
        default=[0, 5],
        type=tuple,
        help="phase range as tuple. maybe phases, there is no telling what the plural is",
    )
    # sr 10
    parser.add_argument(
        "--samp_rate", default=1, type=int, help="steps between integer time points"
    )
    parser.add_argument(
        "--time_lower_lim",
        default=0,
        type=int,
        help="lower limit, zero seems like a real nice choice",
    )
    # upper lim 128
    parser.add_argument(
        "--time_upper_lim",
        default=16,
        type=int,
        help="arbitrary upper limit for series",
    )
    parser.add_argument(
        "--path",
        default="../../raw_data_csv",
        type=str,
        help="path to save csv, dont be to funny my guy",
    )
    parser.add_argument(
        "--name_train", default="sine_wave_train_data.csv", type=str, help="file name"
    )
    parser.add_argument(
        "--name_test", default="sine_wave_test_data.csv", type=str, help="file name"
    )
    # cardinality 1024
    parser.add_argument("--cardinality", default=64, type=int, help="|dataset|,")
    args = parser.parse_args()

    get_sine_toy(
        args.amp,
        args.freq,
        args.phi,
        args.samp_rate,
        args.time_lower_lim,
        args.time_upper_lim,
        args.path,
        args.name_train,
        args.name_test,
        args.cardinality,
    )
