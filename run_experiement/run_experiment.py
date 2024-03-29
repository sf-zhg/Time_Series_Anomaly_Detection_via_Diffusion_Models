import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from TimeSeriesDiffusion.data.get_data import get_dataset
from TimeSeriesDiffusion.trainer import trainer
from TimeSeriesDiffusion.models.unet import Unet
from TimeSeriesDiffusion.models.convnet import ConvNetWithTimeEmbeddings
from TimeSeriesDiffusion.diffusion_model.diffusion_ingridients import (
    DiffusionIngridients,
)
from TimeSeriesDiffusion.diffusion_model.inference import generative_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion model - Anomaly detection")
    parser.add_argument(
        "--dataset", default="sine_wave", type=str, help="dataset to train on"
    )
    parser.add_argument(
        "--data_path", default="raw_data_csv", type=str, help="path to data"
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="optimizer, choose between adam and sgd",
    )
    parser.add_argument(
        "--batch_size",
        default="128",
        type=int,
        help="batch size for training, feel free to put ridiculously large number in case you rich",
    )
    parser.add_argument(
        "--learning_rate", default="0.01", type=float, help="learning rate"
    )
    parser.add_argument(
        "--epochs", default="100", type=int, help="training iterationsn"
    )
    parser.add_argument(
        "--time_steps", default="100", type=int, help="diffusion steps lambda"
    )
    parser.add_argument(
        "--beta_lower",
        default="0.01",
        type=float,
        help="lower bound for beta, needs to be smaller than the upper bound and between [0,1]",
    )
    parser.add_argument(
        "--beta_upper",
        default="0.99",
        type=float,
        help="upper bound in [0,1], has to be larger than beta_lower",
    )
    parser.add_argument(
        "--scheduler",
        default="linear",
        type=str,
        help="variance scheduler, choose linear, quadratic, cosine or sigmoid",
    )
    parser.add_argument(
        "--noise",
        default="gaussian",
        type=str,
        help="what noise, choose between: gaussian, simplex, brownian and cold",
    )
    parser.add_argument(
        "--loss",
        default="l1",
        type=str,
        help="loss type, l1, l2 or huber are valid inputs",
    )
    parser.add_argument("--dim", default="128", type=int, help="dimensions of unet")

    parser.add_argument(
        "--num_samples",
        default="3",
        type=int,
        help="number of samples per label to be generated and shit",
    )
    parser.add_argument(
        "--csv_path",
        default="results_csv_and_plots",
        type=str,
        help="where to save shit",
    )
    parser.add_argument(
        "--noisy_file_name", default="noisy", type=str, help="noisy_file"
    )
    parser.add_argument("--gen_file_name", default="gen", type=str, help="gen_file")
    parser.add_argument(
        "--log_schedule",
        default="10",
        type=int,
        help="log schedule for generating between timesteps",
    )
    parser.add_argument(
        "--anno_threshold",
        default="0.1",
        type=float,
        help="threshold for annomalies, since data is scaled has to be a float",
    )
    parser.add_argument(
        "--model_arch",
        default="convnet",
        type=str,
        help="network architecture, choose between unet, convnet and lstm",
    )

    args = parser.parse_args()

    assert args.optimizer in [
        "adam",
        "sgd",
    ], "Invalid optimizer. Choose between 'adam' and 'sgd'."
    assert args.batch_size > 0, "Batch size must be greater than 0."
    assert args.learning_rate > 0, "Learning rate must be greater than 0."
    assert args.epochs > 0, "Number of epochs must be greater than 0."
    assert args.time_steps > 0, "Number of time steps must be greater than 0."
    assert (
        0 <= args.beta_lower <= 1 and 0 <= args.beta_upper <= 1
    ), "beta_lower and beta_upper must be between 0 and 1 and beta_lower must be smaller than beta_upper."
    assert args.scheduler in [
        "linear",
        "quadratic",
        "cosine",
        "sigmoid",
    ], "Invalid scheduler. Choose between 'linear', 'quadratic', 'cosine', and 'sigmoid'."
    assert args.noise in [
        "gaussian",
        "simplex",
        "brownian",
        "cold",
    ], "Invalid noise type. Choose between 'gaussian', 'simplex', 'brownian', and 'cold'."
    assert args.loss in [
        "l1",
        "l2",
        "huber",
    ], "Invalid loss type. Choose between 'l1', 'l2', and 'huber'."
    assert args.dim > 0, "Dimension of Unet must be greater than 0."
    assert args.num_samples > 0, "Number of samples per label must be greater than 0."
    assert (
        args.log_schedule > 0
    ), "Log schedule for generating between timesteps must be greater than 0."
    assert (
        args.anno_threshold >= 0
    ), "Anomaly threshold must be greater than or equal to 0."

    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_root = args.data_path
    train_data, test_data = get_dataset(dataset=args.dataset, root=full_root)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    for batch in train_loader:
        x, _ = batch
        break

    if args.model_arch == "convnet":
        model = ConvNetWithTimeEmbeddings(dim=32)
    elif args.model_arch == "unet":
        model = Unet(dims=32)
    else:
        raise ValueError(
            "invalid model architecture. 'convnet' and 'unet' can be used."
        )

    model = nn.DataParallel(model)
    diff = DiffusionIngridients(
        model=model,
        time_steps=args.time_steps,
        beta_lower=args.beta_lower,
        beta_upper=args.beta_upper,
        scheduler=args.scheduler,
        noise=args.noise,
        loss=args.loss,
        device=device,
    )

    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists(args.csv_path):
        os.makedirs(args.csv_path)

    writer = SummaryWriter(comment="_pretraining")
    t0 = time.time()

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer(
            data_loader=train_loader,
            optimizer=args.optimizer,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            time_steps=args.time_steps,
            loss=args.loss,
            diffusion_model=diff,
        )
        writer.add_scalar("Train/Loss", train_loss, epoch)

        if epoch % 5 == 0:
            torch.save(
                model.module.state_dict(), "results/pretraining_{}.pth".format(epoch)
            )

            # generate result
            a = generative_inference(
                diffusion_model=diff,
                test_loader=test_loader,
                num_samples=args.num_samples,
                csv_path=args.csv_path,
                generated_file_name=args.gen_file_name,
                noisy_file_name=args.noisy_file_name,
                device=device,
                noise=args.noise,
                log_schedule=args.log_schedule,
                anno_threshold=args.anno_threshold,
                epoch=epoch,
            )
            print(a)

    t1 = time.time() - t0
    print("Time elapsed in hours:", t1 / 3600)
