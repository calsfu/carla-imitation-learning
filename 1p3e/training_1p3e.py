import time
import random
import argparse

import torch

# from network import ClassificationNetwork
from network_1p3e import ClassificationNetwork

from dataset_1p3e import get_dataloader


def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    gpu = torch.device('cuda')

    nr_epochs = 10
    batch_size = 64
    nr_of_classes = 9  # needs to be changed
    start_time = time.time()
    print(data_folder)
    train_loader = get_dataloader(data_folder, batch_size)
    print("Start training...")
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)

            batch_out = infer_action(batch_in)
            loss = cross_entropy_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, save_path)


def cross_entropy_loss(batch_out, batch_gt, epsilon=1e-12):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    
    Args:
        batch_out: torch.Tensor of size (batch_size, C), predicted probabilities
        batch_gt: torch.Tensor of size (batch_size, C), ground truth labels (one-hot encoded)
        epsilon: Small constant to prevent log(0)

    Returns:
        float: Average cross entropy loss for the batch.
    """
    # criterion = torch.nn.CrossEntropyLoss()
    # return criterion(batch_out, batch_gt)
    # softmax
    batch_out = torch.softmax(batch_out, dim=1)

    # clamp 
    batch_out = torch.clamp(batch_out, min=epsilon, max=1. - epsilon)
    
    # Compute cross entropy loss
    loss = -torch.sum(batch_gt * torch.log(batch_out), dim=1)  # Sum over classes
    return torch.mean(loss)  # Average over the batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC518 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./hw1/1p3e/model_1p3e.pth", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)