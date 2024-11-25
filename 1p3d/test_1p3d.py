import time
import random
import argparse

import torch

# from network import ClassificationNetwork
from network_1p3d import ClassificationNetwork
from dataset_1p3d import get_dataloader


def test(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    model = torch.load(save_path)
    model.eval()
    gpu = torch.device('cuda')

    batch_size = 64
    test_loader = get_dataloader(data_folder, batch_size)
    print("Start testing...")
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)

            batch_out = model(batch_in)
            loss = cross_entropy_loss(batch_out, batch_gt)
            total_loss += loss

            _, batch_out = torch.max(batch_out, 1)
            _, batch_gt = torch.max(batch_gt, 1)
            total_acc += (batch_out == batch_gt).sum().item()
        
        print(f"Test Loss: {total_loss}")
        print(f"Test Accuracy: {100 * total_acc / len(test_loader.dataset)}%")




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
    parser.add_argument('-s', '--save_path', default="./hw1/1p3d/model_1p3d.pth", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    test(args.data_folder, args.save_path)