import torch
import torch.nn as nn

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 320x240 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')
        self.model = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11264 , 512),
            nn.ReLU(),
            nn.Linear(512 , 100),
            nn.ReLU(),
            nn.Linear(100 , 50),
            nn.ReLU(),
            nn.Linear(50, 4)
        ).to(gpu)



    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """

        return torch.sigmoid(self.model(observation))



    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector in multi-hot encoding. 
        That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        # actions is [acceleration, steering, braking]
        # C = 4
        # [steer_right, steer_left, throttle, brake]

        throttle = actions[0]
        steering = actions[1]
        brake = actions[2]

        action = torch.zeros(4)

        if steering > 0:
            action[0] = 1
        elif steering < 0:
            action[1] = 1
        if throttle > 0:
            action[2] = 1
        if brake > 0:
            action[3] = 1

        return action
    
    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        # C = 4
        # [steer_right, steer_left, throttle, brake]
        action = [0, 0, 0]
        if scores[0] > 0.5:
            action[0] = 1
        if scores[1] > 0.5 and scores[0] < 0:
            action[1] = 1
        elif scores[1] > 0.5 and scores[0] > 0:
            action[1] = -1
        else:
            action[1] = 0
        if scores[2] > 0.5:
            action[2] = 1
        return action


