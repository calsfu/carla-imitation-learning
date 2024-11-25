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
        self.image_model = torch.nn.Sequential(
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
        ).to(gpu)

        # Assuming concat
        self.model = torch.nn.Sequential(
            nn.Linear(59, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
        ).to(gpu)



    def forward(self, image, sensors):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        image:   torch.Tensor of size (batch_size, height, width, channel)
        sensor_data: torch.Tensor of size (batch_size, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        # run the through image model
        image_out = self.image_model(image)

        # Concatenate the image and sensor data
        out = torch.cat([image_out, sensors], dim=1)

        return self.model(out)



    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        # actions is [acceleration, steering, braking]
        # C = 9
        # [throttle, steer_left and throttle, steer_right and throttle,
        # brake,  brake and steer_left, brake and steer_right,
        # coast, steer_left, steer_right] 

        throttle = actions[0]
        steering = actions[1]
        brake = actions[2]

        action = torch.zeros(9)

        if throttle > 0 and brake == 0: # throttle
            if steering < 0:
                action[1] = 1 # steer_left and throttle
            elif steering > 0:
                action[2] = 1 # steer_right and throttle
            else:
                action[0] = 1 # throttle
        elif throttle == 0 and brake > 0: # brake
            if steering < 0:
                action[4] = 1 # brake and steer_left
            elif steering > 0:
                action[5] = 1 # brake and steer_right
            else:
                action[3] = 1 # brake
        elif throttle == 0 and brake == 0: # coast
            if steering < 0:
                action[7] = 1 # steer_left
            elif steering > 0:
                action[8] = 1 # steer_right 
            else:
                action[6] = 1 # coast

        assert torch.sum(action) == 1

        return action
    
    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """

        # Note: The scores are compounded onto the current values of the car's controls
        # Planning to add the scores to the current values of the car's controls
        if scores[0] == 1:
            return (1, 0, 0)
        elif scores[1] == 1:
            return (1, -1, 0)
        elif scores[2] == 1:
            return (1, 1, 0)
        elif scores[3] == 1:
            return (0, 0, 1)
        elif scores[4] == 1:
            return (0, -1, 1)
        elif scores[5] == 1:
            return (0, 1, 1)
        elif scores[6] == 1:
            return (0, 0, 0)
        elif scores[7] == 1:
            return (0, -1, 0)
        elif scores[8] == 1:
            return (0, 1, 0)
        else:
            raise ValueError("Invalid action class")


