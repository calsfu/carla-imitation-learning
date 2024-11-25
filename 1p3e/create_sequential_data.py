import numpy as np
import glob

def create_sequential_data(self, data_dir):
    data_list = glob.glob(data_dir+'*.npy') #need to change to your data format

    sequential_data = []
    control_data = []
    sequence_len = 3

    for i in range(len(data_list) - sequence_len):
        sequence = []
        control = []
        for j in range(sequence_len):
            data = np.load(data_list[i+j], allow_pickle=True).item()
            observation = data['observation']
            control = data['action'] # will use the last control as the label
            image = observation['camera']
            if self.transform:
                image = self.transform(image)
            sequence.append(image)
        
        sequential_data.append(sequence)
        control_data.append(control) 

    new_dict = {'observation': sequential_data, 'action': control_data}

    return new_dict


