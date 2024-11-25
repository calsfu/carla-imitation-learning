import numpy as np
import matplotlib.pyplot as plt

# Loss values for each epoch
loss_values = [
    160.928436, 92.945847, 92.927711, 92.922394, 92.917274, 92.916931, 
    92.907532, 92.937202, 92.889252, 92.843292, 92.812500, 92.850349, 
    92.878746, 92.868652, 92.842117, 92.857742, 92.845398, 92.829834, 
    92.866180, 92.850098, 92.828857, 92.873535, 92.837494, 92.824265, 
    92.841675, 92.834839, 92.818832, 92.828003, 92.839416, 92.836639, 
    92.826187, 92.848206, 92.810303, 92.820427, 92.831642, 92.815231, 
    92.845062, 92.823456, 92.816605, 92.849808, 92.820183, 92.832832, 
    92.827072, 92.812469, 92.841293, 92.816498, 92.824707, 92.837158, 
    92.828171, 92.823410, 92.833229, 92.820084, 92.840240, 92.812317, 
    92.818619, 92.829269, 92.825912, 92.817795, 92.838989, 92.826935, 
    92.823135, 92.830292, 92.826775, 92.819962, 92.836822, 92.832275, 
    92.828033, 92.841400, 92.814247, 92.827576, 92.835754, 92.819092, 
    92.843048, 92.816963, 92.825226, 92.831482, 92.810715, 92.839935, 
    92.829666, 92.821503, 92.846832, 92.818604, 92.833580, 92.830963, 
    92.825363, 92.812813, 92.840118, 92.823013, 92.832687, 92.829010, 
    92.819199, 92.831818, 92.825150, 92.839081, 92.822845, 92.824676, 
    92.832993, 92.812180, 92.820892, 92.837883, 92.830254, 92.822433
]


# Generate epochs
epochs = np.arange(1, 101)

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, label="Training Loss Over Epochs ", color='b', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multiclass Training Loss")
plt.grid(True)
plt.legend()
plt.show()