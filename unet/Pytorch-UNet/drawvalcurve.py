import matplotlib.pyplot as plt
import numpy as np

results = []
with open("/Users/ziyaoshang/Desktop/MEproject/me744_project/unet/Pytorch-UNet/logs/single2lbs_retrain_logs.txt", "r") as f:
    for line in f:
        if "Dice" in line:
            # extract the dice score which is the last element after splitting by space 
            results.append(float(line.strip().split()[-1]))
    


# five validation rounds per epoch
epochs = len(results) // 5
# still plot all results, but mark epoch boundaries
x_vals = np.arange(len(results))
plt.figure(figsize=(10, 5))
plt.plot(x_vals / 5.0, results, marker='o', label='Dice Coefficient')
# for epoch in range(1, epochs):
#     plt.axvline(x=epoch * 5 - 0.5, color='r', linestyle='--', alpha=0.5)
plt.title('Validation Dice Coefficient Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.ylim(0.5, 1)
plt.grid(True)
plt.legend()
plt.savefig('validation_dice_curve_baseline.png')
# plt.show()