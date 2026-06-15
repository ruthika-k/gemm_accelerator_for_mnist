import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

os.makedirs("exports", exist_ok=True)
os.makedirs("mnist_model", exist_ok=True)

SCALE_FRAC_BITS = 16   # Q16.16 fixed-point 

# =========================================================
# HYPERPARAMETERS
# =========================================================

batch_size = 64
learning_rate = 0.001
epochs = 5

print("Hyperparameters set")

# =========================================================
# DATASET
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("MNIST loaded")

# =========================================================
# MODEL
# =========================================================

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN()

print("Model created")

# =========================================================
# LOSS + OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =========================================================
# TRAINING
# =========================================================

print("Training started")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss {loss.item():.4f}")

    print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

# =========================================================
# EVALUATION
# =========================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# =========================================================
# CALIBRATION HELPERS
# =========================================================

def collect_layer_outputs():
    A1_list = []
    A2_list = []

    with torch.no_grad():
        for i in range(1000):
            img, _ = test_dataset[i]
            x = img.view(1, -1)

            fc1 = model.model[1](x)
            r1  = model.model[2](fc1)
            fc2 = model.model[3](r1)

            A1_list.append(r1.numpy())
            A2_list.append(fc2.numpy())

    return np.concatenate(A1_list), np.concatenate(A2_list)

# =========================================================
# SCALE COMPUTATION
# =========================================================

def scale(x):
    return np.percentile(np.abs(x), 99.9) / 127.0

print("Calibrating...")

A1_vals, A2_vals = collect_layer_outputs()

A1_scale = scale(A1_vals)
A2_scale = scale(A2_vals)

A0_scale = 1/127  # MNIST already 0–1, simple fixed scale

print("A0_scale:", A0_scale)
print("A1_scale:", A1_scale)
print("A2_scale:", A2_scale)

# =========================================================
# EXTRACT WEIGHTS + BIASES
# =========================================================

state = model.state_dict()

w1 = state["model.1.weight"].numpy()
b1 = state["model.1.bias"].numpy()
w2 = state["model.3.weight"].numpy()
b2 = state["model.3.bias"].numpy()

# weight scales
W1_scale = scale(w1)
W2_scale = scale(w2)

# bias scales (derived)
B1_scale = A0_scale * W1_scale
B2_scale = A1_scale * W2_scale

# =========================================================
# QUANTIZATION
# =========================================================

def qint8(x, scale):
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8)

w1_q = qint8(w1, W1_scale)
w2_q = qint8(w2, W2_scale)

# bias scaling 
b1_q = np.round(b1 / B1_scale).astype(np.int32)
b2_q = np.round(b2 / B2_scale).astype(np.int32)

# =========================================================
# EXPORT HELPERS
# =========================================================

def export_int8(x, name):
    with open(name, "w") as f:
        for v in x.flatten():
            f.write(f"{int(v) & 0xFF:02X}\n")

def export_int32(x, name):
    with open(name, "w") as f:
        for v in x.flatten():
            f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")

def export_int8_8wide(data, filename):
    """
    data shape = (num_neurons, num_inputs)

    Output:
        Each line contains 8 weights.

    Layout:
        For each block of 8 neurons:
            For each input index:
                w[n+7][i].... w[n+1][i] w[n+0][i]

    Missing neurons are padded with 00.
    """

    num_neurons, num_inputs = data.shape

    with open(filename, "w") as f:

        for neuron_base in range(0, num_neurons, 8):

            for inp in range(num_inputs):

                line = []

                for lane in range(8):

                    neuron = neuron_base + lane

                    if neuron < num_neurons:
                        val = data[neuron][inp]
                    else:
                        val = 0

                    line.append(f"{int(val) & 0xFF:02X}")
                
                line.reverse()
                f.write("".join(line))
                f.write("\n")

    print(f"Saved {filename}")

# =========================================================
# EXPORT WEIGHTS & BIASES
# =========================================================

export_int8_8wide(w1_q, "exports/layer1_w_int8.txt")
export_int32(b1_q, "exports/layer1_b_int32.txt")

export_int8_8wide(w2_q, "exports/layer2_w_int8.txt")
export_int32(b2_q, "exports/layer2_b_int32.txt")


# =========================================================
# EXPORT TEST DATA
# =========================================================

print("Exporting test data")

def export_test_data():
    with open("exports/test_inputs_int8.txt", "w") as f_in, \
        open("exports/test_labels.txt", "w") as f_lbl:

        for img, label in test_dataset:

            x = img.view(-1).numpy()
            x_q = qint8(x, A0_scale)
            for v in x_q:
                f_in.write(f"{int(v) & 0xFF:02X} ")
            f_in.write("\n")
            f_lbl.write(f"{label}\n")

export_test_data()

# =========================================================
# EXPORT SACLES
# =========================================================
def to_fixed(x, frac_bits=SCALE_FRAC_BITS):
    return int(np.round(x * (1 << frac_bits)))

scales = {
    "A0_scale": A0_scale,
    "A1_scale": A1_scale,
    "A2_scale": A2_scale,
    "W1_scale": W1_scale,
    "W2_scale": W2_scale,
    "B1_scale": B1_scale,
    "B2_scale": B2_scale,
}

with open("exports/scales.txt", "w") as f:
    for k, v in scales.items():
        f.write(f"{to_fixed(v):08X}\n")

print("Saved scales.mem")

# debug print
for k, v in scales.items():
    print(f"{k:10s} = {v:.8f}  -> {to_fixed(v)} (fixed)")

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), "mnist_model/mnist_fp32.pth")

print("DONE: TRAIN + CALIBRATE + EXPORT INT8")