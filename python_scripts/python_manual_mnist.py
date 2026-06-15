import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import sys
import pandas as pd
import os

os.makedirs("performance_results", exist_ok=True)
num_images = int(sys.argv[1])

SCALE_FRAC_BITS = 16

# =========================================================
# MODEL
# =========================================================

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)

# =========================================================
# LOAD MODEL
# =========================================================

model = SimpleNN()
model.load_state_dict(
    torch.load("mnist_model/mnist_fp32.pth", map_location="cpu")
)
model.eval()

# =========================================================
# LOAD DATASET
# =========================================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# =========================================================
# LOAD Q16.16 SCALES
# =========================================================

with open("exports/scales.txt") as f:
    scales = np.array(
        [int(line.strip(), 16) for line in f],
        dtype=np.int32
    )

A0_scale = int(scales[0])
A1_scale = int(scales[1])
A2_scale = int(scales[2])

W1_scale = int(scales[3])
W2_scale = int(scales[4])

B1_scale = int(scales[5])
B2_scale = int(scales[6])

# floating-point versions
A0 = A0_scale / 65536.0
A1 = A1_scale / 65536.0
A2 = A2_scale / 65536.0

W1 = W1_scale / 65536.0
W2 = W2_scale / 65536.0

B1 = B1_scale / 65536.0
B2 = B2_scale / 65536.0

# =========================================================
# EXTRACT WEIGHTS
# =========================================================

state = model.state_dict()

w1 = state["model.1.weight"].cpu().numpy()
b1 = state["model.1.bias"].cpu().numpy()

w2 = state["model.3.weight"].cpu().numpy()
b2 = state["model.3.bias"].cpu().numpy()

# =========================================================
# QUANTIZATION HELPERS
# =========================================================

def qint8(x, scale):
    return np.clip(
        np.round(x / scale),
        -128,
        127
    ).astype(np.int8)

# =========================================================
# QUANTIZE PARAMETERS
# =========================================================

w1_q = qint8(w1, W1)
w2_q = qint8(w2, W2)

b1_q = np.round(
    b1 / B1
).astype(np.int32)

b2_q = np.round(
    b2 / B2
).astype(np.int32)


df = pd.DataFrame(columns=[
    "image_idx",
    "label",
    "pred",
    "result",
])

df.to_csv("performance_results/python_performance.csv", index=False)


for idx in range(num_images):

    img, label = test_dataset[idx]

    # =========================================================
    # QUANTIZE INPUT
    # =========================================================

    x = img.view(-1).numpy()

    x_q = qint8(
        x,
        A0
    )

    # =========================================================
    # REQUANTIZATION MULTIPLIERS
    # =========================================================

    rq1 = (A0_scale * W1_scale) // A1_scale
    rq2 = (A1_scale * W2_scale) // A2_scale

    # =========================================================
    # LAYER 1
    # =========================================================

    fc1 = np.zeros(
        32,
        dtype=np.int32
    )

    for i in range(32):

        acc = np.sum(
            x_q.astype(np.int32) *
            w1_q[i].astype(np.int32),
            dtype=np.int32
        )

        acc += int(b1_q[i])

        # ReLU
        if acc < 0:
            acc = 0

        # RTL:
        # temp = temp * rq1;
        # temp = temp >>> 16;

        acc = acc * rq1
        acc = acc >> SCALE_FRAC_BITS

        # clamp
        if acc > 127:
            acc = 127

        elif acc < -128:
            acc = -128

        fc1[i] = acc

    #debug
    #print("\nLayer1 Activations")

    #for i, v in enumerate(fc1):
    #    print(i, v)

    # =========================================================
    # LAYER 2
    # =========================================================

    fc2 = np.zeros(
        10,
        dtype=np.int32
    )

    for i in range(10):

        acc = np.sum(
            fc1.astype(np.int32) *
            w2_q[i].astype(np.int32),
            dtype=np.int32
        )

        acc += int(b2_q[i])

        acc = acc * rq2
        acc = acc >> SCALE_FRAC_BITS

        fc2[i] = acc

    # =========================================================
    # PREDICTION
    # =========================================================

    pred = int(np.argmax(fc2))

    print("\n================================")
    print("RESULT")
    print("================================")
    print("image idx  :", idx)
    print("Label      :", label)
    print("Prediction :", pred)

    new_row = {
        "image_idx": idx,
        "label": label,
        "pred": int(pred),
        "result": int(pred == label)
    }

    pd.DataFrame([new_row]).to_csv(
        "performance_results/python_performance.csv",
        mode="a",      
        header=False,  
        index=False
    )



df = pd.read_csv("performance_results/python_performance.csv")
accuracy = df["result"].mean() * 100
total_correct = df["result"].sum()
total_incorrect = num_images - total_correct


print(f"Accuracy = {accuracy:.2f}%")
print(f"Total Correct ={total_correct}")
print(f"Total Inorrect ={total_incorrect}")

#debug
#print("\nLayer2 Logits")

#for i, v in enumerate(fc2):
#    print(f"{i}: {v}")