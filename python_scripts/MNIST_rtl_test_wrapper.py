import subprocess
import pandas as pd
import sys
import os

os.makedirs("performance_results", exist_ok=True)
os.makedirs("simulations", exist_ok=True)

n_images = int(sys.argv[1])

img_select_script_path = "python_scripts/select_test_image.py"
test_labels_path = "exports/test_labels.txt"

test_bench = "tb/tb_gemm_mnist.v"
gemm_accelerator = "rtl/gemm_accelerator.v"
mac = "rtl/mac.v"
sram_A = "rtl/sram_A.v"
sram_B = "rtl/sram_B.v"
sram_C = "rtl/sram_C.v"

labels = []

with open(test_labels_path, "r") as f:
    labels = [int(line.strip()) for line in f]

print("Compiling RTL...")

subprocess.run(
    [
        "iverilog",
        "-g2012",
        "-o",
        "simulations/sim.out",
        test_bench,
        gemm_accelerator,
        mac,
        sram_A,
        sram_B,
        sram_C
    ],
    check=True
)

print("Compilation successful")

df = pd.DataFrame(columns=[
    "image_idx",
    "label",
    "pred",
    "result",
    "l1_cycles",
    "l2_cycles",
    "total_cycles"
])

df.to_csv("performance_results/rtl_performance.csv", index=False)

for i in range(n_images):
    img_idx = i
    subprocess.run(
        [
            "python3",
            f"{img_select_script_path}",
            f"{img_idx}"
        ], 
    check=True
    )
    
    subprocess.run(
        [
            "vvp",
            "simulations/sim.out",
            f"+IMAGE_IDX={img_idx}",
            f"+LABEL={labels[img_idx]}"
        ],
        check=True
    )

df = pd.read_csv("performance_results/rtl_performance.csv")
accuracy = df["result"].mean() * 100
total_correct = df["result"].sum()
total_incorrect = n_images - total_correct
avg_l1_cycles = df["l1_cycles"].mean()
avg_l2_cycles = df["l2_cycles"].mean()
avg_total_cycles = df["total_cycles"].mean()


print(f"Accuracy = {accuracy:.2f}%")
print(f"Total Correct ={total_correct}")
print(f"Total Inorrect ={total_incorrect}")
print(f"Average L1 cycles ={avg_l1_cycles}")
print(f"Average L2 cycles ={avg_l2_cycles}")
print(f"Total cycles ={avg_total_cycles}")





    


