# INT8 GEMM Accelerator for MNIST Inference

A parameterizable GEMM (General Matrix Multiplication) accelerator implemented in Verilog and evaluated using a quantized 2-layer MNIST classifier.

The accelerator performs matrix multiplication using parallel INT8 MAC units, while neural-network-specific operations such as bias addition, ReLU activation, requantization, and prediction are handled in the testbench. This separation keeps the compute engine generic and reusable for arbitrary GEMM workloads.

---

## Architecture

* INT8 activations and weights
* INT32 accumulation
* Multiple parallel MAC units
* Packed weight storage optimized for MAC parallelism
* SRAM-based memory system
* Parameterizable matrix dimensions (M, K, N)
* Fixed-point requantization using Q16.16 scaling factors

### Weight Packing

Weights are packed in SRAM according to the number of parallel MAC lanes. Each SRAM word stores multiple INT8 weights, allowing several output columns to be processed simultaneously and improving memory bandwidth utilization.

---

## Neural Network

784 → 32 → ReLU → 10

### Accelerator Responsibilities

* Activation × Weight multiplication
* INT32 accumulation of partial sums
* Writing outputs to SRAM C

### Testbench Responsibilities

* Bias addition
* ReLU activation
* Fixed-point requantization
* Layer-to-layer data movement
* Argmax prediction
* Accuracy measurement
* Cycle-count collection

---

## Quantization

The floating-point model is quantized to INT8 activations and weights.

Per-layer activation, weight, and bias scaling factors are exported during training and stored as Q16.16 fixed-point values. During inference, INT32 accumulated outputs are requantized using these scaling factors before being passed to the next layer.

### Inference Flow

INT8 Activations
        ↓
GEMM Accelerator
(A × W Accumulation)
        ↓
INT32 Partial Sums
        ↓
Bias Addition
        ↓
ReLU
        ↓
Requantization (Q16.16)
        ↓
Next Layer
        ↓
Prediction

---

## Results

| Metric                 | Value          |
| ---------------------- | -------------- |
| Dataset                | MNIST Test Set |
| Precision              | INT8           |
| FP32 Accuracy          | ~96%           |
| INT8 Accuracy          | ~94.6%         |
| Average Layer 1 Cycles | 3180           |
| Average Layer 2 Cycles | 88             |
| Average Total Cycles   | 3268           |

---

## Key Files


---

## Usage

1. Train the MNIST model using pytorch and export the quantized test data, weights & biases

python ./python_scripts/train_mnist_export_int8.py

2. Copy test data corresponding to a single image to selected_image.txt 

python ./python_scripts/select_test_image.py <index of the image>

3. Compile and run verilog simulation

Compile:

iverilog -g2012 -o simulations/sim.out \
tb/tb_gemm_mnist.v \
rtl/gemm_accelerator.v \
rtl/mac.v \
rtl/sram_A.v \
rtl/sram_B.v \
rtl/sram_C.v

Run:

vvp simulations/sim.out +IMAGE_IDX=<index of the image> +LABEL=<label of the image> 

#needs manual pick of the label from ./exports/test_labels.txt

4. Alternatively simulate and get stats for multiple images using the MNIST_rtl_test_wrapper.py script

python ./python_scripts/MNIST_rtl_test_wrapper.py <num of images to be simulated>

#automatically simulates test images from index 0 to n_images-1 and calculates accuracy and cycle counts.  
#n_images <= 10000. 

5. Python simulation that mimics RTL code for comparision:

python ./python_scripts/python_manual_mnist.py <num of images to be simulated>
---

## Future Improvements

* Early termination for partially utilized MAC lanes
* Tiled matrix multiplication for larger workloads
* AXI memory interface
* FPGA implementation and benchmarking
* Systolic-array-based architecture exploration
