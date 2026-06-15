`timescale 1ns/1ps

module tb_gemm;

reg clk;
reg rst_n;
reg start;

reg [9:0]  A_base;
reg [11:0] B_base;
reg [9:0]  C_base;

reg [15:0] M;
reg [15:0] K;
reg [15:0] N;

wire done;

integer i;

//variables to help with stats export
integer stats_file;

integer cycle_counter;
integer l1_start_cycle;
integer l1_end_cycle;
integer l2_start_cycle;
integer l2_end_cycle;
integer total_cycles;

//current inference details
integer image_idx;
integer label;
integer result;

//scaling factors for the activations, weights and biases
reg signed [31:0] scales [0:6];
reg signed [31:0] A0_scale;
reg signed [31:0] A1_scale;
reg signed [31:0] A2_scale;
reg signed [31:0] W1_scale;
reg signed [31:0] W2_scale;
reg signed [31:0] B1_scale;
reg signed [31:0] B2_scale;

// Requantization multipliers (Q16.16)
reg signed [63:0] requant_l1;
reg signed [63:0] requant_l2;


//--------------------------------------------------
// DUT
//--------------------------------------------------

gemm_accelerator dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),

    .A_base(A_base),
    .B_base(B_base),
    .C_base(C_base),

    .M(M),
    .K(K),
    .N(N),

    .done(done)
);

//--------------------------------------------------
// CLOCK
//--------------------------------------------------

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

always @(posedge clk)
    cycle_counter = cycle_counter + 1;

//--------------------------------------------------
// INITILIZE STATS COLLECTION
//--------------------------------------------------
initial begin
    cycle_counter = 0;

    stats_file = $fopen("performance_results/rtl_performance.csv","a");
end

//--------------------------------------------------
// IMPORT SCALING FACTORS -- 32bit Q16.16 fixed point numbers
//--------------------------------------------------
initial begin
    $display("Importing scaling factors...");

    $readmemh(
        "exports/scales.txt",
        scales
    );

    A0_scale = scales[0];
    A1_scale = scales[1];
    A2_scale = scales[2];
    W1_scale = scales[3];
    W2_scale = scales[4];
    B1_scale = scales[5];
    B2_scale = scales[6];

    // Compute requantization multipliers once
    requant_l1 =
        ($signed(A0_scale) * $signed(W1_scale))
        / $signed(A1_scale);

    requant_l2 =
        ($signed(A1_scale) * $signed(W2_scale))
        / $signed(A2_scale);

    $display("A0_scale = %0d", A0_scale);
    $display("A1_scale = %0d", A1_scale);
    $display("A2_scale = %0d", A2_scale);

    $display("W1_scale = %0d", W1_scale);
    $display("W2_scale = %0d", W2_scale);

    $display("B1_scale = %0d", B1_scale);
    $display("B2_scale = %0d", B2_scale);

    $display("requant_l1 = %0d", requant_l1);
    $display("requant_l2 = %0d", requant_l2);
end

//--------------------------------------------------
// STORAGE 
//--------------------------------------------------

reg signed [31:0] l1_out [0:31]; //32 neurons
reg signed [31:0] bias1  [0:31];
reg signed [7:0]  l1_int8  [0:31];

reg signed [31:0] l2_out [0:9]; //10 neurons
reg signed [31:0] bias2  [0:9];

reg signed [63:0] temp;

reg signed [31:0] max_val;
reg [3:0]  pred;

//--------------------------------------------------
// MAIN TEST
//--------------------------------------------------

initial begin
    if(!$value$plusargs("IMAGE_IDX=%d", image_idx))
        image_idx = 0;

    $display("image_idx = %0d", image_idx);

    if(!$value$plusargs("LABEL=%d", label))
        label = 0;

    $display("label =  %0d", label);
end

initial begin

    //--------------------------------------------------
    // ASSERT RESET
    //--------------------------------------------------
    rst_n = 0;
    start = 0;

    A_base = 0;
    B_base = 0;
    C_base = 0;

    //--------------------------------------------------
    // RELEASE RESET
    //--------------------------------------------------

    #20;
    rst_n = 1;

    //--------------------------------------------------
    // ================= LAYER 1 =================
    //--------------------------------------------------

    M = 1;
    K = 784;
    N = 32;

    $display("Loading Layer 1...");

    $readmemh("exports/selected_image.txt",
              dut.sram_A.SRAMs, 0, 783);

    $readmemh("exports/layer1_w_int8.txt",
              dut.sram_B.SRAMs, 0, 3135);

    $readmemh("exports/layer1_b_int32.txt",
              bias1, 0, 31);
    

    //--------------------------------------------------
    // RUN L1
    //--------------------------------------------------

    @(posedge clk);
    start <= 1;

    @(posedge clk);
    start <= 0;
    l1_start_cycle = cycle_counter;

    wait(done == 0);
    $display("Accelerator started");

    wait(done == 1);
    $display("Accelerator finished");

    $display("Layer 1 done");

    l1_end_cycle = cycle_counter;

    $display("L1 cycles = %0d", l1_end_cycle - l1_start_cycle);


    // =====================================================
    // POST PROCESS L1
    // =====================================================
    for (i = 0; i < 32; i = i + 1) begin

        l1_out[i] = $signed(dut.sram_C.SRAMs[i]);    

        // bias add
        temp = l1_out[i] + bias1[i];

        // ReLU
        if (temp < 0)
            temp = 0;

        // REQUANTIZE TO INT8
        temp = temp * requant_l1;
        temp = temp >>> 16;

        // clamp
        if (temp > 127)
            temp = 127;
        else if (temp < -128)
            temp = -128;

        l1_int8[i] = temp[7:0];       
    end

    //--------------------------------------------------
    // FEED L1 → L2
    //--------------------------------------------------

    for (i = 0; i < 32; i = i + 1) begin
        dut.sram_A.SRAMs[i] = l1_int8[i];
    end

    //--------------------------------------------------
    // ================= LAYER 2 =================
    //--------------------------------------------------

    M = 1;
    K = 32;
    N = 10;

    $display("Loading Layer 2...");

    $readmemh("exports/layer2_w_int8.txt",
              dut.sram_B.SRAMs, 0, 63);

    $readmemh("exports/layer2_b_int32.txt",
              bias2, 0, 9);

    //--------------------------------------------------
    // RUN L2
    //--------------------------------------------------

    @(posedge clk);
    start <= 1;

    @(posedge clk);
    start <= 0;
    l2_start_cycle = cycle_counter;

    wait(done == 0);
    $display("Accelerator started");

    wait(done == 1);
    $display("Accelerator finished");

    $display("Layer 2 done");

    l2_end_cycle = cycle_counter;

    $display("L2 cycles = %0d",
         l2_end_cycle - l2_start_cycle);

    //--------------------------------------------------
    // L2 POST PROCESS 
    //--------------------------------------------------

    max_val = 32'h80000000;
    pred = 0;

    for (i = 0; i < 10; i = i + 1) begin

        l2_out[i] = $signed(dut.sram_C.SRAMs[i]);

        temp = l2_out[i] + bias2[i];

        temp = temp * requant_l2;
        temp = temp >>> 16;

        l2_out[i] = temp[31:0];

        $display("OUT[%0d] = %0d", i, l2_out[i]);

        if (l2_out[i] > max_val) begin
            max_val = l2_out[i];
            pred = i;
        end
    end

    //--------------------------------------------------
    // RESULT
    //--------------------------------------------------

    $display("");
    $display("===== FINAL PREDICTION =====");
    $display("Digit = %0d", pred);

    total_cycles = (l1_end_cycle - l1_start_cycle) + (l2_end_cycle - l2_start_cycle);
    result = (pred==label);

    $fwrite(stats_file,
    "%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
    image_idx,
    label,
    pred,
    result,
    l1_end_cycle - l1_start_cycle,
    l2_end_cycle - l2_start_cycle,
    total_cycles
    );

    $fclose(stats_file);

    $display("===== RESULT =====");
    $display("Label      = %0d", label);
    $display("Prediction = %0d", pred);
    $display("Correct    = %0d", result);

    $finish;
end

endmodule