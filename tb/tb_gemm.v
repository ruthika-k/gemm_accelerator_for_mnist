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
integer cycle_counter;
integer start_cycle;
integer end_cycle;

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

//////////////////////////////////////////////////
// Clock
//////////////////////////////////////////////////

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial cycle_counter = 0;

always @(posedge clk)
    cycle_counter = cycle_counter + 1;

//////////////////////////////////////////////////
// Wave dump
//////////////////////////////////////////////////

initial begin
    $dumpfile("waveforms/gemm.vcd");
    $dumpvars(0, tb_gemm);
end

//////////////////////////////////////////////////
// Cycle monitor
//////////////////////////////////////////////////

always @(posedge clk) begin
    $display("--------------------------------------------------");
    $display("t=%0t state=%0d done=%0d",
             $time,
             dut.state,
             dut.done);

    $display("i=%0d j=%0d k=%0d",
             dut.i,
             dut.j,
             dut.k);

    $display("issue=%0d data=%0d mac=%0d",
             dut.issue_valid,
             dut.data_valid,
             dut.mac_valid);

    $display("last_issue=%0d last_data=%0d last_mac=%0d",
             dut.last_issue,
             dut.last_data,
             dut.last_mac);

    $display("A_addr=%0d B_addr=%0d",
             dut.sram_A_addr,
             dut.sram_B_addr);

    $display("A_dout=%0d B_dout=%h",
             dut.sram_A_dout,
             dut.sram_B_dout);

//    $display("A_reg=%0d B_reg=%h",
//             dut.A_reg,
//             dut.B_reg);

    $display("accum0=%0d mac0=%0d",
             dut.accum_reg[0],
             dut.mac_out[0]);

    $display("C_WE=%0d C_addr=%0d C_din=%0d",
             dut.sram_C_WE,
             dut.sram_C_addr,
             dut.sram_C_din);
end

//////////////////////////////////////////////////
// Main test
//////////////////////////////////////////////////

initial begin

    rst_n  = 0;
    start  = 0;

    A_base = 0;
    B_base = 0;
    C_base = 0;

    M = 2;
    K = 2;
    N = 5;

    //--------------------------------------------------
    // Reset
    //--------------------------------------------------

    #20;
    rst_n = 1;

    //--------------------------------------------------
    // Initialize SRAM A
    //--------------------------------------------------
    dut.sram_A.SRAMs[3] = {
        8'd4
    };
    
    dut.sram_A.SRAMs[2] = {
        8'd3
    };

    dut.sram_A.SRAMs[1] = {
        8'd2
    };

    dut.sram_A.SRAMs[0] = {
        8'd1
    };

    //--------------------------------------------------
    // Initialize SRAM B
    //--------------------------------------------------

    dut.sram_B.SRAMs[1] = {
        8'd8,
        8'd7,
        8'd6,
        8'd5,
        8'd4,
        8'd3,
        8'd2,
        8'd1
    };
    
    dut.sram_B.SRAMs[0] = {    
        8'd8,
        8'd7,
        8'd6,
        8'd5,
        8'd4,
        8'd3,
        8'd2,
        8'd1
    };



    //--------------------------------------------------
    // Clear SRAM C
    //--------------------------------------------------

    for(i = 0; i < 16; i = i + 1)
        dut.sram_C.SRAMs[i] = 0;

    //--------------------------------------------------
    // Start accelerator
    //--------------------------------------------------

    @(posedge clk);
    start <= 1;
    start_cycle = cycle_counter;

    @(posedge clk);
    start <= 0;

    wait(done == 0);
    $display("Accelerator started");

    wait(done == 1);
    $display("Accelerator finished");

    $display("Layer 1 done");

    end_cycle = cycle_counter;

    $display("Total cycles = %0d", end_cycle - start_cycle);

    //--------------------------------------------------
    // Dump SRAM C
    //--------------------------------------------------

    $display("");
    $display("===== SRAM C CONTENTS =====");

    for(i = 0; i < 16; i = i + 1)
        $display("C[%0d] = %0d", i, dut.sram_C.SRAMs[i]);

    $display("");
    $finish;

end

endmodule