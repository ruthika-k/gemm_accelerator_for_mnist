module gemm_accelerator(
    input clk,
    input rst_n,

    input start,

    input [9:0]  A_base,
    input [11:0] B_base,
    input [9:0]  C_base,

    input [15:0] M,
    input [15:0] K,
    input [15:0] N,

    output reg done
);

//regs to store the inputs
reg [9:0]  A_base_r;
reg [11:0] B_base_r;
reg [9:0]  C_base_r;

reg [15:0] M_r;
reg [15:0] K_r;
reg [15:0] N_r;

//FSM variables
localparam IDLE    = 3'd0;
localparam INIT    = 3'd1;
localparam COMPUTE = 3'd2;
localparam WRITE   = 3'd3;
localparam DONE    = 3'd4;

reg [2:0] state;
reg [2:0] next_state;

//sram I/O
reg [7:0] sram_A_din;
wire [7:0] sram_A_dout;
reg [9:0] sram_A_addr;
reg [0:0] sram_A_CS, sram_A_WE, sram_A_RD; 

reg [63:0] sram_B_din;
wire [63:0] sram_B_dout;
reg [11:0] sram_B_addr;
reg [0:0] sram_B_CS, sram_B_WE, sram_B_RD; 

reg [31:0] sram_C_din;
wire [31:0] sram_C_dout;
reg [9:0] sram_C_addr;
reg [0:0] sram_C_CS, sram_C_WE, sram_C_RD; 

//counters to loop through the matrix
reg [15:0] i; //iterates through rows of Matrix A
reg [15:0] j; //iterates through columns of Matrix B, N_MAC columns at once 
reg [15:0] k; //iterates through elements in a single row of Matrix A or column of Matrix B
reg [15:0] i_max; // i_max = M-1 
reg [15:0] j_max; // j_max = ceil(N/N_MAC)-1
reg [15:0] k_max; // k_max = K-1

//counter to loop through the MAC units during multicycle WRITE state
reg [3:0] mac_id;

//parameters
localparam N_MAC = 8; // number of parallel mac units
integer idx;

//MAC I/O
reg signed [7:0] A_reg;
reg signed [7:0] B_reg [N_MAC-1:0];
reg signed [31:0] accum_in [N_MAC-1:0];
wire signed [31:0] mac_out [N_MAC-1:0];
reg signed [31:0] accum_reg [N_MAC-1:0];

//Pipelined valid signals 
reg issue_valid; //address issued to sram_A and sram_B in that cycle is valid
reg data_valid;  //data in sram_A_dout and sram_B_dout regs are valid in that cycle
reg mac_valid;   //combinational MAC output is valid

//trackers for last compute cycles
reg last_issue; //address corresponding to the final k value is issued 
reg last_data;  //data corresponding to the last fetch is available in sram_A_dout and sram_B_dout
reg last_mac;   //combinational MAC output for the last fetch is ready

//tracks the completion of WRITE state
reg write_done;

//SRAM instances
SRAM_A sram_A(
    .dataIn(sram_A_din),
    .dataOut(sram_A_dout),
    .Addr(sram_A_addr),
    .CS(sram_A_CS),
    .WE(sram_A_WE),
    .RD(sram_A_RD), 
    .Clk(clk) 
);

SRAM_B sram_B(
    .dataIn(sram_B_din),
    .dataOut(sram_B_dout),
    .Addr(sram_B_addr),
    .CS(sram_B_CS),
    .WE(sram_B_WE),
    .RD(sram_B_RD),
    .Clk(clk) 
);

SRAM_C sram_C(
    .dataIn(sram_C_din),
    .dataOut(sram_C_dout),
    .Addr(sram_C_addr),
    .CS(sram_C_CS),
    .WE(sram_C_WE),
    .RD(sram_C_RD), 
    .Clk(clk) 
);

//MAC instances
genvar g;
generate
    for(g = 0; g < N_MAC; g = g + 1) begin : MAC_ARRAY
        mac mac_inst (
            .a(A_reg),
            .b(B_reg[g]),
            .accum_in(accum_in[g]),
            .accum_out(mac_out[g])
        );
    end
endgenerate

//state transistion
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        state <= IDLE;
    end
    else begin
        state <= next_state;
    end
end

//data path
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        done <= 0; 

        A_base_r <= 0;
        B_base_r <= 0;
        C_base_r <= 0;

        M_r <= 0;
        K_r <= 0;
        N_r <= 0;

        i_max <= 0;
        j_max <= 0;
        k_max <= 0;

        i <= 0;
        j <= 0;
        k <= 0;
        mac_id <= 0;

        issue_valid <= 0;
        data_valid <= 0;
        mac_valid <= 0;

        last_issue <= 0;
        last_data <= 0;
        last_mac <= 0;

        sram_A_addr <= 0;
        sram_B_addr <= 0;
        sram_C_addr <= 0;
        sram_C_din <= 0;

        sram_A_CS <= 0;
        sram_A_WE <= 0;
        sram_A_RD <= 0;

        sram_B_CS <= 0;
        sram_B_WE <= 0;
        sram_B_RD <= 0;

        sram_C_CS <= 0;
        sram_C_WE <= 0;
        sram_C_RD <= 0;

        A_reg <= 0;
        for(idx = 0; idx < N_MAC; idx = idx + 1) begin
            B_reg[idx] <= 0;
            accum_reg[idx] <= 0;
        end
    end
    else begin
        case(state) 
            IDLE:begin //hold reg values from done until start, load the new inputs when start
                if(start) begin
                    done <= 0; //done=1 until start pulse is received
                    A_base_r <= A_base;
                    B_base_r <= B_base;
                    C_base_r <= C_base;
                    M_r <= M;
                    K_r <= K;
                    N_r <= N;
                end 
            end
            INIT:begin //feed all the regs with known reset values
                i_max <= M_r-1;
                j_max <= ((N_r+N_MAC-1)/N_MAC)-1;
                k_max <= K_r-1;

                i <= 0;
                j <= 0;
                k <= 0;

                issue_valid <= 1; //issue_valid = 1 in first compute cycle as the address(combinational) for k=0 is issued
                data_valid <= 0;
                mac_valid <= 0;

                if(K_r==1) //corner case when k_max = K-1 = 0
                    last_issue <= 1;
                else
                    last_issue <= 0;
                last_data <= 0;
                last_mac <= 0;

                sram_C_addr <= 0;
                sram_C_din <= 0;
                mac_id <= 0;

                sram_C_CS <= 0;
                sram_C_WE <= 0;
                sram_C_RD <= 0;

                A_reg <= 0;
                for(idx = 0; idx < N_MAC; idx = idx + 1)begin
                    B_reg[idx] <= 0;
                    accum_reg[idx] <= 0;
                end
            end
            COMPUTE:begin //compute C[i][j*N_MAC : j*N_MAC+N_MAC-1]

                sram_C_CS <= 0;
                sram_C_RD <= 0;
                sram_C_WE <= 0;

                //propagate valid signals
                issue_valid <= (last_issue)? 0:1;
                data_valid <= issue_valid;
                mac_valid <= data_valid;

                //check for last cycle
                if(k_max == 0)
                    last_issue <= 1;
                else if(k==k_max-1)
                    last_issue <= 1;
                last_data <= last_issue;
                last_mac <= last_data;

                //update k-counter
                if(k<k_max) begin
                    k <= k+1;
                end

                //update accumulator reg with combinational mac output
                if(mac_valid) begin
                    for(idx = 0; idx < N_MAC; idx = idx + 1)
                        accum_reg[idx] <= mac_out[idx];
                end

                //copy the contents of SRAM outputs into input regs for MAC units
                if(data_valid) begin
                    A_reg <= sram_A_dout;
                    for(idx = 0; idx < N_MAC; idx = idx + 1)
                        B_reg[idx] <= sram_B_dout[idx*8 +: 8];
                end
            end
            WRITE:begin //write the contents of N_MAC mac units to sram_C, one mac unit value per cycle

                //iterate through each mac unit
                if(mac_id < N_MAC-1) begin
                    mac_id <= mac_id + 1;
                end

                //write to sram_C only for the valid columns of Matrix B
                if(j*N_MAC + mac_id < N_r) begin
                    sram_C_CS <= 1;
                    sram_C_RD <= 0;
                    sram_C_WE <= 1;
                    sram_C_addr <= C_base_r + i*N_r + j*N_MAC + mac_id;
                    sram_C_din <= accum_reg[mac_id];
                end
                else begin
                    sram_C_CS <= 0;
                    sram_C_RD <= 0;
                    sram_C_WE <= 0;                    
                end

                //after last write schedule, reset the registers and valid signals for next compute state or done state, update i,j counters
                if(mac_id == N_MAC-1) begin
                    mac_id <= 0;
                    A_reg <= 0;
                    for(idx = 0; idx < N_MAC; idx = idx + 1)begin
                        accum_reg[idx] <= 0; 
                        B_reg[idx] <= 0;   
                    end

                    data_valid <= 0;
                    mac_valid <= 0;
                    
                    if(k_max==0)
                        last_issue <= 1;
                    else
                        last_issue <= 0;
                    last_data <= 0;
                    last_mac <= 0;
                    
                    if(j<j_max)begin
                        j <= j+1;
                        k <= 0;
                        issue_valid <= 1;
                    end
                    else if(j==j_max && i<i_max)begin
                        j <= 0;
                        i <= i+1;
                        k <= 0;
                        issue_valid <= 1;
                    end
                    else begin
                        j <= 0;
                        i <= 0;
                        k <= 0;
                        issue_valid <= 0;
                    end
                end
            end
            DONE:begin //assert done = 1 and move to IDLE state

                sram_C_CS <= 0;
                sram_C_RD <= 0;
                sram_C_WE <= 0;

                done <= 1;            
            end
        endcase
    end
end


//combinational logic & next state logic
always @(*) begin

    sram_A_CS = (state == COMPUTE);
    sram_A_RD = (state == COMPUTE);
    sram_A_WE = 0;

    sram_B_CS = (state == COMPUTE);
    sram_B_RD = (state == COMPUTE);
    sram_B_WE = 0;

    sram_A_addr = A_base_r + i*K_r + k;
    sram_B_addr = B_base_r + j*K_r + k;

    for(idx = 0; idx < N_MAC; idx = idx + 1)
        accum_in[idx] = accum_reg[idx];

    write_done = (mac_id == N_MAC-1);
    
    next_state = state; //default

    case(state)
        IDLE:begin
            if(start)begin
                next_state = INIT;
            end
        end
        INIT:begin
            next_state = COMPUTE;
        end
        COMPUTE:begin
            if(mac_valid && last_mac)begin
                next_state = WRITE;
            end
        end
        WRITE:begin
            if(!write_done)begin
                next_state = WRITE;
            end
            else if(j==j_max && i==i_max)begin
                next_state = DONE;
            end
            else begin
                next_state = COMPUTE;
            end
        end
        DONE:begin
            next_state = IDLE;
        end        
    endcase
end
endmodule