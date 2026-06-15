module mac (
    input  signed [7:0]  a,
    input  signed [7:0]  b,
    input  signed [31:0] accum_in,

    output signed [31:0] accum_out
);

assign accum_out = accum_in + (a * b);

endmodule