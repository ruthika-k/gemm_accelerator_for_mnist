module SRAM_A(dataIn,dataOut,Addr,CS,WE,RD, Clk );

// parameters for the width 
parameter ADR   = 10;

parameter DAT   = 8;

parameter DPTH  = 1024;

//ports
input   [DAT-1:0]  dataIn;

output reg [DAT-1:0]  dataOut;

input   [ADR-1:0]  Addr;

input CS,WE,RD,Clk;

reg [DAT-1:0] SRAMs [0:DPTH-1];

always @ (posedge Clk)

begin

 if (CS == 1'b1) begin

  if (WE == 1'b1 && RD == 1'b0) begin

   SRAMs [Addr] = dataIn;

  end

  else if (RD == 1'b1 && WE == 1'b0) begin

   dataOut = SRAMs [Addr]; 

  end

  else;

 end

 else;

end
endmodule