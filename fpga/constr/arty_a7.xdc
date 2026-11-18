## Arty A7 (A7-35T and A7-100T share these pins) - phase table demo with a Pmod DA2.
## Pin numbers copied from Digilent's master files (digilent-xdc, Arty-A7-35-Master.xdc
## and Arty-A7-100-Master.xdc, rev. D/E); the two files list the same pins for every
## port used here. The FPGA part is chosen in scripts/build.tcl, not in this file.

## 100 MHz oscillator
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { clk100 }]; #IO_L12P_T1_MRCC_35 Sch=gclk[100]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports { clk100 }];

## Switches: layer to play (all up = whole table)
set_property -dict { PACKAGE_PIN A8    IOSTANDARD LVCMOS33 } [get_ports { sw[0] }]; #Sch=sw[0]
set_property -dict { PACKAGE_PIN C11   IOSTANDARD LVCMOS33 } [get_ports { sw[1] }]; #Sch=sw[1]
set_property -dict { PACKAGE_PIN C10   IOSTANDARD LVCMOS33 } [get_ports { sw[2] }]; #Sch=sw[2]
set_property -dict { PACKAGE_PIN A10   IOSTANDARD LVCMOS33 } [get_ports { sw[3] }]; #Sch=sw[3]

## Button 0: reset
set_property -dict { PACKAGE_PIN D9    IOSTANDARD LVCMOS33 } [get_ports { btn0 }]; #Sch=btn[0]

## Green LEDs LD4..LD7: layer being played
set_property -dict { PACKAGE_PIN H5    IOSTANDARD LVCMOS33 } [get_ports { led[0] }]; #Sch=led[4]
set_property -dict { PACKAGE_PIN J5    IOSTANDARD LVCMOS33 } [get_ports { led[1] }]; #Sch=led[5]
set_property -dict { PACKAGE_PIN T9    IOSTANDARD LVCMOS33 } [get_ports { led[2] }]; #Sch=led[6]
set_property -dict { PACKAGE_PIN T10   IOSTANDARD LVCMOS33 } [get_ports { led[3] }]; #Sch=led[7]

## Pmod header JA, top row: Pmod DA2 (pin 1 SYNC, 2 DINA, 3 DINB, 4 SCLK, 5 GND, 6 VCC)
set_property -dict { PACKAGE_PIN G13   IOSTANDARD LVCMOS33 } [get_ports { da2_sync_n }]; #Sch=ja[1]
set_property -dict { PACKAGE_PIN B11   IOSTANDARD LVCMOS33 } [get_ports { da2_dina }];   #Sch=ja[2]
set_property -dict { PACKAGE_PIN A11   IOSTANDARD LVCMOS33 } [get_ports { da2_dinb }];   #Sch=ja[3]
set_property -dict { PACKAGE_PIN D12   IOSTANDARD LVCMOS33 } [get_ports { da2_sclk }];   #Sch=ja[4]

## Pmod header JD, top row: gate pulse (pin 1) and scope trigger (pin 2)
set_property -dict { PACKAGE_PIN D4    IOSTANDARD LVCMOS33 } [get_ports { gate }]; #Sch=jd[1]
set_property -dict { PACKAGE_PIN D3    IOSTANDARD LVCMOS33 } [get_ports { trig }]; #Sch=jd[2]

## The DAC lines and the two markers come straight from flip-flops: keep those
## flip-flops in the I/O cells so the four DAC lines leave with the same delay.
set_property IOB TRUE [get_ports { da2_sync_n da2_dina da2_dinb da2_sclk gate trig }]

## Button and switches are asynchronous; each goes through a two-flop
## synchronizer in arty_phase_demo.v (rst_ff, sw_ff0/sw_ff1).
set_false_path -from [get_ports { btn0 sw[*] }]

## Outputs: no external device is clocked by clk100. The DAC is clocked by
## da2_sclk, and data, SYNC and SCLK are generated with 40 ns of setup and hold
## between them, far above any skew between pads. The LEDs are for the eye.
set_false_path -to [get_ports { da2_sync_n da2_dina da2_dinb da2_sclk gate trig led[*] }]

## Configuration bank voltage (bank 0 is at 3.3 V on the Arty)
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
