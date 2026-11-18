## Arty A7 (A7-35T and A7-100T share these pins) - lock and readout demo:
## Pmod DA2 on JA, Pmod AD1 on JD, markers on JB.
## Pin numbers copied from Digilent's master files (digilent-xdc, rev. D/E).

## 100 MHz oscillator
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { clk100 }]; #IO_L12P_T1_MRCC_35 Sch=gclk[100]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports { clk100 }];

## Switches: 0 loop closed, 1 drift, 2 DAC B shows the shot value, 3 set-point steps
set_property -dict { PACKAGE_PIN A8    IOSTANDARD LVCMOS33 } [get_ports { sw[0] }]; #Sch=sw[0]
set_property -dict { PACKAGE_PIN C11   IOSTANDARD LVCMOS33 } [get_ports { sw[1] }]; #Sch=sw[1]
set_property -dict { PACKAGE_PIN C10   IOSTANDARD LVCMOS33 } [get_ports { sw[2] }]; #Sch=sw[2]
set_property -dict { PACKAGE_PIN A10   IOSTANDARD LVCMOS33 } [get_ports { sw[3] }]; #Sch=sw[3]

## Button 0: reset
set_property -dict { PACKAGE_PIN D9    IOSTANDARD LVCMOS33 } [get_ports { btn0 }]; #Sch=btn[0]

## Green LEDs LD4..LD7: loop closed, drift on, locked, controller at a rail
set_property -dict { PACKAGE_PIN H5    IOSTANDARD LVCMOS33 } [get_ports { led[0] }]; #Sch=led[4]
set_property -dict { PACKAGE_PIN J5    IOSTANDARD LVCMOS33 } [get_ports { led[1] }]; #Sch=led[5]
set_property -dict { PACKAGE_PIN T9    IOSTANDARD LVCMOS33 } [get_ports { led[2] }]; #Sch=led[6]
set_property -dict { PACKAGE_PIN T10   IOSTANDARD LVCMOS33 } [get_ports { led[3] }]; #Sch=led[7]

## Pmod header JA, top row: Pmod DA2 (pin 1 SYNC, 2 DINA, 3 DINB, 4 SCLK)
set_property -dict { PACKAGE_PIN G13   IOSTANDARD LVCMOS33 } [get_ports { da2_sync_n }]; #Sch=ja[1]
set_property -dict { PACKAGE_PIN B11   IOSTANDARD LVCMOS33 } [get_ports { da2_dina }];   #Sch=ja[2]
set_property -dict { PACKAGE_PIN A11   IOSTANDARD LVCMOS33 } [get_ports { da2_dinb }];   #Sch=ja[3]
set_property -dict { PACKAGE_PIN D12   IOSTANDARD LVCMOS33 } [get_ports { da2_sclk }];   #Sch=ja[4]

## Pmod header JD, top row: Pmod AD1 (pin 1 CS, 2 D0, 3 D1, 4 CLK)
set_property -dict { PACKAGE_PIN D4    IOSTANDARD LVCMOS33 } [get_ports { ad1_cs_n }]; #Sch=jd[1]
set_property -dict { PACKAGE_PIN D3    IOSTANDARD LVCMOS33 } [get_ports { ad1_d0 }];   #Sch=jd[2]
set_property -dict { PACKAGE_PIN F4    IOSTANDARD LVCMOS33 } [get_ports { ad1_d1 }];   #Sch=jd[3]
set_property -dict { PACKAGE_PIN F3    IOSTANDARD LVCMOS33 } [get_ports { ad1_sclk }]; #Sch=jd[4]

## Pmod header JB, top row: gate (pin 1) and scope trigger (pin 2). JB has no
## series resistors: connect only scope probes here, never a driven signal.
set_property -dict { PACKAGE_PIN E15   IOSTANDARD LVCMOS33 } [get_ports { gate }]; #Sch=jb_p[1]
set_property -dict { PACKAGE_PIN E16   IOSTANDARD LVCMOS33 } [get_ports { trig }]; #Sch=jb_n[1]

## Serial lines and markers come straight from flip-flops, and the two ADC data
## lines go straight into one (d0_q, d1_q in ad1_spi.v): keep them in the I/O cells.
set_property IOB TRUE [get_ports { da2_sync_n da2_dina da2_dinb da2_sclk ad1_cs_n ad1_sclk ad1_d0 ad1_d1 gate trig }]

## Button and switches are asynchronous; two-flop synchronizers in arty_lock_demo.v.
set_false_path -from [get_ports { btn0 sw[*] }]

## ADC data: launched by the ADC from ad1_sclk, which this design generates at
## 12.5 MHz. Each bit is read 70 ns after the edge that clocks it out and 10 ns
## before the edge that replaces it, so the path is not timed against clk100.
set_false_path -from [get_ports { ad1_d0 ad1_d1 }]

## Outputs: no external device is clocked by clk100 (see arty_a7.xdc).
set_false_path -to [get_ports { da2_sync_n da2_dina da2_dinb da2_sclk ad1_cs_n ad1_sclk gate trig led[*] }]

## Configuration bank voltage (bank 0 is at 3.3 V on the Arty)
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
