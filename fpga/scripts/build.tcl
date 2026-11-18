# Vivado batch build (non-project flow).
#
#   cd fpga
#   vivado -mode batch -source scripts/build.tcl -tclargs 100 phase   ;# phase table demo, Arty A7-100T
#   vivado -mode batch -source scripts/build.tcl -tclargs 100 lock    ;# lock and readout demo
#   vivado -mode batch -source scripts/build.tcl -tclargs 35  phase   ;# Arty A7-35T
#
# Defaults: board 100, design phase. Every data/*.mem is copied next to the run.
# Output: build/<design>/arty_<design>_demo_<board>.bit plus timing, utilisation
# and I/O reports.
# NOT RUN YET: written without access to Vivado; see fpga/README.md.

set board  [expr {$argc > 0 ? [lindex $argv 0] : "100"}]
set design [expr {$argc > 1 ? [lindex $argv 1] : "phase"}]
array set parts {100 xc7a100tcsg324-1  35 xc7a35ticsg324-1L}
array set xdcs  {phase arty_a7.xdc  lock arty_a7_lock.xdc}
if {![info exists parts($board)]} { error "board must be 35 or 100, got '$board'" }
if {![info exists xdcs($design)]} { error "design must be phase or lock, got '$design'" }
set part $parts($board)
set top  arty_${design}_demo

set root [file normalize [file join [file dirname [info script]] ..]]
set out  [file join $root build $design]
file mkdir $out
# $readmemh("<name>.mem") is resolved from the directory Vivado runs in
foreach mem [glob [file join $root data *.mem]] { file copy -force $mem $out }
cd $out

read_verilog [glob [file join $root rtl *.v]]
read_xdc     [file join $root constr $xdcs($design)]

synth_design -top $top -part $part
opt_design
place_design
route_design

report_timing_summary -file timing_$board.rpt
report_utilization    -file utilization_$board.rpt
report_io             -file io_$board.rpt
check_timing          -file check_timing_$board.rpt

write_bitstream -force ${top}_$board.bit
puts "bitstream: [file join $out ${top}_$board.bit]  (part $part)"
