# Vivado batch build (non-project flow) of the phase table demo.
#
#   cd fpga
#   vivado -mode batch -source scripts/build.tcl -tclargs 100     ;# Arty A7-100T
#   vivado -mode batch -source scripts/build.tcl -tclargs 35      ;# Arty A7-35T
#
# Second optional argument: the table to load (default data/phase_table.mem).
# Output: build/arty_phase_demo_<board>.bit plus timing and utilisation reports.
# NOT RUN YET: written without access to Vivado; see fpga/README.md.

set board [expr {$argc > 0 ? [lindex $argv 0] : "100"}]
array set parts {100 xc7a100tcsg324-1  35 xc7a35ticsg324-1L}
if {![info exists parts($board)]} { error "board must be 35 or 100, got '$board'" }
set part $parts($board)

set root [file normalize [file join [file dirname [info script]] ..]]
set mem  [file normalize [expr {$argc > 1 ? [lindex $argv 1] : [file join $root data phase_table.mem]}]]
set out  [file join $root build]
file mkdir $out
# $readmemh("phase_table.mem") is resolved from the directory Vivado runs in
file copy -force $mem [file join $out phase_table.mem]
cd $out

read_verilog [glob [file join $root rtl *.v]]
read_xdc     [file join $root constr arty_a7.xdc]

synth_design -top arty_phase_demo -part $part
opt_design
place_design
route_design

report_timing_summary -file timing_$board.rpt
report_utilization    -file utilization_$board.rpt
report_io             -file io_$board.rpt
check_timing          -file check_timing_$board.rpt

write_bitstream -force arty_phase_demo_$board.bit
puts "bitstream: [file join $out arty_phase_demo_$board.bit]  (part $part)"
