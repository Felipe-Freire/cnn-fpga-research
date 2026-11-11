# Out-of-context Vivado synthesis (and optional place & route) of an HLS-generated top.
#
# usage:
#   vivado -mode batch -nojournal -source vivado_synth_ooc.tcl \
#          -tclargs <hdl_dir> <top> <part> <clock_period_ns> <out_dir> [impl=0|1]
#
# Out-of-context mode avoids I/O buffer insertion (the io_parallel top has hundreds of pins that
# would never be device pins in the real trigger firmware) and gives the logic cost of the core.

lassign $argv hdl_dir top part clk_ns out_dir do_impl
if {$do_impl eq ""} { set do_impl 0 }
file mkdir $out_dir
set out_dir [file normalize $out_dir]
set hdl_dir [file normalize $hdl_dir]

# $readmemh paths in HLS ROMs are relative to the HDL directory
cd $hdl_dir
read_verilog [glob $hdl_dir/*.v]
set dat_files [glob -nocomplain $hdl_dir/*.dat]
if {[llength $dat_files]} { add_files $dat_files }

set xdc [file join $out_dir clock.xdc]
set fh [open $xdc w]
puts $fh "create_clock -period $clk_ns -name ap_clk \[get_ports ap_clk\]"
close $fh
read_xdc -mode out_of_context $xdc

synth_design -top $top -part $part -mode out_of_context
report_utilization -file [file join $out_dir post_synth_util.rpt]
report_utilization -hierarchical -hierarchical_depth 4 -file [file join $out_dir post_synth_util_hier.rpt]
report_timing_summary -file [file join $out_dir post_synth_timing.rpt]

if {$do_impl} {
    opt_design
    place_design
    route_design
    report_utilization -file [file join $out_dir post_route_util.rpt]
    report_timing_summary -file [file join $out_dir post_route_timing.rpt]
}
puts "OOC_FLOW_DONE"
