# ============================================================================
# TQ Engine RTL Build & Simulation Makefile
# ============================================================================

TOPLEVEL     = tq_engine
TESTBENCH    = tq_engine_tb
RTL_DIR      = rtl
TB_DIR       = tb
SIM_DIR      = sim

RTL_SRCS = $(RTL_DIR)/token_preprocessor.sv \
           $(RTL_DIR)/rht_butterfly.sv \
           $(RTL_DIR)/lloyd_max_quant.sv \
           $(RTL_DIR)/qjl_engine.sv \
           $(RTL_DIR)/tq_engine.sv

TB_SRCS  = $(RTL_DIR)/tq_engine_tb.sv

# ============================================================================
# Simulation: Verilator (open-source, fast)
# ============================================================================
sim-verilator: gen-vectors
	mkdir -p $(SIM_DIR)
	verilator --sv --lint-only $(RTL_SRCS) 2>&1 | head -50
	@echo "Lint complete. For full simulation, use VCS or Icarus below."

# Icarus Verilog (open-source, SystemVerilog support limited)
sim-iverilog: gen-vectors
	mkdir -p $(SIM_DIR)
	iverilog -g2012 -o $(SIM_DIR)/$(TESTBENCH) $(RTL_SRCS) $(TB_SRCS)
	cd $(SIM_DIR) && ./$(TESTBENCH)

# VCS (commercial, full SV support)
sim-vcs: gen-vectors
	mkdir -p $(SIM_DIR)
	vcs -full64 -sverilog -timescale=1ns/1ps \
	    +vcs+lic+wait \
	    -o $(SIM_DIR)/$(TESTBENCH)_vcs \
	    $(RTL_SRCS) $(TB_SRCS)
	$(SIM_DIR)/$(TESTBENCH)_vcs

# ============================================================================
# Lint
# ============================================================================
lint:
	verilator --sv --lint-only --Wall $(RTL_SRCS) 2>&1

# ============================================================================
# Test vector generation
# ============================================================================
gen-vectors:
	mkdir -p $(TB_DIR)
	cd .. && python hardware/rtl/gen_test_vectors.py

# ============================================================================
# DC Synthesis (requires Synopsys DC and cell library)
# ============================================================================
SYNTH_DIR = syn
DC_SCRIPT = $(SYNTH_DIR)/run_dc.tcl
TARGET_FREQ = 400  # MHz

syn:
	mkdir -p $(SYNTH_DIR)
	@echo "Generating DC script..."
	$(MAKE) gen-dc-script
	dc_shell -f $(DC_SCRIPT) | tee $(SYNTH_DIR)/dc.log
	@echo "Synthesis complete. Check $(SYNTH_DIR)/dc.log"

gen-dc-script:
	@cat > $(DC_SCRIPT) << 'TCEOF'
set TARGET_LIBRARY "/path/to/28nm_stdcell.db"  ;# replace with actual path
set DESIGN $(TOPLEVEL)
set CLK_PERIOD [expr 1000.0 / $(TARGET_FREQ)]  ;# ns

set_app_var target_library $$TARGET_LIBRARY
set_app_var link_library "* $$TARGET_LIBRARY"

analyze -format sverilog [list \
    ../rtl/token_preprocessor.sv \
    ../rtl/rht_butterfly.sv \
    ../rtl/lloyd_max_quant.sv \
    ../rtl/qjl_engine.sv \
    ../rtl/tq_engine.sv \
]
elaborate $$DESIGN

create_clock -period $$CLK_PERIOD [get_ports clk]
set_input_delay  [expr $$CLK_PERIOD * 0.1] -clock clk [all_inputs]
set_output_delay [expr $$CLK_PERIOD * 0.1] -clock clk [all_outputs]
set_max_fanout 16 [all_inputs]
set_max_transition [expr $$CLK_PERIOD * 0.15] [current_design]

compile_ultra -no_autoungroup

report_area    > $(SYNTH_DIR)/area.rpt
report_power   > $(SYNTH_DIR)/power.rpt
report_timing  > $(SYNTH_DIR)/timing.rpt
report_qor     > $(SYNTH_DIR)/qor.rpt

write -f verilog -output $(SYNTH_DIR)/$(TOPLEVEL)_netlist.v
write_sdc $(SYNTH_DIR)/$(TOPLEVEL).sdc
TCEOF

# ============================================================================
# Clean
# ============================================================================
clean:
	rm -rf $(SIM_DIR) $(SYNTH_DIR) $(TB_DIR) *.vcd

.PHONY: sim-verilator sim-iverilog sim-vcs lint gen-vectors syn gen-dc-script clean
