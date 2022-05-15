"""
Code to calculate the annual heat consumption, peak power and CO2 savings from upgrades.

Author: Charles Gerike-Roberts 15th May 2022

Inputs:
    - Current Building Characteristics Excel File
    - Upgraded Building Characteristics Excel File
    - General Input Excel File for Building which contains:
        - DHW inputs
        - CO2 Factors
        - Fuel costs
        - Actual fuel bills
        - Heating efficiencies
        - Current heating system info
        - Capex costs for heating.
        - Opex costs for heating.

Outputs:
    - CO2e saved per year.
    - Annual fuel costs.
    - Payback analysis.
    -
"""

import main
import TCM_funcs

# calculate current building characteristics heat consumption and peak power
bc_ex = "Current Building Characteristics.xlsx"
qHVAC_c, dt_c, bcp_c = main.HL(bc_ex)

inp = "Building Inputs.xlsx"

dhw_peak, dhw_cons = TCM_funcs.DHW(inp)

ann_cons, peak_power_space, peak_power_tot = TCM_funcs.heat_cons(qHVAC_c, dhw_peak, dhw_cons, dt_c)
