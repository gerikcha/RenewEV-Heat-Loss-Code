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

# calculate domestic hot water requirements.
inp = "Building Inputs.xlsx"
dhw_peak, dhw_cons = TCM_funcs.DHW(inp)

# calculate current building characteristics heat consumption and peak power
bc_ex_c = "Current Building Characteristics.xlsx"
qHVAC_c, dt_c, bcp_c = main.HL(bc_ex_c)
Q_cons_heat_c, Q_cons_cool_c = TCM_funcs.heat_cons(qHVAC_c, dhw_cons, dt_c)

# calculate current building characteristics heat consumption and peak power
bc_ex_u = "Upgraded Building Characteristics.xlsx"
qHVAC_u, dt_u, bcp_u = main.HL(bc_ex_u)
Q_cons_heat_u, Q_cons_cool_u = TCM_funcs.heat_cons(qHVAC_u, dhw_cons, dt_u)

# analysis of upgrading building and heating system on CO2 emissions and costs.

