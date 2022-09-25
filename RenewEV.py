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
import Analysis
import main
import DHW
import HeatConsumption
import PeakPower
import time
import snoop


# calculate domestic hot water requirements.
inp = "Building Inputs.xlsx"
dhw_peak, dhw_cons, sum_hw = DHW.DHW(inp)

# calculate current building characteristics heat consumption and peak power
bc_ex_c = "Current Building Characteristics.xlsx"
qHVAC_c, dt_c, bcp_c = main.HL(bc_ex_c)
Q_cons_heat_c, Q_cons_cool_c = HeatConsumption.heat_cons(qHVAC_c, dhw_cons, dt_c)
#
# # calculate current building characteristics heat consumption and peak power
# bc_ex_u = "Upgraded Building Characteristics.xlsx"
# qHVAC_u, dt_u, bcp_u = main.HL(bc_ex_u)
# Q_cons_heat_u, Q_cons_cool_u = HeatConsumption.heat_cons(qHVAC_u, dhw_cons, dt_u)


# peak power calculation for current building
bc_ex_c = "Current Building Characteristics.xlsx"
pp_c, bhlc_c, bcp_c = PeakPower.PP(inp, bc_ex_c)

# peak power calculation for upgraded building
bc_ex_u = "Upgraded Building Characteristics.xlsx"
pp_u, bhlc_u, bcp_u = PeakPower.PP(inp, bc_ex_u)

# analysis of upgrading building and heating system on CO2 emissions and costs.
Results = Analysis.Analysis(bhlc_c, bhlc_u, inp, dhw_cons)

pp_tot_c = pp_c + dhw_peak
pp_tot_u = pp_u + dhw_peak

#analyse upgraded benefits
current_loss = bcp_c['PP (W)']
upgraded_loss = bcp_u['PP (W)']
improv = current_loss - upgraded_loss
bcp_u.insert(2, 'Reduction (W)', improv)

# write to excel
print('Current Peak Power is:', pp_c, 'kW')
print('Upgraded Peak Power is:', pp_u, 'kW')


