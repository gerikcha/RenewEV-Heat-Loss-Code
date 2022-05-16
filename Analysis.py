"""
Analysis of the change in annual heat consumption between current and upgraded heating systems
to determine the annual cost, CO2e savings and payback period.

Author: Charles Gerike-Roberts 16/05/22

Inputs:
    - Building inputs excel file name, inp (string)
    - Annual heating consumption for current set-up, Q_cons_heat_c (kWh)
    - Annual heating consumption for upgraded set-up, Q_cons_heat_u (kWh)

Outputs:
    - Table containing:
        - Current annual fuel cost. C_Fuel (£)
        - Current annual heat pump cost. C_HP (£)
        - Upgraded annual fuel cost. U_Fuel (£)
        - Upgraded annual heat pump cost. U_HP (£)
        - Current annual co2e emissions w/ fuel. (t)
        - Current annual co2e emissions w/ HP. (t)
        - Upgraded annual co2e emissions w/fuel. (t)
        - Upgraded annual co2e emissions w/HP. (t)
"""
import pandas as pd
import numpy as np

def Analysis(Q_cons_heat_c, Q_cons_heat_u, inp):
    HS = pd.read_excel(inp, sheet_name='Heating System', na_values=["N"], keep_default_na=True,
                       header=0, usecols="A:B")

    # calculate annual fuel consumption for current and upgraded building
    if HS['Boiler_Eff']['Value'] == 'nan':
        Eff_Tot_F = (1 - HS['Dis_losses']['Value']) * (1 - HS['Plant_losses']['Value']) * (1 - HS['Misc_losses']['Value'])
    else:
        Eff_Tot_F = HS['Boiler_Eff']['Value'] * (1 - HS['Dis_losses']['Value']) * (1 - HS['Plant_losses']['Value']) * (1 - HS['Misc_losses']['Value'])

    Q_heat_c_f = sum(Q_cons_heat_c / Eff_Tot_F)  # increase heating consumption values to account for losses in current fuel
    Q_heat_u_f = sum(Q_cons_heat_u / Eff_Tot_F)

    # calculate annual hp consumption for current and upgraded building
    Eff_Tot_HP = (1 - HS['Dis_losses']['Value']) * (1 - HS['Plant_losses']['Value']) * (1 - HS['Misc_losses']['Value'])

    Q_heat_c_HP = sum(Q_cons_heat_c / (Eff_Tot_HP * HS['HP_COP']['Value']))
    Q_heat_u_HP = sum(Q_cons_heat_u / (Eff_Tot_HP * HS['HP_COP']['Value']))

    # import fuel data
    FD = pd.read_excel(inp, sheet_name='Fuel', na_values=["N"], keep_default_na=True,
                       header=0, usecols="A:C")

    # calculate cost of existing fuel for current and upgraded building fabric
    CF = HS['MS_Current']['Value']
    if HS['L_Fuel_Cost_Unit']['Value'] == 0:
        C_c_fuel_low = (Q_heat_c_f * HS['L_Fuel_Cost']['Value']) + HS['L_Fuel_Stand']['Value']
        C_u_fuel_low = (Q_heat_u_f * HS['L_Fuel_Cost']['Value']) + HS['L_Fuel_Stand']['Value']
        C_c_fuel_high = (Q_heat_c_f * HS['H_Fuel_Cost']['Value']) + HS['H_Fuel_Stand']['Value']
        C_u_fuel_high = (Q_heat_u_f * HS['H_Fuel_Cost']['Value']) + HS['H_Fuel_Stand']['Value']
    elif HS['C_Fuel_Cost_Unit']['Value'] == 1:
        C_c_fuel_low = ((Q_heat_c_f / FD[CF]['CHVL']) * HS['L_Fuel_Cost']['Value']) + HS['L_Fuel_Stand']['Value']
        C_u_fuel_low = ((Q_heat_u_f / FD[CF]['CHVL']) * HS['L_Fuel_Cost']['Value']) + HS['L_Fuel_Stand']['Value']
        C_c_fuel_high = ((Q_heat_c_f / FD[CF]['CHV/L']) * HS['H_Fuel_Cost']['Value']) + HS['H_Fuel_Stand']['Value']
        C_u_fuel_high = ((Q_heat_u_f / FD[CF]['CHV/L']) * HS['H_Fuel_Cost']['Value']) + HS['H_Fuel_Stand']['Value']
    else:
        C_c_fuel_low = ((Q_heat_c_f / FD[CF]['CHVS']) * HS['L_Fuel_Cost']['Value']) + HS['L_Fuel_Stand']['Value']
        C_u_fuel_low = ((Q_heat_u_f / FD[CF]['CHVS']) * HS['L_Fuel_Cost']['Value']) + HS['L_Fuel_Stand']['Value']
        C_c_fuel_high = ((Q_heat_c_f / FD[CF]['CHVS']) * HS['H_Fuel_Cost']['Value']) + HS['H_Fuel_Stand']['Value']
        C_u_fuel_high = ((Q_heat_u_f / FD[CF]['CHVS']) * HS['H_Fuel_Cost']['Value']) + HS['H_Fuel_Stand']['Value']

    # calculate heating costs for heat pump for current and upgraded building




