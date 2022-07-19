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

def Analysis(bhlc_c, bhlc_u, inp, dhw_cons):
    HS = pd.read_excel(inp, sheet_name='Heating System', na_values=["N"], keep_default_na=True,
                       index_col=0, header=0, usecols="A:B")

    # determine annual space heating consumption from HDD data.
    Q_space_heat_c = (bhlc_c * (HS['Value']['HDD'] * 24)) / 1000
    Q_space_heat_u = (bhlc_u * (HS['Value']['HDD'] * 24)) / 1000

    Q_cons_heat_c = Q_space_heat_c + dhw_cons
    Q_cons_heat_u = Q_space_heat_u + dhw_cons

    # calculate annual fuel consumption for current and upgraded building
    if np.isnan(HS['Value']['Boiler_Eff']):
        Eff_Tot_F = (1 - HS['Value']['Dis_losses']) * (1 - HS['Value']['Plant_losses']) * (1 - HS['Value']['Misc_losses'])
    else:
        Eff_Tot_F = HS['Value']['Boiler_Eff'] * (1 - HS['Value']['Dis_losses']) * (1 - HS['Value']['Plant_losses']) * (1 - HS['Value']['Misc_losses'])

    Q_heat_c_f = Q_cons_heat_c / Eff_Tot_F  # increase heating consumption values to account for losses in current fuel
    Q_heat_u_f = Q_cons_heat_u / Eff_Tot_F

    # calculate annual hp consumption for current and upgraded building
    Eff_Tot_HP = (1 - HS['Value']['Dis_losses']) * (1 - HS['Value']['Plant_losses']) * (1 - HS['Value']['Misc_losses'])

    Q_heat_c_HP = Q_cons_heat_c / (Eff_Tot_HP * HS['Value']['HP_SCOP'])
    Q_heat_u_HP = Q_cons_heat_u / (Eff_Tot_HP * HS['Value']['HP_SCOP'])

    # import fuel data
    FD = pd.read_excel(inp, sheet_name='Fuel', na_values=["N"], keep_default_na=True,
                       index_col=0, header=0, usecols="A:D")

    # calculate cost of existing fuel for current and upgraded building fabric
    CF = HS['Value']['MS_Current']
    if HS['Value']['L_Fuel_Cost_Unit'] == 0:
        C_c_fuel_low = (Q_heat_c_f * HS['Value']['L_Fuel_Cost'])/100 + HS['Value']['L_Fuel_Stand']
        C_u_fuel_low = (Q_heat_u_f * HS['Value']['L_Fuel_Cost'])/100 + HS['Value']['L_Fuel_Stand']
        C_c_fuel_high = (Q_heat_c_f * HS['Value']['H_Fuel_Cost'])/100 + HS['Value']['H_Fuel_Stand']
        C_u_fuel_high = (Q_heat_u_f * HS['Value']['H_Fuel_Cost'])/100 + HS['Value']['H_Fuel_Stand']
    elif HS['Value']['C_Fuel_Cost_Unit'] == 1:
        C_c_fuel_low = ((Q_heat_c_f / FD['CHVL'][CF]) * HS['Value']['L_Fuel_Cost'])/100 + HS['Value']['L_Fuel_Stand']
        C_u_fuel_low = ((Q_heat_u_f / FD['CHVL'][CF]) * HS['Value']['L_Fuel_Cost'])/100 + HS['Value']['L_Fuel_Stand']
        C_c_fuel_high = ((Q_heat_c_f / FD['CHVL'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + HS['Value']['H_Fuel_Stand']
        C_u_fuel_high = ((Q_heat_u_f / FD['CHVL'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + HS['Value']['H_Fuel_Stand']
    else:
        C_c_fuel_low = ((Q_heat_c_f / FD['CHVS'][CF]) * HS['Value']['L_Fuel_Cost'])/100 + HS['Value']['L_Fuel_Stand']
        C_u_fuel_low = ((Q_heat_u_f / FD['CHVS'][CF]) * HS['Value']['L_Fuel_Cost'])/100 + HS['Value']['L_Fuel_Stand']
        C_c_fuel_high = ((Q_heat_c_f / FD['CHVS'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + HS['Value']['H_Fuel_Stand']
        C_u_fuel_high = ((Q_heat_u_f / FD['CHVS'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + HS['Value']['H_Fuel_Stand']

    # add calculated values to dataframe
    Analy = {'Result':['M Ann Cons (kWh)', 'M Ann F Cons (kWh)', 'M Ann HP Cons (kWh)', 'M Ann F HP Cons (kWh)',
                       'M L Cost (£)', 'M L F Cost (£)', 'M H Cost (£)', 'M H F Cost (£)'],
             'Value':[Q_heat_c_f, Q_heat_u_f, Q_heat_c_HP, Q_heat_u_HP,
                      C_c_fuel_low, C_u_fuel_low, C_c_fuel_high, C_u_fuel_high]}

    Results = pd.DataFrame(Analy)

    # calculate heating costs for heat pump for current and upgraded building
    C_c_hp_low = (Q_heat_c_HP * HS['Value']['L_Elec_Cost'])/100 + HS['Value']['L_Elec_Stand']
    C_u_hp_low = (Q_heat_u_HP * HS['Value']['L_Elec_Cost']) / 100 + HS['Value']['L_Elec_Stand']
    C_c_hp_high = (Q_heat_c_HP * HS['Value']['H_Elec_Cost']) / 100 + HS['Value']['H_Elec_Stand']
    C_u_hp_high = (Q_heat_u_HP * HS['Value']['H_Elec_Cost']) / 100 + HS['Value']['H_Elec_Stand']

    Analy = {'Result':['M L HP Cost (£)', 'M L HP + F Cost (£)', 'M H HP Cost (£)', 'M H HP + F Cost (£)'],
             'Value':[C_c_hp_low, C_u_hp_low, C_c_hp_high, C_u_hp_high]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate annual CO2e figures
    CO2_c_fuel = (Q_heat_c_f * FD['S_1_E'][CF]) / 1000
    CO2_u_fuel = (Q_heat_u_f * FD['S_1_E'][CF]) / 1000
    CO2_c_HP = (Q_heat_c_HP * FD['S_1_E']['Electricity']) / 1000
    CO2_u_HP = (Q_heat_u_HP * FD['S_1_E']['Electricity']) / 1000

    Analy = {'Result':['M CO2 (t)', 'M F CO2 (t)', 'M HP CO2 (t)', 'M F + HP CO2 (t)'],
             'Value':[CO2_c_fuel, CO2_u_fuel, CO2_c_HP, CO2_u_HP]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate modelled fabric upgrade savings with no change to heating system
    C_savings_f_low = C_c_fuel_low - C_u_fuel_low
    C_savings_f_high = C_c_fuel_high - C_u_fuel_high
    CO2_savings_f = CO2_c_fuel - CO2_u_fuel

    Analy = {'Result':['M F C Savings Low (£)', 'M F C Savings High (£)', 'M F CO2 Savings (t)'],
             'Value':[C_savings_f_low, C_savings_f_high, CO2_savings_f]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate modelled heat pump upgrade savings wit no fabric upgrades
    C_savings_hp_low = C_c_fuel_low - C_c_hp_low
    C_savings_hp_high = C_c_fuel_high - C_c_hp_high
    CO2_savings_hp = CO2_c_fuel - CO2_c_HP

    Analy = {'Result':['M HP C Savings Low (£)', 'M HP C Savings High (£)', 'M HP CO2 Savings (t)'],
             'Value':[C_savings_hp_low, C_savings_hp_high, CO2_savings_hp]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate modelled heat pump + fabric upgrade savings
    C_savings_hp_f_low = C_c_fuel_low - C_u_hp_low
    C_savings_hp_f_high = C_c_fuel_high - C_u_hp_high
    CO2_savings_hp_f = CO2_c_fuel - CO2_u_HP

    Analy = {'Result': ['M F+HP C Savings Low (£)', 'M F+HP C Savings High (£)', 'M F+HP CO2 Savings (t)'],
             'Value': [C_savings_hp_f_low, C_savings_hp_f_high, CO2_savings_hp_f]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    ## savings based on actual bill data
    A_Bill = HS['Value']['A_Cost']
    AS = HS['Value']['L_Fuel_Stand']
    AC = A_Bill - AS
    if HS['Value']['L_Fuel_Cost_Unit'] == 0:
        Q_a_heat_fuel = AC / (HS['Value']['L_Fuel_Cost']/100)
    elif HS['Value']['C_Fuel_Cost_Unit'] == 1:
        Q_a_heat_fuel = AC * ((100 * FD['CHVL'][CF])/HS['Value']['L_Fuel_Cost'])
    else:
        Q_a_heat_fuel = AC * ((100 * FD['CHVS'][CF]) / HS['Value']['L_Fuel_Cost'])

    Q_a_heat = Q_a_heat_fuel * Eff_Tot_F

    P_savings_f = (Q_heat_u_f - Q_heat_c_f)/Q_heat_c_f
    P_savings_hp = (Q_heat_c_HP - Q_heat_c_f)/Q_heat_c_f
    P_savings_hp_f = (Q_heat_u_HP - Q_heat_c_f)/Q_heat_c_f

    Q_a_heat_f = Q_a_heat_fuel * (P_savings_f + 1)
    Q_a_heat_hp = Q_a_heat * (P_savings_hp + 1)
    Q_a_heat_hp_f = Q_a_heat * (P_savings_hp_f + 1)

    Analy = {'Result': ['A Ann Cons (kWh)', 'A F Ann Cons (kWh)', 'A HP Ann Cons (kWh)', 'A F+HP Ann Cons (kWh)',
                        'F % Savings', 'HP % Savings', 'F+HP % Savings'],
             'Value': [Q_a_heat_fuel, Q_a_heat_f, Q_a_heat_hp, Q_a_heat_hp_f,
                       P_savings_f, P_savings_hp, P_savings_hp_f]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # high cost for annual heat consumption
    if HS['Value']['L_Fuel_Cost_Unit'] == 0:
        C_a_fuel_high = (Q_a_heat_fuel * HS['Value']['H_Fuel_Cost'])/100 + AS
    elif HS['Value']['C_Fuel_Cost_Unit'] == 1:
        C_a_fuel_high = ((Q_a_heat_fuel / FD['CHVL'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + AS
    else:
        C_a_fuel_high = ((Q_a_heat_fuel / FD['CHVS'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + AS

    # cost for fabric upgrades

    if HS['Value']['L_Fuel_Cost_Unit'] == 0:
        C_a_fuel_f_low = (Q_a_heat_f * HS['Value']['L_Fuel_Cost'])/100 + AS
        C_a_fuel_f_high = (Q_a_heat_f * HS['Value']['H_Fuel_Cost'])/100 + AS
    elif HS['Value']['C_Fuel_Cost_Unit'] == 1:
        C_a_fuel_f_low = ((Q_a_heat_f / FD['CHVL'][CF]) * HS['Value']['L_Fuel_Cost'])/100 + AS
        C_a_fuel_f_high = ((Q_a_heat_f / FD['CHVL'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + AS
    else:
        C_a_fuel_f_low = ((Q_a_heat_f / FD['CHVS'][CF]) * HS['Value']['L_Fuel_Cost'])/100 + AS
        C_a_fuel_f_high = ((Q_a_heat_f / FD['CHVS'][CF]) * HS['Value']['H_Fuel_Cost'])/100 + AS

    Analy = {'Result': ['A Fuel Cost High (£)', 'A F Cost Low (£)', 'A F Cost Low (£)'],
             'Value': [C_a_fuel_high, C_a_fuel_f_low, C_a_fuel_f_high]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    Q_a_heat_hp = Q_a_heat_hp / (Eff_Tot_HP * HS['Value']['HP_SCOP'])
    Q_a_heat_hp_f = Q_a_heat_hp_f / (Eff_Tot_HP * HS['Value']['HP_SCOP'])

    # cost for heat pump upgrades

    C_a_heat_hp_low = (Q_a_heat_hp * HS['Value']['L_Elec_Cost'])/100 + HS['Value']['L_Elec_Stand']
    C_a_heat_hp_high = (Q_a_heat_hp * HS['Value']['H_Elec_Cost']) / 100 + HS['Value']['H_Elec_Stand']

    # cost for heat pump and fabric upgrades

    C_a_heat_hp_f_low = (Q_a_heat_hp_f * HS['Value']['L_Elec_Cost']) / 100 + HS['Value']['L_Elec_Stand']
    C_a_heat_hp_f_high = (Q_a_heat_hp_f * HS['Value']['H_Elec_Cost']) / 100 + HS['Value']['H_Elec_Stand']

    Analy = {'Result': ['A HP Cost Low (£)', 'A HP Cost High (£)', 'A F+HP Cost Low (£)',
                        'A F+HP Cost High (£)'],
             'Value': [C_a_heat_hp_low, C_a_heat_hp_high, C_a_heat_hp_f_low, C_a_heat_hp_f_high]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate annual co2 emissions for different scenarios
    CO2_a_fuel = (Q_a_heat_fuel * FD['S_1_E'][CF]) / 1000
    CO2_a_f = (Q_a_heat_f * FD['S_1_E'][CF]) / 1000
    CO2_a_HP = (Q_a_heat_hp * FD['S_1_E']['Electricity']) / 1000
    CO2_a_HP_f = (Q_a_heat_hp_f * FD['S_1_E']['Electricity']) / 1000

    Analy = {'Result': ['A CO2 Current (t)', 'A F CO2 (t)', 'A HP CO2(t)', 'A F+HP CO2(t)'],
             'Value': [CO2_a_fuel, CO2_a_f, CO2_a_HP, CO2_a_HP_f]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate savings for fabric upgrades
    C_a_savings_f_low = AC - C_a_fuel_f_low
    C_a_savings_f_high = AC - C_a_fuel_f_high
    CO2_a_savings_f = CO2_a_fuel - CO2_a_f

    Analy = {'Result': ['A F C Savings Low (£)', 'A F C Savings High (£)', 'A F CO2 Savings (t)'],
             'Value': [C_a_savings_f_low, C_a_savings_f_high, CO2_a_savings_f]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate savings for heat upgrades
    C_a_savings_hp_low = AC - C_a_heat_hp_low
    C_a_savings_hp_high = AC - C_a_heat_hp_high
    CO2_a_savings_hp = CO2_a_fuel - CO2_a_HP

    Analy = {'Result': ['A HP C Savings Low (£)', 'A HP C Savings High (£)', 'A HP CO2 Savings (t)'],
             'Value': [C_a_savings_hp_low, C_a_savings_hp_high, CO2_a_savings_hp]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    # calculate savings for fabric + heat upgrades
    C_a_savings_hp_f_low = AC - C_a_heat_hp_f_low
    C_a_savings_hp_f_high = AC - C_a_heat_hp_f_high
    CO2_a_savings_hp_f = CO2_a_fuel - CO2_a_HP_f

    Analy = {'Result': ['A F+HP C Savings Low (£)', 'A F+HP C Savings High (£)', 'A F+HP CO2 Savings (t)'],
             'Value': [C_a_savings_hp_f_low, C_a_savings_hp_f_high, CO2_a_savings_hp_f]}

    Analy = pd.DataFrame(Analy)

    Results = Results.append(Analy, ignore_index=True)

    return(Results)
