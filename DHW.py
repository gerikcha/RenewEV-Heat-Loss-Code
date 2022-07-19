"""
Code to calculate the annual energy consumption and peak power for the DHW system.

Inputs:
    - Building characteristics, DHW sheet.

Outputs:
    - DHW_peak, peak heat load from DHW.
    - DHW_cons, annual energy consumption from DHW.
"""
import numpy as np
import pandas as pd

def DHW(inp):
    DHW = pd.read_excel(inp, sheet_name='DHW', na_values=["N"], keep_default_na=True,
                       index_col=0, header=0)

    n_shower = DHW.loc['n_shower']['Value']
    n_bath = DHW.loc['n_bath']['Value']
    n_wash = DHW.loc['n_wash']['Value']
    n_sink = DHW.loc['n_sink']['Value']
    hw_shower = DHW.loc['hw_shower']['Value']
    hw_bath = DHW.loc['hw_bath']['Value']
    hw_wash = DHW.loc['hw_wash']['Value']
    hw_sink = DHW.loc['hw_sink']['Value']
    t_reheat = DHW.loc['t_reheat']['Value']
    water_in = DHW.loc['water_in']['Value']
    water_out = DHW.loc['water_out']['Value']
    t_daily = DHW.loc['t_daily']['Value']
    t_active = DHW.loc['t_active']['Value']

    sum_hw = n_shower * hw_shower + n_bath * hw_bath + n_wash * hw_wash + n_sink * hw_sink

    v_reheat = sum_hw / (t_reheat * 3600)

    water_temp_diff = water_out - water_in

    dhw_peak = (v_reheat * 4200 * water_temp_diff) / 1000

    dhw_cons = dhw_peak * t_daily * t_active

    return dhw_peak, dhw_cons

    