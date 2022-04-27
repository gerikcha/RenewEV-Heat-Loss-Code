"""
Code to calculate the annual energy consumption and peak power for the DHW system.

Inputs:
    - Building characteristics, DHW sheet.

Outputs:
    - DHW_peak, peak heat load from DHW.
    - DHW_cons, annual energy consumption from DHW.
"""
import numpy as np

def DHW():
    