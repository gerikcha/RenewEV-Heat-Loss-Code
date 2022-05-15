"""
Code to calculate the peak power loss using the BS:EN 12831:2003 simplified method.

Author: Charles Gerike-Roberts 15/05/2022

"""
import numpy as np
import pandas as pd

def PP(bcp, inp):
    ### calculate heat losses due to building fabric
    Gen = pd.read_excel(inp, sheet_name='General', na_values=["N"], keep_default_na=True,
                       header=0)
    ## calculate U-values for walls, floors and roofs
    Rsi_w = 0.13  # convection coefficient resistance value for u-value calculation for inside wall ISO 6946:2007
    Rso_w = 0.04  # convection coefficient resistance value for u-value calculation for outside wall ISO 6946:2007
    Rso_f = 0.04  # convection coefficient resistance value for u-value calculation for outside floor ISO 6946:2007
    Rsi_f = 0.17  # convection coefficient resistance value for u-value calculation for inside floor ISO 6946:2007
    Rso_r = 0.04  # convection coefficient resistance value for u-value calculation for outside roof ISO 6946:2007
    Rsi_r = 0.1  # convection coefficient resistance value for u-value calculation for inside roof ISO 6946:2007



    for i in range(0, len(bcp)):
        if bcp['U-Value'][i] != 'N':
            bcp['U-Value'][i] = bcp['U-Value'][i]
        elif 'Wallex' in bcp['Element Type'][i]:
            if np.isnan(bcp['density_5'][i]):
                if np.isnan(bcp['density_4'][i]):
                    if np.isnan(bcp['density_3'][i]):
                        if np.isnan(bcp['density_2'][i]):
                            R1 = bcp['Thickness'][i] / bcp['conductivity_1'][i]
                            R_tot = Rsi_w + R1 + Rso_w
                            bcp['U-Value'][i] = 1 / R_tot
                        else:
                            R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                            R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                            R_tot = Rsi_w + R1 + R2 + Rso_w
                            bcp['U-Value'][i] = 1 / R_tot
                    else:
                        R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                        R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                        R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                        R_tot = Rsi_w + R1 + R2 + R3 + Rso_w
                        bcp['U-Value'][i] = 1 / R_tot
                else:
                    R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                    R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                    R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                    R4 = bcp['Thickness_4'][i] / bcp['conductivity_4'][i]
                    R_tot = Rsi_w + R1 + R2 + R3 + R4 + Rso_w
                    bcp['U-Value'][i] = 1 / R_tot
            else:
                R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                R4 = bcp['Thickness_4'][i] / bcp['conductivity_4'][i]
                R5 = bcp['Thickness_5'][i] / bcp['conductivity_5'][i]
                R_tot = Rsi_w + R1 + R2 + R3 + R4 + R5 + Rso_w
                bcp['U-Value'][i] = 1 / R_tot
        elif 'Floor' in bcp['Element Type'][i]:
            if np.isnan(bcp['density_5'][i]):
                if np.isnan(bcp['density_4'][i]):
                    if np.isnan(bcp['density_3'][i]):
                        if np.isnan(bcp['density_2'][i]):
                            R1 = bcp['Thickness'][i] / bcp['conductivity_1'][i]
                            R_tot = Rsi_f + R1 + Rso_f
                            bcp['U-Value'][i] = 1 / R_tot
                        else:
                            R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                            R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                            R_tot = Rsi_f + R1 + R2 + Rso_f
                            bcp['U-Value'][i] = 1 / R_tot
                    else:
                        R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                        R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                        R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                        R_tot = Rsi_f + R1 + R2 + R3 + Rso_f
                        bcp['U-Value'][i] = 1 / R_tot
                else:
                    R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                    R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                    R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                    R4 = bcp['Thickness_4'][i] / bcp['conductivity_4'][i]
                    R_tot = Rsi_f + R1 + R2 + R3 + R4 + Rso_f
                    bcp['U-Value'][i] = 1 / R_tot
            else:
                R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                R4 = bcp['Thickness_4'][i] / bcp['conductivity_4'][i]
                R5 = bcp['Thickness_5'][i] / bcp['conductivity_5'][i]
                R_tot = Rsi_f + R1 + R2 + R3 + R4 + R5 + Rso_f
                bcp['U-Value'][i] = 1 / R_tot
        elif 'Roof' in bcp['Element Type'][i]:
            if np.isnan(bcp['density_5'][i]):
                if np.isnan(bcp['density_4'][i]):
                    if np.isnan(bcp['density_3'][i]):
                        if np.isnan(bcp['density_2'][i]):
                            R1 = bcp['Thickness'][i] / bcp['conductivity_1'][i]
                            R_tot = Rsi_r + R1 + Rso_r
                            bcp['U-Value'][i] = 1 / R_tot
                        else:
                            R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                            R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                            R_tot = Rsi_r + R1 + R2 + Rso_r
                            bcp['U-Value'][i] = 1 / R_tot
                    else:
                        R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                        R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                        R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                        R_tot = Rsi_r + R1 + R2 + R3 + Rso_r
                        bcp['U-Value'][i] = 1 / R_tot
                else:
                    R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                    R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                    R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                    R4 = bcp['Thickness_4'][i] / bcp['conductivity_4'][i]
                    R_tot = Rsi_r + R1 + R2 + R3 + R4 + Rso_r
                    bcp['U-Value'][i] = 1 / R_tot
            else:
                R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
                R2 = bcp['Thickness_2'][i] / bcp['conductivity_2'][i]
                R3 = bcp['Thickness_3'][i] / bcp['conductivity_3'][i]
                R4 = bcp['Thickness_4'][i] / bcp['conductivity_4'][i]
                R5 = bcp['Thickness_5'][i] / bcp['conductivity_5'][i]
                R_tot = Rsi_r + R1 + R2 + R3 + R4 + R5 + Rso_r
                bcp['U-Value'][i] = 1 / R_tot
        else:
            bcp['U-Value'][i] = bcp['U-Value'][i]

    ## calculate fabric losses
    fabric_loss = np.zeros(bcp.shape[0])
    for i in range(0, len(bcp)):
        fabric_hlc = bcp['fk'][i] * bcp['U-Value'][i] * bcp['Surface'][i]
        fabric_loss[i] = fabric_hlc * (bcp['D Temp'] - Gen['Value']['Ex_Temp'])

    ## calculate ventilation losses
    vent_loss = np.zeros(bcp.shape[0])
    for i in range(0, len(bcp)):
        if np.isnan(bcp['ACH'][i]):
            vent_loss[i] = 0
        else:
            vent_hlc = 0.34 * bcp['ACH'][i] * bcp['Volume'][i]
            vent_loss[i] = vent_hlc * (bcp['D Temp'] - Gen['Value']['Ex_Temp'])

    ## calculate heating-up capacity
    heat_up = np.zeros(bcp.shape[0])
    for i in range(0, len(bcp)):
        if np.isnan(bcp['ACH'][i]):
            heat_up[i] = 0
        else:
            heat_up[i] = bcp['Surface'][i] * bcp['fRH'][i]

    ## calculate peak power required
    f_losses_tot = np.sum(fabric_loss)
    v_losses_tot = np.sum(vent_loss)
    h_losses_tot = np.sum(heat_up)

    pp = (f_losses_tot + v_losses_tot + h_losses_tot)

    return(pp)