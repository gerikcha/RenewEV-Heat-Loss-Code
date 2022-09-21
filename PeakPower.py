"""
Code to calculate the peak power loss using the BS:EN 12831:2003 simplified method.

Author: Charles Gerike-Roberts 15/05/2022

"""
import numpy as np
import pandas as pd
import TCM_funcs

def PP(inp, bc_ex):
    ### calculate heat losses due to building fabric
    ## define building characteristics
    bc = TCM_funcs.building_characteristics(bc_ex)

    ## add thermophysical properties
    bcp = TCM_funcs.thphprop(bc, bc_ex)

    ## Import general building information
    Gen = pd.read_excel(inp, sheet_name='General', na_values=["N"], keep_default_na=True,
                       index_col=0, header=0)
    ## calculate U-values for walls, floors and roofs
    Rsi_w = 0.13  # convection coefficient resistance value for u-value calculation for inside wall ISO 6946:2007
    Rso_w = 0.04  # convection coefficient resistance value for u-value calculation for outside wall ISO 6946:2007
    Rso_f = 0.04  # convection coefficient resistance value for u-value calculation for outside floor ISO 6946:2007
    Rsi_f = 0.17  # convection coefficient resistance value for u-value calculation for inside floor ISO 6946:2007
    Rso_r = 0.04  # convection coefficient resistance value for u-value calculation for outside roof ISO 6946:2007
    Rsi_r = 0.1  # convection coefficient resistance value for u-value calculation for inside roof ISO 6946:2007



    for i in range(0, len(bcp)):
        n = bcp['U-Value'][i]
        if np.isnan(n):
            e = bcp['Element_Type'][i]
            if 'Wallex' in bcp['Element_Type'][i]:
                if np.isnan(bcp['conductivity_5'][i]):
                    if np.isnan(bcp['conductivity_4'][i]):
                        if np.isnan(bcp['conductivity_3'][i]):
                            if np.isnan(bcp['conductivity_2'][i]):
                                R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
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
            elif 'Floor' == bcp['Element_Type'][i]:
                if np.isnan(bcp['conductivity_5'][i]):
                    if np.isnan(bcp['conductivity_4'][i]):
                        if np.isnan(bcp['conductivity_3'][i]):
                            if np.isnan(bcp['conductivity_2'][i]):
                                R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
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
            else:
                if np.isnan(bcp['conductivity_5'][i]):
                    if np.isnan(bcp['conductivity_4'][i]):
                        if np.isnan(bcp['conductivity_3'][i]):
                            if np.isnan(bcp['conductivity_2'][i]):
                                R1 = bcp['Thickness_1'][i] / bcp['conductivity_1'][i]
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

    ## import room key
    R_K = pd.read_excel(bc_ex, sheet_name='Room Key', na_values=["N"], keep_default_na=True, header=0)


    ## calculate fabric losses
    fabric_loss = np.zeros([bcp.shape[0], 1])
    fabric_hlc = np.zeros([bcp.shape[0], 1])
    room_fabric_loss = np.zeros([R_K.shape[0], 1])
    room_fabric_hlc = np.zeros([R_K.shape[0], 1])
    R_K.insert(1, 'HLC (W/K)', room_fabric_hlc)
    R_K.insert(1, 'PP (W)', room_fabric_loss)
    for i in range(0, len(bcp)):
        e = bcp['Element_Code'][i]
        r, c = np.where(R_K == e)
        r = r[0]
        fabric_hlc[i] = bcp['fk'][i] * bcp['U-Value'][i] * bcp['Surface'][i]
        D_Temp = R_K['Design Temperature'][r]
        if np.isnan(D_Temp):
            R_K['HLC (W/K)'][r] = R_K['HLC (W/K)'][r] + fabric_hlc[i]
        else:
            Ex_Temp = Gen['Value']['Ex_Temp']
            fabric_loss[i] = fabric_hlc[i] * (D_Temp - Ex_Temp)
            R_K['HLC (W/K)'][r] = R_K['HLC (W/K)'][r] + fabric_hlc[i]
            R_K['PP (W)'][r] = R_K['PP (W)'][r] + fabric_loss[i]

    ## insert losses for elements into dataframes
    bcp.insert(2, 'HLC (W/K)', fabric_hlc)
    bcp.insert(2, 'PP (W)', fabric_loss)

    # unheated space losses
    IW = pd.read_excel(bc_ex, sheet_name='IW', na_values=["N"], keep_default_na=True, header=0)
    for i in range(0, len(IW)):
        e = IW['Element Code'][i]
        u = IW['Unheated Room'][i]
        r, c = np.where(R_K == u)
        r = r[0]
        ur_hlc = R_K['HLC (W/K)'][r]
        t, c = np.where(bcp == e)
        t = t[0]
        e_hlc = bcp['HLC (W/K)'][t]
        bu = ur_hlc / (e_hlc + ur_hlc)
        x, c = np.where(R_K == e)
        x = x[0]
        r_hlc = R_K['HLC (W/K)'][x]
        R_K['HLC (W/K)'][x] = (r_hlc - e_hlc) + (e_hlc * bu)
        bcp['HLC (W/K)'][t] = e_hlc * bu

    for i in range(0, len(R_K)):
        if np.isnan(R_K['Design Temperature'][i]):
            R_K['PP (W)'][i] = 0
            R_K['HLC (W/K)'][i] = 0
        else:
            R_K['PP (W)'][i] = R_K['PP (W)'][i]
            R_K['HLC (W/K)'][i] = R_K['HLC (W/K)'][i]

    ## calculate ventilation losses
    vent_loss = np.zeros(R_K.shape[0])
    vent_hlc = np.zeros(R_K.shape[0])
    for i in range(0, len(R_K)):
        if np.isnan(R_K['Design Temperature'][i]):
            vent_hlc[i] = 0
            vent_loss[i] = 0
        else:
            vent_hlc[i] = 0.34 * R_K['ACH (1/h)'][i] * R_K['Volume (m3)'][i]
            vent_loss[i] = vent_hlc[i] * (R_K['Design Temperature'][i] - Gen['Value']['Ex_Temp'])

    ## chimney ventilation losses
    vent_chim_bhlc = Gen['Value']['Chim_Flow'] * 0.33
    vent_chim_loss = vent_chim_bhlc * (18 - Gen['Value']['Ex_Temp'])

    ## calculate heating-up capacity
    heat_up = np.zeros(R_K.shape[0])
    for i in range(0, len(R_K)):
        if np.isnan(R_K['Design Temperature'][i]):
            heat_up[i] = 0
        else:
            heat_up[i] = R_K['Floor Area (m2)'][i] * R_K['fRH'][i]
    ## calculate peak power required
    f_losses_tot = np.sum(R_K['PP (W)'])
    v_losses_tot = np.sum(vent_loss) + vent_chim_loss
    h_losses_tot = np.sum(heat_up)
    pp = (f_losses_tot + v_losses_tot + h_losses_tot) / 1000

    ## calculate bhlc
    f_bhlc_tot = np.sum(R_K['HLC (W/K)'])
    v_bhlc_tot = np.sum(vent_hlc) + vent_chim_bhlc
    bhlc = f_bhlc_tot + v_bhlc_tot

    return(pp, bhlc, bcp)