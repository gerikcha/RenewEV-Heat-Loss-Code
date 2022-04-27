"""
Started on 28 October 2021.
Authors: L.Beber, E.Regev, C.Gerike-Roberts
Code which models the dynamic thermal transfer in a building.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TCM_funcs
import dm4bem
import copy
import Element_Types

# global constants
σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics
bc = TCM_funcs.building_characteristics()

# Define Inputs
ip = TCM_funcs.inputs()
Kpf = ip.loc['Kpf']['Value']
Kpc = ip.loc['Kpc']['Value']
Kph = ip.loc['Kph']['Value']

# Add thermo-physical properties
bcp = TCM_funcs.thphprop(bc)

# Create bcp of all elements apart from windows, doors and skylights
bcp_nodorwinsky = bcp[bcp['Element_Type'] != 'Window']  # remove window rows from bcp
bcp_nodorwinsky = bcp_nodorwinsky[bcp_nodorwinsky['Element_Type'] != 'Door']  # remove door rows from bcp
bcp_nodorwinsky = bcp_nodorwinsky[bcp_nodorwinsky['Element_Type'] != 'Skylight']  # remove skylight rows from bcp
bcp_nodorwinsky = bcp_nodorwinsky.reset_index() # reset row index of dataframe

#Create a bcp each for all window, door and skylight elements
bcp_win = bcp.loc[bcp['Element_Type'] == 'Window'] # create bcp for window elements
bcp_win = bcp_win.reset_index() # reset row index of dataframe
bcp_dor = bcp.loc[bcp['Element_Type'] == 'Door'] # create bcp for door elements
bcp_dor = bcp_dor.reset_index() # reset row index of dataframe
bcp_sky = bcp.loc[bcp['Element_Type'] == 'Skylight'] # create bcp for skylight elements
bcp_sky = bcp_sky.reset_index() # reset row index of dataframe

# Determine solar radiation for each element
rad_surf_tot, t = TCM_funcs.rad(bcp, ip)
rad_surf_tot_nodorwinsky, t_nodorwinsky = TCM_funcs.rad(bcp_nodorwinsky, ip)
rad_surf_tot_win, t_win = TCM_funcs.rad(bcp_win, ip)
rad_surf_tot_dor, t_dor = TCM_funcs.rad(bcp_dor, ip)
rad_surf_tot_sky, t_sky = TCM_funcs.rad(bcp_sky, ip)

# Thermal Circuits
TCd = {}
TCd.update({str(0): Element_Types.indoor_air(bcp_nodorwinsky, ip, rad_surf_tot_nodorwinsky)}) # inside air
TCd.update({str(1): Element_Types.ventilation(ip, Kpf, rad_surf_tot_nodorwinsky)})  # ventilation and heating
uc = 2                                                                          # variable to track how many heat flows have been used
IG = np.zeros([rad_surf_tot.shape[0], 1])                                    # set the radiation entering through windows to zero
tcd_n = 1

for i in range(0, len(bcp_win)):
    TCd_i, IGR = Element_Types.window(bcp_win.loc[i, :], rad_surf_tot_win, i)
    TCd.update({str(tcd_n + i + 1): TCd_i})
    IG = IG + IGR
    tcd_n = tcd_n + 1

for i in range(0, len(bcp_dor)):
    TCd_i = Element_Types.door(bcp_dor.loc[i, :], rad_surf_tot_dor, i)
    TCd.update({str(tcd_n + i + 1): TCd_i})
    tcd_n = tcd_n + 1

for i in range(0, len(bcp_sky)):
    TCd_i, IGR = Element_Types.skylight(bcp_sky.loc[i, :], rad_surf_tot_sky, i)
    TCd.update({str(tcd_n + i + 1): TCd_i})
    IG = IG + IGR
    tcd_n = tcd_n + 1

tcd_dorwinsky = tcd_n
tcd_n = tcd_n + 1 # recalibrate tcd number tracker

for i in range(0, len(bcp_nodorwinsky)):
    if bcp_nodorwinsky.Element_Type[i] == 'Wallex - 1 Layer':
        TCd_i, uca = Element_Types.Ex_Wall_1(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Wallex - 2 Layers':
        TCd_i, uca = Element_Types.Ex_Wall_2(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Wallex - 3 Layers':
        TCd_i, uca = Element_Types.Ex_Wall_3(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Wallex - 4 Layers':
        TCd_i, uca = Element_Types.Ex_Wall_4(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Wallex - 5 Layers':
        TCd_i, uca = Element_Types.Ex_Wall_5(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Roof - 1 Layer':
        TCd_i, uca = Element_Types.Roof_1(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Roof - 2 Layers':
        TCd_i, uca = Element_Types.Roof_2(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Roof - 3 Layers':
        TCd_i, uca = Element_Types.Roof_3(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Roof - 4 Layers':
        TCd_i, uca = Element_Types.Roof_4(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Roof - 5 Layers':
        TCd_i, uca = Element_Types.Roof_5(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky, uc)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Floor - 1 Layer':
        TCd_i = Element_Types.Floor_1(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Floor - 2 Layers':
        TCd_i = Element_Types.Floor_2(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Floor - 3 Layers':
        TCd_i = Element_Types.Floor_3(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Floor - 4 Layers':
        TCd_i = Element_Types.Floor_4(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    elif bcp_nodorwinsky.Element_Type[i] == 'Floor - 5 Layers':
        TCd_i = Element_Types.Floor_5(bcp_nodorwinsky.loc[i, :], ip, rad_surf_tot_nodorwinsky)
        TCd.update({str(tcd_n): TCd_i})
        tcd_n = tcd_n + 1
    uc = uca                                                                    # update heat flow tracker

IR_Surf = bcp_nodorwinsky.shape[0]
IG = IG / IR_Surf                                                        #divide total indoor radiation by number of indoor surfaces

TCd_f = copy.deepcopy(TCd)

for i in range(0, len(bcp_nodorwinsky)):
        TCd_i = TCM_funcs.indoor_rad(bcp_nodorwinsky.loc[i, :], TCd_f[str(tcd_dorwinsky + i + 1)], IG)
        TCd_f[str(tcd_dorwinsky + i + 1)] = TCd_i

TCd_h = copy.deepcopy(TCd_f)
TCd_c = copy.deepcopy(TCd)

for i in range(0, len(bcp_nodorwinsky)):
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(tcd_dorwinsky + i + 1)])
        TCd_c[str(tcd_dorwinsky + i + 1)] = TCd_i

TCd_c[str(1)] = Element_Types.ventilation(ip, Kpc, rad_surf_tot_nodorwinsky)
TCd_h[str(1)] = Element_Types.ventilation(ip, Kph, rad_surf_tot_nodorwinsky)

TCd_f = pd.DataFrame(TCd_f)
TCd_c = pd.DataFrame(TCd_c)
TCd_h = pd.DataFrame(TCd_h)

u, rad_surf_tot = TCM_funcs.u_assembly(TCd_f, rad_surf_tot)
u_c, rad_surf_tot = TCM_funcs.u_assembly_c(TCd_c, rad_surf_tot)
AssX = TCM_funcs.assembly(TCd_f,tcd_dorwinsky,tcd_n)

TCd_f = TCd_f.drop('Q')
TCd_f = TCd_f.drop('T')
TCd_c = TCd_c.drop('Q')
TCd_c = TCd_c.drop('T')
TCd_h = TCd_h.drop('Q')
TCd_h = TCd_h.drop('T')

TCd_f = pd.DataFrame.to_dict(TCd_f)
TCd_c = pd.DataFrame.to_dict(TCd_c)
TCd_h = pd.DataFrame.to_dict(TCd_h)


TCAf = dm4bem.TCAss(TCd_f, AssX)
TCAc = dm4bem.TCAss(TCd_c, AssX)
TCAh = dm4bem.TCAss(TCd_h, AssX)

qHVAC = TCM_funcs.solver(TCAf, TCAc, TCAh, ip, u, u_c, t, Kpc, Kph, rad_surf_tot)

dhw_peak, dhw_cons = TCM_funcs.DHW()
