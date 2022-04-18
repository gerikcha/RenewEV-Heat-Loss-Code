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

# global constants
σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics
bc = TCM_funcs.building_characteristics()

# Define Inputs
ip = TCM_funcs.inputs()

# Add thermo-physical properties
bcp = TCM_funcs.thphprop(bc)

# Determine solar radiation for each element
rad_surf_tot, t = TCM_funcs.rad(bcp, ip)

# Thermal Circuits
TCd = {}
TCd.update({str(0): TCM_funcs.indoor_air(bcp.Surface, ip, rad_surf_tot)}) # inside air
TCd.update({str(1): TCM_funcs.ventilation(V, Vdot, Kpf, T_set, rad_surf_tot)})  # ventilation and heating
uc = 2                                                                          # variable to track how many heat flows have been used
IG = np.zeros([rad_surf_tot.shape[0], 1])                                       # set the radiation entering through windows to zero
for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i, uca = TCM_funcs.solid_wall_w_ins(bcp.loc[i, :], ip, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    elif bcp.Element_Type[i] == 'DG':
        TCd_i, uca, IGR = TCM_funcs.window(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
        IG = IG + IGR                                                         # update total radiation coming through windows
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i, uca = TCM_funcs.susp_floor(bcp.loc[i, :], h, V, rad_surf_tot, uc, Tg)
        TCd.update({str(i+2): TCd_i})
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i, uca = TCM_funcs.flat_roof_w_in(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    uc = uca                                                                    # update heat flow tracker

IG = IG / IR_Surf                                                           #divide total indoor radiation by number of indoor surfaces

TCd_f = copy.deepcopy(TCd)

for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd_f[str(i+2)], IG)
        TCd_f[str(i+2)] = TCd_i
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd_f[str(i+2)], IG)
        TCd_f[str(i + 2)] = TCd_i
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd_f[str(i+2)], IG)
        TCd_f[str(i + 2)] = TCd_i

TCd_h = copy.deepcopy(TCd_f)
TCd_c = copy.deepcopy(TCd)

for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(i+2)])
        TCd_c[str(i+2)] = TCd_i
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(i+2)])
        TCd_c[str(i + 2)] = TCd_i
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(i+2)])
        TCd_c[str(i + 2)] = TCd_i

TCd_c[str(1)] = TCM_funcs.ventilation(V, Vdot, Kpc, T_set, rad_surf_tot)
TCd_h[str(1)] = TCM_funcs.ventilation(V, Vdot, Kph, T_set, rad_surf_tot)

TCd_f = pd.DataFrame(TCd_f)
TCd_c = pd.DataFrame(TCd_c)
TCd_h = pd.DataFrame(TCd_h)

u, rad_surf_tot = TCM_funcs.u_assembly(TCd_f, rad_surf_tot)
u_c, rad_surf_tot = TCM_funcs.u_assembly_c(TCd_c, rad_surf_tot)
AssX = TCM_funcs.assembly(TCd_f)

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

TCM_funcs.solver(TCAf, TCAc, TCAh, dt, u, u_c, t, Tisp, DeltaT, DeltaBlind, Kpc, Kph, rad_surf_tot)