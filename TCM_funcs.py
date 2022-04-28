"""
Created on We Nov 10 2021

@author: L. Beber, C. Gerike-Roberts, E. Regev

File with all the functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem


def building_characteristics():
    """
    This code is designed to read an excel file which contains the characteristics of the building
    and create a data frame from it.
    """

    bc = pd.read_excel("Building Characteristics.xlsx", sheet_name='Elements', na_values=["N"], keep_default_na=True,
                       header=0)

    return bc

def inputs():
    ip = pd.read_excel("Building Characteristics.xlsx", sheet_name='Inputs', na_values=["N"], keep_default_na=True,
                       index_col=0, usecols="A:B")
    return ip

def thphprop(BCdf):
    """
    Parameters
    ----------
    BCdf : data frame of building characteristics
        DESCRIPTION.
        Data Frame of building characteristics. Example:
                BCdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth', ]

    Returns
    -------
    Bdf : data frame
        DESCRIPTION.
        data frame of the Building characteristics with associated thermophysical properties
                Bdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth',
                'Density 1', 'specific heat 1', 'conductivity 1', 'LW emissivity 1', 'SW transmittance 1',
                'SW absorptivity 1', 'albedo 1', 'Density 2', 'specific heat 2', 'conductivity 2', 'LW emissivity 2',
                'SW transmittance 2', 'SW absorptivity 2', 'albedo 2', 'Density 3', 'specific heat 3', 'conductivity 3',
                'LW emissivity 3', 'SW transmittance 3', 'SW absorptivity 3', 'albedo 3']
    """

    # Thermo-physical and radiative properties - source data frame
    # ----------------------------------------------------------

    """ Incropera et al. (2011) Fundamentals of heat and mass transfer, 7 ed,
        Table A3,
            concrete (stone mix) p. 993
            insulation polystyrene extruded (R-12) p.990
            glass plate p.993
            Clay tile, hollow p.989
            Wood, oak p.989
            Soil p.994
    """

    thphp = pd.read_excel("Building Characteristics.xlsx", sheet_name='Materials', header=0, usecols="A:H")

    # add empty columns for thermo-physical properties
    BCdf = BCdf.reindex(columns=BCdf.columns.to_list() + ['rad_s', 'density_1', 'specific_heat_1', 'conductivity_1',
                                                          'LW_emissivity_1', 'SW_transmittance_1', 'SW_absorptivity_1',
                                                          'albedo_1', 'density_2', 'specific_heat_2', 'conductivity_2',
                                                          'LW_emissivity_2', 'SW_transmittance_2', 'SW_absorptivity_2',
                                                          'albedo_2', 'density_3', 'specific_heat_3', 'conductivity_3',
                                                          'LW_emissivity_3', 'SW_transmittance_3', 'SW_absorptivity_3',
                                                          'albedo_3', 'density_4', 'specific_heat_4', 'conductivity_4',
                                                          'LW_emissivity_4', 'SW_transmittance_4', 'SW_absorptivity_4',
                                                          'albedo_4', 'density_5', 'specific_heat_5', 'conductivity_5',
                                                          'LW_emissivity_5', 'SW_transmittance_5', 'SW_absorptivity_5',
                                                          'albedo_5'])

    # fill columns with properties for the given materials 1-5 of each element
    for i in range(0, len(BCdf)):
        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_1'] == thphp.Material[j]:
                BCdf.loc[i, 'density_1'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_1'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_1'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_1'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_1'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_1'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_1'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_2'] == thphp.Material[j]:
                BCdf.loc[i, 'density_2'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_2'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_2'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_2'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_2'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_2'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_2'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_3'] == thphp.Material[j]:
                BCdf.loc[i, 'density_3'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_3'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_3'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_3'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_3'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_3'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_3'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_4'] == thphp.Material[j]:
                BCdf.loc[i, 'density_4'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_4'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_4'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_4'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_4'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_4'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_4'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_5'] == thphp.Material[j]:
                BCdf.loc[i, 'density_5'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_5'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_5'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_5'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_5'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_5'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_5'] = thphp.Albedo[j]

    return BCdf


def rad(bcp, ip):
    # Simulation with weather data
    # ----------------------------
    albedo_sur = ip.loc['Albedo_sur']['Value']
    latitude = ip.loc['Latitude']['Value']
    dt = ip.loc['dt']['Value']
    WF = ip.loc['WF']['Value']
    t_start = ip.loc['t_start']['Value']
    t_end = ip.loc['t_end']['Value']

    filename = WF
    start_date = t_start
    end_date = t_end

    # Read weather data from Energyplus .epw file
    [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
    weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    del data
    weather.index = weather.index.map(lambda t: t.replace(year=2000))
    weather = weather[(weather.index >= start_date) & (
            weather.index < end_date)]
    # Solar radiation on a tilted surface South
    Φt = {}
    for k in range(0, len(bcp)):
        surface_orientationS = {'slope': bcp.loc[k, 'Slope'],
                                'azimuth': bcp.loc[k, 'Azimuth'],
                                'latitude': latitude}
        rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientationS, albedo_sur)
        Φt.update({str(k + 2): rad_surf.sum(axis=1)})

    Φt = pd.DataFrame(Φt)
    # Interpolate weather data for time step dt
    data = pd.concat([weather['temp_air'], Φt], axis=1)
    data = data.resample(str(dt) + 'S').interpolate(method='linear')
    data = data.rename(columns={'temp_air': 'To'})

    # time
    t = dt * np.arange(data.shape[0])

    return data, t

def indoor_rad(bcp_r, TCd, IG):
    Q = TCd['Q']
    lim = np.shape(Q)[1]
    for i in range(0, lim):
        if Q[0, i] == -1:
            if np.isnan(bcp_r['SW_absorptivity_5']):
                if np.isnan(bcp_r['SW_absorptivity_4']):
                    if np.isnan(bcp_r['SW_absorptivity_3']):
                        if np.isnan(bcp_r['SW_absorptivity_2']):
                                x = bcp_r['SW_absorptivity_1'] * IG
                                Q[:, i] = x[:, 0]
                        else:
                                x = bcp_r['SW_absorptivity_2'] * IG
                                Q[:, i] = x[:, 0]
                    else:
                            x = bcp_r['SW_absorptivity_3'] * IG
                            Q[:, i] = x[:, 0]
                else:
                    x = bcp_r['SW_absorptivity_4'] * IG
                    Q[:, i] = x[:, 0]
            else:
                x = bcp_r['SW_absorptivity_5'] * IG
                Q[:, i] = x[:, 0]
    TCd['Q'] = Q  # replace Q in TCd with new Q

    return TCd

def indoor_rad_c(TCd_c):
    Q = TCd_c['Q']
    lim = np.shape(Q)[1]
    for i in range(0, lim):
        if Q[0, i] == -1:
            Q[:, i] = 0

    TCd_c['Q'] = Q  # replace Q in TCd with new Q

    return TCd_c


def u_assembly(TCd, rad_surf_tot):
    rad_surf_tot = rad_surf_tot.loc[:, rad_surf_tot.any()]
    u = np.empty((len(rad_surf_tot), 1))  # create u matrix
    for i in range(0, TCd.shape[1]):
        TCd_i = TCd[str(i)]
        T = TCd_i['T']
        T = T[:, ~np.isnan(T).any(axis=0)]
        if np.shape(T)[1] == 0:
            print('No Temp')
        else:
            u = np.append(u, T, axis=1)

    u = np.delete(u, 0, 1)

    for j in range(0, TCd.shape[1]):
        TCd_j = TCd[str(j)]
        Q = TCd_j['Q']
        Q = Q[:, ~np.isnan(Q).any(axis=0)]
        if np.shape(Q)[1] == 0:
            print('No Heat Flow')
        else:
            u = np.append(u, Q, axis=1)

    u = pd.DataFrame(u)

    return u, rad_surf_tot

def u_assembly_c(TCd_c, rad_surf_tot):
    rad_surf_tot = rad_surf_tot.loc[:, rad_surf_tot.any()]
    u_c = np.empty((len(rad_surf_tot), 1))  # create u matrix
    for i in range(0, TCd_c.shape[1]):
        TCd_i = TCd_c[str(i)]
        T = TCd_i['T']
        T = T[:, ~np.isnan(T).any(axis=0)]
        if np.shape(T)[1] == 0:
            print('No Temp')
        else:
            u_c = np.append(u_c, T, axis=1)

    u_c = np.delete(u_c, 0, 1)

    for j in range(0, TCd_c.shape[1]):
        TCd_j = TCd_c[str(j)]
        Q = TCd_j['Q']
        Q = Q[:, ~np.isnan(Q).any(axis=0)]
        u_c = np.append(u_c, Q, axis=1)

    u_c = pd.DataFrame(u_c)

    return u_c, rad_surf_tot


def assembly(TCd, tcd_dorwinsky, tcd_n):
    """
    Description: The assembly function is used to define how the nodes in the disassembled thermal circuits
    are merged together.

    Inputs: TCd

    Outputs: AssX
    """
    TCd_last_node = np.zeros(TCd.shape[1] - 1)  # define size of matrix for last node in each TC
    TCd_element_numbers = np.arange(1, TCd.shape[1], 1)  # create vector which contains the number for each element

    # compute number of last node of each thermal circuit and input into thermal circuit sizes matrix
    for i in range(0, len([TCd_last_node][0])):
        TCd_last_node[i] = len(TCd[str(i + 1)]['A'][0]) - 1

    print(TCd_last_node)

    IA_nodes = np.arange(len(TCd[str(0)]['A'][0]))  # create vector with the nodes for inside air
    print(IA_nodes)

    # create assembly matrix containing ventilation, windows, doors and skylights
    AssX = np.zeros((len(TCd_last_node), 4))
    Ven_Nodes = [1, 0, 0, 0]
    AssX[0] = Ven_Nodes
    for i in range(0, (tcd_dorwinsky - 1)):
        AssX[i+1, 0] = TCd_element_numbers[i + 1]  # set first column of row to element
        AssX[i+1, 1] = TCd_last_node[i + 1]  # set second column to last node of that element
        AssX[i+1, 2] = 0  # set third column to inside air element
        AssX[i+1, 3] = 0  # set 4th column to element of inside air which connects to corresponding element

    # insert walls, floors and roofs into assembly matrix
    for i in range(tcd_dorwinsky, (tcd_n - 1)):
        AssX[i, 0] = TCd_element_numbers[i]  # set first column of row to element
        AssX[i, 1] = TCd_last_node[i]  # set second column to last node of that element
        AssX[i, 2] = 0  # set third column to inside air element
        AssX[i, 3] = IA_nodes[(i - (tcd_dorwinsky - 1))]  # set 4th column to element of inside air which connects to corresponding element

    AssX = AssX.astype(int)

    print(AssX)

    return AssX


def solver(TCAf, TCAc, TCAh, ip, u, u_c, t, Kpc, Kph, rad_surf_tot):
    [Af, Bf, Cf, Df] = dm4bem.tc2ss(TCAf['A'], TCAf['G'], TCAf['b'], TCAf['C'], TCAf['f'], TCAf['y'])
    [Ac, Bc, Cc, Dc] = dm4bem.tc2ss(TCAc['A'], TCAc['G'], TCAc['b'], TCAc['C'], TCAc['f'], TCAc['y'])
    [Ah, Bh, Ch, Dh] = dm4bem.tc2ss(TCAh['A'], TCAh['G'], TCAh['b'], TCAh['C'], TCAh['f'], TCAh['y'])

    # define values from input tensor
    dt = ip.loc['dt']['Value']
    Tisp = ip.loc['Tisp']['Value']
    DeltaT = ip.loc['T_cooling']['Value'] - Tisp
    DeltaBlind = ['DeltaBlind']['Value']

    if DeltaBlind == -1:
        u_c = u
    else:
        u_c = u_c

    # Maximum time-step
    dtmax = min(-2. / np.linalg.eig(Af)[0])
    print(f'Maximum time step f: {dtmax:.2f} s')

    if dtmax >= dt:
        raise ValueError('Free cooling time-step unstable.')

    dtmax = min(-2. / np.linalg.eig(Ac)[0])
    print(f'Maximum time step c: {dtmax:.2f} s')

    if dtmax >= dt:
        raise ValueError('Cooling time-step unstable.')

    dtmax = min(-2. / np.linalg.eig(Ah)[0])
    print(f'Maximum time step h: {dtmax:.2f} s')

    if dtmax >= dt:
        raise ValueError('Heating time-step unstable.')

    # Step response
    # -------------
    duration = 3600 * 24 * 1  # [s]
    # number of steps
    n = int(np.floor(duration / dt))

    t_ss = np.arange(0, n * dt, dt)  # time

    # Vectors of state and input (in time)
    n_tC = Af.shape[0]  # no of state variables (temps with capacity)
    # u = [To To To Tsp Phio Phii Qaux Phia]
    u_ss = np.zeros([(u.shape[1]), n])
    u_ss[0:3, :] = np.ones([3, n])
    u_ss[4:6, :] = 1

    # initial values for temperatures obtained by explicit and implicit Euler
    temp_exp = np.zeros([n_tC, t_ss.shape[0]])
    temp_imp = np.zeros([n_tC, t_ss.shape[0]])

    I = np.eye(n_tC)
    for k in range(n - 1):
        temp_exp[:, k + 1] = (I + dt * Ac) @ \
                             temp_exp[:, k] + dt * Bc @ u_ss[:, k]
        temp_imp[:, k + 1] = np.linalg.inv(I - dt * Ac) @ \
                             (temp_imp[:, k] + dt * Bc @ u_ss[:, k])

    y_exp = Cc @ temp_exp + Dc @ u_ss
    y_imp = Cc @ temp_imp + Dc @ u_ss

    fig, axs = plt.subplots(2, 1)
    # axs[0].plot(t_ss / 3600, y_exp.T, t_ss / 3600, y_imp.T)
    # axs[0].set(ylabel='$T_i$ [°C]', title='Step input: To = 1°C')

    # initial values for temperatures
    temp_exp = np.zeros([n_tC, t.shape[0]])
    temp_imp = np.zeros([n_tC, t.shape[0]])
    Tisp = Tisp * np.ones(u.shape[0])
    y = np.zeros(u.shape[0])
    y[0] = Tisp[0]
    qHVAC = 0 * np.ones(u.shape[0])

    # integration in time
    I = np.eye(n_tC)
    for k in range(u.shape[0] - 1):
        if y[k] > Tisp[k] + DeltaBlind:
            us = u_c
        else:
            us = u
        if y[k] > DeltaT + Tisp[k]:
            temp_exp[:, k + 1] = (I + dt * Ac) @ temp_exp[:, k] \
                                 + dt * Bc @ us.iloc[k, :]
            y[k + 1] = Cc @ temp_exp[:, k + 1] + Dc @ us.iloc[k + 1]
            qHVAC[k + 1] = Kpc * (Tisp[k + 1] - y[k + 1])
        elif y[k] < Tisp[k]:
            temp_exp[:, k + 1] = (I + dt * Ah) @ temp_exp[:, k] \
                                 + dt * Bh @ us.iloc[k, :]
            y[k + 1] = Ch @ temp_exp[:, k + 1] + Dh @ us.iloc[k + 1]
            qHVAC[k + 1] = Kph * (Tisp[k + 1] - y[k + 1])
        else:
            temp_exp[:, k + 1] = (I + dt * Af) @ temp_exp[:, k] \
                                 + dt * Bf @ us.iloc[k, :]
            y[k + 1] = Cf @ temp_exp[:, k + 1] + Df @ us.iloc[k]
            qHVAC[k + 1] = 0

    # plot indoor and outdoor temperature
    axs[0].plot(t / 3600, y, label='$T_{indoor}$')
    axs[0].plot(t / 3600, rad_surf_tot['To'], label='$T_{outdoor}$')
    axs[0].set(xlabel='Time [h]',
               ylabel='Temperatures [°C]',
               title='Simulation for weather')
    axs[0].legend(loc='upper right')

    # plot total solar radiation and HVAC heat flow
    del rad_surf_tot['To']
    Φt = rad_surf_tot.sum(axis=1)
    axs[1].plot(t / 3600, qHVAC, label='$q_{HVAC}$')
    axs[1].plot(t / 3600, Φt, label='$Φ_{total}$')
    axs[1].set(xlabel='Time [h]',
               ylabel='Heat flows [W]')
    axs[1].legend(loc='upper right')
    plt.ylim(-1500, 6000)
    fig.tight_layout()

    plt.show()

    return qHVAC

def DHW():
    DHW = pd.read_excel("Building Characteristics.xlsx", sheet_name='DHW', na_values=["N"], keep_default_na=True,
                       header=0)

    n_shower = DHW.loc['n_shower']['Value']
    n_bath = DHW.loc['n_bath']['Value']
    n_wash = DHW.loc['n_wash']['Value']
    n_sink = DHW.loc['n_sink']['Value']
    hw_shower = DHW.loc['hw_shower']['Value']
    hw_bath = DHW.loc['hw_shower']['Value']
    hw_wash = DHW.loc['hw_shower']['Value']
    hw_sink = DHW.loc['hw_shower']['Value']
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

def heat_cons(qHVAC, dhw_peak, dhw_cons, dt):
    qHVAC_diff = np.diff(qHVAC)
    qHVAC_red = qHVAC
    for i in range(0, qHVAC_diff.shape[0]):
        a = int(qHVAC_diff[i])
        if a in range(1, 5):
            break
        else:
            qHVAC_red = np.delete(qHVAC_red, 0)

    dt_h = dt / 3600
    ann_cons_init = qHVAC_red[0] * (dt_h / 2)
    ann_cons_space_end = qHVAC_red[-1] * (dt_h / 2)
    ann_cons = 0
    for i in range(1, (qHVAC_red.shape[0] - 1)):
        ann_cons_part = qHVAC_red[i] * dt_h
        ann_cons = ann_cons + ann_cons_part

    ann_cons = ann_cons + dhw_cons + ann_cons_init + ann_cons_space_end

    peak_power_space = max(qHVAC_red)

    peak_power_tot = peak_power_space + dhw_peak

    return ann_cons, peak_power_space, peak_power_tot
