"""
Code to calculate the annual heating consumption and cooling consumption of a building.

Author: Charles Gerike-Roberts 16/05/22

Inputs:
    - HVAC heat flow from dynamic solver, qHVAC (W)
    - Annual heat consumption from DHW heating, dhw_cons (kWh)
    - Time step from dynamic solver, dt (s)

Outputs:
    - Annual heat consumption, Q_cons_heat (kWh)
    - Annual cool consumption, Q_cons_cool (kWh)

"""
import numpy as np

def heat_cons(qHVAC, dhw_cons, dt):
    qHVAC_diff = np.diff(qHVAC)
    qHVAC_red = qHVAC/1000
    for i in range(0, qHVAC_diff.shape[0]):
        a = int(qHVAC_diff[i])
        if a in range(1, 5):
            break
        else:
            qHVAC_red = np.delete(qHVAC_red, 0)

    qHVAC_heat = np.zeros(qHVAC_red.shape[0])
    qHVAC_cool = np. zeros(qHVAC_red.shape[0])

    for i in range(0, qHVAC_heat.shape[0]):
        if qHVAC_red[i] >= 0:
            qHVAC_heat[i] = qHVAC_red[i]
        else:
            qHVAC_cool[i] = qHVAC_red[i]

    dt_h = dt / 3600
    Q_cons_init_heat = qHVAC_heat[0] * (dt_h / 2)
    Q_cons_space_end_heat = qHVAC_heat[-1] * (dt_h / 2)
    Q_cons_init_cool = qHVAC_cool[0] * (dt_h / 2)
    Q_cons_space_end_cool = qHVAC_cool[-1] * (dt_h / 2)
    Q_cons_heat = 0
    Q_cons_cool = 0
    for i in range(1, (qHVAC_heat.shape[0] - 1)):
        Q_cons_part = qHVAC_heat[i] * dt_h
        Q_cons_heat = Q_cons_heat + Q_cons_part

    for i in range(1, (qHVAC_cool.shape[0] - 1)):
        Q_cons_part = qHVAC_cool[i] * dt_h
        Q_cons_cool = Q_cons_cool + Q_cons_part

    Q_cons_heat = Q_cons_heat + dhw_cons + Q_cons_init_heat + Q_cons_space_end_heat
    Q_cons_cool = Q_cons_cool + Q_cons_init_cool + Q_cons_space_end_cool

    return Q_cons_heat, Q_cons_cool