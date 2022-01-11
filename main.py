import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


"""
List of available channels: - 
Time Into Test
Event Time
OP Speed 1
OP Torque 1
OP Speed 2
OP Torque 2
IP Speed 1
IP Torque 1
GBox T1
GBox T2
GBox T3
[V2] Pressure GBX out
[V3] Pressure Filter out
[V4] Temperature GBX out
[V5] Temperature Filter out
[V9] Pri. GBox Oil Temp
[P1] Pri.Gbox Press
Raw Oil Flow
Avg Oil Flow
Oil Flow Revs
TCM_CluPosnRel
TCM_DrvModReqd
TCM_GearActv
TCM_CluSts
TCM CluPosnBas
TCM_ClutchFault
TCM_CluSlipSts
TCM_CluTqDes
TCM_GearTar
TCM_GearActtnActv
TCM_GearEngd
TCM_CluTqTrf
TCM_TqEstimd
TCM_CluTqCalcd
TCM_ClutchTrq
TCM_GearActuatorFault
TCM_FaultReaction1
TCM_FaultReaction2
TCM_ClutchFault
TCM_BSW_ElecFault
TCM_SysPLow
TCM_ClutchOpenTime
TCM_ClutchOverTemp
TCM_ClutchHighTemp
TCM_HydOilOverTemp
TCM_LubPOutOfRange
TCM_SumpOilOverTemp
TCM_SumpOilHighTemp
TCM_CDS_triggered
TCM_FaultState
TCM GearActr1PosnBas
TCM GearActr2PosnBas
TCM GearActr3PosnBas
TCM GearActr4PosnBas
TCM_TC0FctSt
TCM CluPosnBas
TCM_GearEngd
TCM_GLBFct_St
TCM OutShaftNBas
TCM MaiShaftNBas
TCM InShaftNBas
TCM ClushaftNBas
TCM_FaultState_TCM
TCM_OilPmpEnaDutyCycOutp
TCM_OilPmpEnaPerdOutp
TCM_SysPBas
TCM_HydoilTBas
1306 Oil In - TCM_SumpoilTBas
1306 Oil Out - TCM_Sumpoil2TBas
AxleTorque
LockingTorque
OPSpeedDelta
"""


def get_data(path=''):
    """
    Defines the file path for analysis.
    File path can be passed as variable or left blank.
    If left blank, the function will open a file dialog allowing user to select a file
    """
    if(path == ''):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
    else:
        file_path = path
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    file_name = file_dir.split('/')[-1]

    print("#---------------------------------#")
    print("# File: ", file_name)
    print("#---------------------------------#")
    print("Location:", file_path)
    print("#---------------------------------#")

    df = pd.read_csv(file_path, sep=",", header=[3, 4], encoding='ISO-8859-1')
    return df, file_path, file_dir, file_name


def add_calculated_channels(data):
    """
    Adds calculated channels to the dataframe
    """
    # Calculated Channels
    # ------------------
    sr = calc_sample_rate(data)
    data['AxleTorque'] = data['OP Torque 1'] + data['OP Torque 2']
    data['LockingTorque'] = data['OP Torque 1'] - data['OP Torque 2']
    data['OPSpeedDelta'] = data['OP Speed 1'] - data['OP Speed 2']

    return data, sr


def set_axis(plots, axis, label, start, end, major, minor):
    """
    Function for setting plot label, axis major and minor ticks and formats the gridlines
    """
    for plot in plots:
        if major:
            major_ticks = np.arange(start, end + 1, major)
            if axis == 'x':
                plot.set_xlabel(label)
                plot.set_xlim([start, end])
                plot.set_xticks(major_ticks)
            elif axis == 'y':
                plot.set_ylabel(label)
                plot.set_ylim([start, end])
                plot.set_yticks(major_ticks)

        if minor:
            minor_ticks = np.arange(start, end + 1, minor)
            if axis == 'x':
                plot.set_xticks(minor_ticks, minor=True)
            elif axis == 'y':
                plot.set_yticks(minor_ticks, minor=True)

        if major and minor:
            plot.grid(which='both')
        plot.grid(which='minor', alpha=0.4)
        plot.grid(which='major', alpha=0.8)


def set_start(data):
    """
    Finds the start of the EoL test run and removes data beforehand. 
    Adjusts 'Event Time' channel to 0s at start
    """
    EoL_start_1 = np.argwhere(np.array(data['IP Speed 1']) < 10).flatten().tolist()
    EoL_start_1 = [i for i in EoL_start_1 if i != 0]

    EoL_start_2 = np.argwhere(np.array(data['[V9] Pri. GBox Oil Temp']) < 52).flatten().tolist()
    EoL_start_2 = [i for i in EoL_start_2 if i != 0]

    EoL_start_3 = np.argwhere(np.array(data['[V9] Pri. GBox Oil Temp']) > 42).flatten().tolist()
    EoL_start_3 = [i for i in EoL_start_3 if i != 0]

    EoL_start = sorted(list(set(EoL_start_1).intersection(EoL_start_2, EoL_start_3)))[-1]

    data = data.loc[EoL_start:len(data)].copy()
    data.reset_index(drop=True, inplace=True)
    data["Event Time"] = data.loc[:, "Event Time"] - data.loc[0, "Event Time"]
    return data


def find_points(file, sample_rate):
    """
    Finds a start and end index for each 
    speed step with a 5s buffer form ramped sections
    """
    max_idx = sorted(np.argwhere(np.array(file['IP Speed 1']) > 5995).flatten().tolist())[-1]
    points = []
    for i in range(6):
        step_temp = np.argwhere(np.array(file.loc[0:max_idx, 'IP Speed 1']) > (i+1) * 950).flatten().tolist()
        step_temp_2 = np.argwhere(np.array(file.loc[0:max_idx, 'IP Speed 1']) < (i+1) * 1000 + 350).flatten().tolist()
        step_temp_sort = sorted(list(set(step_temp).intersection(step_temp_2)))
        points.append(step_temp_sort[1] + (int(sample_rate) * 5))
        points.append(step_temp_sort[-1] - (int(sample_rate) * 5))

    points_grouped = (list(zip(points[::2], points[1::2])))
    return points, points_grouped


def calc_sample_rate(data):
    """
    Calculates the sample rate from a given dataset
    """
    return round(1 / data['Event Time'].diff().mean().values[0], 1)


def generate_plotting_data(data, points):
    """
    Uses the start and end points found in the find_points() function to extract the data and
    calculate averages for each section.
    Returns a dataframe with averages of all channels for each speed step
    """
    extracted_data = []
    for point in points:
        start = point[0]
        end = point[1]
        extracted_data.append(pd.DataFrame(data.loc[start:end].mean().to_dict(),index=[data.index.values[-1]]))
    return pd.concat(extracted_data)


def plot_channel(data, file_name, chart, x_axis, y_axis, label, marker=''):
    print(f'Plotting {y_axis}')
    chart.plot(
        data[x_axis],
        data[y_axis],
        # color=colour,
        label=f'{file_name}: {label}',
        marker=marker
    )


def plot_points(data, chart, x_axis, y_axis, points=''):
    print(f'Plotting {y_axis}')
    if points == '':
        chart.plot(
            data[x_axis],
            data[y_axis],
            'o',
            linestyle = 'None'
        )
    else:
        chart.plot(
            data[x_axis].iloc[points],
            data[y_axis].iloc[points],
            'o',
            linestyle = 'None'
        )

# TODO:
# -----------------------
# o  Add colour management for plotting when plotting multiple files for comparison
# o  Determine correct output for graphs
# o  Add print to PDF or savefig function to save the output into each raw_data folder 
# -----------------------


#  Open Files
# -----------------------
raw_data = []
raw_data.append(get_data('//DSUK01/Company Shared Documents/Projects/1306/XT REPORTS'
    '/XT-14972 - PRO6 Noise Investigation/R&D testing/1306-027'
    '/2021-12-10 - 1306-027/1306-027_EOL_TEST_Run1'
    '/Trace 01314 10 12 2021 17_29_29.&0M_001.CSV'))
raw_data.append(get_data('//DSUK01/Company Shared Documents/Projects/1306/XT REPORTS'
    '/XT-14972 - PRO6 Noise Investigation/R&D testing/1306-027'
    '/2021-12-21 - 1306-027/1306-027_EOL_TEST_Run3'
    '/Trace 01335 21 12 2021 15_35_39.&11_001.CSV'))
raw_data.append(get_data('//DSUK01/Company Shared Documents/Projects/1306/XT REPORTS'
    '/XT-14972 - PRO6 Noise Investigation/R&D testing/1306-027'
    '/2022-01-04 - 1306-027/1306-027_EOL TEST_Run5'
    '/Trace 01356 04 01 2022 18_26_02.&1L_001.CSV'))


# Figure 1 - Summary plot
# -----------------------
figsize = (16, 9)
fig, ax = plt.subplots(3, figsize=figsize)
xlim = 200
xmaj = math.floor(xlim / 20)
xminor = math.floor(xmaj / 5)

set_axis(ax, 'x', 'Time [s]', 0, xlim, xmaj, xminor)
axSecondary = ax[0].twinx()
set_axis([axSecondary], 'y', 'Temperature [degC]', 20, 70, 5, 2.5)

fig2, ax2 = plt.subplots(2, 2, figsize=figsize)

foutput = []
plotting_data = []

for rdata, fpath, fdir, fname in raw_data:
    foutput.append(f'{fdir}/{fname}.png')

    rdata, sr = add_calculated_channels(rdata)
    rdata = set_start(rdata)
    points, points_grouped = find_points(rdata, sr)
    plotting_data = generate_plotting_data(rdata, points_grouped)

    # Fig 1, Plot 1 - IP Speed & Temperature
    plot_channel(rdata, fname, axSecondary, 'Event Time', '[V9] Pri. GBox Oil Temp', 'Oil Temperature [degC]')
    plot_channel(rdata, fname, ax[0], 'Event Time', 'IP Speed 1', 'Input Speed [rpm]')
    plot_points(rdata, ax[0], 'Event Time', 'IP Speed 1', points)
    plot_points(plotting_data, ax[0], 'Event Time', 'IP Speed 1')

    # Fig 1, Plot 2 - IP Torque
    plot_channel(rdata, fname, ax[1], 'Event Time', 'IP Torque 1', 'Input Speed [rpm]')
    plot_points(plotting_data, ax[1], 'Event Time', 'IP Torque 1')

    # Fig 1, Plot 3
    plot_channel(rdata, fname, ax[2], 'Event Time', 'AxleTorque', 'Axle Torque [Nm]')

    # Fig 2, Top-Left
    plot_channel(plotting_data, fname, ax2[0, 0], 'IP Speed 1', 'IP Torque 1', 'IP Torque [Nm]', 'o')
    plot_channel(plotting_data, fname, ax2[0, 1], 'IP Speed 1', 'Raw Oil Flow', 'Flow Rate [l/min]', 'o')
    plot_channel(plotting_data, fname, ax2[1, 0], 'IP Speed 1', '[P1] Pri.Gbox Press', 'Oil Pressure [bar]', 'o')
    plot_channel(plotting_data, fname, ax2[1, 1], 'IP Speed 1', '[V9] Pri. GBox Oil Temp', 'Oil Temperature [degC]', 'o')
    plot_channel(plotting_data, fname, ax2[1, 1], 'IP Speed 1', 'GBox T2', 'RH Flange Temp [degC]', 'o')
    plot_channel(plotting_data, fname, ax2[1, 1], 'IP Speed 1', 'GBox T3', 'LH Flange Temp [degC]', 'o')



ax[0].legend(loc=2, facecolor="white") # loc=2 == 'upper left'
ax[0].set_zorder(1)
ax[0].set_frame_on(False)
axSecondary.set_frame_on(True)
axSecondary.legend(loc=1, facecolor="white") # loc=1 == 'upper right'
ax[0].set_title("Input Speed & Oil Temperature", loc='left')
set_axis([ax[0]], 'y', 'Speed [rpm]', 0, 10000, 1000, 500)

ax[1].set_title("Input Torque", loc='left')
ax[1].legend(loc=1, facecolor="white")
set_axis([ax[1]], 'y', 'Torque [Nm]', 0, 30, 10, 2)


# Plot 3
# -----------------------

ax[2].set_title("Axle Torque (LH + RH)", loc='left')
ax[2].legend(loc=1, facecolor="white")
set_axis([ax[2]], 'y', 'Torque [Nm]', 0, 30, 10, 2)

fig.suptitle(f'{fdir}', fontsize=10)
plt.subplots_adjust(left=0.05, bottom=0.07, right=0.965, top=0.9, wspace=0.2, hspace=0.4)
# plt.savefig(foutput, format='png', bbox_inches='tight', dpi=150)
# plt.savefig(foutput2, format='png', bbox_inches='tight', dpi=150)


# # Figure 2 - EoL Plot
# -----------------------
# fig2, ax2 = plt.subplots(2, 2, figsize=figsize)

# Plot 1 - IP Torque
ax2[0, 0].set_title("IP Torque Comparison", loc='left')
ax2[0, 0].legend(loc=2, facecolor="white")
set_axis([ax2[0, 0]], 'y', 'Torque [Nm]', 0, 30, 10, 2)

# Plot 2 - Oil Flow
ax2[0, 1].set_title("Oil Flow Rate Comparison", loc='left')
ax2[0, 1].legend(loc=2, facecolor="white")
set_axis([ax2[0, 1]], 'y', 'Oil Flow Rate [L/min]', 0, 20, 2, 1)

# Plot 3 - Oil Pressure
ax2[1, 0].set_title("Oil Pressure Comparison", loc='left')
ax2[1, 0].legend(loc=2, facecolor="white")
set_axis([ax2[1, 0]], 'y', 'Oil Pressure [bar]', 0, 3, 0.5, 0.25)

# Plot 4 - Oil Temperature
ax2[1, 1].set_title("Oil & OP Flange Temperature Comparison", loc='left')
ax2[1, 1].legend(loc=2, facecolor="white")
set_axis([ax2[1, 1]], 'y', 'Temperature [degC]', 20, 100, 10, 2.5)

for axs in ax2:
    set_axis(axs, 'x', 'IP Speed [rpm]', 0, 8000, 1000, 200)
# fig2.suptitle(f'{fname} vs {fname2}', fontsize=16)
plt.subplots_adjust(left=0.05, bottom=0.07, right=0.965, top=0.9, wspace=0.2, hspace=0.4)
plt.show()