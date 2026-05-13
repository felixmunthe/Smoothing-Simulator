# -*- coding: utf-8 -*-
"""
Classical Smoothing Methods

Author: Munthe, Felix A.
Created on Friday, 15 September 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import Pseudopressure_Conversion
from scipy import stats
from svg_to_emf import convert_svg_to_emf

# ----- Import Production Data -----
def read_production(file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    time = []
    downhole_rate = []
    
    # Process each subsequent line and append the values
    for line in lines[2:]:
        values = line.split()
        if len(values) >= 2:
            time.append(float(values[0]))
            downhole_rate.append(float(values[1]))
    
    return time, downhole_rate

# ----- Import Noisy RNP Data -----
def read_noisy_RNP (file_path):

    # Step 1: Open the text file
    with open (file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    noisy_time = []
    noisy_RNP = []
    
    # Step 3: Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            noisy_time.append(float(values[0]))
            noisy_RNP.append(float(values[1]))
    
    return noisy_time, noisy_RNP

# ----- Main Execution -----
def main():
    # Production data file path
    downhole_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\downhole_gas_rate.txt"
    surface_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\surface_gas_rate.txt"
    volume_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Outlier Detection Paper\Outlier Detection Simulator\Synthetic Model\surface_gas_volume.txt"

    # Define actual time and rate data
    time, downhole_rate = read_production(downhole_file_path) # hr, Mcf/d
    time, surface_rate = read_production(surface_file_path) # hr, Mscf/d
    time, volume = read_production(volume_file_path) # hr, scf
    time = np.array(time) / 24 # days
    time_prod = time[1:]
    volume = np.array(volume) * 0.001 # Mscf
    tmb = [volume[i] / surface_rate[i] for i in range (1, len(time))] # days

    # Generate RNP coordinates
    RNP_data = [Pseudopressure_Conversion.delta_pseudopressure[i] / downhole_rate[i] for i in range(1, len(time))]
    RNP_coordinates = np.column_stack((time_prod, RNP_data))
    print(RNP_data[0])

    # Export true RNP data
    file_path = "C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Smoothing Paper/Smoothing Simulator/true_RNP.txt"
    with open(file_path, "w") as file:
        # Write the header
        file.write("t(days)\tRNP(psia2/cp-d/Mscf)\n")
        # Write the data to the file
        for x, y in zip(time_prod, RNP_data):
            file.write(f"{x}\t{y}\n")

    # Noisy RNP data file path
    noisy_RNP_file_path = f"C:/Users/felix/OneDrive/Documents/Kuliah/2. Master's Texas A&M University/Publications/Conference Paper/Outlier Detection Paper/Outlier Detection Simulator/Noisy RNP Model/All/25%/noisy_data_5.txt"
    noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
    noisy_data = np.column_stack((noisy_time, noisy_RNP))

    # Transform RNP data into log value
    log_noisy_time = np.log(noisy_time)
    log_noisy_RNP = np.log(noisy_RNP)

    # Perform Linear Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_noisy_time, log_noisy_RNP)
    regression_RNP = [slope * log_noisy_time[i] + intercept for i in range(len(log_noisy_time))]

    # Perform Polynomial Regression
    degree1 = 3
    coefficients1 = np.polyfit(log_noisy_time, log_noisy_RNP, degree1)
    polynomial_function1 = np.poly1d(coefficients1)
    x_poly1 = np.linspace(min(log_noisy_time), max(log_noisy_time), 100)
    y_poly1 = polynomial_function1(x_poly1)

    degree2 = 10
    coefficients2 = np.polyfit(log_noisy_time, log_noisy_RNP, degree2)
    polynomial_function2 = np.poly1d(coefficients2)
    x_poly2 = np.linspace(min(log_noisy_time), max(log_noisy_time), 100)
    y_poly2 = polynomial_function2(x_poly2)

    # Perform Moving Average
    window_size1 = 5
    moving_average1 = np.convolve(noisy_RNP, np.ones(window_size1) / window_size1, mode = 'valid')
    
    window_size2 = 20
    moving_average2 = np.convolve(noisy_RNP, np.ones(window_size2) / window_size2, mode = 'valid')

    # Plot noisy RNP vs time
    plt.figure()
    plt.plot(noisy_time, noisy_RNP, 'o', label = 'data')
    plt.plot(time_prod, RNP_data, '-', label = 'true line', linewidth = '3')
    plt.plot(noisy_time, noisy_RNP, ':', label = 'linear interpolation', linewidth = '3', alpha = 0.9)
    plt.plot(noisy_time, np.exp(regression_RNP), '-.', label = 'linear regression', linewidth = '3', alpha = 0.85)
    plt.plot(np.exp(x_poly1), np.exp(y_poly1), '--', label = 'polynomial regression (degree = 3)', linewidth = '3', alpha = 0.8)
    #plt.plot(np.exp(x_poly2), np.exp(y_poly2), '--', label = 'polynomial regression (degree = 10)', linewidth = '3', alpha = 0.8)
    plt.plot(noisy_time[window_size1-1:], moving_average1, '-', label = 'moving average (window = 5)', linewidth = '3', alpha = 0.75)
    #plt.plot(noisy_time[window_size2-1:], moving_average2, '-', label = 'moving average (window = 20)', linewidth = '3', alpha = 0.75)
    plt.xlabel("$t$, day", fontsize = 20)
    plt.ylabel("$\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\cdot$d/Mscf", fontsize = 20)
    plt.xscale("log")
    plt.yscale("log")
    # plt.title("Rate-normalized Pseudo-pressure (RNP) vs. Time", fontsize = 24)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
    plt.legend(fontsize = 20)
    plt.show()

    svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_2.svg"
    emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_2.emf"
    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

    convert_svg_to_emf(svg_path, emf_path, inkscape_path=inkscape_path)

    return

main()
