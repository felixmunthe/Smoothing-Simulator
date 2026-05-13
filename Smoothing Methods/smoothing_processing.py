# -*- coding: utf-8 -*-
"""
B-Spline Smoothing Methods

Author: Munthe, Felix A.
Created on Wednesday, 1 November 2023
"""

import os
import json
import numpy as np
from smoothing_methods import (savgol, lowess_smooth, gam, kernel, bspline)

# Import True RNP Data
def read_true_RNP(file_path):

    # Step 1: Open the text file
    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    # Step 2: Process the data and create an array
    true_time = []
    true_RNP = []
    
    # Process each subsequent line and append the values
    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            true_time.append(float(values[0]))
            true_RNP.append(float(values[1]))
    
    return true_time, true_RNP

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

# ----- Create / Load Train-Test Split -----
def train_test_split(split_file_path, flow_regime_scenarios, noise_levels, number_data_set, train_ratio = 0.7, random_seed = 42):

    # If split file already exists, load it
    if os.path.exists(split_file_path):
        with open(split_file_path, "r", encoding = "utf-8") as file:
            split_data = json.load(file)
        return split_data

    # Otherwise, create split once and save it
    split_data = {}

    for s_idx, flow_regime_scenario in enumerate(flow_regime_scenarios):
        split_data[flow_regime_scenario] = {}

        for n_idx, noise_level in enumerate(noise_levels):
            rng = np.random.default_rng(random_seed + s_idx * 100 + n_idx)

            file_ids = np.arange(1, number_data_set + 1)
            rng.shuffle(file_ids)

            number_train = int(round(train_ratio * number_data_set))
            train_file_ids = sorted(file_ids[:number_train].tolist())
            test_file_ids = sorted(file_ids[number_train:].tolist())

            split_data[flow_regime_scenario][noise_level] = {
                "train_file_ids": train_file_ids,
                "test_file_ids": test_file_ids
            }

    with open(split_file_path, "w", encoding = "utf-8") as file:
        json.dump(split_data, file, indent = 4)

    return split_data

# ----- Compute SSE -----
def compute_sse(smoothed_RNP, true_RNP):
    sse = 0
    relative_residual = np.zeros(len(true_RNP), dtype = float)

    for j in range(len(smoothed_RNP)):
        relative_residual[j] = (smoothed_RNP[j] - true_RNP[j]) / true_RNP[j]
        sse += relative_residual[j] ** 2

    return sse, relative_residual

# ----- Main Execution -----
def main():
    # True RNP data file path
    true_RNP_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\true_RNP.txt"
    
    # Define time and RNP true data
    true_time, true_RNP = read_true_RNP(true_RNP_file_path) # days, psia2/cp-d/Mscf
    true_time = np.array(true_time) # days
    true_RNP = np.array(true_RNP) # psia2/cp-d/Mscf

    # Specify the folder path and file name for result file
    noisy_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Noisy RNP Model"
    output_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\Train Results"

    # Create root output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    flow_regime_scenarios = ["Transient", "Transition", "BDF", "Transient-Transition", "Transition-BDF", "Transient-BDF","All"]
    noise_levels = ["25%", "50%", "75%"]

    # Process the smoothing over the range of noisy data
    number_data_set = 10
    train_ratio = 0.7
    random_seed = 42

    # Define table headers
    headers = {
        "Data_Set": "Data_Set",
        "Noise_Level": "Noise_Level",
        "SSE": "SSE"
    }

    # Create / load saved train-test split
    split_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\train_test_split.json"

    split_data = train_test_split(split_file_path, flow_regime_scenarios, noise_levels, number_data_set, train_ratio = train_ratio, random_seed = random_seed)

    # Hyperparameter settings
    savgol_windows = [5, 7, 9, 11, 13, 15, 17, 19]
    lowess_windows = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    gam_lam_values = [0.001, 0.01, 0.1, 1, 10, 100]
    gaussian_sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    bspline_splines = [10, 20, 40, 60, 80, 100]

    # Loop over all scenarios and noise levels
    for flow_regime_scenario in flow_regime_scenarios:
        for noise_level in noise_levels:

            print("===================================================")
            print("Flow Regime Scenario:", flow_regime_scenario)
            print("Noise Level:", noise_level)

            train_file_ids = split_data[flow_regime_scenario][noise_level]["train_file_ids"]
            test_file_ids = split_data[flow_regime_scenario][noise_level]["test_file_ids"]

            print("Train File IDs:", train_file_ids)
            print("Test File IDs:", test_file_ids)

            # Output folder
            folder_path = f"{output_path}/{flow_regime_scenario}/{noise_level}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Export split info
            split_info_file_path = f"{folder_path}train_test_split_{flow_regime_scenario}_{noise_level}.txt"
            with open(split_info_file_path, "w", encoding = "utf-8") as file:
                file.write(f"Flow Regime Scenario\t{flow_regime_scenario}\n")
                file.write(f"Noise Level\t{noise_level}\n")
                file.write(f"Train File IDs\t{train_file_ids}\n")
                file.write(f"Test File IDs\t{test_file_ids}\n")

            # Summary store for best results of each method
            best_method_store = []

            # =========================================================
            # Savitzky-Golay
            # =========================================================
            print("----- Tuning Savitzky-Golay -----")
            best_sse = float("inf")
            best_window = None
            best_degree = None

            result_file_path = f"{folder_path}Savitzky_Golay_tuning_results.txt"
            with open(result_file_path, "w", encoding = "utf-8") as file:
                file.write("Window\tDegree\tAverage_SSE\n")

                for window in savgol_windows:
                    for degree in range(1, min(4, window)):
                        sse_store = []

                        for i in train_file_ids:
                            noisy_RNP_file_path = f"{noisy_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                            noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
                            noisy_time = np.array(noisy_time)
                            noisy_RNP = np.array(noisy_RNP)

                            smoothed_RNP = savgol(noisy_time, noisy_RNP, window, degree)
                            sse, relative_residual = compute_sse(smoothed_RNP, true_RNP)
                            sse_store.append(sse)

                        average_sse = np.mean(sse_store)
                        file.write(f"{window}\t{degree}\t{average_sse}\n")

                        if average_sse < best_sse:
                            best_sse = average_sse
                            best_window = window
                            best_degree = degree

            best_method_store.append(["Savitzky-Golay", best_sse, f"Window = {best_window}; Degree = {best_degree}"])

            # =========================================================
            # LoWeSS
            # =========================================================
            print("----- Tuning LoWeSS -----")
            best_sse = float("inf")
            best_window = None

            result_file_path = f"{folder_path}LoWeSS_tuning_results.txt"
            with open(result_file_path, "w", encoding = "utf-8") as file:
                file.write("Window\tAverage_SSE\n")

                for window in lowess_windows:
                    sse_store = []

                    for i in train_file_ids:
                        noisy_RNP_file_path = f"{noisy_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                        noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
                        noisy_time = np.array(noisy_time)
                        noisy_RNP = np.array(noisy_RNP)

                        smoothed_RNP = lowess_smooth(noisy_time, noisy_RNP, window, robust = False)
                        sse, relative_residual = compute_sse(smoothed_RNP, true_RNP)
                        sse_store.append(sse)

                    average_sse = np.mean(sse_store)
                    file.write(f"{window}\t{average_sse}\n")

                    if average_sse < best_sse:
                        best_sse = average_sse
                        best_window = window
            
            best_method_store.append(["LoWeSS", best_sse, f"Window = {best_window}"])

            # =========================================================
            # Robust LoWeSS
            # =========================================================
            print("----- Tuning Robust LoWeSS -----")
            best_sse = float("inf")
            best_window = None

            result_file_path = f"{folder_path}Robust_LoWeSS_tuning_results.txt"
            with open(result_file_path, "w", encoding = "utf-8") as file:
                file.write("Window\tAverage_SSE\n")

                for window in lowess_windows:
                    sse_store = []

                    for i in train_file_ids:
                        noisy_RNP_file_path = f"{noisy_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                        noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
                        noisy_time = np.array(noisy_time)
                        noisy_RNP = np.array(noisy_RNP)

                        smoothed_RNP = lowess_smooth(noisy_time, noisy_RNP, window, robust = True)
                        sse, relative_residual = compute_sse(smoothed_RNP, true_RNP)
                        sse_store.append(sse)

                    average_sse = np.mean(sse_store)
                    file.write(f"{window}\t{average_sse}\n")

                    if average_sse < best_sse:
                        best_sse = average_sse
                        best_window = window

            best_method_store.append(["Robust LoWeSS", best_sse, f"Window = {best_window}"])

            # =========================================================
            # Linear GAM
            # =========================================================
            print("----- Tuning Linear GAM -----")
            best_sse = float("inf")
            best_lam = None

            result_file_path = f"{folder_path}Linear_GAM_tuning_results.txt"
            with open(result_file_path, "w", encoding = "utf-8") as file:
                file.write("Lambda\tAverage_SSE\n")

                for lam_value in gam_lam_values:
                    sse_store = []

                    for i in train_file_ids:
                        noisy_RNP_file_path = f"{noisy_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                        noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
                        noisy_time = np.array(noisy_time)
                        noisy_RNP = np.array(noisy_RNP)

                        smoothed_RNP = gam(noisy_time, noisy_RNP, lam_value)
                        sse, relative_residual = compute_sse(smoothed_RNP, true_RNP)
                        sse_store.append(sse)

                    average_sse = np.mean(sse_store)
                    file.write(f"{lam_value}\t{average_sse}\n")

                    if average_sse < best_sse:
                        best_sse = average_sse
                        best_lam = lam_value

            best_method_store.append(["Linear GAM", best_sse, f"Lambda = {best_lam}"])

            # =========================================================
            # Gaussian Kernel
            # =========================================================
            print("----- Tuning Gaussian Kernel -----")
            best_sse = float("inf")
            best_sigma = None

            result_file_path = f"{folder_path}Gaussian_Kernel_tuning_results.txt"
            with open(result_file_path, "w", encoding = "utf-8") as file:
                file.write("Sigma\tAverage_SSE\n")

                for sigma in gaussian_sigmas:
                    sse_store = []

                    for i in train_file_ids:
                        noisy_RNP_file_path = f"{noisy_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                        noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
                        noisy_time = np.array(noisy_time)
                        noisy_RNP = np.array(noisy_RNP)

                        smoothed_RNP = kernel(noisy_time, noisy_RNP, sigma)
                        sse, relative_residual = compute_sse(smoothed_RNP, true_RNP)
                        sse_store.append(sse)

                    average_sse = np.mean(sse_store)
                    file.write(f"{sigma}\t{average_sse}\n")

                    if average_sse < best_sse:
                        best_sse = average_sse
                        best_sigma = sigma

            best_method_store.append(["Gaussian Kernel", best_sse, f"Sigma = {best_sigma}"])

            # =========================================================
            # B-Spline
            # =========================================================
            print("----- Tuning B-Spline -----")
            best_sse = float("inf")
            best_n_splines = None

            result_file_path = f"{folder_path}B_Spline_tuning_results.txt"
            with open(result_file_path, "w", encoding = "utf-8") as file:
                file.write("n_splines\tAverage_SSE\n")

                for n_splines_value in bspline_splines:
                    sse_store = []

                    for i in train_file_ids:
                        noisy_RNP_file_path = f"{noisy_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                        noisy_time, noisy_RNP = read_noisy_RNP(noisy_RNP_file_path)
                        noisy_time = np.array(noisy_time)
                        noisy_RNP = np.array(noisy_RNP)

                        smoothed_RNP = bspline(noisy_time, noisy_RNP, n_splines_value)
                        sse, relative_residual = compute_sse(smoothed_RNP, true_RNP)
                        sse_store.append(sse)

                    average_sse = np.mean(sse_store)
                    file.write(f"{n_splines_value}\t{average_sse}\n")

                    if average_sse < best_sse:
                        best_sse = average_sse
                        best_n_splines = n_splines_value

            best_method_store.append(["B-Spline", best_sse, f"n_splines = {best_n_splines}"])

            # =========================================================
            # Export Best Summary
            # =========================================================
            summary_file_path = f"{folder_path}Best_tuning_summary.txt"
            with open(summary_file_path, "w", encoding = "utf-8") as file:
                file.write("Method\tAverage_SSE\tBest_Parameter\n")
                for row in best_method_store:
                    file.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")

            print("-> Completed:", flow_regime_scenario, noise_level)

    print("--- Hyperparameter Tuning Complete ---")
    return

main()