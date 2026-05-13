# -*- coding: utf-8 -*-
"""
Smoothed RNP Plot

Author: Munthe, Felix A.
Created on Wednesday, 1 November 2023
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import Pseudopressure_Conversion
from scipy.stats import linregress
from svg_to_emf import convert_svg_to_emf

# Import True RNP Data
def read_true_RNP(file_path):

    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    true_time = []
    true_RNP = []

    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            true_time.append(float(values[0]))
            true_RNP.append(float(values[1]))

    return true_time, true_RNP

# Import Noisy RNP Data
def read_noisy_RNP(file_path):

    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    noisy_time = []
    noisy_RNP = []

    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            noisy_time.append(float(values[0]))
            noisy_RNP.append(float(values[1]))

    return noisy_time, noisy_RNP

# Import Smoothed RNP Data
def read_smoothed_RNP(file_path):

    with open(file_path, "r", encoding = "utf-8") as file:
        lines = file.readlines()

    smoothed_time = []
    smoothed_RNP = []

    for line in lines[1:]:
        values = line.split()
        if len(values) >= 2:
            smoothed_time.append(float(values[0]))
            smoothed_RNP.append(float(values[1]))

    return smoothed_time, smoothed_RNP

# Read saved train-test split
def read_train_test_split(split_file_path):
    with open(split_file_path, "r", encoding = "utf-8") as file:
        split_data = json.load(file)
    return split_data

# ----- Function to calculate the slope within a window -----
def calculate_slope(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value

def calculate_slope_through_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0 or np.sum(x**2) == 0:
        raise ValueError("Cannot fit slope through origin: invalid x or y.")

    slope = np.sum(x * y) / np.sum(x ** 2)

    y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        r_value = np.sqrt(max(0.0, 1 - ss_res / ss_tot))
    else:
        r_value = np.nan

    return slope, 0.0, r_value

def fit_fixed_slope_intercept(time_vals, rnp_vals, slope = 0.5):
    time_vals = np.asarray(time_vals, dtype = float)
    rnp_vals = np.asarray(rnp_vals, dtype = float)

    log_t = np.log10(time_vals)
    log_rnp = np.log10(rnp_vals)

    # Best intercept for a fixed slope in least-squares sense
    intercept = np.mean(log_rnp - slope * log_t)
    return intercept

def evaluate_fixed_slope_window(time_vals, rnp_vals, slope = 0.5, point_tolerance = 0.08):
    time_vals = np.asarray(time_vals, dtype = float)
    rnp_vals = np.asarray(rnp_vals, dtype = float)

    intercept = fit_fixed_slope_intercept(time_vals, rnp_vals, slope = slope)

    log_t = np.log10(time_vals)
    log_rnp = np.log10(rnp_vals)
    log_pred = intercept + slope * log_t

    residuals = log_rnp - log_pred

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    center_bias = np.abs(np.mean(residuals))
    inlier_fraction = np.mean(np.abs(residuals) <= point_tolerance)

    ss_res = np.sum((log_rnp - log_pred) ** 2)
    ss_tot = np.sum((log_rnp - np.mean(log_rnp)) ** 2)

    if ss_tot > 0:
        fixed_r2 = 1 - ss_res / ss_tot
    else:
        fixed_r2 = np.nan

    return intercept, rmse, mae, center_bias, inlier_fraction, fixed_r2

def search_best_fixed_slope_interval(
    time_vals,
    rnp_vals,
    slope = 0.5,
    min_points = 8,
    min_log_span = 0.8,
    t_min = None,
    t_max = None,
    point_tolerance = 0.08,
    min_inlier_fraction = 0.70,
    max_center_bias = 0.03,
    min_fixed_r2 = 0.70
    ):

    """
    Search for the best contiguous interval for a fixed slope in log-log space.

    Selection logic:
    1. Fit the intercept for slope = 0.5
    2. Evaluate whether the fitted line lies within the actual data trend
    3. Prefer intervals with:
       - high inlier fraction
       - low RMSE
       - low center bias
       - large log span
       - many points
    """

    time_vals = np.asarray(time_vals, dtype = float)
    rnp_vals = np.asarray(rnp_vals, dtype = float)

    mask = np.isfinite(time_vals) & np.isfinite(rnp_vals) & (time_vals > 0) & (rnp_vals > 0)

    if t_min is not None:
        mask &= time_vals >= t_min
    if t_max is not None:
        mask &= time_vals <= t_max

    time_vals = time_vals[mask]
    rnp_vals = rnp_vals[mask]

    sort_idx = np.argsort(time_vals)
    time_vals = time_vals[sort_idx]
    rnp_vals = rnp_vals[sort_idx]

    n = len(time_vals)
    candidates = []

    for i in range(n):
        for j in range(i + min_points - 1, n):
            t_window = time_vals[i:j+1]
            rnp_window = rnp_vals[i:j+1]

            log_span = np.log10(t_window[-1]) - np.log10(t_window[0])
            if log_span < min_log_span:
                continue

            intercept, rmse, mae, center_bias, inlier_fraction, fixed_r2 = evaluate_fixed_slope_window(
                t_window,
                rnp_window,
                slope = slope,
                point_tolerance = point_tolerance
            )

            candidates.append({
                "start_idx": i,
                "end_idx": j,
                "start_time": t_window[0],
                "end_time": t_window[-1],
                "n_points": len(t_window),
                "log_span": log_span,
                "intercept": intercept,
                "rmse": rmse,
                "mae": mae,
                "center_bias": center_bias,
                "inlier_fraction": inlier_fraction,
                "fixed_r2": fixed_r2
            })

    if not candidates:
        raise ValueError("No valid interval found. Try reducing min_points or min_log_span.")

    # First, try candidates that satisfy all quality conditions
    filtered = [
        c for c in candidates
        if c["inlier_fraction"] >= min_inlier_fraction
        and c["center_bias"] <= max_center_bias
        and (np.isnan(c["fixed_r2"]) or c["fixed_r2"] >= min_fixed_r2)
    ]

    # If none satisfy all conditions, fall back to all candidates
    if len(filtered) == 0:
        filtered = candidates

    # Rank candidates:
    # 1. highest inlier fraction
    # 2. lowest RMSE
    # 3. lowest center bias
    # 4. largest log span
    # 5. most points
    best = sorted(
        filtered,
        key = lambda c: (-c["inlier_fraction"], c["rmse"], c["center_bias"], -c["log_span"], -c["n_points"])
    )[0]

    # Build fitted line
    t_line = np.logspace(np.log10(best["start_time"]), np.log10(best["end_time"]), 200)
    rnp_line = 10 ** (best["intercept"] + slope * np.log10(t_line))

    best["line_t"] = t_line
    best["line_rnp"] = rnp_line
    best["slope"] = slope
    best["n_candidates"] = len(candidates)
    best["n_filtered"] = len(filtered)

    return best

# ----- Main Execution -----
def main():

    method_name = "Linear_GAM"

    # ----- Plot settings -----
    plot_results = True
    plot_flow_regime_scenario = "All"
    plot_noise_level = "50%"
    plot_dataset_ids = [9]   # None = plot all test datasets in All/50%, or use [9] for one dataset only
    save_emf = False

    # True RNP data file path
    true_RNP_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\true_RNP.txt"
    true_time, true_RNP = read_true_RNP(true_RNP_file_path) # days, psia2/cp-d/Mscf
    true_time = np.array(true_time) # days
    true_RNP = np.array(true_RNP) # days
    
    noisy_root_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Noisy RNP Model"
    smoothed_root_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\Test Results"

    split_file_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Smoothing Methods\train_test_split.json"

    output_root_path = rf"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Curve Fit Results\{method_name}"

    flow_regime_scenarios = ["Transient", "Transition", "BDF", "Transient-Transition", "Transition-BDF", "Transient-BDF", "All"]
    noise_levels = ["25%", "50%", "75%"]

    desired_slope = 0.5

    split_data = read_train_test_split(split_file_path)

    pseudotime = np.array(Pseudopressure_Conversion.pseudotime)
    pseudotime_linear = np.array(Pseudopressure_Conversion.pseudotime_linear)

    # if pseudotime has one extra point
    if len(pseudotime_linear) == len(true_time) + 1:
        pseudotime_linear = pseudotime_linear[1:]

    # if pseudotime has one extra point
    if len(pseudotime) == len(true_time) + 1:
        pseudotime = pseudotime[1:]

    for flow_regime_scenario in flow_regime_scenarios:
        for noise_level in noise_levels:

            test_file_ids = split_data[flow_regime_scenario][noise_level]["test_file_ids"]

            output_folder = f"{output_root_path}/{flow_regime_scenario}/{noise_level}/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            summary_rows = []

            for i in test_file_ids:
                print("===================================================")
                print("Method:", method_name)
                print("Scenario:", flow_regime_scenario)
                print("Noise Level:", noise_level)
                print("Dataset:", i)

                noisy_file_path = f"{noisy_root_path}/{flow_regime_scenario}/{noise_level}/noisy_data_{i}.txt"
                smoothed_file_path = f"{smoothed_root_path}/{flow_regime_scenario}/{noise_level}/{method_name}/smoothed_RNP_{i}.txt"

                noisy_time, noisy_RNP = read_noisy_RNP(noisy_file_path)
                smoothed_time, smoothed_RNP = read_smoothed_RNP(smoothed_file_path)

                noisy_time = np.array(noisy_time)
                noisy_RNP = np.array(noisy_RNP)
                smoothed_time = np.array(smoothed_time)
                smoothed_RNP = np.array(smoothed_RNP)

                # Filter up to 5000 days for log-log slope search
                noisy_mask = noisy_time <= 7000
                smoothed_mask = smoothed_time <= 7000

                filtered_noisy_time = noisy_time[noisy_mask]
                filtered_noisy_RNP = noisy_RNP[noisy_mask]
                filtered_smoothed_time = smoothed_time[smoothed_mask]
                filtered_smoothed_RNP = smoothed_RNP[smoothed_mask]
                
                best_noisy = search_best_fixed_slope_interval(
                    filtered_noisy_time,
                    filtered_noisy_RNP,
                    slope = desired_slope,
                    min_points = 8,
                    min_log_span = 0.8,
                    t_min = 0.25,
                    t_max = 7000,
                    point_tolerance = 0.08,
                    min_inlier_fraction = 0.70,
                    max_center_bias = 0.03,
                    min_fixed_r2 = 0.70
                )

                best_smoothed = search_best_fixed_slope_interval(
                    filtered_smoothed_time,
                    filtered_smoothed_RNP,
                    slope = desired_slope,
                    min_points = 8,
                    min_log_span = 0.8,
                    t_min = 0.25,
                    t_max = 7000,
                    point_tolerance = 0.08,
                    min_inlier_fraction = 0.70,
                    max_center_bias = 0.03,
                    min_fixed_r2 = 0.70
                )

                true_start_time = 0
                true_end_time = 4000.0

                # Square-root pseudo-time analysis
                pseudo_linear_true = pseudotime_linear[:len(true_time)]
                pseudo_linear_noisy = pseudotime_linear[:len(noisy_time)]
                pseudo_linear_smoothed = pseudotime_linear[:len(smoothed_time)]

                # Plain pseudo-time arrays for validation
                pseudo_true = pseudotime[:len(true_time)]

                # true_mask_linear = (true_time >= best_true["start_time"]) & (true_time <= best_true["end_time"])
                true_mask_linear = (true_time >= true_start_time) & (true_time <= true_end_time)
                true_linear_superposition_pseudotime = pseudo_linear_true[true_mask_linear]
                true_linear_RNP = true_RNP[true_mask_linear]

                true_sqrt_linear_superposition_pseudotime = np.sqrt(true_linear_superposition_pseudotime)

                noisy_mask_linear = (noisy_time > 0) & (noisy_time <= best_noisy["end_time"])
                smoothed_mask_linear = (smoothed_time > 0) & (smoothed_time <= best_smoothed["end_time"])
                true_mask_linear = (true_time > true_start_time) & (true_time <= true_end_time)

                noisy_linear_superposition_pseudotime = pseudo_linear_noisy[noisy_mask_linear]
                smoothed_linear_superposition_pseudotime = pseudo_linear_smoothed[smoothed_mask_linear]

                noisy_linear_RNP = noisy_RNP[noisy_mask_linear]
                smoothed_linear_RNP = smoothed_RNP[smoothed_mask_linear]

                noisy_sqrt_linear_superposition_pseudotime = np.sqrt(noisy_linear_superposition_pseudotime)
                smoothed_sqrt_linear_superposition_pseudotime = np.sqrt(smoothed_linear_superposition_pseudotime)

                # ----- Validation using plain pseudo-time -----
                true_pseudotime_selected = pseudo_true[true_mask_linear]
                true_sqrt_pseudotime = np.sqrt(true_pseudotime_selected)

                true_linear_slope, true_linear_intercept, true_linear_r_value = calculate_slope_through_origin(
                    true_sqrt_linear_superposition_pseudotime,
                    true_linear_RNP
                )

                noisy_linear_slope, noisy_linear_intercept, noisy_linear_r_value = calculate_slope_through_origin(
                    noisy_sqrt_linear_superposition_pseudotime,
                    noisy_linear_RNP
                )

                smoothed_linear_slope, smoothed_linear_intercept, smoothed_linear_r_value = calculate_slope_through_origin(
                    smoothed_sqrt_linear_superposition_pseudotime,
                    smoothed_linear_RNP
                )

                # ----- Validation slopes using sqrt(pseudotime) -----
                true_pseudo_slope, true_pseudo_intercept, true_pseudo_r_value = calculate_slope_through_origin(
                    true_sqrt_pseudotime,
                    true_linear_RNP
                )

                # Reservoir properties
                T = 212 + 459.67 # Rankine
                h = 100
                phi = 0.05
                miugi = 0.026527595
                cti = Pseudopressure_Conversion.cti

                true_LFP = 4 * (200.8 * T) / (true_linear_slope * np.sqrt(phi * miugi * cti))
                noisy_LFP = 4 * (200.8 * T) / (noisy_linear_slope * np.sqrt(phi * miugi * cti))
                smoothed_LFP = 4 * (200.8 * T) / (smoothed_linear_slope * np.sqrt(phi * miugi * cti))
                
                # ----- Validation LFP using sqrt(pseudotime) -----
                true_pseudo_LFP = 4 * (200.8 * T) / (true_pseudo_slope * np.sqrt(phi * miugi * cti))

                # ----- Optional plotting for All / 50% only -----
                should_plot = (
                    plot_results
                    and flow_regime_scenario == plot_flow_regime_scenario
                    and noise_level == plot_noise_level
                    and (plot_dataset_ids is None or i in plot_dataset_ids)
                )

                if should_plot:
                    plot_folder = f"{output_folder}Plots/"
                    if not os.path.exists(plot_folder):
                        os.makedirs(plot_folder)

                    fitted_noisy_linear_RNP = [
                        noisy_linear_intercept + noisy_linear_slope * noisy_sqrt_linear_superposition_pseudotime[k]
                        for k in range(len(noisy_sqrt_linear_superposition_pseudotime))
                    ]

                    fitted_smoothed_linear_RNP = [
                        smoothed_linear_intercept + smoothed_linear_slope * smoothed_sqrt_linear_superposition_pseudotime[k]
                        for k in range(len(smoothed_sqrt_linear_superposition_pseudotime))
                    ]

                    # Plot 1: log-log RNP vs time
                    plt.figure()
                    plt.plot(noisy_time, noisy_RNP, 'o', label = 'noisy RNP')
                    plt.plot(smoothed_time, smoothed_RNP, 'x', label = 'smoothed RNP')
                    plt.plot(best_noisy["line_t"], best_noisy["line_rnp"], '-', label = 'fitted noisy RNP', linewidth = 3)
                    plt.plot(best_smoothed["line_t"], best_smoothed["line_rnp"], '-', label = 'fitted smoothed RNP', linewidth = 3)
                    plt.xlabel("$t$, day", fontsize = 20)
                    plt.ylabel("$\\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\\cdot$d/Mscf", fontsize = 20)
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.minorticks_on()
                    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
                    plt.grid(True, which = 'major', linestyle = '-', linewidth = 1)
                    plt.grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                    plt.legend(fontsize = 20)
                    plt.tight_layout()
                    plt.show()

                    svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_14.svg"
                    emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_14.emf"
                    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                    convert_svg_to_emf(svg_path, emf_path, inkscape_path = inkscape_path)

                    # Full scale
                    plot_noisy_time = np.sqrt(pseudo_linear_noisy)
                    plot_noisy_RNP = noisy_RNP
                    plot_smoothed_time = np.sqrt(pseudo_linear_smoothed)
                    plot_smoothed_RNP = smoothed_RNP

                    # Plot 2: side-by-side RNP vs square-root linear superposition pseudo-time
                    fig, axes = plt.subplots(1, 2, figsize = (16, 6), sharey = False)

                    # ----- Left: original/full scale -----
                    axes[0].plot(plot_noisy_time, plot_noisy_RNP, 'o', label = 'noisy RNP')
                    axes[0].plot(plot_smoothed_time, plot_smoothed_RNP, 'x', label = 'smoothed RNP')
                    axes[0].plot(noisy_sqrt_linear_superposition_pseudotime, fitted_noisy_linear_RNP, '-', label = 'fitted noisy RNP', linewidth = 3)
                    axes[0].plot(smoothed_sqrt_linear_superposition_pseudotime, fitted_smoothed_linear_RNP, '-', label = 'fitted smoothed RNP', linewidth = 3)
                    axes[0].set_xlabel("$\\sqrt{t_{a,LS}}$, $day^{1/2}$", fontsize = 20)
                    axes[0].set_ylabel("$\\Delta$m(p)/$q_{g}$, $psia^{2}$/cp$\\cdot$d/Mscf", fontsize = 20)
                    axes[0].minorticks_on()
                    axes[0].set_xlim(left = 0)
                    axes[0].set_ylim(bottom = 0)
                    axes[0].tick_params(axis = 'both', which = 'major', labelsize = 18)
                    axes[0].grid(True, which = 'major', linestyle = '-', linewidth = 1)
                    axes[0].grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                    axes[0].legend(fontsize = 14)
                    # axes[0].set_title("Original Scale", fontsize = 18)

                    # ----- Right: filtered scale -----
                    axes[1].plot(plot_noisy_time, plot_noisy_RNP, 'o', label = 'noisy RNP')
                    axes[1].plot(plot_smoothed_time, plot_smoothed_RNP, 'x', label = 'smoothed RNP')
                    axes[1].plot(noisy_sqrt_linear_superposition_pseudotime, fitted_noisy_linear_RNP, '-', label = 'fitted noisy RNP', linewidth = 3)
                    axes[1].plot(smoothed_sqrt_linear_superposition_pseudotime, fitted_smoothed_linear_RNP, '-', label = 'fitted smoothed RNP', linewidth = 3)
                    axes[1].set_xlabel("$\\sqrt{t_{a,LS}}$, $day^{1/2}$", fontsize = 20)
                    axes[1].minorticks_on()
                    axes[1].set_xlim(left = 0, right = 150)
                    axes[1].set_ylim(bottom = 0, top = 0.1e10)
                    axes[1].tick_params(axis = 'both', which = 'major', labelsize = 18)
                    axes[1].grid(True, which = 'major', linestyle = '-', linewidth = 1)
                    axes[1].grid(True, which = 'minor', linestyle = ':', linewidth = 0.5)
                    axes[1].legend(fontsize = 14)
                    # axes[1].set_title("Filtered Scale ($t < 5000$ days)", fontsize = 18)
                    plt.show()

                    svg_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_15.svg"
                    emf_path = r"C:\Users\felix\OneDrive\Documents\Kuliah\2. Master's Texas A&M University\Publications\Conference Paper\Smoothing Paper\Smoothing Simulator\Figures\Fig_15.emf"
                    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"

                    convert_svg_to_emf(svg_path, emf_path, inkscape_path=inkscape_path)

                # ----- Export per dataset -----
                dataset_result_path = f"{output_folder}curve_fit_result_{i}.txt"
                with open(dataset_result_path, "w", encoding = "utf-8") as file:
                    file.write(f"Method\t{method_name}\n")
                    file.write(f"Flow Regime Scenario\t{flow_regime_scenario}\n")
                    file.write(f"Noise Level\t{noise_level}\n")
                    file.write(f"Dataset\t{i}\n\n")

                    file.write("Noisy Log-Log Fixed Slope Fit\n")
                    file.write(f"Slope\t{best_noisy['slope']}\n")
                    file.write(f"Start Time\t{best_noisy['start_time']}\n")
                    file.write(f"End Time\t{best_noisy['end_time']}\n")
                    file.write(f"RMSE\t{best_noisy['rmse']}\n")
                    file.write(f"MAE\t{best_noisy['mae']}\n")
                    file.write(f"Center Bias\t{best_noisy['center_bias']}\n")
                    file.write(f"Inlier Fraction\t{best_noisy['inlier_fraction']}\n")
                    file.write(f"Fixed R2\t{best_noisy['fixed_r2']}\n\n")

                    file.write("Smoothed Log-Log Fixed Slope Fit\n")
                    file.write(f"Slope\t{best_smoothed['slope']}\n")
                    file.write(f"Start Time\t{best_smoothed['start_time']}\n")
                    file.write(f"End Time\t{best_smoothed['end_time']}\n")
                    file.write(f"RMSE\t{best_smoothed['rmse']}\n")
                    file.write(f"MAE\t{best_smoothed['mae']}\n")
                    file.write(f"Center Bias\t{best_smoothed['center_bias']}\n")
                    file.write(f"Inlier Fraction\t{best_smoothed['inlier_fraction']}\n")
                    file.write(f"Fixed R2\t{best_smoothed['fixed_r2']}\n\n")

                    file.write("Square-Root Time Results\n")
                    file.write(f"Noisy Linear Slope\t{noisy_linear_slope}\n")
                    file.write(f"Noisy R Value\t{noisy_linear_r_value}\n")
                    file.write(f"Noisy LFP\t{noisy_LFP}\n")
                    file.write(f"Smoothed Linear Slope\t{smoothed_linear_slope}\n")
                    file.write(f"Smoothed R Value\t{smoothed_linear_r_value}\n")
                    file.write(f"Smoothed LFP\t{smoothed_LFP}\n")

                summary_rows.append([
                    i,
                    best_noisy["start_time"],
                    best_noisy["end_time"],
                    best_noisy["rmse"],
                    best_noisy["inlier_fraction"],
                    best_noisy["fixed_r2"],
                    best_smoothed["start_time"],
                    best_smoothed["end_time"],
                    best_smoothed["rmse"],
                    best_smoothed["inlier_fraction"],
                    best_smoothed["fixed_r2"],
                    noisy_linear_slope,
                    noisy_linear_r_value,
                    noisy_LFP,
                    smoothed_linear_slope,
                    smoothed_linear_r_value,
                    smoothed_LFP
                ])

            # ----- Export summary for one scenario/noise -----
            summary_file_path = f"{output_folder}curve_fit_summary.txt"
            with open(summary_file_path, "w", encoding = "utf-8") as file:
                file.write(
                    "Dataset\t"
                    "Noisy_Start\tNoisy_End\tNoisy_RMSE\tNoisy_InlierFrac\tNoisy_FixedR2\t"
                    "Smoothed_Start\tSmoothed_End\tSmoothed_RMSE\tSmoothed_InlierFrac\tSmoothed_FixedR2\t"
                    "Noisy_LinearSlope\tNoisy_RValue\tNoisy_LFP\t"
                    "Smoothed_LinearSlope\tSmoothed_RValue\tSmoothed_LFP\n"
                )

                for row in summary_rows:
                    file.write("\t".join(str(v) for v in row) + "\n")
    print('True slope = ', true_linear_slope)
    print("True LFP superposition = ", true_LFP)
    print('True LFP pseudotime = ', true_pseudo_LFP)
    print("--- Batch curve fitting complete ---")
    return

main()
