import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----- Savitzky-Golay -----
def savgol(noisy_time, noisy_RNP, window, degree):
    # Work in log-log space
    log_time = np.log(noisy_time)
    log_RNP = np.log(noisy_RNP)

    # Interpolate to uniform log-time grid
    uniform_log_time = np.linspace(np.min(log_time), np.max(log_time), len(log_time))
    interp_function = interp1d(log_time, log_RNP, kind = "linear", fill_value = "extrapolate")
    uniform_log_RNP = interp_function(uniform_log_time)

    # Apply Savitzky-Golay on uniformly spaced log-time data
    smoothed_uniform_log_RNP = savgol_filter(
        uniform_log_RNP,
        window_length = window,
        polyorder = degree,
        mode = "interp"
    )

    # Interpolate back to original log-time points
    back_interp_function = interp1d(
        uniform_log_time,
        smoothed_uniform_log_RNP,
        kind = "linear",
        fill_value = "extrapolate"
    )
    smoothed_log_RNP = back_interp_function(log_time)

    smoothed_RNP = np.exp(smoothed_log_RNP)
    return smoothed_RNP

# ----- LoWeSS -----
def lowess_smooth(noisy_time, noisy_RNP, window, robust = False):
    log_time = np.log(noisy_time)
    log_RNP = np.log(noisy_RNP)

    frac = window / len(noisy_time)
    frac = min(max(frac, 0.01), 1.0)

    if robust == False:
        smoothed_log_RNP = lowess(
            log_RNP,
            log_time,
            frac = frac,
            it = 0,
            return_sorted = False
        )
    else:
        smoothed_log_RNP = lowess(
            log_RNP,
            log_time,
            frac = frac,
            it = 5,
            return_sorted = False
        )

    smoothed_RNP = np.exp(smoothed_log_RNP)
    return smoothed_RNP

# ----- Linear GAM -----
def gam(noisy_time, noisy_RNP, lam_value):
    from pygam import LinearGAM

    X = np.log(noisy_time).reshape(-1,1)
    y = np.log(noisy_RNP)

    gam_model = LinearGAM(lam = lam_value)
    gam_model.fit(X, y)

    smoothed_log_RNP = gam_model.predict(X)
    smoothed_RNP = np.exp(smoothed_log_RNP)

    return smoothed_RNP

# ----- Gaussian Kernel -----
def kernel(noisy_time, noisy_RNP, sigma):
    log_time = np.log(noisy_time)
    log_RNP = np.log(noisy_RNP)

    smoothed_log_RNP = np.zeros(len(log_RNP), dtype = float)

    for i in range(len(log_time)):
        weights = np.exp(-0.5 * ((log_time - log_time[i]) / sigma) ** 2)
        weights = weights / np.sum(weights)
        smoothed_log_RNP[i] = np.sum(weights * log_RNP)

    smoothed_RNP = np.exp(smoothed_log_RNP)
    return smoothed_RNP

# ----- B-Spline -----
def bspline(noisy_time, noisy_RNP, n_splines_value):
    from pygam import LinearGAM

    X = np.log(noisy_time).reshape(-1,1)
    y = np.log(noisy_RNP)

    gam_model = LinearGAM(n_splines = n_splines_value)
    gam_model.fit(X, y)

    smoothed_log_RNP = gam_model.predict(X)
    smoothed_RNP = np.exp(smoothed_log_RNP)

    return smoothed_RNP