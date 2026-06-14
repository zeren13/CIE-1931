
# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

import colour
from colour import MSDS_CMFS

# Optional: Savitzky-Golay if SciPy is installed
try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")
matplotlib.rcParams.update({'font.size': 12})
DIALOG_DECORATOR = getattr(st, 'dialog', None) or getattr(st, 'experimental_dialog', None)
HAS_DIALOG = callable(DIALOG_DECORATOR)

st.title("CIE 1931 - Chromaticity coordinates from emission spectra")

# ----------------- Sidebar (global) -----------------
st.sidebar.header("Global parameters")
wl_min_default = st.sidebar.number_input("Minimum wavelength (nm)", min_value=200, max_value=10000, value=380)
wl_max_default = st.sidebar.number_input("Maximum wavelength (nm)", min_value=200, max_value=10000, value=780)
interp_interval_default = st.sidebar.selectbox("Interpolation interval (nm)", options=[1, 2, 5, 10], index=2)
dpi_save = st.sidebar.number_input("DPI for TIFF export", min_value=72, max_value=1200, value=600)

# ---- Diagram customization ----
st.sidebar.markdown("---")
st.sidebar.subheader("Diagram customization")

plot_title = st.sidebar.text_input("Plot title", value="CIE 1931 chromaticity diagram")
x_axis_label = st.sidebar.text_input("X-axis label", value="x-chromaticity coordinate")
y_axis_label = st.sidebar.text_input("Y-axis label", value="y-chromaticity coordinate")

def _safe_mpl_text(s: str) -> str:
    """Avoid Matplotlib failures caused by invalid mathtext.

    Matplotlib tries to interpret mathtext when it sees '$'. If the user enters
    one '$' by itself, tight_layout can fail.
    """
    s = "" if s is None else str(s)
    if s.count("$") % 2 == 1:
        s = s.replace("$", "")
    return s

title_color = st.sidebar.color_picker("Title color", "#000000")
axes_color = st.sidebar.color_picker("Axes and tick color", "#000000")
locus_label_color = st.sidebar.color_picker("Wavelength label color (numbers)", "#000000")

title_font_family = st.sidebar.selectbox("Title font", options=["sans-serif", "serif", "monospace"], index=0)
title_font_size = st.sidebar.number_input("Title font size", min_value=8, max_value=48, value=14)

tick_font_family = st.sidebar.selectbox("Tick font", options=["sans-serif", "serif", "monospace"], index=0)
tick_font_size = st.sidebar.number_input("Tick size", min_value=6, max_value=24, value=10)

locus_numbers_font_family = st.sidebar.selectbox("Wavelength number font (locus)", options=["sans-serif", "serif", "monospace"], index=0)
locus_numbers_font_size = st.sidebar.number_input("Wavelength number size (locus)", min_value=6, max_value=24, value=8)

axes_linewidth = st.sidebar.slider("Axis spine width", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

show_point_labels = st.sidebar.checkbox("Show labels next to each point", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Legend")
legend_loc = st.sidebar.selectbox("Location", options=[
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center", "center"
], index=0)
legend_font_size = st.sidebar.number_input("Font size", min_value=6, max_value=24, value=8)
legend_ncols = st.sidebar.slider("Columns", min_value=1, max_value=4, value=1)
legend_box_color = st.sidebar.color_picker("Background", "#FFFFFF")
legend_box_alpha = st.sidebar.slider("Background opacity", min_value=0.0, max_value=1.0, value=0.85)
legend_box_linewidth = st.sidebar.slider("Border width", min_value=0.0, max_value=4.0, value=0.8, step=0.1)

# ---- Dominant wavelength / purity ----
st.sidebar.markdown("---")
st.sidebar.subheader("Dominant wavelength / purity")

wp_mode = st.sidebar.selectbox("White point", ["E (0.3333, 0.3333)", "D65 (0.3127, 0.3290)", "Custom"], index=0)
if wp_mode.startswith("E"):
    WHITE_POINT = (0.3333, 0.3333)
elif wp_mode.startswith("D65"):
    WHITE_POINT = (0.3127, 0.3290)
else:
    wp_x = st.sidebar.number_input("White point x", min_value=0.0, max_value=1.0, value=0.3333, step=0.0001, format="%.4f")
    wp_y = st.sidebar.number_input("White point y", min_value=0.0, max_value=1.0, value=0.3333, step=0.0001, format="%.4f")
    WHITE_POINT = (float(wp_x), float(wp_y))

# ---- Advanced analysis ----
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis tools")
show_xyz_columns = st.sidebar.checkbox("Include XYZ and u'v' coordinates", value=True)
show_spectral_metrics = st.sidebar.checkbox("Include spectral metrics", value=True)
show_distance_matrix = st.sidebar.checkbox("Show chromaticity distance matrix", value=True)
show_quality_flags = st.sidebar.checkbox("Show data quality flags", value=True)
show_peak_guides = st.sidebar.checkbox("Show peak/FWHM guides in spectra plot", value=True)

# ----------------- Uploaders -----------------
st.markdown("### Upload data")
st.caption("CSV: one or more files. XLSX: one or more files (all sheets are processed).")

csv_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
xlsx_files = st.file_uploader("Upload one or more Excel files (.xlsx)", type=["xlsx"], accept_multiple_files=True)

# ----------------- Helpers: file reading -----------------
def uploaded_to_bytes(uploaded):
    if uploaded is None:
        return b""
    try:
        return uploaded.getvalue()
    except Exception:
        try:
            uploaded.seek(0)
        except Exception:
            pass
        return uploaded.read()

def read_csv_flexible_bytes(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise ValueError("The file is empty or could not be read.")
    bio = io.BytesIO(raw)

    trials = [
        (";", ","), (";", "."),
        (",", ","), (",", "."),
        (None, ","), (None, "."),
    ]
    last_err = None
    for sep, dec in trials:
        try:
            bio.seek(0)
            df = pd.read_csv(bio, sep=sep, engine="python" if sep is None else "c", decimal=dec)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e

    try:
        bio.seek(0)
        df = pd.read_csv(bio, sep=r"\s+", engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception as e:
        last_err = e

    raise ValueError(f"Could not read the CSV file (separator/decimal format). Last error: {last_err}")

@st.cache_data(show_spinner=False)
def load_csv_df(raw: bytes) -> pd.DataFrame:
    return read_csv_flexible_bytes(raw)

@st.cache_data(show_spinner=False)
def list_excel_sheets(raw: bytes):
    xls = pd.ExcelFile(io.BytesIO(raw))
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def load_excel_sheet(raw: bytes, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet)

# ----------------- Helpers: columns (auto-detection) -----------------
_WL_HINTS = [
    "wavelength", "wavelength (nm)", "lambda", "wl", "nm",
    "wavelength", "wavelength", "wavelength", "wave"
]
_INT_HINTS = [
    "intensity", "intensity (a.u.)", "a.u.", "au", "counts", "cps", "signal",
    "emission", "fluorescence", "fluorescencia"
]

def _norm_colname(c):
    return str(c).strip().lower().replace("\ufeff", "").replace("\u00b5", "u")

def guess_columns(df: pd.DataFrame):
    cols = list(df.columns)
    if len(cols) < 2:
        return None, None

    norm = [_norm_colname(c) for c in cols]

    wl_idx = None
    int_idx = None

    for i, n in enumerate(norm):
        if any(h in n for h in _WL_HINTS):
            wl_idx = i
            break
    for i, n in enumerate(norm):
        if any(h in n for h in _INT_HINTS):
            int_idx = i
            break

    if wl_idx is None or int_idx is None or wl_idx == int_idx:
        numeric_scores = []
        for i, c in enumerate(cols):
            s = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            numeric_scores.append((i, float(s.notna().mean())))
        numeric_scores.sort(key=lambda x: x[1], reverse=True)
        if wl_idx is None and numeric_scores:
            wl_idx = numeric_scores[0][0]
        if int_idx is None and len(numeric_scores) > 1:
            int_idx = numeric_scores[1][0]
        if wl_idx == int_idx and len(numeric_scores) > 1:
            int_idx = numeric_scores[1][0]

    return cols[wl_idx] if wl_idx is not None else None, cols[int_idx] if int_idx is not None else None

def to_numeric_series(s: pd.Series):
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

# ----------------- CMF arrays (no deepcopy) -----------------
@st.cache_data(show_spinner=False)
def cmfs_arrays():
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

    if hasattr(cmfs, "wavelengths"):
        wls = np.asarray(cmfs.wavelengths, dtype=float)
    elif hasattr(cmfs, "domain"):
        wls = np.asarray(cmfs.domain, dtype=float)
    else:
        wls = np.asarray(cmfs.index, dtype=float)

    vals = np.asarray(cmfs.values, dtype=float)
    if vals.shape[0] == 3 and vals.shape[1] == wls.shape[0]:
        vals = vals.T
    return wls, vals[:, 0], vals[:, 1], vals[:, 2]

CMF_WLS, CMF_X, CMF_Y, CMF_Z = cmfs_arrays()
CMF_MIN, CMF_MAX = float(np.min(CMF_WLS)), float(np.max(CMF_WLS))

@st.cache_data(show_spinner=False)
def locus_arrays(wl_start=380, wl_end=780):
    m = (CMF_WLS >= wl_start) & (CMF_WLS <= wl_end)
    wls = CMF_WLS[m]
    xbar, ybar, zbar = CMF_X[m], CMF_Y[m], CMF_Z[m]
    den = xbar + ybar + zbar
    den = np.where(den == 0, np.nan, den)
    x = xbar / den
    y = ybar / den
    xy = np.column_stack([x, y])
    ok = np.isfinite(xy).all(axis=1)
    return wls[ok].astype(float), xy[ok]

LOCUS_WLS_F, LOCUS_XY = locus_arrays(380, 780)

# ----------------- Geometry: dominant wavelength / purity -----------------
def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def _ray_segment_intersection(w, v, p0, p1, eps=1e-12):
    s = p1 - p0
    rxs = _cross2(v, s)
    if abs(rxs) < eps:
        return None
    q_p = p0 - w
    t = _cross2(q_p, s) / rxs
    u = _cross2(q_p, v) / rxs
    if t >= 0 and 0 <= u <= 1:
        return t, u
    return None

def dominant_wavelength_and_purity(x, y, white_point=(0.3333, 0.3333)):
    w = np.array([white_point[0], white_point[1]], dtype=float)
    p = np.array([x, y], dtype=float)
    v = p - w
    if not np.isfinite(v).all() or (abs(v[0]) < 1e-15 and abs(v[1]) < 1e-15):
        return np.nan, np.nan, "Undefined"

    best = None  # (t, kind, i, u, q)
    for i in range(len(LOCUS_XY) - 1):
        p0 = LOCUS_XY[i]
        p1 = LOCUS_XY[i + 1]
        hit = _ray_segment_intersection(w, v, p0, p1)
        if hit is None:
            continue
        t, u = hit
        if best is None or t < best[0]:
            best = (t, "locus", i, u, w + t * v)

    # purple line
    p0 = LOCUS_XY[-1]
    p1 = LOCUS_XY[0]
    hit = _ray_segment_intersection(w, v, p0, p1)
    if hit is not None:
        t, u = hit
        if best is None or t < best[0]:
            best = (t, "purple", -1, u, w + t * v)

    if best is None:
        return np.nan, np.nan, "No intersection"

    _, kind, i, u, q = best
    d_wp_p = float(np.linalg.norm(p - w))
    d_wp_q = float(np.linalg.norm(q - w))
    purity = (d_wp_p / d_wp_q) * 100.0 if d_wp_q > 0 else np.nan

    if kind == "locus":
        wl0 = float(LOCUS_WLS_F[i])
        wl1 = float(LOCUS_WLS_F[i + 1])
        wl = wl0 + u * (wl1 - wl0)
        return wl, purity, "Dominant"

    # complementary wavelength (opposite ray)
    v2 = -v
    best2 = None
    for j in range(len(LOCUS_XY) - 1):
        p0 = LOCUS_XY[j]
        p1 = LOCUS_XY[j + 1]
        hit2 = _ray_segment_intersection(w, v2, p0, p1)
        if hit2 is None:
            continue
        t2, u2 = hit2
        if best2 is None or t2 < best2[0]:
            best2 = (t2, j, u2)

    if best2 is None:
        return np.nan, purity, "Purple (no complementary)"

    _, j, u2 = best2
    wl0 = float(LOCUS_WLS_F[j])
    wl1 = float(LOCUS_WLS_F[j + 1])
    wl_comp = wl0 + u2 * (wl1 - wl0)
    return wl_comp, purity, "Complementary"

# ----------------- Numerical integration: spectrum -> color coordinates -----------------
def spectrum_to_xyz_xy(wl_grid, intensity_grid):
    xbar = np.interp(wl_grid, CMF_WLS, CMF_X)
    ybar = np.interp(wl_grid, CMF_WLS, CMF_Y)
    zbar = np.interp(wl_grid, CMF_WLS, CMF_Z)

    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    X = float(integrate(intensity_grid * xbar, wl_grid))
    Y = float(integrate(intensity_grid * ybar, wl_grid))
    Z = float(integrate(intensity_grid * zbar, wl_grid))

    S = X + Y + Z
    if not np.isfinite(S) or S <= 0:
        raise ValueError("Invalid XYZ values (intensities are near zero in the selected range).")
    return X, Y, Z, X / S, Y / S

def spectrum_to_xy(wl_grid, intensity_grid):
    _, _, _, x, y = spectrum_to_xyz_xy(wl_grid, intensity_grid)
    return x, y

def xy_to_upvp(x, y):
    den = (-2.0 * x) + (12.0 * y) + 3.0
    if not np.isfinite(den) or abs(den) < 1e-15:
        return np.nan, np.nan
    return (4.0 * x) / den, (9.0 * y) / den

def estimate_cct_mccamy(x, y):
    """Approximate CCT in kelvin using McCamy's formula; useful as a quick guide."""
    den = 0.1858 - y
    if not np.isfinite(den) or abs(den) < 1e-15:
        return np.nan
    n = (x - 0.3320) / den
    cct = (-449.0 * n**3) + (3525.0 * n**2) - (6823.3 * n) + 5520.33
    return float(cct) if np.isfinite(cct) and cct > 0 else np.nan

def spectral_metrics(wl_grid, intensity_grid):
    wl = np.asarray(wl_grid, dtype=float)
    it = np.asarray(intensity_grid, dtype=float)
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")

    def _linear_crossing(x0, y0, x1, y1, target):
        if abs(y1 - y0) < 1e-15:
            return float(x0)
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

    area = float(integrate(it, wl))
    abs_area = float(integrate(np.abs(it), wl))
    peak_idx = int(np.nanargmax(it))
    peak_wl = float(wl[peak_idx])
    peak_intensity = float(it[peak_idx])
    centroid = float(integrate(wl * it, wl) / area) if area > 0 else np.nan

    half_max = peak_intensity / 2.0
    above = np.where(it >= half_max)[0]
    fwhm = np.nan
    left_half = np.nan
    right_half = np.nan
    if len(above) >= 2 and peak_intensity > 0:
        left_i = int(above[0])
        right_i = int(above[-1])

        if left_i > 0:
            left_half = _linear_crossing(wl[left_i - 1], it[left_i - 1], wl[left_i], it[left_i], half_max)
        else:
            left_half = float(wl[left_i])

        if right_i < len(it) - 1:
            right_half = _linear_crossing(wl[right_i], it[right_i], wl[right_i + 1], it[right_i + 1], half_max)
        else:
            right_half = float(wl[right_i])

        fwhm = float(right_half - left_half)

    negative_fraction = float(np.mean(it < 0)) if len(it) else np.nan
    dynamic_range = float(peak_intensity / np.nanmedian(np.abs(it))) if np.nanmedian(np.abs(it)) > 0 else np.nan

    return {
        "Peak_nm": peak_wl,
        "Peak_intensity": peak_intensity,
        "Centroid_nm": centroid,
        "FWHM_nm": fwhm,
        "FWHM_left_nm": left_half,
        "FWHM_right_nm": right_half,
        "Integrated_area": area,
        "Integrated_abs_area": abs_area,
        "Negative_fraction_%": negative_fraction * 100.0,
        "Dynamic_range_peak_to_median": dynamic_range,
    }

def quality_flags(metrics, cfg):
    flags = []
    if np.isfinite(metrics.get("Negative_fraction_%", np.nan)) and metrics["Negative_fraction_%"] > 0:
        flags.append("negative intensities")
    if np.isfinite(metrics.get("Peak_nm", np.nan)):
        edge_margin = max(2.0 * float(cfg["interp"]), 1.0)
        if metrics["Peak_nm"] <= cfg["wl_min"] + edge_margin or metrics["Peak_nm"] >= cfg["wl_max"] - edge_margin:
            flags.append("peak near range edge")
    if np.isfinite(metrics.get("FWHM_nm", np.nan)) and metrics["FWHM_nm"] <= float(cfg["interp"]):
        flags.append("FWHM near interpolation limit")
    if not flags:
        flags.append("ok")
    return "; ".join(flags)

# ----------------- Preprocessing -----------------
def preprocess_spectrum(df, wl_col, int_col,
                        wl_min_local, wl_max_local, interp_interval_local,
                        clip_negative=False,
                        baseline_subtract_min=False,
                        smooth_method="None",
                        smooth_window=11,
                        smooth_poly=3,
                        normalize_mode="None"):
    out = pd.DataFrame()
    out["wavelength"] = to_numeric_series(df[wl_col])
    out["intensity"] = to_numeric_series(df[int_col])
    out = out.dropna()
    out = out[(out["wavelength"] >= wl_min_local) & (out["wavelength"] <= wl_max_local)].copy()
    if out.empty:
        raise ValueError("No data found in the selected wavelength range.")
    out = out.sort_values("wavelength")
    out = out.groupby("wavelength", as_index=False)["intensity"].mean()

    if baseline_subtract_min:
        out["intensity"] = out["intensity"] - float(out["intensity"].min())

    if clip_negative:
        out["intensity"] = out["intensity"].clip(lower=0)

    wl_grid = np.arange(wl_min_local, wl_max_local + 1e-9, interp_interval_local, dtype=float)
    intensity_grid = np.interp(
        wl_grid,
        out["wavelength"].values.astype(float),
        out["intensity"].values.astype(float),
        left=0.0, right=0.0
    )

    if smooth_method == "Moving average":
        w = int(max(3, smooth_window))
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w, dtype=float) / w
        intensity_grid = np.convolve(intensity_grid, kernel, mode="same")
    elif smooth_method == "Savitzky-Golay" and _HAS_SCIPY:
        n = int(len(intensity_grid))
        w = min(int(max(5, smooth_window)), n if n % 2 == 1 else n - 1)
        if w % 2 == 0:
            w -= 1
        p = int(max(2, smooth_poly))
        if w >= 3 and p >= w:
            p = w - 1
        if w >= 3 and p < w:
            intensity_grid = savgol_filter(intensity_grid, window_length=w, polyorder=p, mode="interp")

    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    if normalize_mode == "Max = 1":
        m = float(np.max(np.abs(intensity_grid)))
        if m > 0:
            intensity_grid = intensity_grid / m
    elif normalize_mode == "Area = 1":
        area = float(integrate(np.abs(intensity_grid), wl_grid))
        if area > 0:
            intensity_grid = intensity_grid / area

    if float(np.max(np.abs(intensity_grid))) <= 0:
        raise ValueError("Zero intensity in the selected range (check columns and range).")

    return wl_grid, intensity_grid

# ----------------- Build datasets -----------------
datasets = []

if csv_files:
    for i, f in enumerate(csv_files):
        raw = uploaded_to_bytes(f)
        if not raw:
            continue
        try:
            df = load_csv_df(raw)
            datasets.append({"id": f"csv_{i}", "name": f.name, "df": df})
        except Exception as e:
            st.error(f"Error reading CSV {f.name}: {e}")

if xlsx_files:
    for j, f in enumerate(xlsx_files):
        raw = uploaded_to_bytes(f)
        if not raw:
            continue
        try:
            sheets = list_excel_sheets(raw)
        except Exception as e:
            st.error(f"Error reading Excel {f.name}: {e}")
            continue
        for k, sh in enumerate(sheets):
            try:
                df = load_excel_sheet(raw, sh)
                datasets.append({"id": f"xlsx_{j}_{k}", "name": f"{f.name} - {sh}", "df": df})
            except Exception as e:
                st.error(f"Error reading {f.name} | sheet {sh}: {e}")

# ----------------- Per-dataset configuration -----------------
def init_cfg(ds):
    df = ds["df"]
    cols = list(df.columns)
    wl_guess, int_guess = guess_columns(df)
    return {
        "label": ds["name"],
        "color": "#000000",
        "marker": "o",
        "size": 120,
        "wl_col": wl_guess if wl_guess in cols else (cols[0] if cols else None),
        "int_col": int_guess if int_guess in cols else (cols[1] if len(cols) > 1 else (cols[0] if cols else None)),
        "wl_min": int(wl_min_default),
        "wl_max": int(wl_max_default),
        "interp": int(interp_interval_default),
        "baseline_subtract_min": False,
        "clip_negative": False,
        "smooth_method": "None",
        "smooth_window": 11,
        "smooth_poly": 3,
        "normalize": "None",
    }

def get_cfg(ds):
    key = f"cfg_{ds['id']}"
    if key not in st.session_state:
        st.session_state[key] = init_cfg(ds)
    cfg = st.session_state[key]
    cols = list(ds["df"].columns)
    if cfg.get("wl_col") not in cols and cols:
        cfg["wl_col"] = cols[0]
    if cfg.get("int_col") not in cols and cols:
        cfg["int_col"] = cols[1] if len(cols) > 1 else cols[0]
    st.session_state[key] = cfg
    return cfg


def render_cfg_form(ds, cfg, key_prefix="dlg_"):
    """Per-sample configuration form.
    key_prefix avoids widget-key collisions and lets Streamlit manage state reliably.
    """
    ds_id = ds["id"]
    keyp = f"{key_prefix}{ds_id}_"
    cols = list(ds["df"].columns)

    st.markdown("Columns")
    wl_col = st.selectbox("Wavelength column", options=cols,
                          index=cols.index(cfg["wl_col"]) if cfg["wl_col"] in cols else 0,
                          key=f"{keyp}wlcol")
    int_default_index = cols.index(cfg["int_col"]) if cfg["int_col"] in cols else min(1, len(cols) - 1)
    int_col = st.selectbox("Intensity column", options=cols,
                           index=int_default_index,
                           key=f"{keyp}intcol")

    st.markdown("Point style")
    label = st.text_input("Label", value=cfg["label"], key=f"{keyp}label")
    color = st.color_picker("Color", value=cfg["color"], key=f"{keyp}color")
    marker_opts = ['o','s','^','x','D','*','v']
    marker = st.selectbox("Marker", options=marker_opts,
                          index=marker_opts.index(cfg["marker"]) if cfg["marker"] in marker_opts else 0,
                          key=f"{keyp}marker")
    size = st.slider("Size (s)", min_value=20, max_value=500, value=int(cfg["size"]), key=f"{keyp}size")

    st.markdown("Calculation range")
    wl_min = st.number_input("Min wavelength (nm)", min_value=200, max_value=10000, value=int(cfg["wl_min"]), key=f"{keyp}wlmin")
    wl_max = st.number_input("Max wavelength (nm)", min_value=200, max_value=10000, value=int(cfg["wl_max"]), key=f"{keyp}wlmax")
    interp_opts = [1,2,5,10]
    interp = st.selectbox("Interval (nm)", options=interp_opts,
                          index=interp_opts.index(int(cfg["interp"])) if int(cfg["interp"]) in interp_opts else 1,
                          key=f"{keyp}interp")

    st.markdown("Preprocessing")
    baseline_subtract_min = st.checkbox("Baseline: subtract minimum (shift to 0)", value=bool(cfg["baseline_subtract_min"]), key=f"{keyp}base")
    clip_negative = st.checkbox("Clip negative intensities to 0", value=bool(cfg["clip_negative"]), key=f"{keyp}clip")

    smooth_options = ["None", "Moving average"] + (["Savitzky-Golay"] if _HAS_SCIPY else [])
    smooth_method = st.selectbox("Smoothing", options=smooth_options,
                                 index=smooth_options.index(cfg["smooth_method"]) if cfg["smooth_method"] in smooth_options else 0,
                                 key=f"{keyp}smooth")
    smooth_window = st.slider("Window (smoothing)", min_value=3, max_value=101, value=int(cfg["smooth_window"]), step=2, key=f"{keyp}win")
    smooth_poly = st.slider("Polynomial order (Savitzky-Golay)", min_value=2, max_value=7, value=int(cfg["smooth_poly"]), step=1, key=f"{keyp}poly")
    normalize = st.selectbox("Normalization", options=["None", "Max = 1", "Area = 1"],
                             index=["None","Max = 1","Area = 1"].index(cfg["normalize"]) if cfg["normalize"] in ["None","Max = 1","Area = 1"] else 0,
                             key=f"{keyp}norm")

    return {
        "label": label,
        "color": color,
        "marker": marker,
        "size": int(size),
        "wl_col": wl_col,
        "int_col": int_col,
        "wl_min": int(wl_min),
        "wl_max": int(wl_max),
        "interp": int(interp),
        "baseline_subtract_min": bool(baseline_subtract_min),
        "clip_negative": bool(clip_negative),
        "smooth_method": smooth_method,
        "smooth_window": int(smooth_window),
        "smooth_poly": int(smooth_poly),
        "normalize": normalize,
    }


# ----------------- Per-sample configuration (modal) -----------------
if "open_cfg_id" not in st.session_state:
    st.session_state["open_cfg_id"] = None

def request_open_config(ds_id: str):
    st.session_state["open_cfg_id"] = ds_id
    st.rerun()

def handle_config_dialog(datasets_by_id):
    """If a sample is selected, open the dialog and stop the rest of the script.
    This prevents the app from calculating with stale config during the same run.
    """
    ds_id = st.session_state.get("open_cfg_id", None)
    if not ds_id:
        return

    ds = datasets_by_id.get(ds_id)
    if ds is None:
        st.session_state["open_cfg_id"] = None
        return

    cfg = get_cfg(ds)
    title = f"Configure: {ds['name']}"

    if HAS_DIALOG:
        @DIALOG_DECORATOR(title)
        def _dlg():
            with st.form(key=f"form_{ds_id}"):
                new_cfg = render_cfg_form(ds, cfg, key_prefix="dlg_")
                cA, cB = st.columns(2)
                with cA:
                    submitted = st.form_submit_button("Save")
                with cB:
                    cancelled = st.form_submit_button("Cancel")
            if submitted:
                st.session_state[f"cfg_{ds_id}"] = new_cfg
                st.session_state["open_cfg_id"] = None
                st.rerun()
            if cancelled:
                st.session_state["open_cfg_id"] = None
                st.rerun()

        _dlg()
        st.stop()
    else:
        with st.expander(title, expanded=True):
            new_cfg = render_cfg_form(ds, cfg, key_prefix="dlg_")
            cA, cB = st.columns(2)
            if cA.button("Save", key=f"save_{ds_id}"):
                st.session_state[f"cfg_{ds_id}"] = new_cfg
                st.session_state["open_cfg_id"] = None
                st.rerun()
            if cB.button("Cancel", key=f"cancel_{ds_id}"):
                st.session_state["open_cfg_id"] = None
                st.rerun()
        st.stop()

datasets_by_id = {ds['id']: ds for ds in datasets}
handle_config_dialog(datasets_by_id)

# ----------------- CIE figure
fig, ax = plt.subplots(figsize=(7, 7))
try:
    try:
        fig_cie, ax_cie = colour.plotting.plot_chromaticity_diagram_CIE1931(show=False)
    except TypeError:
        fig_cie, ax_cie = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
    ax.clear()
    plt.close(fig)
    fig, ax = fig_cie, ax_cie
except Exception:
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)

fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.12)

for txt in list(ax.texts):
    try:
        txt.set_clip_on(False)
        txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
    except Exception:
        pass
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_clip_on(False)

fontprop_title = FontProperties(family=title_font_family, size=title_font_size)
fontprop_ticks = FontProperties(family=tick_font_family, size=tick_font_size)
fontprop_locus = FontProperties(family=locus_numbers_font_family, size=locus_numbers_font_size)

try:
    t = ax.set_title(_safe_mpl_text(plot_title), color=title_color)
    t.set_fontproperties(fontprop_title)
except Exception:
    ax.set_title(_safe_mpl_text(plot_title), color=title_color)

try:
    tx = ax.set_xlabel(_safe_mpl_text(x_axis_label), color=axes_color)
    ty = ax.set_ylabel(_safe_mpl_text(y_axis_label), color=axes_color)
    tx.set_fontproperties(fontprop_ticks)
    ty.set_fontproperties(fontprop_ticks)
except Exception:
    ax.set_xlabel(_safe_mpl_text(x_axis_label), color=axes_color)
    ax.set_ylabel(_safe_mpl_text(y_axis_label), color=axes_color)

for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    try:
        lbl.set_fontproperties(fontprop_ticks)
        lbl.set_color(axes_color)
        lbl.set_clip_on(False)
    except Exception:
        pass

for txt in list(ax.texts):
    try:
        txt.set_clip_on(False)
        txt.set_fontproperties(fontprop_locus)
        txt.set_color(locus_label_color)
    except Exception:
        pass

try:
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
except Exception:
    pass

# ----------------- Process -----------------
results = []
spectra_to_plot = []

if datasets:
    st.markdown("### Samples")
    st.caption("Per-sample options: Configure button (popup dialog). Spectra are shown in a separate expander.")

    for ds in datasets:
        cfg = get_cfg(ds)

        c1, c2, c3 = st.columns([6, 2, 4])
        with c1:
            st.write(ds["name"])
        with c2:
            if st.button("Configure", key=f"btn_cfg_{ds['id']}"):
                request_open_config(ds["id"])
        with c3:
            st.write(f"Wavelength: {cfg['wl_min']}-{cfg['wl_max']} nm | interp: {cfg['interp']} nm")

        try:
            df = ds["df"]
            if cfg["wl_min"] >= cfg["wl_max"]:
                raise ValueError("Minimum wavelength must be lower than maximum wavelength.")
            if cfg["wl_min"] < CMF_MIN or cfg["wl_max"] > CMF_MAX:
                raise ValueError(f"The range must be within the CMF domain: {CMF_MIN:.0f}-{CMF_MAX:.0f} nm.")
            if cfg["wl_col"] not in df.columns or cfg["int_col"] not in df.columns:
                raise ValueError("Selected columns do not exist in this file/sheet.")

            wl_grid, intensity_grid = preprocess_spectrum(
                df, cfg["wl_col"], cfg["int_col"],
                cfg["wl_min"], cfg["wl_max"], cfg["interp"],
                clip_negative=cfg["clip_negative"],
                baseline_subtract_min=cfg["baseline_subtract_min"],
                smooth_method=cfg["smooth_method"],
                smooth_window=cfg["smooth_window"],
                smooth_poly=cfg["smooth_poly"],
                normalize_mode=cfg["normalize"],
            )

            X_val, Y_val, Z_val, x_val, y_val = spectrum_to_xyz_xy(wl_grid, intensity_grid)
            XYZ_sum = X_val + Y_val + Z_val
            X_norm = X_val / XYZ_sum
            Y_norm = Y_val / XYZ_sum
            Z_norm = Z_val / XYZ_sum
            up_val, vp_val = xy_to_upvp(x_val, y_val)
            cct_val = estimate_cct_mccamy(x_val, y_val)
            spec_metrics = spectral_metrics(wl_grid, intensity_grid)
            flags = quality_flags(spec_metrics, cfg)
            wl_dom, purity, wl_kind = dominant_wavelength_and_purity(x_val, y_val, white_point=WHITE_POINT)

            ax.scatter([x_val], [y_val], c=[cfg["color"]], marker=cfg["marker"], s=cfg["size"], label=cfg["label"])
            if show_point_labels:
                ax.text(x_val + 0.008, y_val, cfg["label"],
                        fontsize=tick_font_size, fontfamily=tick_font_family,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))

            row = {
                "Label": cfg["label"],
                "x": float(x_val),
                "y": float(y_val),
                "Wavelength_nm": (float(wl_dom) if np.isfinite(wl_dom) else np.nan),
                "Wavelength_type": wl_kind,
                "Excitation_purity_%": (float(purity) if np.isfinite(purity) else np.nan),
                "wl_min_nm": int(cfg["wl_min"]),
                "wl_max_nm": int(cfg["wl_max"]),
                "interp_nm": int(cfg["interp"]),
                "baseline_subtract_min": bool(cfg["baseline_subtract_min"]),
                "clip_negative": bool(cfg["clip_negative"]),
                "smooth_method": cfg["smooth_method"],
                "smooth_window": int(cfg["smooth_window"]),
                "smooth_poly": int(cfg["smooth_poly"]),
                "normalize": cfg["normalize"],
                "wl_column": cfg["wl_col"],
                "intensity_column": cfg["int_col"],
            }

            if show_xyz_columns:
                row.update({
                    "X": float(X_val),
                    "Y": float(Y_val),
                    "Z": float(Z_val),
                    "X_norm": float(X_norm),
                    "Y_norm": float(Y_norm),
                    "Z_norm": float(Z_norm),
                    "u_prime": float(up_val) if np.isfinite(up_val) else np.nan,
                    "v_prime": float(vp_val) if np.isfinite(vp_val) else np.nan,
                    "CCT_McCamy_K": float(cct_val) if np.isfinite(cct_val) else np.nan,
                })

            if show_spectral_metrics:
                row.update(spec_metrics)

            if show_quality_flags:
                row["Quality_flags"] = flags

            results.append(row)

            spectra_to_plot.append({
                "label": cfg["label"],
                "wl": wl_grid,
                "it": intensity_grid,
                "color": cfg["color"],
                "peak_nm": spec_metrics["Peak_nm"],
                "fwhm_left_nm": spec_metrics["FWHM_left_nm"],
                "fwhm_right_nm": spec_metrics["FWHM_right_nm"],
                "half_max": spec_metrics["Peak_intensity"] / 2.0,
            })

            st.success(f"{cfg['label']}: x={x_val:.4f}, y={y_val:.4f} | peak={spec_metrics['Peak_nm']:.1f} nm | wavelength ({wl_kind})={wl_dom:.1f} nm | purity={purity:.1f}%")
        except Exception as e:
            st.error(f"{cfg['label']}: {e}")
else:
    st.info("Upload files to begin.")

# ----------------- Outputs -----------------
if results:
    try:
        legend = ax.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_ncols)
        frame = legend.get_frame()
        frame.set_facecolor(legend_box_color)
        frame.set_alpha(legend_box_alpha)
        frame.set_linewidth(legend_box_linewidth)
    except Exception:
        pass

    results_df = pd.DataFrame(results)

    st.markdown("### Results summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", len(results_df))
    m2.metric("Mean x", f"{results_df['x'].mean():.4f}")
    m3.metric("Mean y", f"{results_df['y'].mean():.4f}")
    if "Peak_nm" in results_df.columns:
        m4.metric("Mean peak", f"{results_df['Peak_nm'].mean():.1f} nm")
    else:
        m4.metric("White point", f"{WHITE_POINT[0]:.4f}, {WHITE_POINT[1]:.4f}")

    tab_cie, tab_spectra, tab_table, tab_compare, tab_export = st.tabs([
        "CIE diagram", "Spectra", "Table", "Compare", "Export"
    ])

    with tab_cie:
        st.markdown("### CIE 1931 diagram")
        try:
            plt.tight_layout()
        except Exception:
            pass
        st.pyplot(fig)

    with tab_spectra:
        st.markdown("### Emission spectra processed for calculation")
        if spectra_to_plot:
            fig_s, ax_s = plt.subplots(figsize=(8, 4.5))
            for s in spectra_to_plot:
                ax_s.plot(s["wl"], s["it"], label=s["label"], color=s["color"])
                if show_peak_guides:
                    ax_s.axvline(s["peak_nm"], color=s["color"], linestyle=":", linewidth=1.0, alpha=0.8)
                    if np.isfinite(s["fwhm_left_nm"]) and np.isfinite(s["fwhm_right_nm"]):
                        ax_s.hlines(s["half_max"], s["fwhm_left_nm"], s["fwhm_right_nm"],
                                    color=s["color"], linestyle="--", linewidth=1.0, alpha=0.8)
            ax_s.set_xlabel("Wavelength (nm)")
            ax_s.set_ylabel("Intensity (a.u.)")
            ax_s.grid(alpha=0.25)
            try:
                ax_s.legend(fontsize=8, loc="best")
            except Exception:
                pass
            st.pyplot(fig_s)
        else:
            st.info("No spectra to display.")

    with tab_table:
        st.markdown("### Coordinate and spectral table")
        st.dataframe(results_df, use_container_width=True)

    with tab_compare:
        st.markdown("### Sample comparison")
        if show_distance_matrix and len(results_df) > 1:
            labels = results_df["Label"].astype(str).tolist()
            xy = results_df[["x", "y"]].to_numpy(dtype=float)
            dist = np.zeros((len(xy), len(xy)), dtype=float)
            for i in range(len(xy)):
                for j in range(len(xy)):
                    dist[i, j] = float(np.linalg.norm(xy[i] - xy[j]))
            dist_df = pd.DataFrame(dist, index=labels, columns=labels)
            st.markdown("Chromaticity distance in CIE 1931 xy space")
            st.dataframe(dist_df.style.format("{:.5f}"), use_container_width=True)
        elif len(results_df) <= 1:
            st.info("Upload at least two samples to compare chromaticity distances.")
        else:
            st.info("Enable the distance matrix in the sidebar to see pairwise comparisons.")

        if {"Peak_nm", "FWHM_nm", "Centroid_nm"}.issubset(results_df.columns):
            st.markdown("Spectral metrics overview")
            overview_cols = ["Label", "Peak_nm", "Centroid_nm", "FWHM_nm", "Integrated_area"]
            st.dataframe(results_df[overview_cols], use_container_width=True)

    with tab_export:
        st.markdown("### Downloads")
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button("Download CSV table", data=csv_buf.getvalue(), file_name="CIE1931_coordinates.csv", mime="text/csv")

        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="results", index=False)
            if show_distance_matrix and len(results_df) > 1:
                labels = results_df["Label"].astype(str).tolist()
                xy = results_df[["x", "y"]].to_numpy(dtype=float)
                dist = np.zeros((len(xy), len(xy)), dtype=float)
                for i in range(len(xy)):
                    for j in range(len(xy)):
                        dist[i, j] = float(np.linalg.norm(xy[i] - xy[j]))
                pd.DataFrame(dist, index=labels, columns=labels).to_excel(writer, sheet_name="xy_distances")
        xlsx_buf.seek(0)
        st.download_button(
            "Download Excel workbook",
            data=xlsx_buf.getvalue(),
            file_name="CIE1931_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="tiff", dpi=dpi_save)
        img_buf.seek(0)
        st.download_button("Download TIFF diagram", data=img_buf.getvalue(), file_name="CIE1931_diagram.tiff", mime="image/tiff")
