
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

# Opcional: Savitzky–Golay si SciPy está instalado
try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")
matplotlib.rcParams.update({'font.size': 12})

st.title("CIE 1931 — Coordenadas cromáticas desde espectros de emisión")

# ----------------- Sidebar: parámetros globales -----------------
st.sidebar.header("Parámetros globales")
wl_min_default = st.sidebar.number_input("λ mínimo (nm)", min_value=200, max_value=10000, value=380)
wl_max_default = st.sidebar.number_input("λ máximo (nm)", min_value=200, max_value=10000, value=780)
interp_interval_default = st.sidebar.selectbox("Intervalo de interpolación (nm)", options=[1, 2, 5, 10], index=2)
dpi_save = st.sidebar.number_input("DPI para exportar imagen TIFF", min_value=72, max_value=1200, value=600)

# ----------------- Sidebar: personalización del diagrama -----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Personalización del diagrama")

plot_title = st.sidebar.text_input("Título del gráfico", value="Diagrama cromático CIE 1931")
x_axis_label = st.sidebar.text_input("Etiqueta eje X", value="x-chromaticity coordinate")
y_axis_label = st.sidebar.text_input("Etiqueta eje Y", value="y-chromaticity coordinate")

title_color = st.sidebar.color_picker("Color del título", "#000000")
axes_color = st.sidebar.color_picker("Color de ejes y ticks", "#000000")
locus_label_color = st.sidebar.color_picker("Color de etiquetas λ (números)", "#000000")

title_font_family = st.sidebar.selectbox("Fuente del título", options=["sans-serif", "serif", "monospace"], index=0)
title_font_size = st.sidebar.number_input("Tamaño fuente título", min_value=8, max_value=48, value=14)

tick_font_family = st.sidebar.selectbox("Fuente ticks", options=["sans-serif", "serif", "monospace"], index=0)
tick_font_size = st.sidebar.number_input("Tamaño ticks", min_value=6, max_value=24, value=10)

locus_numbers_font_family = st.sidebar.selectbox("Fuente números λ (locus)", options=["sans-serif", "serif", "monospace"], index=0)
locus_numbers_font_size = st.sidebar.number_input("Tamaño números λ (locus)", min_value=6, max_value=24, value=8)

axes_linewidth = st.sidebar.slider("Grosor de ejes (spines)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

show_point_labels = st.sidebar.checkbox("Mostrar etiquetas junto a cada punto", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Leyenda")
legend_loc = st.sidebar.selectbox("Ubicación", options=[
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center", "center"
], index=0)
legend_font_size = st.sidebar.number_input("Tamaño fuente", min_value=6, max_value=24, value=8)
legend_ncols = st.sidebar.slider("Columnas", min_value=1, max_value=4, value=1)
legend_box_color = st.sidebar.color_picker("Fondo", "#FFFFFF")
legend_box_alpha = st.sidebar.slider("Opacidad fondo", min_value=0.0, max_value=1.0, value=0.85)
legend_box_linewidth = st.sidebar.slider("Grosor borde", min_value=0.0, max_value=4.0, value=0.8, step=0.1)

# ----------------- Sidebar: λ dominante/pureza (opción 6) -----------------
st.sidebar.markdown("---")
st.sidebar.subheader("λ dominante / pureza (mejorado)")

wp_mode = st.sidebar.selectbox("White point", ["E (0.3333, 0.3333)", "D65 (0.3127, 0.3290)", "Custom"], index=0)
if wp_mode.startswith("E"):
    WHITE_POINT = (0.3333, 0.3333)
elif wp_mode.startswith("D65"):
    WHITE_POINT = (0.3127, 0.3290)
else:
    wp_x = st.sidebar.number_input("White point x", min_value=0.0, max_value=1.0, value=0.3333, step=0.0001, format="%.4f")
    wp_y = st.sidebar.number_input("White point y", min_value=0.0, max_value=1.0, value=0.3333, step=0.0001, format="%.4f")
    WHITE_POINT = (float(wp_x), float(wp_y))

# ----------------- Uploaders -----------------
st.markdown("### Subir datos")
st.markdown("CSV: uno o varios archivos. XLSX: uno o varios archivos (se procesan hojas).")

csv_files = st.file_uploader("Sube uno o varios archivos CSV", type=["csv"], accept_multiple_files=True)
xlsx_files = st.file_uploader("Sube uno o varios archivos Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

# ----------------- Helpers: lectura -----------------
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

def read_csv_flexible(uploaded_file):
    raw = uploaded_to_bytes(uploaded_file)
    if not raw:
        raise ValueError("Archivo vacío o no se pudo leer.")
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

    raise ValueError(f"No pude leer el CSV (separador/decimal). Último error: {last_err}")

# ----------------- Helpers: columnas (opción 2) -----------------
_WL_HINTS = [
    "wavelength", "wavelength (nm)", "lambda", "wl", "nm",
    "longitud de onda", "longitud_de_onda", "longitud", "onda"
]
_INT_HINTS = [
    "intensity", "intensity (a.u.)", "a.u.", "au", "counts", "cps", "signal",
    "emission", "fluorescence", "fluorescencia"
]

def _norm_colname(c):
    return str(c).strip().lower().replace("\ufeff", "").replace("µ", "u")

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

# ----------------- CMFs arrays (sin deepcopy) -----------------
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

# ----------------- Geometría: λ dominante/pureza (opción 6) -----------------
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

    t_hit, kind, i, u, q = best
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

# ----------------- Integración numérica: espectro -> xy -----------------
def spectrum_to_xy(wl_grid, intensity_grid):
    xbar = np.interp(wl_grid, CMF_WLS, CMF_X)
    ybar = np.interp(wl_grid, CMF_WLS, CMF_Y)
    zbar = np.interp(wl_grid, CMF_WLS, CMF_Z)

    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    X = float(integrate(intensity_grid * xbar, wl_grid))
    Y = float(integrate(intensity_grid * ybar, wl_grid))
    Z = float(integrate(intensity_grid * zbar, wl_grid))

    S = X + Y + Z
    if not np.isfinite(S) or S <= 0:
        raise ValueError("XYZ inválido (intensidades ~0 en el rango).")
    return X / S, Y / S

# ----------------- Preprocesamiento (opción 3) -----------------
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
        raise ValueError("No hay datos en el rango seleccionado.")
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
        w = int(max(5, smooth_window))
        if w % 2 == 0:
            w += 1
        p = int(max(2, smooth_poly))
        if p >= w:
            p = w - 1
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
        raise ValueError("Intensidad nula en el rango (revisa columnas / rango).")

    return wl_grid, intensity_grid

# ----------------- Plot base (con diagrama CIE) -----------------
fig, ax = plt.subplots(figsize=(7, 7))
try:
    fig_cie, ax_cie = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
    ax.clear()
    plt.close(fig)
    fig, ax = fig_cie, ax_cie
except Exception:
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)

# evitar cortes de textos
fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.12)
for txt in list(ax.texts):
    try:
        txt.set_clip_on(False)
        txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
    except Exception:
        pass
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_clip_on(False)

# aplicar estilos
fontprop_title = FontProperties(family=title_font_family, size=title_font_size)
fontprop_ticks = FontProperties(family=tick_font_family, size=tick_font_size)
fontprop_locus = FontProperties(family=locus_numbers_font_family, size=locus_numbers_font_size)

try:
    t = ax.set_title(plot_title, color=title_color)
    t.set_fontproperties(fontprop_title)
except Exception:
    ax.set_title(plot_title, color=title_color)

try:
    tx = ax.set_xlabel(x_axis_label, color=axes_color)
    ty = ax.set_ylabel(y_axis_label, color=axes_color)
    tx.set_fontproperties(fontprop_ticks)
    ty.set_fontproperties(fontprop_ticks)
except Exception:
    ax.set_xlabel(x_axis_label, color=axes_color)
    ax.set_ylabel(y_axis_label, color=axes_color)

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

# ----------------- Procesamiento por dataset -----------------
results = []

def dataset_block(df_raw, dataset_key, default_label):
    cols = list(df_raw.columns)
    if len(cols) < 2:
        st.error("Esta hoja/archivo no tiene al menos 2 columnas.")
        return

    wl_guess, int_guess = guess_columns(df_raw)

    st.markdown("Columnas (autodetectadas, pero puedes cambiar)")
    wl_col = st.selectbox(
        "Columna de λ (nm)",
        options=cols,
        index=(cols.index(wl_guess) if wl_guess in cols else 0),
        key=f"wlcol_{dataset_key}"
    )
    int_col = st.selectbox(
        "Columna de intensidad",
        options=cols,
        index=(cols.index(int_guess) if int_guess in cols else min(1, len(cols)-1)),
        key=f"intcol_{dataset_key}"
    )

    st.markdown("Estilo del punto")
    label = st.text_input("Label", value=default_label, key=f"label_{dataset_key}")
    color = st.color_picker("Color", "#000000", key=f"color_{dataset_key}")
    marker = st.selectbox("Marker", options=['o','s','^','x','D','*','v'], index=0, key=f"marker_{dataset_key}")
    size = st.slider("Tamaño (s)", min_value=20, max_value=500, value=120, key=f"size_{dataset_key}")

    st.markdown("Rango para cálculo")
    wl_min_local = st.number_input("λ min (nm)", min_value=200, max_value=10000, value=int(wl_min_default), key=f"wlmin_{dataset_key}")
    wl_max_local = st.number_input("λ max (nm)", min_value=200, max_value=10000, value=int(wl_max_default), key=f"wlmax_{dataset_key}")
    interp_local = st.selectbox("Intervalo (nm)", options=[1,2,5,10],
                               index=[1,2,5,10].index(int(interp_interval_default)) if int(interp_interval_default) in [1,2,5,10] else 2,
                               key=f"interp_{dataset_key}")

    st.markdown("Preprocesamiento")
    baseline_subtract_min = st.checkbox("Baseline: restar el mínimo (llevar a 0)", value=False, key=f"base_{dataset_key}")
    clip_negative = st.checkbox("Recortar intensidades negativas a 0", value=False, key=f"clip_{dataset_key}")

    smooth_options = ["None", "Moving average"]
    if _HAS_SCIPY:
        smooth_options.append("Savitzky-Golay")
    smooth_method = st.selectbox("Suavizado", options=smooth_options, index=0, key=f"smooth_{dataset_key}")
    smooth_window = st.slider("Ventana (suavizado)", min_value=3, max_value=101, value=11, step=2, key=f"win_{dataset_key}")
    smooth_poly = st.slider("Orden polinómico (Savitzky-Golay)", min_value=2, max_value=7, value=3, step=1, key=f"poly_{dataset_key}")
    normalize_mode = st.selectbox("Normalización", options=["None", "Max = 1", "Area = 1"], index=0, key=f"norm_{dataset_key}")

    # Vista previa (rápida)
    with st.expander("Vista previa del espectro", expanded=False):
        try:
            wl_preview = to_numeric_series(df_raw[wl_col])
            it_preview = to_numeric_series(df_raw[int_col])
            tmp = pd.DataFrame({"wl": wl_preview, "it": it_preview}).dropna()
            tmp = tmp[(tmp["wl"] >= wl_min_local) & (tmp["wl"] <= wl_max_local)]
            if not tmp.empty:
                fig2, ax2 = plt.subplots()
                ax2.plot(tmp["wl"].values, tmp["it"].values)
                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Intensity")
                st.pyplot(fig2)
            else:
                st.info("Sin datos en el rango para vista previa.")
        except Exception:
            st.info("No se pudo mostrar la vista previa.")

    # Calcular
    try:
        if wl_min_local >= wl_max_local:
            raise ValueError("λ min debe ser menor que λ max.")
        if wl_min_local < CMF_MIN or wl_max_local > CMF_MAX:
            raise ValueError(f"El rango debe estar dentro de las CMFs: {CMF_MIN:.0f}–{CMF_MAX:.0f} nm.")

        wl_grid, intensity_grid = preprocess_spectrum(
            df_raw, wl_col, int_col,
            wl_min_local, wl_max_local, interp_local,
            clip_negative=clip_negative,
            baseline_subtract_min=baseline_subtract_min,
            smooth_method=smooth_method,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
            normalize_mode=normalize_mode
        )

        x_val, y_val = spectrum_to_xy(wl_grid, intensity_grid)
        wl_dom, purity, wl_kind = dominant_wavelength_and_purity(x_val, y_val, white_point=WHITE_POINT)

        ax.scatter([x_val], [y_val], c=[color], marker=marker, s=size, label=label)
        if show_point_labels:
            ax.text(x_val + 0.008, y_val, label,
                    fontsize=tick_font_size, fontfamily=tick_font_family,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))

        results.append({
            "Label": label,
            "x": float(x_val),
            "y": float(y_val),
            "Wavelength_nm": (float(wl_dom) if np.isfinite(wl_dom) else np.nan),
            "Wavelength_type": wl_kind,
            "Excitation_purity_%": (float(purity) if np.isfinite(purity) else np.nan),
            "wl_min_nm": int(wl_min_local),
            "wl_max_nm": int(wl_max_local),
            "interp_nm": int(interp_local),
            "baseline_subtract_min": bool(baseline_subtract_min),
            "clip_negative": bool(clip_negative),
            "smooth_method": smooth_method,
            "smooth_window": int(smooth_window),
            "smooth_poly": int(smooth_poly),
            "normalize": normalize_mode
        })

        st.success(f"OK: x={x_val:.4f}, y={y_val:.4f} | λ({wl_kind})={wl_dom:.1f} nm | pureza={purity:.1f}%")

    except Exception as e:
        st.error(f"Error en {default_label}: {e}")

# ----------------- Ejecutar: CSV -----------------
if csv_files:
    st.markdown("### CSV")
    for i, f in enumerate(csv_files):
        try:
            df = read_csv_flexible(f)
        except Exception as e:
            st.error(f"Error leyendo {f.name}: {e}")
            continue
        with st.expander(f"{f.name}", expanded=(i == 0)):
            dataset_block(df, dataset_key=f"csv_{i}", default_label=f.name)

# ----------------- Ejecutar: XLSX -----------------
if xlsx_files:
    st.markdown("### XLSX (todas las hojas)")
    for j, f in enumerate(xlsx_files):
        raw = uploaded_to_bytes(f)
        if not raw:
            st.error(f"Archivo vacío: {f.name}")
            continue
        try:
            xls = pd.ExcelFile(io.BytesIO(raw))
        except Exception as e:
            st.error(f"No se pudo leer {f.name}: {e}")
            continue

        for k, sheet in enumerate(xls.sheet_names):
            try:
                df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
            except Exception as e:
                st.error(f"Error leyendo {f.name} | {sheet}: {e}")
                continue
            with st.expander(f"{f.name} — hoja: {sheet}", expanded=(j == 0 and k == 0)):
                dataset_block(df, dataset_key=f"xlsx_{j}_{k}", default_label=f"{f.name} - {sheet}")

# ----------------- Salidas -----------------
if results:
    try:
        legend = ax.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_ncols)
        frame = legend.get_frame()
        frame.set_facecolor(legend_box_color)
        frame.set_alpha(legend_box_alpha)
        frame.set_linewidth(legend_box_linewidth)
    except Exception:
        pass

    st.markdown("### Diagrama")
    plt.tight_layout()
    st.pyplot(fig)

    results_df = pd.DataFrame(results)
    st.markdown("### Tabla de coordenadas")
    st.dataframe(results_df)

    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button("Descargar tabla CSV", data=csv_buf.getvalue(), file_name="coordenadas_CIE1931.csv", mime="text/csv")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="tiff", dpi=dpi_save)
    img_buf.seek(0)
    st.download_button("Descargar diagrama TIFF", data=img_buf.getvalue(), file_name="diagrama_CIE1931.tiff", mime="image/tiff")
else:
    st.info("Sube archivos para calcular coordenadas.")
