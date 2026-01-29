
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

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")
matplotlib.rcParams.update({'font.size': 12})
DIALOG_DECORATOR = getattr(st, 'dialog', None) or getattr(st, 'experimental_dialog', None)
HAS_DIALOG = callable(DIALOG_DECORATOR)

st.title("CIE 1931 — Coordenadas cromáticas desde espectros de emisión")

# ----------------- Sidebar (global) -----------------
st.sidebar.header("Parámetros globales")
wl_min_default = st.sidebar.number_input("λ mínimo (nm)", min_value=200, max_value=10000, value=380)
wl_max_default = st.sidebar.number_input("λ máximo (nm)", min_value=200, max_value=10000, value=780)
interp_interval_default = st.sidebar.selectbox("Intervalo de interpolación (nm)", options=[1, 2, 5, 10], index=2)
dpi_save = st.sidebar.number_input("DPI para exportar imagen TIFF", min_value=72, max_value=1200, value=600)

# ---- Personalización del diagrama (se conserva) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Personalización del diagrama")

plot_title = st.sidebar.text_input("Título del gráfico", value="Diagrama cromático CIE 1931")
x_axis_label = st.sidebar.text_input("Etiqueta eje X", value="x-chromaticity coordinate")
y_axis_label = st.sidebar.text_input("Etiqueta eje Y", value="y-chromaticity coordinate")

def _safe_mpl_text(s: str) -> str:
    """Evita caídas de Matplotlib por mathtext inválido.

    Matplotlib intenta interpretar mathtext si ve '$'. Si el usuario escribe un
    '$' suelto (o impar), puede explotar en tight_layout.
    """
    s = "" if s is None else str(s)
    if s.count("$") % 2 == 1:
        s = s.replace("$", "")
    return s

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

# ---- λ dominante / pureza (se conserva) ----
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
st.caption("CSV: uno o varios archivos. XLSX: uno o varios archivos (se procesan todas las hojas).")

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

def read_csv_flexible_bytes(raw: bytes) -> pd.DataFrame:
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

# ----------------- Helpers: columnas (autodetección) -----------------
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

# ----------------- Geometría: λ dominante/pureza -----------------
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

# ----------------- Preprocesamiento -----------------
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

# ----------------- Construir datasets -----------------
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
            st.error(f"Error leyendo CSV {f.name}: {e}")

if xlsx_files:
    for j, f in enumerate(xlsx_files):
        raw = uploaded_to_bytes(f)
        if not raw:
            continue
        try:
            sheets = list_excel_sheets(raw)
        except Exception as e:
            st.error(f"Error leyendo Excel {f.name}: {e}")
            continue
        for k, sh in enumerate(sheets):
            try:
                df = load_excel_sheet(raw, sh)
                datasets.append({"id": f"xlsx_{j}_{k}", "name": f"{f.name} — {sh}", "df": df})
            except Exception as e:
                st.error(f"Error leyendo {f.name} | hoja {sh}: {e}")

# ----------------- Config por dataset -----------------
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
    """Formulario de configuración por muestra.
    key_prefix evita colisiones con otros widgets y permite que Streamlit maneje bien el estado.
    """
    ds_id = ds["id"]
    keyp = f"{key_prefix}{ds_id}_"

    st.markdown("Columnas")
    wl_col = st.selectbox("Columna de longitud de onda", options=list(ds["df"].columns),
                          index=list(ds["df"].columns).index(cfg["wl_col"]) if cfg["wl_col"] in ds["df"].columns else 0,
                          key=f"{keyp}wlcol")
    int_col = st.selectbox("Columna de intensidad", options=list(ds["df"].columns),
                           index=list(ds["df"].columns).index(cfg["int_col"]) if cfg["int_col"] in ds["df"].columns else 1,
                           key=f"{keyp}intcol")

    st.markdown("Estilo del punto")
    label = st.text_input("Label", value=cfg["label"], key=f"{keyp}label")
    color = st.color_picker("Color", value=cfg["color"], key=f"{keyp}color")
    marker_opts = ['o','s','^','x','D','*','v']
    marker = st.selectbox("Marker", options=marker_opts,
                          index=marker_opts.index(cfg["marker"]) if cfg["marker"] in marker_opts else 0,
                          key=f"{keyp}marker")
    size = st.slider("Tamaño (s)", min_value=20, max_value=500, value=int(cfg["size"]), key=f"{keyp}size")

    st.markdown("Rango para cálculo")
    wl_min = st.number_input("λ min (nm)", min_value=200, max_value=10000, value=int(cfg["wl_min"]), key=f"{keyp}wlmin")
    wl_max = st.number_input("λ max (nm)", min_value=200, max_value=10000, value=int(cfg["wl_max"]), key=f"{keyp}wlmax")
    interp_opts = [1,2,5,10]
    interp = st.selectbox("Intervalo (nm)", options=interp_opts,
                          index=interp_opts.index(int(cfg["interp"])) if int(cfg["interp"]) in interp_opts else 1,
                          key=f"{keyp}interp")

    st.markdown("Preprocesamiento")
    baseline_subtract_min = st.checkbox("Baseline: restar el mínimo (llevar a 0)", value=bool(cfg["baseline_subtract_min"]), key=f"{keyp}base")
    clip_negative = st.checkbox("Recortar intensidades negativas a 0", value=bool(cfg["clip_negative"]), key=f"{keyp}clip")

    smooth_options = ["None", "Moving average"] + (["Savitzky-Golay"] if _HAS_SCIPY else [])
    smooth_method = st.selectbox("Suavizado", options=smooth_options,
                                 index=smooth_options.index(cfg["smooth_method"]) if cfg["smooth_method"] in smooth_options else 0,
                                 key=f"{keyp}smooth")
    smooth_window = st.slider("Ventana (suavizado)", min_value=3, max_value=101, value=int(cfg["smooth_window"]), step=2, key=f"{keyp}win")
    smooth_poly = st.slider("Orden polinómico (Savitzky-Golay)", min_value=2, max_value=7, value=int(cfg["smooth_poly"]), step=1, key=f"{keyp}poly")
    normalize = st.selectbox("Normalización", options=["None", "Max = 1", "Area = 1"],
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


# ----------------- Configuración por muestra (modal) -----------------
if "open_cfg_id" not in st.session_state:
    st.session_state["open_cfg_id"] = None

def request_open_config(ds_id: str):
    st.session_state["open_cfg_id"] = ds_id
    st.rerun()

def handle_config_dialog(datasets_by_id):
    """Si hay una muestra seleccionada, abre el diálogo y detiene el resto del script.
    Esto evita que la app siga calculando con la config vieja en el mismo run.
    """
    ds_id = st.session_state.get("open_cfg_id", None)
    if not ds_id:
        return

    ds = datasets_by_id.get(ds_id)
    if ds is None:
        st.session_state["open_cfg_id"] = None
        return

    cfg = get_cfg(ds)
    title = f"Configurar: {ds['name']}"

    if HAS_DIALOG:
        @DIALOG_DECORATOR(title)
        def _dlg():
            with st.form(key=f"form_{ds_id}"):
                new_cfg = render_cfg_form(ds, cfg, key_prefix="dlg_")
                cA, cB = st.columns(2)
                with cA:
                    submitted = st.form_submit_button("Guardar")
                with cB:
                    cancelled = st.form_submit_button("Cancelar")
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
            if cA.button("Guardar", key=f"save_{ds_id}"):
                st.session_state[f"cfg_{ds_id}"] = new_cfg
                st.session_state["open_cfg_id"] = None
                st.rerun()
            if cB.button("Cancelar", key=f"cancel_{ds_id}"):
                st.session_state["open_cfg_id"] = None
                st.rerun()
        st.stop()

datasets_by_id = {ds['id']: ds for ds in datasets}
handle_config_dialog(datasets_by_id)

# ----------------- Figura CIE
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

# ----------------- Procesar -----------------
results = []
spectra_to_plot = []

if datasets:
    st.markdown("### Muestras")
    st.caption("Opciones por muestra: botón Configurar (ventana emergente). Espectros: en desplegable aparte.")

    for ds in datasets:
        cfg = get_cfg(ds)

        c1, c2, c3 = st.columns([6, 2, 4])
        with c1:
            st.write(ds["name"])
        with c2:
            if st.button("Configurar", key=f"btn_cfg_{ds['id']}"):
                request_open_config(ds["id"])
        with c3:
            st.write(f"λ: {cfg['wl_min']}–{cfg['wl_max']} nm | interp: {cfg['interp']} nm")

        try:
            df = ds["df"]
            if cfg["wl_min"] >= cfg["wl_max"]:
                raise ValueError("λ min debe ser menor que λ max.")
            if cfg["wl_min"] < CMF_MIN or cfg["wl_max"] > CMF_MAX:
                raise ValueError(f"El rango debe estar dentro de las CMFs: {CMF_MIN:.0f}–{CMF_MAX:.0f} nm.")
            if cfg["wl_col"] not in df.columns or cfg["int_col"] not in df.columns:
                raise ValueError("Columnas seleccionadas no existen en este archivo/hoja.")

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

            x_val, y_val = spectrum_to_xy(wl_grid, intensity_grid)
            wl_dom, purity, wl_kind = dominant_wavelength_and_purity(x_val, y_val, white_point=WHITE_POINT)

            ax.scatter([x_val], [y_val], c=[cfg["color"]], marker=cfg["marker"], s=cfg["size"], label=cfg["label"])
            if show_point_labels:
                ax.text(x_val + 0.008, y_val, cfg["label"],
                        fontsize=tick_font_size, fontfamily=tick_font_family,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))

            results.append({
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
            })

            spectra_to_plot.append({
                "label": cfg["label"],
                "wl": wl_grid,
                "it": intensity_grid,
                "color": cfg["color"]
            })

            st.success(f"{cfg['label']}: x={x_val:.4f}, y={y_val:.4f} | λ({wl_kind})={wl_dom:.1f} nm | pureza={purity:.1f}%")
        except Exception as e:
            st.error(f"{cfg['label']}: {e}")
else:
    st.info("Sube archivos para comenzar.")

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

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Diagrama CIE 1931")
        try:
            plt.tight_layout()
        except Exception:
            pass
        st.pyplot(fig)

    with col_right:
        with st.expander("Espectros de emisión (procesados para el cálculo)", expanded=False):
            if spectra_to_plot:
                fig_s, ax_s = plt.subplots(figsize=(3.5, 3.5))
                for s in spectra_to_plot:
                    ax_s.plot(s["wl"], s["it"], label=s["label"], color=s["color"])
                ax_s.set_xlabel("Wavelength (nm)")
                ax_s.set_ylabel("Intensity (a.u.)")
                ax_s.tick_params(labelsize=8)
                try:
                    ax_s.legend(fontsize=7, loc="best")
                except Exception:
                    pass
                st.pyplot(fig_s)
            else:
                st.info("No hay espectros para mostrar.")

    st.markdown("### Tabla de coordenadas")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button("Descargar tabla CSV", data=csv_buf.getvalue(), file_name="coordenadas_CIE1931.csv", mime="text/csv")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="tiff", dpi=dpi_save)
    img_buf.seek(0)
    st.download_button("Descargar diagrama TIFF", data=img_buf.getvalue(), file_name="diagrama_CIE1931.tiff", mime="image/tiff")
