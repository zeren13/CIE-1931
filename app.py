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

# Opcional: Savitzky-Golay si SciPy esta instalado
try:
    from scipy.signal import savgol_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="SpectraLab Toolkit")
matplotlib.rcParams.update({'font.size': 12})
DIALOG_DECORATOR = getattr(st, 'dialog', None) or getattr(st, 'experimental_dialog', None)
HAS_DIALOG = callable(DIALOG_DECORATOR)

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Inicio"


def go_to_page(page_name: str):
    st.session_state["active_page"] = page_name
    st.rerun()


def set_info_section(section_name: str):
    st.session_state["cie_info_section"] = section_name
    st.rerun()


# ============================================================
# Utilidades compartidas
# (unica implementacion, usada por todas las paginas: antes existian
# 2-3 copias casi identicas de cada una de estas funciones)
# ============================================================

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
        raise ValueError("El archivo esta vacio o no se pudo leer.")
    trials = [
        (";", ","), (";", "."),
        (",", ","), (",", "."),
        (None, ","), (None, "."),
    ]
    last_err = None
    for sep, dec in trials:
        try:
            bio = io.BytesIO(raw)
            df = pd.read_csv(bio, sep=sep, engine="python" if sep is None else "c", decimal=dec)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
    try:
        bio = io.BytesIO(raw)
        df = pd.read_csv(bio, sep=r"\s+", engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception as e:
        last_err = e
    raise ValueError(f"No se pudo leer el CSV (separador/decimal). Ultimo error: {last_err}")


@st.cache_data(show_spinner=False)
def load_csv_df(raw: bytes) -> pd.DataFrame:
    return read_csv_flexible_bytes(raw)


@st.cache_data(show_spinner=False)
def list_excel_sheets(raw: bytes):
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


@st.cache_data(show_spinner=False)
def load_excel_sheet(raw: bytes, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet)


def to_numeric_series(s: pd.Series):
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")


_WL_HINTS = ["wavelength", "wavelength (nm)", "lambda", "wl", "nm", "wave"]
_INT_HINTS = [
    "intensity", "intensity (a.u.)", "a.u.", "au", "counts", "cps", "signal",
    "emission", "fluorescence", "fluorescencia", "abs", "excitation",
]


def _norm_colname(c):
    return str(c).strip().lower().replace("\ufeff", "").replace("\u00b5", "u")


def guess_columns(df: pd.DataFrame):
    """Adivina columna de longitud de onda e intensidad por nombre, con
    fallback a la fraccion de valores numericos si el nombre no ayuda."""
    cols = list(df.columns)
    if not cols:
        return None, None
    if len(cols) < 2:
        return cols[0], cols[0]

    norm = [_norm_colname(c) for c in cols]
    wl_idx, int_idx = None, None
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
            s = to_numeric_series(df[c])
            numeric_scores.append((i, float(s.notna().mean())))
        numeric_scores.sort(key=lambda x: x[1], reverse=True)
        if wl_idx is None and numeric_scores:
            wl_idx = numeric_scores[0][0]
        if int_idx is None and len(numeric_scores) > 1:
            int_idx = numeric_scores[1][0]
        if wl_idx == int_idx and len(numeric_scores) > 1:
            int_idx = numeric_scores[1][0]

    return (cols[wl_idx] if wl_idx is not None else cols[0],
            cols[int_idx] if int_idx is not None else cols[-1])


def spectrum_metrics(wl, intensity):
    wl = np.asarray(wl, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    peak_idx = int(np.nanargmax(intensity))
    peak_wl = float(wl[peak_idx])
    peak_intensity = float(intensity[peak_idx])
    area = float(integrate(intensity, wl))
    half = peak_intensity / 2.0
    above = np.where(intensity >= half)[0]
    fwhm = np.nan
    if len(above) >= 2:
        fwhm = float(wl[int(above[-1])] - wl[int(above[0])])
    return peak_wl, peak_intensity, area, fwhm


def filter_and_normalize(wl_raw, int_raw, wl_min, wl_max, normalize="None"):
    """Filtra por rango y normaliza sin interpolar a una malla (usado en el
    visor, donde se quiere respetar el muestreo original del instrumento)."""
    data = pd.DataFrame({
        "wavelength": to_numeric_series(wl_raw),
        "intensity": to_numeric_series(int_raw),
    }).dropna()
    data = data[(data["wavelength"] >= wl_min) & (data["wavelength"] <= wl_max)].copy()
    if data.empty:
        raise ValueError("No hay datos dentro del rango seleccionado.")
    data = data.sort_values("wavelength").groupby("wavelength", as_index=False)["intensity"].mean()
    wl = data["wavelength"].to_numpy(dtype=float)
    intensity = data["intensity"].to_numpy(dtype=float)

    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    if normalize == "Max = 1":
        max_val = float(np.nanmax(np.abs(intensity)))
        if max_val > 0:
            intensity = intensity / max_val
    elif normalize == "Area = 1":
        area_abs = float(integrate(np.abs(intensity), wl))
        if area_abs > 0:
            intensity = intensity / area_abs
    return wl, intensity


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
        return np.nan, np.nan, "Indefinido"

    best = None  # (t, kind, i, u, q)
    for i in range(len(LOCUS_XY) - 1):
        p0, p1 = LOCUS_XY[i], LOCUS_XY[i + 1]
        hit = _ray_segment_intersection(w, v, p0, p1)
        if hit is None:
            continue
        t, u = hit
        if best is None or t < best[0]:
            best = (t, "locus", i, u, w + t * v)

    p0, p1 = LOCUS_XY[-1], LOCUS_XY[0]
    hit = _ray_segment_intersection(w, v, p0, p1)
    if hit is not None:
        t, u = hit
        if best is None or t < best[0]:
            best = (t, "purple", -1, u, w + t * v)

    if best is None:
        return np.nan, np.nan, "Sin interseccion"

    _, kind, i, u, q = best
    d_wp_p = float(np.linalg.norm(p - w))
    d_wp_q = float(np.linalg.norm(q - w))
    purity = (d_wp_p / d_wp_q) * 100.0 if d_wp_q > 0 else np.nan

    if kind == "locus":
        wl0, wl1 = float(LOCUS_WLS_F[i]), float(LOCUS_WLS_F[i + 1])
        wl = wl0 + u * (wl1 - wl0)
        return wl, purity, "Dominante"

    # longitud de onda complementaria (rayo opuesto)
    v2 = -v
    best2 = None
    for j in range(len(LOCUS_XY) - 1):
        p0, p1 = LOCUS_XY[j], LOCUS_XY[j + 1]
        hit2 = _ray_segment_intersection(w, v2, p0, p1)
        if hit2 is None:
            continue
        t2, u2 = hit2
        if best2 is None or t2 < best2[0]:
            best2 = (t2, j, u2)

    if best2 is None:
        return np.nan, purity, "Purpura (sin complementaria)"

    _, j, u2 = best2
    wl0, wl1 = float(LOCUS_WLS_F[j]), float(LOCUS_WLS_F[j + 1])
    wl_comp = wl0 + u2 * (wl1 - wl0)
    return wl_comp, purity, "Complementaria"


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
        raise ValueError("Valores XYZ invalidos (las intensidades son casi cero en el rango seleccionado).")
    return X / S, Y / S


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
        raise ValueError("No hay datos dentro del rango de longitud de onda seleccionado.")
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
        raise ValueError("Intensidad cero en el rango seleccionado (revisa columnas y rango).")

    return wl_grid, intensity_grid


def _safe_mpl_text(s: str) -> str:
    """Evita fallos de Matplotlib causados por mathtext invalido.

    Matplotlib intenta interpretar mathtext cuando ve '$'. Si el usuario
    escribe un solo '$' suelto, tight_layout puede fallar.
    """
    s = "" if s is None else str(s)
    if s.count("$") % 2 == 1:
        s = s.replace("$", "")
    return s


def show_and_close(fig):
    """st.pyplot + cierre explicito de la figura para evitar fugas de
    memoria en sesiones largas con muchos reruns."""
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Navegacion global (sidebar) - visible y consistente en TODAS las paginas
# ============================================================
PAGES = ["Inicio", "Analisis CIE 1931", "Visor de espectros", "Rendimiento cuantico", "Sobre CIE"]
st.sidebar.header("Navegacion")
for _page_name in PAGES:
    _is_active = st.session_state["active_page"] == _page_name
    if st.sidebar.button(_page_name, type="primary" if _is_active else "secondary",
                         use_container_width=True, key=f"nav_{_page_name}"):
        if _page_name == "Sobre CIE":
            st.session_state["cie_info_section"] = "Que son"
        go_to_page(_page_name)
st.sidebar.markdown("---")

# ============================================================
# Pagina: Inicio
# ============================================================
if st.session_state["active_page"] == "Inicio":
    st.title("SpectraLab Toolkit")
    st.caption("Conjunto de herramientas para visualizar, comparar y analizar datos espectroscopicos.")

    st.markdown("### Modulos principales")
    st.write(
        "La app queda organizada como una suite: cada modulo resuelve una tarea distinta, "
        "pero todos comparten la misma idea de cargar datos, procesarlos y exportar resultados."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### CIE 1931")
        st.write("Calcula coordenadas cromaticas, longitud dominante, pureza y grafica el diagrama CIE.")
        if st.button("Ir al analisis CIE 1931", type="primary"):
            go_to_page("Analisis CIE 1931")
    with c2:
        st.markdown("#### Visor de espectros")
        st.write("Carga espectros de absorcion, emision o excitacion en solucion o solido y comparalos.")
        if st.button("Abrir visor de espectros"):
            go_to_page("Visor de espectros")
    with c3:
        st.markdown("#### Rendimiento cuantico")
        st.write("Calcula rendimiento cuantico relativo usando muestra, referencia, absorbancia y area.")
        if st.button("Abrir rendimiento cuantico"):
            go_to_page("Rendimiento cuantico")
    with c4:
        st.markdown("#### Aprender CIE")
        st.write("Consulta que es CIE 1931 y revisa un ejemplo guiado de calculo.")
        if st.button("Aprender sobre CIE 1931"):
            st.session_state["cie_info_section"] = "Que son"
            go_to_page("Sobre CIE")

    st.markdown("### Flujo de trabajo")
    f1, f2, f3 = st.columns(3)
    f1.info("1. Sube archivos CSV/XLSX con longitud de onda e intensidad.")
    f2.info("2. Configura columnas, tipo de espectro, rango y normalizacion.")
    f3.info("3. Visualiza graficas, calcula parametros y descarga resultados.")

    st.stop()

# ============================================================
# Pagina: Sobre CIE
# ============================================================
if st.session_state["active_page"] == "Sobre CIE":
    if "cie_info_section" not in st.session_state:
        st.session_state["cie_info_section"] = "Que son"

    st.title("Sobre CIE 1931")
    main_col, nav_col = st.columns([3, 1])

    with nav_col:
        st.markdown("### Secciones")
        current_section = st.session_state["cie_info_section"]
        if st.button("Que son", type="primary" if current_section == "Que son" else "secondary", use_container_width=True):
            set_info_section("Que son")
        if st.button("Como se calculan", type="primary" if current_section == "Como se calculan" else "secondary", use_container_width=True):
            set_info_section("Como se calculan")

    with main_col:
        if st.session_state["cie_info_section"] == "Que son":
            st.markdown("### Que son las coordenadas CIE 1931")
            st.write(
                "CIE 1931 es uno de los sistemas colorimetricos mas usados para representar colores "
                "a partir de la respuesta visual humana. Fue publicado en 1931 por la Commission "
                "Internationale de l'Eclairage, tambien conocida como CIE."
            )
            st.write(
                "El diagrama CIE 1931 usa dos coordenadas, x e y, para describir la cromaticidad "
                "de una fuente luminosa. En este caso, la fuente es el espectro de emision que subes "
                "a la aplicacion."
            )
            st.write(
                "La ventaja de este sistema es que separa la informacion de color de la intensidad total. "
                "Por eso dos espectros con intensidades distintas pueden tener coordenadas x,y similares "
                "si su distribucion espectral produce una percepcion de color parecida."
            )
            st.markdown("### Para que sirve en espectros de emision")
            st.write(
                "En materiales luminiscentes, LEDs, fosforos, colorantes o complejos emisores, "
                "las coordenadas CIE permiten comparar rapidamente el color emitido por diferentes muestras."
            )
        else:
            st.markdown("### Como se calculan")
            st.write(
                "El calculo parte del espectro de emision: longitud de onda contra intensidad. "
                "Ese espectro se combina con las funciones de igualacion de color del observador "
                "estandar CIE 1931 de 2 grados."
            )
            st.latex(r"X = \int I(\lambda)\,\overline{x}(\lambda)\,d\lambda")
            st.latex(r"Y = \int I(\lambda)\,\overline{y}(\lambda)\,d\lambda")
            st.latex(r"Z = \int I(\lambda)\,\overline{z}(\lambda)\,d\lambda")
            st.write(
                "Aqui, I(lambda) es la intensidad del espectro de emision, y xbar, ybar, zbar "
                "son las funciones colorimetricas CIE. En el codigo, estas integrales se calculan "
                "numericamente con la regla trapezoidal."
            )
            st.write("Despues se normalizan los valores XYZ para obtener las coordenadas cromaticas:")
            st.latex(r"x = \frac{X}{X + Y + Z}")
            st.latex(r"y = \frac{Y}{X + Y + Z}")
            st.write(
                "Esas coordenadas x,y son las que se ubican sobre el diagrama CIE 1931. "
                "La longitud dominante y la pureza se estiman trazando una linea desde el punto blanco "
                "hasta la coordenada de la muestra y buscando su interseccion con el borde del diagrama."
            )
            st.markdown("### Ejemplo interactivo")
            st.write(
                "Sube un archivo con dos columnas: longitud de onda en nm e intensidad. "
                "La aplicacion interpola el espectro, multiplica cada punto por las funciones CIE "
                "y suma esas contribuciones para obtener X, Y y Z."
            )

            data_source = st.radio(
                "Datos para el ejemplo",
                options=["Usar ejemplo incluido", "Subir mis datos"],
                horizontal=True,
                key="cie_example_source",
            )

            example_file = None
            if data_source == "Subir mis datos":
                example_file = st.file_uploader(
                    "Subir datos para el ejemplo",
                    type=["csv", "xlsx"],
                    key="cie_example_file"
                )

            if data_source == "Usar ejemplo incluido" or example_file is not None:
                try:
                    if data_source == "Usar ejemplo incluido":
                        sample_wl = np.arange(380, 781, 5, dtype=float)
                        sample_intensity = (
                            1.00 * np.exp(-0.5 * ((sample_wl - 525.0) / 32.0) ** 2)
                            + 0.35 * np.exp(-0.5 * ((sample_wl - 610.0) / 45.0) ** 2)
                        )
                        df_example = pd.DataFrame({
                            "wavelength_nm": sample_wl,
                            "intensity": sample_intensity,
                        })
                    else:
                        raw_example = example_file.getvalue()
                        if example_file.name.lower().endswith(".xlsx"):
                            sheet_names = list_excel_sheets(raw_example)
                            sheet_name = st.selectbox("Hoja de Excel", options=sheet_names, key="cie_example_sheet")
                            df_example = load_excel_sheet(raw_example, sheet_name)
                        else:
                            df_example = read_csv_flexible_bytes(raw_example)

                    wl_guess, int_guess = guess_columns(df_example)
                    cols = list(df_example.columns)
                    c_a, c_b, c_c = st.columns(3)
                    with c_a:
                        wl_col = st.selectbox(
                            "Columna de longitud de onda",
                            options=cols,
                            index=cols.index(wl_guess) if wl_guess in cols else 0,
                            key="cie_example_wl_col"
                        )
                    with c_b:
                        int_col = st.selectbox(
                            "Columna de intensidad",
                            options=cols,
                            index=cols.index(int_guess) if int_guess in cols else min(1, len(cols) - 1),
                            key="cie_example_int_col"
                        )
                    with c_c:
                        interval = st.selectbox("Intervalo de interpolacion (nm)", options=[1, 2, 5, 10], index=2, key="cie_example_interval")

                    c_d, c_e = st.columns(2)
                    with c_d:
                        wl_min = st.number_input("Longitud minima (nm)", min_value=200, max_value=10000, value=380, key="cie_example_wl_min")
                    with c_e:
                        wl_max = st.number_input("Longitud maxima (nm)", min_value=200, max_value=10000, value=780, key="cie_example_wl_max")

                    data = pd.DataFrame({
                        "wavelength": to_numeric_series(df_example[wl_col]),
                        "intensity": to_numeric_series(df_example[int_col]),
                    }).dropna()
                    data = data[(data["wavelength"] >= wl_min) & (data["wavelength"] <= wl_max)].copy()
                    if data.empty:
                        raise ValueError("No hay datos dentro del rango seleccionado.")
                    data = data.sort_values("wavelength").groupby("wavelength", as_index=False)["intensity"].mean()

                    wl_grid = np.arange(float(wl_min), float(wl_max) + 1e-9, float(interval), dtype=float)
                    intensity_grid = np.interp(
                        wl_grid,
                        data["wavelength"].values.astype(float),
                        data["intensity"].values.astype(float),
                        left=0.0,
                        right=0.0,
                    )

                    xbar = np.interp(wl_grid, CMF_WLS, CMF_X)
                    ybar = np.interp(wl_grid, CMF_WLS, CMF_Y)
                    zbar = np.interp(wl_grid, CMF_WLS, CMF_Z)
                    x_product = intensity_grid * xbar
                    y_product = intensity_grid * ybar
                    z_product = intensity_grid * zbar
                    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
                    X = float(integrate(x_product, wl_grid))
                    Y = float(integrate(y_product, wl_grid))
                    Z = float(integrate(z_product, wl_grid))
                    total = X + Y + Z
                    if not np.isfinite(total) or total <= 0:
                        raise ValueError("La suma X+Y+Z no es valida. Revisa columnas, rango o intensidades.")
                    x_coord = X / total
                    y_coord = Y / total

                    st.markdown("#### Paso 1. Preparar el espectro")
                    st.write(
                        "Primero se ordenan los datos, se eliminan valores no numericos y se interpola "
                        "la intensidad al intervalo seleccionado. Con esto todos los calculos usan una "
                        "malla regular de longitudes de onda."
                    )
                    st.latex(r"I(\lambda) \rightarrow I(380), I(385), I(390), \ldots, I(780)")
                    st.write(
                        f"En este ejemplo se usan {len(wl_grid)} puntos entre "
                        f"{float(wl_min):.0f} y {float(wl_max):.0f} nm, cada {float(interval):.0f} nm."
                    )

                    st.markdown("#### Paso 2. Multiplicar por las funciones CIE")
                    st.write(
                        "En cada longitud de onda se multiplica la intensidad del espectro por las tres "
                        "funciones colorimetricas del observador estandar CIE 1931."
                    )
                    st.latex(r"I(\lambda)\overline{x}(\lambda),\quad I(\lambda)\overline{y}(\lambda),\quad I(\lambda)\overline{z}(\lambda)")

                    st.markdown("#### Paso 3. Integrar para obtener X, Y y Z")
                    st.write(
                        "La integracion suma el area bajo cada una de esas tres curvas. En la aplicacion "
                        "se usa integracion trapezoidal."
                    )
                    st.latex(r"X = \int I(\lambda)\overline{x}(\lambda)d\lambda")
                    st.latex(r"Y = \int I(\lambda)\overline{y}(\lambda)d\lambda")
                    st.latex(r"Z = \int I(\lambda)\overline{z}(\lambda)d\lambda")
                    st.write("Reemplazando con los valores calculados para este espectro:")
                    st.latex(fr"X \approx {X:.4g}")
                    st.latex(fr"Y \approx {Y:.4g}")
                    st.latex(fr"Z \approx {Z:.4g}")

                    st.markdown("#### Paso 4. Normalizar a coordenadas cromaticas")
                    st.write(
                        "Como X, Y y Z todavia dependen de la intensidad total, se dividen por "
                        "la suma X+Y+Z para quedarse solo con la cromaticidad."
                    )
                    st.latex(fr"X + Y + Z = {X:.4g} + {Y:.4g} + {Z:.4g} = {total:.4g}")
                    st.latex(fr"x = \frac{{X}}{{X+Y+Z}} = \frac{{{X:.4g}}}{{{total:.4g}}} = {x_coord:.4f}")
                    st.latex(fr"y = \frac{{Y}}{{X+Y+Z}} = \frac{{{Y:.4g}}}{{{total:.4g}}} = {y_coord:.4f}")

                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("X", f"{X:.4g}")
                    r2.metric("Y", f"{Y:.4g}")
                    r3.metric("Z", f"{Z:.4g}")
                    r4.metric("x", f"{x_coord:.4f}")
                    r5.metric("y", f"{y_coord:.4f}")

                    plot_col, cie_col = st.columns(2)
                    with plot_col:
                        st.markdown("#### Espectro de emision")
                        fig_example, ax_example = plt.subplots(figsize=(5, 4))
                        ax_example.plot(wl_grid, intensity_grid, color="#1f77b4", label="Espectro interpolado")
                        ax_example.set_xlabel("Longitud de onda (nm)")
                        ax_example.set_ylabel("Intensidad (u.a.)")
                        ax_example.grid(alpha=0.25)
                        ax_example.legend(loc="best")
                        show_and_close(fig_example)

                    with cie_col:
                        st.markdown("#### Punto en el diagrama CIE")
                        fig_cie_example, ax_cie_example = plt.subplots(figsize=(5, 4))
                        try:
                            try:
                                fig_cie_example, ax_cie_example = colour.plotting.plot_chromaticity_diagram_CIE1931(show=False)
                            except TypeError:
                                fig_cie_example, ax_cie_example = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
                        except Exception:
                            locus_mask = (CMF_WLS >= 380) & (CMF_WLS <= 780)
                            lx = CMF_X[locus_mask]
                            ly = CMF_Y[locus_mask]
                            lz = CMF_Z[locus_mask]
                            den = lx + ly + lz
                            ax_cie_example.plot(lx / den, ly / den, color="black", linewidth=1.0)
                            ax_cie_example.set_xlim(0, 0.8)
                            ax_cie_example.set_ylim(0, 0.9)
                        ax_cie_example.scatter([x_coord], [y_coord], color="#d62728", s=90, zorder=5, label="Muestra")
                        ax_cie_example.text(x_coord + 0.01, y_coord, f"x={x_coord:.3f}\ny={y_coord:.3f}", fontsize=8)
                        ax_cie_example.set_xlabel("x")
                        ax_cie_example.set_ylabel("y")
                        try:
                            ax_cie_example.legend(loc="best", fontsize=8)
                        except Exception:
                            pass
                        show_and_close(fig_cie_example)

                    st.markdown("#### Tabla del calculo")
                    calc_df = pd.DataFrame({
                        "wavelength_nm": wl_grid,
                        "I(lambda)": intensity_grid,
                        "xbar(lambda)": xbar,
                        "ybar(lambda)": ybar,
                        "zbar(lambda)": zbar,
                        "I*xbar": x_product,
                        "I*ybar": y_product,
                        "I*zbar": z_product,
                    })
                    st.dataframe(calc_df.head(30), use_container_width=True)
                    st.caption(
                        "La tabla muestra las primeras filas. Las columnas I*xbar, I*ybar e I*zbar "
                        "son las que se integran numericamente para obtener X, Y y Z."
                    )
                except Exception as e:
                    st.error(f"No se pudo calcular el ejemplo: {e}")
            else:
                st.info("Sube un CSV o XLSX para ver el calculo aplicado a tus propios datos.")

    st.stop()

# ============================================================
# Pagina: Visor de espectros
# ============================================================
if st.session_state["active_page"] == "Visor de espectros":
    st.title("Visor de espectros")
    st.caption("Compara espectros de absorcion, emision o excitacion en solucion y solido.")

    viewer_files = st.file_uploader(
        "Sube archivos CSV o XLSX",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="viewer_files",
    )

    viewer_rows = []
    viewer_spectra = []
    if viewer_files:
        for i, f in enumerate(viewer_files):
            raw = f.getvalue()
            try:
                if f.name.lower().endswith(".xlsx"):
                    sheet_names = list_excel_sheets(raw)
                    selected_sheet = st.selectbox(f"Hoja para {f.name}", options=sheet_names, key=f"viewer_sheet_{i}")
                    df_view = load_excel_sheet(raw, selected_sheet)
                    base_name = f"{f.name} - {selected_sheet}"
                else:
                    df_view = load_csv_df(raw)
                    base_name = f.name

                wl_guess, int_guess = guess_columns(df_view)
                cols = list(df_view.columns)
                with st.expander(f"Configurar {base_name}", expanded=i == 0):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        label = st.text_input("Etiqueta", value=base_name, key=f"viewer_label_{i}")
                        wl_col = st.selectbox("Columna longitud de onda", options=cols, index=cols.index(wl_guess) if wl_guess in cols else 0, key=f"viewer_wl_{i}")
                    with c2:
                        int_col = st.selectbox("Columna intensidad", options=cols, index=cols.index(int_guess) if int_guess in cols else min(1, len(cols) - 1), key=f"viewer_int_{i}")
                        spectrum_type = st.selectbox("Tipo", options=["Absorcion", "Emision", "Excitacion"], key=f"viewer_type_{i}")
                    with c3:
                        sample_state = st.selectbox("Medio", options=["Solucion", "Solido"], key=f"viewer_state_{i}")
                        line_color = st.color_picker("Color", value=["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"][i % 5], key=f"viewer_color_{i}")

                    c4, c5, c6 = st.columns(3)
                    with c4:
                        normalize = st.selectbox("Normalizacion", options=["None", "Max = 1", "Area = 1"], key=f"viewer_norm_{i}")
                    with c5:
                        wl_min_v = st.number_input("Min nm", min_value=100, max_value=10000, value=200, key=f"viewer_min_{i}")
                    with c6:
                        wl_max_v = st.number_input("Max nm", min_value=100, max_value=10000, value=900, key=f"viewer_max_{i}")

                wl, intensity = filter_and_normalize(df_view[wl_col], df_view[int_col], wl_min_v, wl_max_v, normalize)
                peak_wl, peak_intensity, area, fwhm = spectrum_metrics(wl, intensity)
                viewer_spectra.append({
                    "label": label,
                    "wl": wl,
                    "intensity": intensity,
                    "color": line_color,
                })
                viewer_rows.append({
                    "Etiqueta": label,
                    "Tipo": spectrum_type,
                    "Medio": sample_state,
                    "Pico_nm": peak_wl,
                    "Intensidad_pico": peak_intensity,
                    "Area": area,
                    "FWHM_nm": fwhm,
                })
            except Exception as e:
                st.error(f"{f.name}: {e}")

    if viewer_spectra:
        left, right = st.columns([2, 1], gap="large")
        with left:
            fig_v, ax_v = plt.subplots(figsize=(8, 4.8))
            for s in viewer_spectra:
                ax_v.plot(s["wl"], s["intensity"], label=s["label"], color=s["color"], linewidth=1.8)
            ax_v.set_xlabel("Longitud de onda (nm)")
            ax_v.set_ylabel("Intensidad / Absorbancia")
            ax_v.grid(alpha=0.25)
            try:
                ax_v.legend(fontsize=8, loc="best")
            except Exception:
                pass
            show_and_close(fig_v)
        with right:
            st.markdown("### Resumen")
            viewer_df = pd.DataFrame(viewer_rows)
            st.dataframe(viewer_df, use_container_width=True, hide_index=True)
            csv_viewer = io.StringIO()
            viewer_df.to_csv(csv_viewer, index=False)
            st.download_button("Descargar resumen CSV", data=csv_viewer.getvalue(), file_name="spectra_viewer_summary.csv", mime="text/csv")
    else:
        st.info("Sube uno o mas espectros para iniciar la visualizacion.")
    st.stop()

# ============================================================
# Pagina: Rendimiento cuantico
# ============================================================
if st.session_state["active_page"] == "Rendimiento cuantico":
    st.title("Rendimiento cuantico relativo")
    st.caption("Calcula Phi de una muestra comparandola con una referencia.")

    st.latex(r"\Phi_x = \Phi_{ref}\left(\frac{I_x}{I_{ref}}\right)\left(\frac{A_{ref}}{A_x}\right)\left(\frac{n_x^2}{n_{ref}^2}\right)")
    st.write(
        "Usa areas integradas de emision, absorbancias a la longitud de onda de excitacion "
        "e indices de refraccion. Mantener absorbancias bajas ayuda a reducir errores por filtro interno."
    )

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("### Muestra")
        sample_area = st.number_input("Area integrada muestra", min_value=0.0, value=1.0, format="%.6f")
        sample_abs = st.number_input("Absorbancia muestra", min_value=0.000001, value=0.05, format="%.6f")
        sample_n = st.number_input("Indice refraccion muestra", min_value=1.0, value=1.333, format="%.6f")
    with q2:
        st.markdown("### Referencia")
        ref_phi = st.number_input("Phi referencia", min_value=0.0, max_value=1.0, value=0.55, format="%.6f")
        ref_area = st.number_input("Area integrada referencia", min_value=0.000001, value=1.0, format="%.6f")
        ref_abs = st.number_input("Absorbancia referencia", min_value=0.000001, value=0.05, format="%.6f")
        ref_n = st.number_input("Indice refraccion referencia", min_value=1.0, value=1.333, format="%.6f")

    phi = ref_phi * (sample_area / ref_area) * (ref_abs / sample_abs) * ((sample_n ** 2) / (ref_n ** 2))
    st.metric("Rendimiento cuantico de la muestra", f"{phi:.4f}", f"{phi * 100:.2f}%")
    st.latex(
        fr"\Phi_x = {ref_phi:.4g}\left(\frac{{{sample_area:.4g}}}{{{ref_area:.4g}}}\right)"
        fr"\left(\frac{{{ref_abs:.4g}}}{{{sample_abs:.4g}}}\right)"
        fr"\left(\frac{{{sample_n:.4g}^2}}{{{ref_n:.4g}^2}}\right) = {phi:.4g}"
    )
    if phi > 1:
        st.warning("El resultado es mayor que 1. Revisa areas, absorbancias, referencia o correcciones experimentales.")
    st.stop()

# ============================================================
# Pagina: Analisis CIE 1931 (principal)
# ============================================================
st.title("CIE 1931 - Coordenadas de cromaticidad a partir de espectros de emision")

# ----------------- Sidebar (parametros de esta pagina) -----------------
st.sidebar.header("Parametros globales")
wl_min_default = st.sidebar.number_input("Longitud de onda minima (nm)", min_value=200, max_value=10000, value=380)
wl_max_default = st.sidebar.number_input("Longitud de onda maxima (nm)", min_value=200, max_value=10000, value=780)
interp_interval_default = st.sidebar.selectbox("Intervalo de interpolacion (nm)", options=[1, 2, 5, 10], index=2)
dpi_save = st.sidebar.number_input("DPI para exportar TIFF", min_value=72, max_value=1200, value=600)

# ---- Personalizacion del diagrama ----
st.sidebar.markdown("---")
st.sidebar.subheader("Personalizacion del diagrama")

plot_title = st.sidebar.text_input("Titulo del grafico", value="Diagrama de cromaticidad CIE 1931")
x_axis_label = st.sidebar.text_input("Etiqueta eje X", value="Coordenada de cromaticidad x")
y_axis_label = st.sidebar.text_input("Etiqueta eje Y", value="Coordenada de cromaticidad y")

title_color = st.sidebar.color_picker("Color del titulo", "#000000")
axes_color = st.sidebar.color_picker("Color de ejes y marcas", "#000000")
locus_label_color = st.sidebar.color_picker("Color de numeros de longitud de onda", "#000000")

title_font_family = st.sidebar.selectbox("Fuente del titulo", options=["sans-serif", "serif", "monospace"], index=0)
title_font_size = st.sidebar.number_input("Tamano de fuente del titulo", min_value=8, max_value=48, value=14)

tick_font_family = st.sidebar.selectbox("Fuente de marcas", options=["sans-serif", "serif", "monospace"], index=0)
tick_font_size = st.sidebar.number_input("Tamano de marcas", min_value=6, max_value=24, value=10)

locus_numbers_font_family = st.sidebar.selectbox("Fuente de numeros (locus)", options=["sans-serif", "serif", "monospace"], index=0)
locus_numbers_font_size = st.sidebar.number_input("Tamano de numeros (locus)", min_value=6, max_value=24, value=8)

axes_linewidth = st.sidebar.slider("Grosor de ejes", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

show_point_labels = st.sidebar.checkbox("Mostrar etiquetas junto a cada punto", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Leyenda")
legend_loc = st.sidebar.selectbox("Ubicacion", options=[
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center", "center"
], index=0)
legend_font_size = st.sidebar.number_input("Tamano de fuente", min_value=6, max_value=24, value=8)
legend_ncols = st.sidebar.slider("Columnas", min_value=1, max_value=4, value=1)
legend_box_color = st.sidebar.color_picker("Fondo", "#FFFFFF")
legend_box_alpha = st.sidebar.slider("Opacidad de fondo", min_value=0.0, max_value=1.0, value=0.85)
legend_box_linewidth = st.sidebar.slider("Grosor del borde", min_value=0.0, max_value=4.0, value=0.8, step=0.1)

# ---- Longitud dominante / pureza ----
st.sidebar.markdown("---")
st.sidebar.subheader("Longitud de onda dominante / pureza")

wp_mode = st.sidebar.selectbox("Punto blanco", ["E (0.3333, 0.3333)", "D65 (0.3127, 0.3290)", "Personalizado"], index=0)
if wp_mode.startswith("E"):
    WHITE_POINT = (0.3333, 0.3333)
elif wp_mode.startswith("D65"):
    WHITE_POINT = (0.3127, 0.3290)
else:
    wp_x = st.sidebar.number_input("Punto blanco x", min_value=0.0, max_value=1.0, value=0.3333, step=0.0001, format="%.4f")
    wp_y = st.sidebar.number_input("Punto blanco y", min_value=0.0, max_value=1.0, value=0.3333, step=0.0001, format="%.4f")
    WHITE_POINT = (float(wp_x), float(wp_y))

# ----------------- Carga de archivos -----------------
st.markdown("### Subir datos")
st.caption("CSV: uno o mas archivos. XLSX: uno o mas archivos (se procesan todas las hojas).")

csv_files = st.file_uploader("Sube uno o mas archivos CSV", type=["csv"], accept_multiple_files=True)
xlsx_files = st.file_uploader("Sube uno o mas archivos Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

# ----------------- Construccion de datasets -----------------
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
            st.error(f"Error al leer CSV {f.name}: {e}")

if xlsx_files:
    for j, f in enumerate(xlsx_files):
        raw = uploaded_to_bytes(f)
        if not raw:
            continue
        try:
            sheets = list_excel_sheets(raw)
        except Exception as e:
            st.error(f"Error al leer Excel {f.name}: {e}")
            continue
        for k, sh in enumerate(sheets):
            try:
                df = load_excel_sheet(raw, sh)
                datasets.append({"id": f"xlsx_{j}_{k}", "name": f"{f.name} - {sh}", "df": df})
            except Exception as e:
                st.error(f"Error al leer {f.name} | hoja {sh}: {e}")

# ----------------- Configuracion por dataset -----------------
def init_cfg(ds):
    df = ds["df"]
    cols = list(df.columns)
    wl_guess, int_guess = guess_columns(df)
    return {
        "label": ds["name"],
        "color": "#000000",
        "spectrum_color": "#1f77b4",
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
    if "spectrum_color" not in cfg:
        cfg["spectrum_color"] = cfg.get("color", "#1f77b4")
    st.session_state[key] = cfg
    return cfg


def render_cfg_form(ds, cfg, key_prefix="dlg_"):
    """Formulario de configuracion por muestra.
    key_prefix evita colisiones de keys de widgets entre dialogos."""
    ds_id = ds["id"]
    keyp = f"{key_prefix}{ds_id}_"
    cols = list(ds["df"].columns)

    st.markdown("Columnas")
    wl_col = st.selectbox("Columna de longitud de onda", options=cols,
                          index=cols.index(cfg["wl_col"]) if cfg["wl_col"] in cols else 0,
                          key=f"{keyp}wlcol")
    int_default_index = cols.index(cfg["int_col"]) if cfg["int_col"] in cols else min(1, len(cols) - 1)
    int_col = st.selectbox("Columna de intensidad", options=cols,
                           index=int_default_index,
                           key=f"{keyp}intcol")

    st.markdown("Estilo del punto")
    label = st.text_input("Etiqueta", value=cfg["label"], key=f"{keyp}label")
    color = st.color_picker("Color", value=cfg["color"], key=f"{keyp}color")
    spectrum_color = st.color_picker("Color de la linea del espectro", value=cfg["spectrum_color"], key=f"{keyp}spectrum_color")
    marker_opts = ['o', 's', '^', 'x', 'D', '*', 'v']
    marker = st.selectbox("Marcador", options=marker_opts,
                          index=marker_opts.index(cfg["marker"]) if cfg["marker"] in marker_opts else 0,
                          key=f"{keyp}marker")
    size = st.slider("Tamano (s)", min_value=20, max_value=500, value=int(cfg["size"]), key=f"{keyp}size")

    st.markdown("Rango de calculo")
    wl_min = st.number_input("Longitud de onda minima (nm)", min_value=200, max_value=10000, value=int(cfg["wl_min"]), key=f"{keyp}wlmin")
    wl_max = st.number_input("Longitud de onda maxima (nm)", min_value=200, max_value=10000, value=int(cfg["wl_max"]), key=f"{keyp}wlmax")
    interp_opts = [1, 2, 5, 10]
    interp = st.selectbox("Intervalo (nm)", options=interp_opts,
                          index=interp_opts.index(int(cfg["interp"])) if int(cfg["interp"]) in interp_opts else 1,
                          key=f"{keyp}interp")

    st.markdown("Preprocesamiento")
    baseline_subtract_min = st.checkbox("Linea base: restar minimo (desplazar a 0)", value=bool(cfg["baseline_subtract_min"]), key=f"{keyp}base")
    clip_negative = st.checkbox("Recortar intensidades negativas a 0", value=bool(cfg["clip_negative"]), key=f"{keyp}clip")

    smooth_options = ["None", "Moving average"] + (["Savitzky-Golay"] if _HAS_SCIPY else [])
    smooth_method = st.selectbox("Suavizado", options=smooth_options,
                                 index=smooth_options.index(cfg["smooth_method"]) if cfg["smooth_method"] in smooth_options else 0,
                                 key=f"{keyp}smooth")
    smooth_window = st.slider("Ventana (suavizado)", min_value=3, max_value=101, value=int(cfg["smooth_window"]), step=2, key=f"{keyp}win")
    smooth_poly = st.slider("Orden del polinomio (Savitzky-Golay)", min_value=2, max_value=7, value=int(cfg["smooth_poly"]), step=1, key=f"{keyp}poly")
    normalize = st.selectbox("Normalizacion", options=["None", "Max = 1", "Area = 1"],
                             index=["None", "Max = 1", "Area = 1"].index(cfg["normalize"]) if cfg["normalize"] in ["None", "Max = 1", "Area = 1"] else 0,
                             key=f"{keyp}norm")

    return {
        "label": label,
        "color": color,
        "spectrum_color": spectrum_color,
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


# ----------------- Configuracion por muestra (modal) -----------------
if "open_cfg_id" not in st.session_state:
    st.session_state["open_cfg_id"] = None


def request_open_config(ds_id: str):
    st.session_state["open_cfg_id"] = ds_id
    st.rerun()


def handle_config_dialog(datasets_by_id):
    """Si hay una muestra seleccionada, abre el dialogo y detiene el resto
    del script. Esto evita calcular con configuracion desactualizada en la
    misma ejecucion."""
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

# ----------------- Figura CIE -----------------
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

# ----------------- Procesar muestras -----------------
results = []
spectra_to_plot = []

if datasets:
    st.markdown("### Muestras")
    st.caption("Opciones por muestra: boton Configurar (ventana emergente). Los resultados se muestran en el resumen debajo.")

    for ds in datasets:
        cfg = get_cfg(ds)

        c1, c2, c3 = st.columns([6, 2, 4])
        with c1:
            st.write(ds["name"])
        with c2:
            if st.button("Configurar", key=f"btn_cfg_{ds['id']}"):
                request_open_config(ds["id"])
        with c3:
            st.write(f"Longitud de onda: {cfg['wl_min']}-{cfg['wl_max']} nm | interp: {cfg['interp']} nm")

        try:
            df = ds["df"]
            if cfg["wl_min"] >= cfg["wl_max"]:
                raise ValueError("La longitud de onda minima debe ser menor que la maxima.")
            if cfg["wl_min"] < CMF_MIN or cfg["wl_max"] > CMF_MAX:
                raise ValueError(f"El rango debe estar dentro del dominio de las CMF: {CMF_MIN:.0f}-{CMF_MAX:.0f} nm.")
            if cfg["wl_col"] not in df.columns or cfg["int_col"] not in df.columns:
                raise ValueError("Las columnas seleccionadas no existen en este archivo/hoja.")

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
                "Etiqueta": cfg["label"],
                "x": float(x_val),
                "y": float(y_val),
                "Longitud_onda_nm": (float(wl_dom) if np.isfinite(wl_dom) else np.nan),
                "Tipo_longitud_onda": wl_kind,
                "Pureza_excitacion_%": (float(purity) if np.isfinite(purity) else np.nan),
                "wl_min_nm": int(cfg["wl_min"]),
                "wl_max_nm": int(cfg["wl_max"]),
                "interp_nm": int(cfg["interp"]),
                "linea_base_restar_min": bool(cfg["baseline_subtract_min"]),
                "recorte_negativos": bool(cfg["clip_negative"]),
                "metodo_suavizado": cfg["smooth_method"],
                "ventana_suavizado": int(cfg["smooth_window"]),
                "orden_polinomio": int(cfg["smooth_poly"]),
                "normalizacion": cfg["normalize"],
                "columna_wl": cfg["wl_col"],
                "columna_intensidad": cfg["int_col"],
            })

            spectra_to_plot.append({
                "label": cfg["label"],
                "wl": wl_grid,
                "it": intensity_grid,
                "color": cfg.get("spectrum_color", cfg["color"])
            })

            st.success(f"{cfg['label']}: x={x_val:.4f}, y={y_val:.4f} | longitud de onda ({wl_kind})={wl_dom:.1f} nm | pureza={purity:.1f}%")
        except Exception as e:
            st.error(f"{cfg['label']}: {e}")
else:
    st.info("Sube archivos para comenzar.")

# ----------------- Resultados -----------------
if results:
    try:
        legend = ax.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_ncols)
        frame = legend.get_frame()
        frame.set_facecolor(legend_box_color)
        frame.set_alpha(legend_box_alpha)
        frame.set_linewidth(legend_box_linewidth)
    except Exception:
        pass

    st.markdown("### Resumen del analisis")
    results_df = pd.DataFrame(results)
    minimal_cols = [
        "Etiqueta",
        "x",
        "y",
        "Longitud_onda_nm",
        "Tipo_longitud_onda",
        "Pureza_excitacion_%",
    ]
    minimal_cols = [c for c in minimal_cols if c in results_df.columns]
    table_df = results_df[minimal_cols].copy()
    for c in ["x", "y"]:
        if c in table_df.columns:
            table_df[c] = table_df[c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    for c in ["Longitud_onda_nm", "Pureza_excitacion_%"]:
        if c in table_df.columns:
            table_df[c] = table_df[c].map(lambda v: f"{v:.1f}" if np.isfinite(v) else "")

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("### Diagrama CIE 1931")
        try:
            plt.tight_layout()
        except Exception:
            pass
        show_and_close(fig)

    with right_col:
        st.markdown("### Espectros de emision")
        if spectra_to_plot:
            fig_s, ax_s = plt.subplots(figsize=(5.5, 3.2))
            for s in spectra_to_plot:
                ax_s.plot(s["wl"], s["it"], label=s["label"], color=s["color"], linewidth=1.8)
            ax_s.set_xlabel("Longitud de onda (nm)")
            ax_s.set_ylabel("Intensidad (u.a.)")
            ax_s.tick_params(labelsize=9)
            ax_s.grid(alpha=0.25)
            try:
                ax_s.legend(fontsize=7, loc="best")
            except Exception:
                pass
            show_and_close(fig_s)
        else:
            st.info("No hay espectros para mostrar.")

        st.markdown("### Tabla de coordenadas")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        with st.expander("Descargas", expanded=False):
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            st.download_button("Descargar tabla CSV completa", data=csv_buf.getvalue(), file_name="CIE1931_coordenadas.csv", mime="text/csv")

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="tiff", dpi=dpi_save)
            img_buf.seek(0)
            st.download_button("Descargar diagrama TIFF", data=img_buf.getvalue(), file_name="CIE1931_diagrama.tiff", mime="image/tiff")
else:
    if datasets:
        plt.close(fig)
