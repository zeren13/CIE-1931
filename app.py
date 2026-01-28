
# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

import colour
from colour import MSDS_CMFS
from colour.colorimetry import SpectralShape
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_xy


# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")
matplotlib.rcParams.update({"font.size": 12})

st.title("CIE 1931 — Coordenadas cromáticas desde espectros de emisión (múltiples archivos)")


# ----------------- Sidebar params -----------------
st.sidebar.header("Parámetros globales")
wl_min = st.sidebar.number_input("λ mínimo (nm)", min_value=200, max_value=10000, value=380)
wl_max = st.sidebar.number_input("λ máximo (nm)", min_value=200, max_value=10000, value=780)
interp_interval = st.sidebar.selectbox("Intervalo de interpolación (nm)", options=[1, 5, 10], index=0)
dpi_save = st.sidebar.number_input("DPI para exportar imagen TIFF", min_value=72, max_value=1200, value=600)

plot_title = st.sidebar.text_input("Título del gráfico", value="Diagrama cromático CIE 1931")
x_axis_label = st.sidebar.text_input("Etiqueta eje X", value="x")
y_axis_label = st.sidebar.text_input("Etiqueta eje Y", value="y")
title_color = st.sidebar.color_picker("Color del título", "#000000")
axes_color = st.sidebar.color_picker("Color de ejes y ticks", "#000000")

title_font_family = st.sidebar.selectbox("Fuente del título", options=["sans-serif", "serif", "monospace"], index=0)
title_font_size = st.sidebar.number_input("Tamaño fuente título", min_value=8, max_value=48, value=14)

tick_font_family = st.sidebar.selectbox("Fuente números de ejes (ticks)", options=["sans-serif", "serif", "monospace"], index=0)
tick_font_size = st.sidebar.number_input("Tamaño números de ejes", min_value=6, max_value=24, value=10)

locus_numbers_font_family = st.sidebar.selectbox("Fuente números λ (etiquetas locus)", options=["sans-serif", "serif", "monospace"], index=0)
locus_numbers_font_size = st.sidebar.number_input("Tamaño números λ", min_value=6, max_value=24, value=8)

axes_linewidth = st.sidebar.slider("Grosor de ejes (spines)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

locus_label_color = st.sidebar.color_picker("Color de etiquetas λ (números)", "#000000")

show_point_labels = st.sidebar.checkbox("Mostrar etiquetas junto a cada punto", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración de la leyenda (lista de archivos)")
legend_loc = st.sidebar.selectbox(
    "Ubicación de la leyenda",
    options=[
        "best", "upper right", "upper left", "lower left", "lower right",
        "right", "center left", "center right", "lower center", "upper center", "center"
    ],
    index=0
)
legend_font_size = st.sidebar.number_input("Tamaño fuente leyenda", min_value=6, max_value=24, value=8)
legend_ncols = st.sidebar.slider("Columnas de la leyenda", min_value=1, max_value=4, value=1)
legend_box_color = st.sidebar.color_picker("Color fondo cuadro leyenda", "#FFFFFF")
legend_box_alpha = st.sidebar.slider("Opacidad fondo leyenda", min_value=0.0, max_value=1.0, value=0.85)
legend_box_linewidth = st.sidebar.slider("Grosor borde cuadro leyenda", min_value=0.0, max_value=4.0, value=0.8, step=0.1)


# ----------------- Uploaders -----------------
st.markdown("#### 1) Subir archivos")
st.markdown("- CSV simples (varios): dos primeras columnas = λ e intensidad (separador ; o ,).")
st.markdown("- XLSX (varias hojas): se procesan todas las hojas (si alguna no sirve, se reporta).")

csv_files = st.file_uploader("Sube uno o varios archivos CSV", type=["csv"], accept_multiple_files=True)
xlsx_files = st.file_uploader("Sube uno o varios archivos Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)


# ----------------- Helpers -----------------
def _uploaded_to_bytes(uploaded):
    """Devuelve bytes del archivo subido sin depender del puntero interno."""
    if uploaded is None:
        return b""
    if hasattr(uploaded, "getvalue"):
        return uploaded.getvalue()
    try:
        uploaded.seek(0)
    except Exception:
        pass
    return uploaded.read()

def read_csv_flexible(uploaded_file):
    """Lee CSV intentando separadores comunes y manejando coma decimal."""
    raw = _uploaded_to_bytes(uploaded_file)
    if not raw:
        raise ValueError("Archivo vacío.")
    bio = io.BytesIO(raw)

    # 1) común en tus datos: sep=';' y coma decimal
    for sep, dec in [(";", ","), (";", "."), (",", ","), (",", "."), (None, ","), (None, ".")]:
        try:
            bio.seek(0)
            df = pd.read_csv(bio, sep=sep, engine="python" if sep is None else "c", decimal=dec)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    # 2) último intento: whitespace
    try:
        bio.seek(0)
        df = pd.read_csv(bio, sep=r"\s+", engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    raise ValueError("No pude leer el CSV (revisa separador/decimal).")

def preprocess_df(df):
    """Toma primeras 2 columnas como wavelength/intensity y limpia NaN."""
    df = df.copy()
    if df.shape[1] < 2:
        raise ValueError("El archivo no tiene al menos 2 columnas.")
    df = df.iloc[:, :2]
    df.columns = ["wavelength", "intensity"]
    df["wavelength"] = pd.to_numeric(df["wavelength"].astype(str).str.replace(",", "."), errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"].astype(str).str.replace(",", "."), errors="coerce")
    df = df.dropna()
    return df

@st.cache_resource
def get_cmfs():
    return MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

@st.cache_resource
def get_locus_from_cmfs():
    """
    Construye la locus espectral (x,y) a partir de las CMFs.
    Evita crear SpectralDistribution para cada λ (más estable y rápido).
    """
    cmfs = get_cmfs().copy().align(SpectralShape(380, 780, 1))
    # domain / wavelengths
    wls = None
    for attr in ("wavelengths", "domain"):
        if hasattr(cmfs, attr):
            wls = np.asarray(getattr(cmfs, attr))
            break
    if wls is None:
        # fallback: intentar extraer del índice
        wls = np.asarray(cmfs.index)

    vals = np.asarray(cmfs.values)
    # asegurar Nx3
    if vals.shape[0] == 3 and vals.shape[1] == wls.shape[0]:
        vals = vals.T

    denom = np.sum(vals, axis=1)
    # evitar división por cero
    denom = np.where(denom == 0, np.nan, denom)
    x = vals[:, 0] / denom
    y = vals[:, 1] / denom
    xy = np.column_stack([x, y])

    # limpiar NaN (por si hay)
    m = np.isfinite(xy).all(axis=1)
    return wls[m].astype(int), xy[m]

_locus_wls, _locus_xy = get_locus_from_cmfs()
_cmfs = get_cmfs()

def dominant_wavelength_from_xy(x, y):
    d = np.sqrt((_locus_xy[:, 0] - x) ** 2 + (_locus_xy[:, 1] - y) ** 2)
    idx = int(np.argmin(d))
    return int(_locus_wls[idx])


# ----------------- Plot base -----------------
fig, ax = plt.subplots(figsize=(7, 7))
try:
    fig_cie, ax_cie = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
    plt.close(fig)
    fig, ax = fig_cie, ax_cie
except Exception:
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)

# evitar cortes
fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.12)
for txt in list(ax.texts):
    try:
        txt.set_clip_on(False)
        txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
    except Exception:
        pass
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_clip_on(False)

# ----------------- Estilos de texto -----------------
fontprop_title = FontProperties(family=title_font_family, size=title_font_size)
fontprop_ticks = FontProperties(family=tick_font_family, size=tick_font_size)
fontprop_locus = FontProperties(family=locus_numbers_font_family, size=locus_numbers_font_size)

# título y ejes
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

# ticks
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    try:
        lbl.set_fontproperties(fontprop_ticks)
        lbl.set_color(axes_color)
        lbl.set_clip_on(False)
    except Exception:
        pass

# locus labels (números λ)
for txt in list(ax.texts):
    try:
        txt.set_clip_on(False)
        txt.set_fontproperties(fontprop_locus)
        txt.set_color(locus_label_color)
    except Exception:
        pass

# spines
try:
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
except Exception:
    pass


# ----------------- Core processing -----------------
results = []

def process_and_plot(df, label, color, marker, size, wl_min_local, wl_max_local, interp_interval_local):
    df = preprocess_df(df)

    if wl_min_local >= wl_max_local:
        raise ValueError("λ mínimo debe ser menor que λ máximo.")

    mask = (df["wavelength"] >= wl_min_local) & (df["wavelength"] <= wl_max_local)
    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError("No hay datos en el rango seleccionado.")

    # ordenar y colapsar duplicados (importante para interpolación estable)
    df = df.sort_values("wavelength")
    df = df.groupby("wavelength", as_index=False)["intensity"].mean()

    if df.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 puntos para calcular coordenadas.")

    # construir SPD e interpolar
    sd = SpectralDistribution(dict(zip(df["wavelength"].values, df["intensity"].values)))
    sd_int = sd.copy().interpolate(SpectralShape(wl_min_local, wl_max_local, interp_interval_local))

    XYZ = sd_to_XYZ(sd_int, cmfs=_cmfs)
    xy = XYZ_to_xy(XYZ)

    x_val, y_val = float(xy[0]), float(xy[1])

    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        raise ValueError("Coordenadas no finitas (revisa el espectro: intensidades/rango).")

    wl_dom = dominant_wavelength_from_xy(x_val, y_val)

    # dibujar punto
    ax.scatter([x_val], [y_val], c=[color], marker=marker, s=size, label=label)

    if show_point_labels:
        try:
            ax.text(
                x_val + 0.008, y_val, label,
                fontsize=tick_font_size, fontfamily=tick_font_family,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0)
            )
        except Exception:
            pass

    return {"Label": label, "x": x_val, "y": y_val, "Dominant_wavelength_nm": wl_dom}


# ----------------- CSV processing -----------------
if csv_files:
    st.subheader("CSV files")
    for i, file in enumerate(csv_files):
        with st.expander(f"Configurar: {file.name}", expanded=(i == 0)):
            label = st.text_input(f"Label (CSV {i})", value=file.name, key=f"csv_label_{i}")
            color = st.color_picker(f"Color (CSV {i})", "#000000", key=f"csv_color_{i}")
            marker = st.selectbox(f"Marker (CSV {i})", options=["o", "s", "^", "x", "D", "*", "v"], index=0, key=f"csv_marker_{i}")
            size = st.slider(f"Tamaño (CSV {i})", min_value=10, max_value=400, value=80, key=f"csv_size_{i}")
            wl_min_local = st.number_input(f"λ min (CSV {i})", min_value=200, max_value=10000, value=int(wl_min), key=f"csv_wlmin_{i}")
            wl_max_local = st.number_input(f"λ max (CSV {i})", min_value=200, max_value=10000, value=int(wl_max), key=f"csv_wlmax_{i}")
            interp_local = st.selectbox(f"Interpolation interval (CSV {i})", [1, 5, 10], index=0, key=f"csv_interp_{i}")

            try:
                df_tmp = read_csv_flexible(file)
                res = process_and_plot(df_tmp, label, color, marker, size, wl_min_local, wl_max_local, interp_local)
                results.append(res)
                st.success(f"Procesado: {label} -> x={res['x']:.4f}, y={res['y']:.4f}, λ_dom={res['Dominant_wavelength_nm']} nm")
            except Exception as e:
                st.error(f"Error procesando {file.name}: {e}")


# ----------------- XLSX processing -----------------
if xlsx_files:
    st.subheader("XLSX files (multiple sheets supported)")
    for j, file in enumerate(xlsx_files):
        raw = _uploaded_to_bytes(file)
        if not raw:
            st.error(f"Archivo vacío: {file.name}")
            continue

        try:
            xls = pd.ExcelFile(io.BytesIO(raw))
            for k, sheet_name in enumerate(xls.sheet_names):
                with st.expander(f"{file.name} - hoja: {sheet_name}", expanded=(j == 0 and k == 0)):
                    label = st.text_input(
                        f"Label ({file.name} - {sheet_name})",
                        value=f"{file.name} - {sheet_name}",
                        key=f"xlsx_label_{j}_{k}"
                    )
                    color = st.color_picker(f"Color ({file.name} - {sheet_name})", "#000000", key=f"xlsx_color_{j}_{k}")
                    marker = st.selectbox(
                        f"Marker ({file.name} - {sheet_name})",
                        options=["o", "s", "^", "x", "D", "*", "v"],
                        index=0,
                        key=f"xlsx_marker_{j}_{k}"
                    )
                    size = st.slider(f"Tamaño ({file.name} - {sheet_name})", min_value=10, max_value=400, value=80, key=f"xlsx_size_{j}_{k}")
                    wl_min_local = st.number_input(
                        f"λ min ({file.name} - {sheet_name})",
                        min_value=200, max_value=10000, value=int(wl_min),
                        key=f"xlsx_wlmin_{j}_{k}"
                    )
                    wl_max_local = st.number_input(
                        f"λ max ({file.name} - {sheet_name})",
                        min_value=200, max_value=10000, value=int(wl_max),
                        key=f"xlsx_wlmax_{j}_{k}"
                    )
                    interp_local = st.selectbox(
                        f"Interpolation interval ({file.name} - {sheet_name})",
                        [1, 5, 10],
                        index=0,
                        key=f"xlsx_interp_{j}_{k}"
                    )
                    try:
                        df_tmp = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)
                        res = process_and_plot(df_tmp, label, color, marker, size, wl_min_local, wl_max_local, interp_local)
                        results.append(res)
                        st.success(f"Procesado: {label} -> x={res['x']:.4f}, y={res['y']:.4f}, λ_dom={res['Dominant_wavelength_nm']} nm")
                    except Exception as e:
                        st.error(f"Error procesando {file.name} - {sheet_name}: {e}")

        except Exception as e:
            st.error(f"No se pudo leer {file.name} como Excel: {e}")


# ----------------- Results and downloads -----------------
if results:
    try:
        legend = ax.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_ncols)
        frame = legend.get_frame()
        frame.set_facecolor(legend_box_color)
        frame.set_alpha(legend_box_alpha)
        frame.set_linewidth(legend_box_linewidth)
    except Exception:
        try:
            ax.legend(loc="best", fontsize=legend_font_size)
        except Exception:
            pass

    plt.tight_layout()
    st.pyplot(fig)

    results_df = pd.DataFrame(results)
    st.subheader("Tabla de coordenadas")
    st.dataframe(results_df)

    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Descargar tabla CSV",
        data=csv_buf.getvalue(),
        file_name="coordenadas_CIE1931.csv",
        mime="text/csv"
    )

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="tiff", dpi=dpi_save)
    img_buf.seek(0)
    st.download_button(
        "📥 Descargar diagrama TIFF (alta resolución)",
        data=img_buf.getvalue(),
        file_name="diagrama_CIE1931.tiff",
        mime="image/tiff"
    )
else:
    st.info("Aún no hay datasets procesados. Sube archivos CSV o XLSX y configura cada dataset.")
