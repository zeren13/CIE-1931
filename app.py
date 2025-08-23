# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import colour
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_xy
from colour import MSDS_CMFS
from colour.colorimetry import SpectralShape
import matplotlib

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")
matplotlib.rcParams.update({'font.size': 12})

st.title("CIE 1931 — Coordenadas cromáticas desde espectros de emisión (múltiples archivos)")

# Sidebar params
st.sidebar.header("Parámetros globales")
wl_min = st.sidebar.number_input("λ mínimo (nm)", min_value=200, max_value=10000, value=380)
wl_max = st.sidebar.number_input("λ máximo (nm)", min_value=200, max_value=10000, value=780)
interp_interval = st.sidebar.selectbox("Intervalo de interpolación (nm)", options=[1, 5, 10], index=0)
dpi_save = st.sidebar.number_input("DPI para exportar imagen TIFF", min_value=72, max_value=1200, value=600)

# NUEVO: Campos para título, nombres y colores de ejes
plot_title = st.sidebar.text_input("Título del gráfico", value="Diagrama cromático CIE 1931")
x_axis_label = st.sidebar.text_input("Etiqueta eje X", value="x")
y_axis_label = st.sidebar.text_input("Etiqueta eje Y", value="y")
title_color = st.sidebar.color_picker("Color del título", "#000000")
axes_color = st.sidebar.color_picker("Color de ejes y ticks", "#000000")

st.markdown("#### 1) Subir archivos")
st.markdown("- **CSV** simples (varios): columnas `wavelength (nm)` y `intensity` (o 1ª y 2ª columnas).")
st.markdown("- **XLSX** (varias hojas): selecciona la hoja para cada archivo.")

csv_files = st.file_uploader("Sube uno o varios archivos CSV", type=["csv"], accept_multiple_files=True)
xlsx_files = st.file_uploader("Sube uno o varios archivos Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

# Helpers -----------------
def read_csv_flexible(file_like):
    """Lee CSV intentando ; o , y reemplazando coma decimal si hace falta."""
    try:
        content = file_like.read()
        if isinstance(content, (bytes, bytearray)):
            text = content.decode('utf-8', errors='replace')
        else:
            text = str(content)
        from io import StringIO
        test = StringIO(text.replace(',', '.'))
        df = pd.read_csv(test, sep=';')
        if df.shape[1] == 1:
            test = StringIO(text.replace(',', '.'))
            df = pd.read_csv(test, sep=',')
        if df.shape[1] == 1:
            test = StringIO(text.replace(',', '.'))
            df = pd.read_csv(test, sep=r'\s+', engine='python')
        return df
    except Exception as e:
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(text), sep=None, engine='python')
            return df
        except Exception as e2:
            raise e

def preprocess_df(df):
    df = df.copy()
    if df.shape[1] < 2:
        raise ValueError("El archivo no tiene al menos 2 columnas.")
    df = df.iloc[:, :2]
    df.columns = ['wavelength', 'intensity']
    df['wavelength'] = pd.to_numeric(df['wavelength'].astype(str).str.replace(',', '.'), errors='coerce')
    df['intensity'] = pd.to_numeric(df['intensity'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna()
    return df

# Precompute locus (chromaticity curve)
_locus_wls = np.arange(380, 781, 1)
_locus_xy = []
_cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

for wl in _locus_wls:
    values = {w: 0.0 for w in _locus_wls}
    values[wl] = 1.0
    sd_tmp = SpectralDistribution(values)
    sd_int = sd_tmp.interpolate(SpectralShape(380, 780, 1))
    XYZ_tmp = sd_to_XYZ(sd_int, cmfs=_cmfs)
    xy_tmp = XYZ_to_xy(XYZ_tmp)
    _locus_xy.append(xy_tmp)

_locus_xy = np.array(_locus_xy)

def dominant_wavelength_from_xy(x, y):
    d = np.sqrt(( _locus_xy[:,0] - x)**2 + (_locus_xy[:,1] - y)**2)
    idx = np.argmin(d)
    return int(_locus_wls[idx])

# Prepare plotting area
fig, ax = plt.subplots(figsize=(7,7))
try:
    fig_cie, ax_cie = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
    ax.clear()
    plt.close(fig)
    fig = fig_cie
    ax = ax_cie
except Exception:
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

# Opcional: poner líneas en el centro
ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0.5, color='gray', linewidth=0.8, linestyle='--')


# ======================== FIX: evitar que se corten los números ========================
# 1) Márgenes más amplios para no cortar labels / ticks
fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.12)
# 2) Asegurar que textos del diagrama no se recorten y sean legibles
for txt in ax.texts:
    txt.set_clip_on(False)
    txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
# 3) Por si acaso, evitar clipping también en los ticks
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_clip_on(False)
# ======================================================================================

# Aplicar título, etiquetas y colores desde sidebar
ax.set_title(plot_title, color=title_color)
ax.set_xlabel(x_axis_label, color=axes_color)
ax.set_ylabel(y_axis_label, color=axes_color)
ax.tick_params(axis='x', colors=axes_color)
ax.tick_params(axis='y', colors=axes_color)

results = []

def process_and_plot(df, label, color, marker, size, wl_min_local, wl_max_local, interp_interval_local):
    df = preprocess_df(df)
    mask = (df['wavelength'] >= wl_min_local) & (df['wavelength'] <= wl_max_local)
    df = df.loc[mask]
    if df.empty:
        raise ValueError("No hay datos en el rango seleccionado.")
    sd = SpectralDistribution(dict(zip(df['wavelength'].values, df['intensity'].values)))
    sd_int = sd.copy().interpolate(SpectralShape(wl_min_local, wl_max_local, interp_interval_local))
    XYZ = sd_to_XYZ(sd_int, cmfs=_cmfs)
    xy = XYZ_to_xy(XYZ)
    x_val, y_val = float(xy[0]), float(xy[1])
    wl_dom = dominant_wavelength_from_xy(x_val, y_val)
    ax.plot(x_val, y_val, marker=marker, color=color, markersize=size/10, linestyle='None', label=label)
    return {"Label": label, "x": x_val, "y": y_val, "Dominant_wavelength_nm": wl_dom}

# Procesar CSV
if csv_files:
    st.subheader("CSV files")
    for i, file in enumerate(csv_files):
        with st.expander(f"Configurar: {file.name}", expanded=(i==0)):
            label = st.text_input(f"Label (CSV {i})", value=file.name, key=f"csv_label_{i}")
            color = st.color_picker(f"Color (CSV {i})", "#000000", key=f"csv_color_{i}")
            marker = st.selectbox(f"Marker (CSV {i})", options=['o','s','^','x','D','*','v'], index=0, key=f"csv_marker_{i}")
            size = st.slider(f"Tamaño (px) (CSV {i})", min_value=6, max_value=200, value=50, key=f"csv_size_{i}")
            wl_min_local = st.number_input(f"λ min (CSV {i})", min_value=200, max_value=10000, value=wl_min, key=f"csv_wlmin_{i}")
            wl_max_local = st.number_input(f"λ max (CSV {i})", min_value=200, max_value=10000, value=wl_max, key=f"csv_wlmax_{i}")
            interp_local = st.selectbox(f"Interpolation interval (CSV {i})", [1,5,10], index=0, key=f"csv_interp_{i}")
            try:
                df_tmp = read_csv_flexible(file)
                res = process_and_plot(df_tmp, label, color, marker, size, wl_min_local, wl_max_local, interp_local)
                results.append(res)
                st.success(f"Procesado: {label} -> x={res['x']:.4f}, y={res['y']:.4f}, λ_dom={res['Dominant_wavelength_nm']} nm")
            except Exception as e:
                st.error(f"Error procesando {file.name}: {e}")

# Procesar XLSX
if xlsx_files:
    st.subheader("XLSX files (multiple sheets supported)")
    for j, file in enumerate(xlsx_files):
        try:
            xls = pd.ExcelFile(file)
            for k, sheet_name in enumerate(xls.sheet_names):
                with st.expander(f"{file.name} - hoja: {sheet_name}", expanded=(j==0 and k==0)):
                    label = st.text_input(f"Label ({file.name} - {sheet_name})", value=f"{file.name} - {sheet_name}", key=f"xlsx_label_{j}_{k}")
                    color = st.color_picker(f"Color ({file.name} - {sheet_name})", "#000000", key=f"xlsx_color_{j}_{k}")
                    marker = st.selectbox(f"Marker ({file.name} - {sheet_name})", options=['o','s','^','x','D','*','v'], index=0, key=f"xlsx_marker_{j}_{k}")
                    size = st.slider(f"Tamaño (px) ({file.name} - {sheet_name})", min_value=6, max_value=200, value=50, key=f"xlsx_size_{j}_{k}")
                    wl_min_local = st.number_input(f"λ min ({file.name} - {sheet_name})", min_value=200, max_value=10000, value=wl_min, key=f"xlsx_wlmin_{j}_{k}")
                    wl_max_local = st.number_input(f"λ max ({file.name} - {sheet_name})", min_value=200, max_value=10000, value=wl_max, key=f"xlsx_wlmax_{j}_{k}")
                    interp_local = st.selectbox(f"Interpolation interval ({file.name} - {sheet_name})", [1,5,10], index=0, key=f"xlsx_interp_{j}_{k}")
                    try:
                        df_tmp = pd.read_excel(file, sheet_name=sheet_name)
                        res = process_and_plot(df_tmp, label, color, marker, size, wl_min_local, wl_max_local, interp_local)
                        results.append(res)
                        st.success(f"Procesado: {label} -> x={res['x']:.4f}, y={res['y']:.4f}, λ_dom={res['Dominant_wavelength_nm']} nm")
                    except Exception as e:
                        st.error(f"Error procesando {file.name} - {sheet_name}: {e}")
        except Exception as e:
            st.error(f"No se pudo leer {file.name} como Excel: {e}")

if results:
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()  # evita que se corten números y ejes
    st.pyplot(fig)

    results_df = pd.DataFrame(results)
    st.subheader("Tabla de coordenadas")
    st.dataframe(results_df)

    csv_buf = io.StringIO()
    results_df.to_csv(csv_buf, index=False)
    st.download_button("📥 Descargar tabla CSV", data=csv_buf.getvalue(), file_name="coordenadas_CIE1931.csv", mime="text/csv")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='tiff', dpi=dpi_save)
    img_buf.seek(0)
    st.download_button("📥 Descargar diagrama TIFF (alta resolución)", data=img_buf.getvalue(), file_name="diagrama_CIE1931.tiff", mime="image/tiff")
else:
    st.info("Aún no hay datasets procesados. Sube archivos CSV o XLSX y configura cada dataset.")
