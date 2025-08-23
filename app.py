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
