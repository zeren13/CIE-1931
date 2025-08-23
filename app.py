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

# ----------------- Sidebar -----------------
st.sidebar.header("Opciones de visualización")
interp_interval = st.sidebar.number_input(
    "Intervalo de interpolación (nm)", min_value=1, max_value=10, value=1
)

# Subida de archivos CSV
st.sidebar.subheader("Subir archivos CSV (una hoja)")
uploaded_csvs = st.sidebar.file_uploader(
    "Selecciona uno o varios CSV",
    type=["csv"],
    accept_multiple_files=True,
    key="csv",
)

# Subida de archivos XLSX
st.sidebar.subheader("Subir archivos XLSX (varias hojas)")
uploaded_excels = st.sidebar.file_uploader(
    "Selecciona uno o varios XLSX",
    type=["xlsx"],
    accept_multiple_files=True,
    key="excel",
)

# ----------------- Main -----------------
st.title("Generador de coordenadas CIE 1931 (x, y)")

# Configuración de gráfico
graph_title = st.text_input("Título de la gráfica", "Diagrama CIE 1931")
x_label = st.text_input("Nombre del eje X", "x")
y_label = st.text_input("Nombre del eje Y", "y")

# Crear figura
fig, ax = plt.subplots(figsize=(8, 8))

# Dibujar contorno del diagrama CIE 1931
cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
wavelengths = np.arange(380, 781, interp_interval)
xy_coords = []

for wl in wavelengths:
    sd = SpectralDistribution({wl: 1.0}, name=f"{wl} nm")
    sd.interpolate(SpectralShape(380, 780, interp_interval))
    XYZ = sd_to_XYZ(sd, cmfs=cmfs)
    xy = XYZ_to_xy(XYZ)
    xy_coords.append(xy)

xy_coords = np.array(xy_coords)
ax.plot(xy_coords[:, 0], xy_coords[:, 1], color="black")
ax.set_title(graph_title)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.9)

# Etiquetar algunas longitudes de onda
for i, wl in enumerate(wavelengths[::20]):
    xy = xy_coords[::20][i]
    ax.text(
        xy[0],
        xy[1],
        str(wl),
        fontsize=8,
        ha="center",
        va="center",
    )

# ----------------- Aquí está el FIX -----------------
# Ajustar márgenes para que no se corten etiquetas
fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)

# Poner fondo blanco a los labels de longitudes de onda
for txt in ax.texts:
    txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
# ---------------------------------------------------

# Procesar archivos CSV
if uploaded_csvs:
    for uploaded_file in uploaded_csvs:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] >= 2:
            wl = df.iloc[:, 0].to_numpy()
            inten = df.iloc[:, 1].to_numpy()

            sd = SpectralDistribution(dict(zip(wl, inten)))
            sd.interpolate(SpectralShape(380, 780, interp_interval))
            XYZ = sd_to_XYZ(sd, cmfs=cmfs)
            xy = XYZ_to_xy(XYZ)

            ax.plot(xy[0], xy[1], "o", label=uploaded_file.name)

# Procesar archivos XLSX
if uploaded_excels:
    for uploaded_file in uploaded_excels:
        xls = pd.ExcelFile(uploaded_file)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if df.shape[1] >= 2:
                wl = df.iloc[:, 0].to_numpy()
                inten = df.iloc[:, 1].to_numpy()

                sd = SpectralDistribution(dict(zip(wl, inten)))
                sd.interpolate(SpectralShape(380, 780, interp_interval))
                XYZ = sd_to_XYZ(sd, cmfs=cmfs)
                xy = XYZ_to_xy(XYZ)

                ax.plot(xy[0], xy[1], "o", label=f"{uploaded_file.name}-{sheet_name}")

ax.legend(fontsize=8)
st.pyplot(fig)
