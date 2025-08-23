# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import colour
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_xy

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")

st.title("CIE 1931 - Multi Spectra Plotter")

# Sidebar config
st.sidebar.header("Plot settings")
plot_title = st.sidebar.text_input("Plot title", "CIE 1931 Chromaticity Diagram")
x_axis_label = st.sidebar.text_input("X-axis label", "x")
y_axis_label = st.sidebar.text_input("Y-axis label", "y")
title_color = st.sidebar.color_picker("Title color", "#000000")
axes_color = st.sidebar.color_picker("Axes color", "#000000")

# File upload
st.sidebar.header("Upload your spectra")
uploaded_csv = st.sidebar.file_uploader("Upload CSV (wavelength, intensity)", type=["csv"], accept_multiple_files=True)
uploaded_xlsx = st.sidebar.file_uploader("Upload XLSX (multiple sheets allowed)", type=["xlsx"], accept_multiple_files=True)

spectra = []

# Procesar CSV
if uploaded_csv:
    for file in uploaded_csv:
        df = pd.read_csv(file)
        if len(df.columns) >= 2:
            label = st.sidebar.text_input(f"Label for {file.name}", file.name)
            spectra.append((df, label))

# Procesar XLSX
if uploaded_xlsx:
    for file in uploaded_xlsx:
        xls = pd.ExcelFile(file)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if len(df.columns) >= 2:
                label = st.sidebar.text_input(f"Label for {file.name} - {sheet_name}", f"{file.name}-{sheet_name}")
                spectra.append((df, label))

# ----------------- Plot -----------------
if spectra:
    # Prepare plotting area
    fig, ax = plt.subplots(figsize=(7,7))
    try:
        fig_cie, ax_cie = colour.plotting.plot_chromaticity_diagram_CIE1931(
            standalone=False,
            spectral_labels=True
        )
        ax.clear()
        plt.close(fig)
        fig = fig_cie
        ax = ax_cie
    except Exception:
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)

    # Ajustar márgenes para que no se corten labels
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)

    # Hacer visibles las etiquetas de longitudes de onda
    for txt in ax.texts:
        txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))

    # Procesar espectros y calcular coordenadas
    for df, label in spectra:
        wl = df.iloc[:, 0].astype(float).values
        inten = df.iloc[:, 1].astype(float).values
        data = dict(zip(wl, inten))

        try:
            sd = SpectralDistribution(data, name=label)
            XYZ = sd_to_XYZ(sd)
            xy = XYZ_to_xy(XYZ)
            ax.plot(xy[0], xy[1], 'o', label=label)
            ax.text(xy[0]+0.01, xy[1]+0.01, label, fontsize=8, color="black")
        except Exception as e:
            st.error(f"Error processing {label}: {e}")

    # Aplicar título y ejes personalizados
    ax.set_title(plot_title, color=title_color)
    ax.set_xlabel(x_axis_label, color=axes_color)
    ax.set_ylabel(y_axis_label, color=axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)

    ax.legend()
    st.pyplot(fig)
else:
    st.info("Upload at least one CSV or XLSX file to plot.")
