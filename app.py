# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colour
from colour import MSDS_CMFS, SpectralDistribution, sd_to_XYZ, XYZ_to_xy

# ----------------- Config -----------------
st.set_page_config(layout="wide", page_title="CIE 1931 - Multi Spectra")

# ----------------- App Title -----------------
st.title("CIE 1931 Chromaticity Diagram - Multi Spectra")

# ----------------- User Inputs -----------------
x_label = st.text_input("X axis label:", value="x")
y_label = st.text_input("Y axis label:", value="y")
title = st.text_input("Graph title:", value="CIE 1931 Chromaticity Diagram")

# File upload
uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=["csv", "xlsx"], accept_multiple_files=True)

# ----------------- Function to process spectrum -----------------
def process_spectrum(data):
    data = data.dropna()
    wavelengths = data.iloc[:, 0].values
    intensities = data.iloc[:, 1].values

    # Create spectral distribution
    sd = SpectralDistribution(dict(zip(wavelengths, intensities)))
    sd = sd.copy().align(MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].shape)

    # Convert to XYZ and then xy
    XYZ = sd_to_XYZ(sd, MSDS_CMFS["CIE 1931 2 Degree Standard Observer"])
    xy = XYZ_to_xy(XYZ)
    return xy

# ----------------- Main -----------------
if uploaded_files:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw chromaticity diagram background
    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl = np.arange(380, 781, 5)
    XYZ = colour.wavelength_to_XYZ(wl, cmfs)
    xy = XYZ_to_xy(XYZ)

    ax.plot(xy[..., 0], xy[..., 1], 'k-', linewidth=1)
    ax.fill(xy[..., 0], xy[..., 1], facecolor="none", edgecolor="black")

    # Plot labels with background for readability
    for i, wavelength in enumerate(wl):
        if wavelength % 20 == 0:
            ax.text(
                xy[i, 0], xy[i, 1], str(wavelength),
                fontsize=9, color="black",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.7)
            )

    # Plot user spectra
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            data = pd.read_csv(file)
            xy_point = process_spectrum(data)
            ax.plot(xy_point[0], xy_point[1], 'o', markersize=10, label=file.name)
        elif file.name.endswith(".xlsx"):
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                data = pd.read_excel(file, sheet_name=sheet_name)
                xy_point = process_spectrum(data)
                ax.plot(xy_point[0], xy_point[1], 'o', markersize=10, label=f"{file.name} - {sheet_name}")

    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="lower right")

    st.pyplot(fig)
