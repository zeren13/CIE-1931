import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_xyY

st.title("CIE 1931 Chromaticity Diagram")

# Configuración personalizada
titulo = st.text_input("Título de la gráfica", "CIE 1931 Chromaticity Diagram")
xlabel = st.text_input("Nombre del eje X", "x")
ylabel = st.text_input("Nombre del eje Y", "y")
titulo_color = st.color_picker("Color del título", "#000000")
ejes_color = st.color_picker("Color de los ejes", "#000000")

# Subida de archivo
uploaded_file = st.file_uploader("Sube un archivo (.csv o .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        sheet_name = st.text_input("Nombre de la hoja (default = primera)", None)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    if "wavelength" in df.columns and "intensity" in df.columns:
        data = dict(zip(df["wavelength"], df["intensity"]))
        sd = SpectralDistribution(data, name="Sample")
        XYZ = sd_to_XYZ(sd)
        xyY = XYZ_to_xyY(XYZ)
        x, y, _ = xyY

        # Graficar diagrama CIE
        fig, ax = plt.subplots(figsize=(8, 6))

        # Dibujar diagrama base
        cmf = pd.read_csv(
            "https://raw.githubusercontent.com/colour-science/colour/master/colour/examples/data/cie_1931_chromaticity.csv"
        )
        ax.plot(cmf["x"], cmf["y"], color="black")

        # Dibujar punto
        ax.scatter(x, y, color="orange", s=100, label="Muestra")

        # Personalizar
        ax.set_title(titulo, color=titulo_color, fontsize=14)
        ax.set_xlabel(xlabel, color=ejes_color, fontsize=12)
        ax.set_ylabel(ylabel, color=ejes_color, fontsize=12)
        ax.tick_params(axis="x", colors=ejes_color)
        ax.tick_params(axis="y", colors=ejes_color)

        ax.legend()
        plt.tight_layout()  # <- evita que se corten números

        st.pyplot(fig)

        # Mostrar coordenadas
        st.write(f"**Coordenadas CIE 1931:** x = {x:.4f}, y = {y:.4f}")
    else:
        st.error("El archivo debe contener las columnas 'wavelength' y 'intensity'.")
