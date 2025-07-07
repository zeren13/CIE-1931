
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colour

st.title("CIE 1931 - Coordenadas Cromatograficas")

uploaded_file = st.file_uploader("Sube tu archivo de Espectro de emisión (CSV separado por ';')", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer archivo como texto y reemplazar coma decimal por punto
        lines = uploaded_file.read().decode("utf-8").splitlines()
        lines = [line.replace(",", ".") for line in lines]

        # Leer usando pandas
        from io import StringIO
        data = StringIO("\n".join(lines))
        df = pd.read_csv(data, sep=";")
        df.columns = ['wavelength', 'intensity']
        df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        df = df.dropna()

        # Filtrar al rango visible
        df = df[(df['wavelength'] >= 380) & (df['wavelength'] <= 780)]

        # Crear distribución espectral e interpolar
        sd = colour.SpectralDistribution(dict(zip(df['wavelength'], df['intensity'])))
        sd = sd.copy().interpolate(colour.SpectralShape(380, 780, 1))

        # Cargar funciones de igualación
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

        # Calcular coordenadas
        XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs)
        xy = colour.XYZ_to_xy(XYZ)

        st.success(f"Coordenadas CIE 1931: x = {xy[0]:.4f}, y = {xy[1]:.4f}")

        # Graficar
        fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
        ax.plot(xy[0], xy[1], 'o', color='black', markersize=4, label='Coordenada cromatografica')
        plt.xlabel("x-chromaticity coordinate")
        plt.ylabel("y-chromaticity coordinate")
        plt.title("CIE 1931 Chromaticity Diagram")
        plt.legend()
        st.pyplot(fig)

        # Guardar como TIFF
        fig.savefig("cie1931_emision_pl.tiff", dpi=600, format='tiff')
        with open("cie1931_emision_pl.tiff", "rb") as f:
            st.download_button("📥 Descargar imagen TIFF", f, file_name="cie1931_emision_pl.tiff", mime="image/tiff")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
