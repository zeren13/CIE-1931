
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colour

st.title("CIE 1931 - Coordenadas Cromatograficas (Múltiples Archivos)")

uploaded_files = st.file_uploader(
    "Sube uno o varios archivos CSV de emisión (separados por ';')",
    type=["csv"], accept_multiple_files=True)

xy_coords = []

if uploaded_files:
    fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            st.markdown(f"### Archivo {idx+1}: {uploaded_file.name}")
            label = st.text_input(f"Nombre para estas coordenadas", value=f"Espectro {idx+1}", key=idx)

            # Leer y procesar el archivo
            lines = uploaded_file.read().decode("utf-8").splitlines()
            lines = [line.replace(",", ".") for line in lines]
            from io import StringIO
            data = StringIO("\n".join(lines))
            df = pd.read_csv(data, sep=";")
            df.columns = ['wavelength', 'intensity']
            df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
            df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
            df = df.dropna()
            df = df[(df['wavelength'] >= 380) & (df['wavelength'] <= 780)]

            if df.empty:
                st.warning(f"El archivo {uploaded_file.name} no tiene datos en el rango visible.")
                continue

            # Calcular coordenadas cromáticas
            sd = colour.SpectralDistribution(dict(zip(df['wavelength'], df['intensity']))) 
            sd = sd.copy().interpolate(colour.SpectralShape(380, 780, 1))
            cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
            XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs)
            xy = colour.XYZ_to_xy(XYZ)
            xy_coords.append((xy[0], xy[1], label))

            # Mostrar coordenadas y agregar al gráfico
            st.success(f"{label}: x = {xy[0]:.4f}, y = {xy[1]:.4f}")
            ax.plot(xy[0], xy[1], 'o', label=label, markersize=4)

        except Exception as e:
            st.error(f"❌ Error en el archivo {uploaded_file.name}: {e}")

    # Finalizar gráfico
    if xy_coords:
        plt.xlabel("x-chromaticity coordinate")
        plt.ylabel("y-chromaticity coordinate")
        plt.title("CIE 1931 Chromaticity Diagram")
        plt.legend()
        st.pyplot(fig)

        # Exportar imagen
        fig.savefig("cie1931_multi.tiff", dpi=600, format='tiff')
        with open("cie1931_multi.tiff", "rb") as f:
            st.download_button("📥 Descargar imagen TIFF", f, file_name="cie1931_multi.tiff", mime="image/tiff")
