import base64
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


APP_NAME = "SpectraLab Toolkit"
APP_VERSION = "Beta 1.8"
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logog.jpg"
UNIVERSITY_LOGO_PATH = APP_DIR / "logo_uniandes.svg"
CIE_LOCUS_PATH = APP_DIR / "CIE_cc_1931_2deg.csv"


st.set_page_config(
    page_title=f"{APP_NAME} — Documentación",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #566572;
        --line: #dce3e8;
        --soft: #f4f7f8;
        --blue: #256b8a;
        --blue-soft: #eaf5fa;
        --green: #257052;
        --green-soft: #edf8f2;
        --violet: #6453a5;
        --violet-soft: #f1eefb;
        --accent: #9e2f2b;
        --accent-soft: #f8eceb;
        --warn: #8a5a00;
        --warn-soft: #fff8e8;
    }
    .stApp { color: var(--ink); }
    .main .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, h4 { letter-spacing: -0.015em; }
    h1 { margin-bottom: .35rem; }
    h2 { margin-top: 2.2rem; }
    p, li { line-height: 1.62; }
    div[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    div[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    div[role="radiogroup"] label {
        border-radius: 8px;
        padding: .28rem .45rem;
    }
    .doc-kicker {
        color: var(--accent);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .doc-lead {
        color: var(--muted);
        font-size: 1.08rem;
        max-width: 800px;
        margin: 0 0 1.5rem 0;
    }
    .version-badge {
        display: inline-block;
        border: 1px solid #d7aaa7;
        background: var(--accent-soft);
        color: #7c211e;
        border-radius: 999px;
        padding: .25rem .7rem;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: .8rem;
    }
    .hero {
        border: 1px solid #d8e6ec;
        border-radius: 18px;
        padding: 2.25rem 2.4rem;
        background: linear-gradient(135deg, #fff 0%, #eef8fb 58%, #f9efee 100%);
        box-shadow: 0 10px 30px rgba(25, 61, 78, .07);
        margin-bottom: 1.4rem;
    }
    .hero h1 { font-size: clamp(2rem, 5vw, 3.3rem); margin: 0 0 .4rem 0; }
    .hero p { color: var(--muted); max-width: 780px; font-size: 1.08rem; margin: 0; }
    .doc-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #fff;
        padding: 1.2rem 1.25rem;
        height: 100%;
        margin-bottom: .75rem;
        box-shadow: 0 4px 14px rgba(31, 54, 68, .045);
    }
    .doc-card .module-icon { font-size: 1.45rem; display: block; margin-bottom: .55rem; }
    .doc-card h3 { font-size: 1.05rem; margin: 0 0 .35rem 0; }
    .doc-card p { color: var(--muted); margin: 0; font-size: .94rem; }
    .doc-card .learn-result { display:block; margin-top:.65rem; color:var(--blue); font-size:.84rem; font-weight:700; }
    .notice, .caution, .method {
        border-radius: 9px;
        padding: .95rem 1.1rem;
        margin: 1rem 0;
    }
    .notice { border-left: 4px solid #4d718a; background: #eef5f8; }
    .caution { border-left: 4px solid #bc7b00; background: var(--warn-soft); }
    .method { border-left: 4px solid var(--accent); background: var(--accent-soft); }
    .notice p, .caution p, .method p { margin: 0; }
    .plain-language {
        border: 1px solid #badbe8;
        border-radius: 12px;
        background: var(--blue-soft);
        padding: 1rem 1.15rem;
        margin: 1rem 0 1.3rem;
    }
    .plain-language b { color: #185875; }
    .plain-language p { margin: 0; }
    .learn-goals {
        border: 1px solid #cae5d7;
        background: var(--green-soft);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: .9rem 0 1.4rem;
    }
    .learn-goals h3 { color: var(--green); font-size: 1rem; margin: 0 0 .5rem; }
    .learn-goals ul { margin: 0; padding-left: 1.2rem; }
    .learn-goals li { margin: .2rem 0; }
    .step-flow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .65rem;
        margin: .8rem 0 1.4rem;
    }
    .step-box {
        border: 1px solid var(--line);
        background: #fff;
        border-radius: 12px;
        padding: .9rem;
        min-height: 118px;
    }
    .step-number {
        display: inline-grid;
        place-items: center;
        width: 1.65rem;
        height: 1.65rem;
        border-radius: 50%;
        background: var(--violet);
        color: #fff;
        font-weight: 800;
        font-size: .78rem;
        margin-bottom: .5rem;
    }
    .step-box b { display:block; margin-bottom:.25rem; }
    .step-box span { color:var(--muted); font-size:.86rem; line-height:1.4; }
    .guided-example {
        border: 1px solid #d8d1f0;
        background: var(--violet-soft);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: 1rem 0;
    }
    .guided-example h3 { color: #51418d; font-size: 1rem; margin: 0 0 .4rem; }
    .guided-example p { margin: 0; }
    .concept-figure {
        border: 1px solid #d8e1e7;
        border-radius: 16px;
        background: linear-gradient(180deg, #ffffff 0%, #f7fafb 100%);
        padding: 1rem 1rem .8rem;
        margin: 1rem 0 1.45rem;
        box-shadow: 0 6px 20px rgba(31, 54, 68, .055);
    }
    .concept-figure h3 {
        font-size: 1rem;
        margin: 0 0 .65rem;
        color: var(--ink);
    }
    .concept-figure img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 11px;
        background: #fff;
    }
    .concept-figure figcaption {
        color: var(--muted);
        font-size: .83rem;
        line-height: 1.45;
        margin-top: .65rem;
    }
    .reading-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .7rem;
        margin: .85rem 0 1.35rem;
    }
    .reading-box {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: .9rem 1rem;
        background: #fff;
    }
    .reading-box b { display:block; color:var(--blue); margin-bottom:.3rem; }
    .reading-box span { color:var(--muted); font-size:.88rem; line-height:1.48; }
    .demo-note {
        border: 1px solid #cfe0e8;
        border-left: 4px solid var(--blue);
        background: #f7fbfd;
        border-radius: 10px;
        color: #3f5967;
        font-size: .88rem;
        padding: .78rem .9rem;
        margin: .15rem 0 1.1rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #cfdde5;
        border-radius: 16px;
        overflow: hidden;
        background: #fbfdfe;
        box-shadow: 0 8px 24px rgba(31, 54, 68, .055);
        margin: 1.15rem 0 1.45rem;
    }
    div[data-testid="stExpander"] details > summary {
        background: linear-gradient(90deg, #eef7fa 0%, #f8fbfc 100%);
        color: #174f68;
        font-weight: 750;
        padding: .9rem 1rem;
    }
    div[data-testid="stExpander"] details[open] > summary {
        border-bottom: 1px solid #dce7ec;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #dbe5ea;
        border-radius: 12px;
        background: #fff;
        padding: .7rem .9rem;
    }
    .result-panel {
        border: 1px solid #ccdce4;
        border-radius: 14px;
        background: #ffffff;
        padding: 1rem 1.1rem;
        margin: .7rem 0 1rem;
    }
    .result-panel .result-label {
        display: block;
        color: var(--muted);
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .result-panel .result-value {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 780;
    }
    .route-chip {
        display: inline-block;
        border: 1px solid #c9dce5;
        border-radius: 999px;
        background: #f4fafc;
        color: #245b73;
        font-size: .82rem;
        font-weight: 700;
        padding: .34rem .7rem;
        margin: .15rem .28rem .15rem 0;
    }
    .question-strip {
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:.7rem;
        margin:1rem 0 1.5rem;
    }
    .question-strip div { border-radius:12px; padding:1rem; background:#fff; border:1px solid var(--line); }
    .question-strip b { color:var(--accent); display:block; margin-bottom:.25rem; }
    .question-strip span { color:var(--muted); font-size:.9rem; }
    .label {
        color: var(--muted);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .scope-table {
        width: 100%;
        border-collapse: collapse;
        margin: .75rem 0 1.25rem 0;
        font-size: .94rem;
    }
    .scope-table th, .scope-table td {
        border-bottom: 1px solid var(--line);
        padding: .7rem .65rem;
        text-align: left;
        vertical-align: top;
    }
    .scope-table th { background: var(--soft); }
    .status-ok { color: #1b6d45; font-weight: 700; }
    .status-approx { color: #8a5a00; font-weight: 700; }
    .status-invalid { color: #982f2a; font-weight: 700; }
    .institutional {
        border-top: 1px solid var(--line);
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        color: var(--muted);
    }
    .small { color: var(--muted); font-size: .88rem; }
    @media (max-width: 720px) {
        .main .block-container { padding: 1.2rem 1rem 3rem; }
        .hero { padding: 1.25rem; }
        .scope-table { font-size: .85rem; }
        .step-flow, .question-strip, .reading-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SECTIONS = [
    "Inicio",
    "Visor de espectros",
    "Análisis de picos",
    "Band gap óptico",
    "Color y colorimetría",
    "Rendimiento cuántico",
    "Datos y diagnósticos",
    "Validación y referencias",
    "Historial de cambios",
]

SECTION_ICONS = {
    "Inicio": "🧭",
    "Visor de espectros": "📈",
    "Análisis de picos": "⛰️",
    "Band gap óptico": "⚡",
    "Color y colorimetría": "🎨",
    "Rendimiento cuántico": "💡",
    "Datos y diagnósticos": "🗂️",
    "Validación y referencias": "✅",
    "Historial de cambios": "🕘",
}


def page_header(title: str, subtitle: str, kicker: str = "Aprende con SpectraLab") -> None:
    st.markdown(f'<div class="doc-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="doc-lead">{subtitle}</p>', unsafe_allow_html=True)


def caution(text: str) -> None:
    st.markdown(f'<div class="caution"><p><b>Advertencia.</b> {text}</p></div>', unsafe_allow_html=True)


def notice(text: str) -> None:
    st.markdown(f'<div class="notice"><p>{text}</p></div>', unsafe_allow_html=True)


def method(text: str) -> None:
    st.markdown(f'<div class="method"><p><b>Principio científico.</b> {text}</p></div>', unsafe_allow_html=True)


def tool_card(icon: str, title: str, description: str, result: str) -> None:
    st.markdown(
        f'<div class="doc-card"><span class="module-icon">{icon}</span><h3>{title}</h3>'
        f'<p>{description}</p><span class="learn-result">Aprenderás: {result}</span></div>',
        unsafe_allow_html=True,
    )


def plain_language(text: str) -> None:
    st.markdown(
        f'<div class="plain-language"><p><b>En palabras simples.</b> {text}</p></div>',
        unsafe_allow_html=True,
    )


def learning_goals(*goals: str) -> None:
    items = "".join(f"<li>{goal}</li>" for goal in goals)
    st.markdown(
        f'<div class="learn-goals"><h3>Al terminar esta sección podrás</h3><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def step_flow(*steps: tuple[str, str]) -> None:
    cards = "".join(
        f'<div class="step-box"><span class="step-number">{index}</span><b>{title}</b><span>{description}</span></div>'
        for index, (title, description) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="step-flow">{cards}</div>', unsafe_allow_html=True)


def guided_example(title: str, text: str) -> None:
    st.markdown(
        f'<div class="guided-example"><h3>Ejemplo guiado · {title}</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def concept_figure(title: str, svg: str, caption: str, view_box: str = "0 0 900 300") -> None:
    svg_document = f'''<svg viewBox="{view_box}" role="img" aria-label="{title}" xmlns="http://www.w3.org/2000/svg">
      <style>
        .v-title {{ font: 700 18px system-ui, sans-serif; fill: #17212b; }}
        .v-label {{ font: 650 15px system-ui, sans-serif; fill: #17212b; }}
        .v-small {{ font: 13px system-ui, sans-serif; fill: #566572; }}
        .v-tiny {{ font: 11px system-ui, sans-serif; fill: #566572; }}
        .v-axis {{ stroke: #73818c; stroke-width: 1.4; }}
        .v-grid {{ stroke: #e5ebef; stroke-width: 1; }}
        .v-arrow {{ stroke: #6453a5; stroke-width: 2.5; fill: none; }}
        .v-box {{ fill: #ffffff; stroke: #cfdbe2; stroke-width: 1.5; }}
        .v-blue {{ fill: #eaf5fa; stroke: #256b8a; stroke-width: 1.5; }}
        .v-green {{ fill: #edf8f2; stroke: #257052; stroke-width: 1.5; }}
        .v-violet {{ fill: #f1eefb; stroke: #6453a5; stroke-width: 1.5; }}
        .v-warn {{ fill: #fff8e8; stroke: #bc7b00; stroke-width: 1.5; }}
      </style>
      {svg}
    </svg>'''
    encoded_svg = base64.b64encode(svg_document.encode("utf-8")).decode("ascii")
    st.markdown(
        f'''<figure class="concept-figure">
        <h3>{title}</h3>
        <img src="data:image/svg+xml;base64,{encoded_svg}" alt="{title}" />
        <figcaption><b>Cómo leerla.</b> {caption}</figcaption>
        </figure>''',
        unsafe_allow_html=True,
    )


def reading_guide(input_text: str, process_text: str, result_text: str) -> None:
    st.markdown(
        f'''<div class="reading-grid">
          <div class="reading-box"><b>1 · Entrada</b><span>{input_text}</span></div>
          <div class="reading-box"><b>2 · Qué hace el programa</b><span>{process_text}</span></div>
          <div class="reading-box"><b>3 · Qué puedes concluir</b><span>{result_text}</span></div>
        </div>''',
        unsafe_allow_html=True,
    )


def demo_note(text: str) -> None:
    st.markdown(
        f'<div class="demo-note"><b>Simulación didáctica.</b> {text} No debe usarse como resultado experimental.</div>',
        unsafe_allow_html=True,
    )


def style_demo_figure(fig: go.Figure, x_title: str, y_title: str, height: int = 350) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=48, r=22, t=44, b=50),
        xaxis_title=x_title,
        yaxis_title=y_title,
        font=dict(family="Arial, sans-serif", size=13, color="#283640"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,.88)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showline=True, linecolor="#9eb0ba", gridcolor="#e8eef1", zeroline=False)
    fig.update_yaxes(showline=True, linecolor="#9eb0ba", gridcolor="#e8eef1", zeroline=False)


PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "scrollZoom": False,
}


def viewer_demo() -> None:
    with st.expander("▶ Prueba interactiva · procesamiento de un espectro"):
        demo_note("La curva se genera con una semilla fija para que puedas repetir la comparación.")
        controls = st.columns(3)
        window = controls[0].slider("Ventana de suavizado", 1, 25, 7, 2, key="demo_viewer_window")
        normalization = controls[1].selectbox(
            "Normalización", ["Ninguna", "Máximo = 1", "Área = 1"], key="demo_viewer_norm"
        )
        show_raw = controls[2].checkbox("Mostrar medición original", True, key="demo_viewer_raw")

        wavelengths = np.linspace(400.0, 700.0, 301)
        clean = np.exp(-0.5 * ((wavelengths - 535.0) / 24.0) ** 2)
        clean += 0.42 * np.exp(-0.5 * ((wavelengths - 610.0) / 36.0) ** 2)
        rng = np.random.default_rng(1931)
        measured = np.clip(clean + rng.normal(0.0, 0.035, wavelengths.size), 0.0, None)
        if window > 1:
            padded = np.pad(measured, window // 2, mode="edge")
            processed = np.convolve(padded, np.ones(window) / window, mode="valid")
        else:
            processed = measured.copy()

        if normalization == "Máximo = 1":
            processed = processed / max(float(np.max(processed)), 1e-12)
        elif normalization == "Área = 1":
            processed = processed / max(float(np.trapezoid(processed, wavelengths)), 1e-12)

        fig = go.Figure()
        if show_raw:
            fig.add_trace(go.Scatter(x=wavelengths, y=measured, name="Medición simulada", line=dict(color="#9e2f2b", width=1.4)))
        fig.add_trace(go.Scatter(x=wavelengths, y=processed, name="Resultado procesado", line=dict(color="#256b8a", width=3)))
        style_demo_figure(fig, "Longitud de onda (nm)", "Señal")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption("Observa si una ventana grande elimina ruido o también borra el hombro cercano a 610 nm.")


def peaks_demo() -> None:
    with st.expander("▶ Prueba interactiva · bandas superpuestas"):
        demo_note("Las componentes son funciones matemáticas conocidas, no especies químicas identificadas.")
        controls = st.columns(2)
        separation = controls[0].slider("Separación entre centros (nm)", 10, 110, 48, key="demo_peaks_sep")
        show_components = controls[1].checkbox("Mostrar componentes", True, key="demo_peaks_components")
        wavelengths = np.linspace(430.0, 650.0, 400)
        center_1, center_2 = 540.0 - separation / 2, 540.0 + separation / 2
        component_1 = np.exp(-0.5 * ((wavelengths - center_1) / 24.0) ** 2)
        component_2 = 0.78 * np.exp(-0.5 * ((wavelengths - center_2) / 31.0) ** 2)
        total = component_1 + component_2 + 0.04
        fig = go.Figure()
        if show_components:
            fig.add_trace(go.Scatter(x=wavelengths, y=component_1 + 0.04, name="Componente 1", line=dict(color="#256b8a", dash="dash")))
            fig.add_trace(go.Scatter(x=wavelengths, y=component_2 + 0.04, name="Componente 2", line=dict(color="#9e2f2b", dash="dash")))
        fig.add_trace(go.Scatter(x=wavelengths, y=total, name="Suma observada", line=dict(color="#17212b", width=4)))
        style_demo_figure(fig, "Longitud de onda (nm)", "Intensidad relativa")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        if separation < 45:
            st.warning("Los centros están muy próximos: la banda puede parecer un único máximo y el ajuste será más ambiguo.")
        else:
            st.info("La separación permite distinguir mejor el hombro, pero todavía no identifica qué origina cada componente.")


def wavelength_to_rgb(wavelength: float) -> str:
    """Approximate display colour for wavelength markers; locus geometry is official CIE data."""
    if 380 <= wavelength < 440:
        red, green, blue = -(wavelength - 440) / 60, 0.0, 1.0
    elif wavelength < 490:
        red, green, blue = 0.0, (wavelength - 440) / 50, 1.0
    elif wavelength < 510:
        red, green, blue = 0.0, 1.0, -(wavelength - 510) / 20
    elif wavelength < 580:
        red, green, blue = (wavelength - 510) / 70, 1.0, 0.0
    elif wavelength < 645:
        red, green, blue = 1.0, -(wavelength - 645) / 65, 0.0
    else:
        red, green, blue = 1.0, 0.0, 0.0
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / 40
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / 80
    else:
        factor = 1.0
    channels = [int(round(255 * (max(channel * factor, 0.0) ** 0.8))) for channel in (red, green, blue)]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def static_cie_coordinates_figure() -> None:
    data = np.loadtxt(CIE_LOCUS_PATH, delimiter=",")
    visible = data[(data[:, 0] >= 380) & (data[:, 0] <= 700)]
    wavelengths, locus_x, locus_y = visible[:, 0], visible[:, 1], visible[:, 2]
    display_points = visible[::4]
    display_colors = [wavelength_to_rgb(value) for value in display_points[:, 0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=locus_x,
            y=locus_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(235,240,243,.72)",
            line=dict(color="#34444f", width=2.2),
            name="Locus espectral CIE 1931",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_points[:, 1],
            y=display_points[:, 2],
            mode="markers",
            marker=dict(size=7, color=display_colors, line=dict(width=0)),
            name="Longitudes de onda",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[locus_x[-1], locus_x[0]],
            y=[locus_y[-1], locus_y[0]],
            mode="lines",
            line=dict(color="#8b4a8d", width=2),
            name="Línea púrpura",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.3127],
            y=[0.3290],
            mode="markers+text",
            marker=dict(size=13, color="#ffffff", line=dict(color="#17212b", width=2)),
            text=["Blanco D65<br>(0.3127, 0.3290)"],
            textposition="bottom right",
            name="Blanco D65",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.300],
            y=[0.600],
            mode="markers+text",
            marker=dict(size=15, color="#2f9d58", line=dict(color="#ffffff", width=2)),
            text=["Ejemplo<br>(0.300, 0.600)"],
            textposition="top right",
            name="Coordenada de ejemplo",
            hoverinfo="skip",
        )
    )
    for label_wavelength in (420, 460, 480, 500, 520, 540, 560, 580, 600, 620, 700):
        index = int(np.argmin(np.abs(wavelengths - label_wavelength)))
        fig.add_annotation(
            x=locus_x[index],
            y=locus_y[index],
            text=str(label_wavelength),
            showarrow=False,
            xshift=11,
            yshift=7,
            font=dict(size=10, color="#43535e"),
        )
    style_demo_figure(fig, "Coordenada de cromaticidad x", "Coordenada de cromaticidad y", 560)
    fig.update_layout(
        title=dict(text="Diagrama de cromaticidad CIE 1931 — observador de 2°", font=dict(size=18)),
        hovermode=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(range=[0, 0.8])
    fig.update_yaxes(range=[0, 0.9], scaleanchor="x", scaleratio=1)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"staticPlot": True, "displaylogo": False, "responsive": True},
    )
    st.caption(
        "Locus espectral construido con las coordenadas oficiales CIE 1931 de 2°. "
        "El punto verde es un ejemplo de lectura y no un resultado experimental. "
        "Fuente: CIE 2019, DOI 10.25039/CIE.DS.mifmy4x4. Los colores en pantalla son orientativos."
    )


def quantum_yield_demo() -> None:
    with st.expander("Laboratorio interactivo · rendimiento cuántico relativo"):
        demo_note(
            "Representa dos soluciones medidas con la misma excitación y configuración. Cambia una variable por vez "
            "para reconocer su efecto sobre el resultado."
        )
        primary = st.columns(3)
        phi_ref = primary[0].slider("Φ de la referencia", 0.05, 1.00, 0.55, 0.01, key="demo_qy_phi_ref")
        area_ratio = primary[1].slider("Relación de áreas Ix / Iref", 0.10, 2.50, 1.00, 0.05, key="demo_qy_area")
        absorbance_sample = primary[2].slider("Absorbancia de muestra Ax", 0.01, 0.30, 0.06, 0.01, key="demo_qy_ax")
        secondary = st.columns(3)
        absorbance_ref = secondary[0].slider("Absorbancia de referencia Aref", 0.01, 0.30, 0.06, 0.01, key="demo_qy_aref")
        n_sample = secondary[1].slider("Índice del solvente nx", 1.30, 1.60, 1.33, 0.01, key="demo_qy_nx")
        n_ref = secondary[2].slider("Índice del solvente nref", 1.30, 1.60, 1.33, 0.01, key="demo_qy_nref")
        absorbed_sample = 1.0 - 10.0 ** (-absorbance_sample)
        absorbed_ref = 1.0 - 10.0 ** (-absorbance_ref)
        phi_sample = phi_ref * area_ratio * (absorbed_ref / absorbed_sample) * (n_sample / n_ref) ** 2

        sensitivity_a = np.linspace(0.01, 0.30, 160)
        sensitivity_phi = phi_ref * area_ratio * (absorbed_ref / (1.0 - 10.0 ** (-sensitivity_a))) * (n_sample / n_ref) ** 2
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sensitivity_a,
                y=sensitivity_phi,
                line=dict(color="#256b8a", width=3),
                name="Sensibilidad a Ax",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[absorbance_sample],
                y=[phi_sample],
                mode="markers",
                marker=dict(size=14, color="#9e2f2b", line=dict(color="#ffffff", width=2)),
                name="Configuración actual",
            )
        )
        style_demo_figure(fig, "Absorbancia de muestra Ax", "Φ relativo calculado", 320)
        fig.update_layout(title=dict(text="Sensibilidad del resultado a la absorbancia", font=dict(size=15)), hovermode="closest")
        fig.update_yaxes(rangemode="tozero")
        result_plot, result_value = st.columns([1.45, 0.75])
        result_plot.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        with result_value:
            st.metric("Φ de la muestra", f"{phi_sample:.3f}")
            st.metric("Fracción absorbida", f"{100 * absorbed_sample:.1f} %")
            st.caption("Resultado didáctico calculado con el método relativo.")

        if absorbance_sample > 0.10 or absorbance_ref > 0.10:
            st.warning("Una o ambas absorbancias superan 0,10: aumenta el riesgo de filtro interno y reabsorción.")
        if phi_sample > 1.0:
            st.error("El resultado supera 1. Revisa unidades, linealidad, correcciones instrumentales y equivalencia experimental.")
        else:
            st.success("El valor permanece en el intervalo físico 0–1; todavía deben comprobarse las condiciones experimentales.")
        st.caption("Mantén Ix/Iref fijo y cambia Ax: la curva muestra por qué el área de emisión no puede interpretarse sola.")


def diagnostics_demo() -> None:
    with st.expander("▶ Prueba interactiva · diagnóstico de cobertura"):
        demo_note("El ejemplo reproduce únicamente la regla de cobertura colorimétrica descrita en esta página.")
        coverage = st.slider("Intervalo realmente medido (nm)", 350, 830, (400, 700), 5, key="demo_diag_coverage")
        points = st.number_input("Número de puntos válidos", min_value=0, max_value=5000, value=301, step=1, key="demo_diag_points")
        if points < 2:
            st.error("INVALID · No hay puntos suficientes para integrar o interpolar.")
        elif coverage[0] <= 380 and coverage[1] >= 780:
            st.success("VALID · Cobertura completa entre 380 y 780 nm.")
        elif coverage[0] <= 400 and coverage[1] >= 700:
            st.warning("APPROXIMATE · Cobertura mínima aceptada; el resultado debe informar la aproximación.")
        else:
            st.error("INVALID · Falta una parte sustancial del intervalo visible requerido.")


with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(f"Documentación · {APP_VERSION}")
    selected = st.radio(
        "Secciones",
        SECTIONS,
        format_func=lambda item: f"{SECTION_ICONS[item]}  {item}",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Esta página explica el alcance y las limitaciones del programa portable. "
        "No sustituye la revisión experimental ni la bibliografía especializada."
    )


if selected == "Inicio":
    st.markdown(
        f"""
        <div class="hero">
          <span class="version-badge">{APP_VERSION}</span>
          <h1>{APP_NAME}</h1>
          <p>Aprende a escoger la herramienta adecuada, preparar tus datos y
          explicar cada resultado sin perder el rigor científico.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Autores e institución")
    identity_text, identity_logos = st.columns([1, 1.35], gap="large")
    with identity_text:
        st.markdown("**Autores:** Camilo Serrano y Mario Macias")
        st.markdown(
            "Grupo de Investigación en Cristalografía y Química de Materiales — "
            "CrisQuimMat — Universidad de los Andes."
        )
        st.markdown("**Citación:** _Publicación en preparación._")
        st.markdown(
            "**Contacto:** [ma.maciasl@uniandes.edu.co](mailto:ma.maciasl@uniandes.edu.co) · "
            "[camiloserrano02@hotmail.com](mailto:camiloserrano02@hotmail.com)"
        )
    with identity_logos:
        crisquimmat_logo, university_logo = st.columns(2, gap="medium")
        with crisquimmat_logo:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width="stretch")
            else:
                st.warning("No se encontró el logo de CrisQuimMat.")
        with university_logo:
            if UNIVERSITY_LOGO_PATH.exists():
                st.image(str(UNIVERSITY_LOGO_PATH), width="stretch")
            else:
                st.warning("No se encontró el logo de la Universidad de los Andes.")

    notice(
        "Los cálculos se realizan en el programa portable para Windows o macOS. "
        "Aquí encontrarás una ruta de aprendizaje, ejemplos y criterios para saber cuándo un resultado es confiable."
    )

    st.subheader("Empieza por tu pregunta")
    st.markdown(
        """
        <div class="question-strip">
          <div><b>¿Quiero observar o comparar?</b><span>Comienza por el Visor. Desde allí puedes enviar la muestra a otros análisis.</span></div>
          <div><b>¿Quiero obtener un parámetro?</b><span>Usa Picos, Band gap, Color y colorimetría o Rendimiento cuántico según tu medición.</span></div>
          <div><b>¿Puedo confiar en el resultado?</b><span>Revisa siempre el diagnóstico, el intervalo medido y las condiciones experimentales.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Herramientas disponibles")
    row_1 = st.columns(3)
    with row_1[0]:
        tool_card("📈", "Visor de espectros", "Compara y prepara espectros sin inventar datos.", "distinguir medición y procesamiento")
    with row_1[1]:
        tool_card("⛰️", "Análisis de picos", "Separa matemáticamente bandas superpuestas.", "leer centros, anchos, áreas y residuos")
    with row_1[2]:
        tool_card("⚡", "Band gap óptico", "Explora Eg mediante modelos de Tauc.", "evitar confundir una hipótesis con una conclusión")
    row_2 = st.columns(2)
    with row_2[0]:
        tool_card("🎨", "Color y colorimetría", "Integra CIE 1931, emisiones y color de objetos en una sola ruta.", "escoger el cálculo según el tipo de medición")
    with row_2[1]:
        tool_card("💡", "Rendimiento cuántico", "Compara una solución con una referencia conocida.", "reconocer las condiciones del método relativo")

    st.subheader("Ruta recomendada")
    step_flow(
        ("Pregunta", "Define qué propiedad quieres observar o estimar."),
        ("Datos", "Comprueba tipo de espectro, unidades y cobertura."),
        ("Análisis", "Aplica el mínimo procesamiento necesario."),
        ("Interpretación", "Revisa diagnóstico, incertidumbre y límites."),
    )

elif selected == "Visor de espectros":
    page_header(
        "Visor de espectros",
        "Visualización comparativa y preprocesamiento controlado de espectros de absorción, emisión o excitación.",
    )
    plain_language(
        "Es la mesa de trabajo inicial: permite mirar varios espectros juntos, limpiarlos con cuidado y decidir qué análisis tiene sentido después."
    )
    learning_goals(
        "reconocer qué columnas necesita el programa",
        "diferenciar recorte, suavizado y normalización",
        "saber cuándo una FWHM global deja de ser válida",
    )
    st.subheader("Del archivo a una comparación")
    step_flow(
        ("Carga", "Selecciona longitud de onda y señal."),
        ("Revisa", "Comprueba tipo, medio y rango realmente medido."),
        ("Procesa", "Aplica solo las correcciones justificadas."),
        ("Compara", "Observa máximos, áreas y formas espectrales."),
    )
    concept_figure(
        "Imagen guía · qué cambia al procesar un espectro",
        """
        <line x1="55" y1="230" x2="405" y2="230" class="v-axis"/><line x1="55" y1="230" x2="55" y2="35" class="v-axis"/>
        <text x="230" y="260" text-anchor="middle" class="v-small">Longitud de onda</text>
        <text x="22" y="135" transform="rotate(-90 22 135)" text-anchor="middle" class="v-small">Señal</text>
        <polyline points="55,217 75,204 95,211 115,183 135,193 155,145 175,155 195,90 215,112 235,63 255,86 275,119 295,109 315,155 335,145 355,191 375,181 405,213" fill="none" stroke="#9e2f2b" stroke-width="2"/>
        <path d="M55 215 C110 207 125 187 158 151 C190 116 211 61 244 77 C280 94 297 144 330 161 C360 178 382 205 405 214" fill="none" stroke="#256b8a" stroke-width="4"/>
        <line x1="235" y1="63" x2="235" y2="230" stroke="#6453a5" stroke-dasharray="6 5"/>
        <text x="246" y="54" class="v-small">máximo</text>
        <rect x="478" y="48" width="370" height="168" rx="16" class="v-blue"/>
        <text x="505" y="82" class="v-label">Rojo · datos medidos</text><text x="505" y="107" class="v-small">Conservan ruido, escala e intensidad original.</text>
        <text x="505" y="143" class="v-label">Azul · datos procesados</text><text x="505" y="168" class="v-small">Facilitan comparar la forma, pero dependen</text><text x="505" y="189" class="v-small">de los parámetros elegidos.</text>
        """,
        "La curva procesada ayuda a ver tendencias, pero no reemplaza la medición original. Comprueba siempre que el suavizado no desplace máximos ni borre hombros.",
        "0 0 900 285",
    )
    reading_guide(
        "Columnas de longitud de onda y señal, con el tipo de espectro y sus unidades.",
        "Recorta, interpola solo dentro del dominio medido y aplica las correcciones seleccionadas.",
        "Puedes comparar forma, posición de máximos y áreas; la intensidad absoluta se pierde si normalizas.",
    )
    viewer_demo()
    method(
        "El visor trabaja únicamente con los puntos medidos dentro del intervalo disponible. "
        "Si el rango solicitado es más amplio que los datos, los extremos no se rellenan ni se extrapolan."
    )
    st.subheader("Datos requeridos")
    st.markdown(
        """
        - Una columna numérica de longitud de onda, normalmente en nanómetros.
        - Una columna numérica de señal: absorbancia, intensidad u otra magnitud declarada por el usuario.
        - Al menos dos puntos válidos y longitudes de onda no duplicadas después de la limpieza.
        """
    )
    st.subheader("Procesamiento disponible")
    st.markdown(
        """
        - Recorte al intervalo medido seleccionado.
        - Interpolación interna sobre una malla regular, sin extender los extremos.
        - Sustracción opcional del mínimo como línea base.
        - Recorte opcional de valores negativos.
        - Suavizado mediante promedio móvil o Savitzky–Golay.
        - Normalización al máximo o al área integrada.
        - Escala, desplazamiento vertical y personalización gráfica.
        """
    )
    caution(
        "El suavizado y la corrección de línea base modifican los datos. Deben registrarse sus parámetros y comprobarse "
        "contra el espectro original. La normalización elimina información sobre intensidad absoluta."
    )
    guided_example(
        "Dos emisiones con intensidades distintas",
        "Si normalizas ambas a máximo = 1 podrás comparar la posición y la forma de las bandas, pero ya no podrás afirmar cuál muestra emitió con mayor intensidad absoluta.",
    )
    st.subheader("Métricas")
    st.markdown(
        "El programa informa máximo, intensidad máxima y área integrada. La FWHM global solo se presenta cuando "
        "se detecta un pico aislado; si existen varios máximos significativos, debe utilizarse Análisis de picos."
    )


elif selected == "Análisis de picos":
    page_header(
        "Análisis de picos y deconvolución",
        "Separación matemática de bandas superpuestas con evaluación explícita de la calidad del ajuste.",
    )
    plain_language(
        "Cuando una banda ancha parece contener varios máximos, esta herramienta prueba si puede describirse como la suma de componentes más simples."
    )
    learning_goals(
        "entender qué representa una componente ajustada",
        "comparar perfiles Gaussianos, Lorentzianos y pseudo-Voigt",
        "usar residuos y métricas sin sobreinterpretar el ajuste",
    )
    st.subheader("Ruta del ajuste")
    step_flow(
        ("Intervalo", "Aísla la región espectral de interés."),
        ("Picos", "Detecta máximos o propone centros manuales."),
        ("Modelo", "Escoge perfil y línea base."),
        ("Diagnóstico", "Examina componentes, residuos y métricas."),
    )
    concept_figure(
        "Imagen guía · una banda observada y sus componentes matemáticas",
        """
        <line x1="55" y1="225" x2="845" y2="225" class="v-axis"/><line x1="55" y1="225" x2="55" y2="35" class="v-axis"/>
        <path d="M55 220 C120 218 160 202 205 156 C244 117 267 73 310 83 C351 92 369 148 398 161 C431 175 447 102 492 83 C546 60 571 117 608 151 C653 193 705 216 845 220" fill="none" stroke="#17212b" stroke-width="5"/>
        <path d="M55 221 C170 220 205 205 240 140 C270 84 306 63 343 140 C365 186 390 216 450 220" fill="none" stroke="#256b8a" stroke-width="3" stroke-dasharray="7 5"/>
        <path d="M315 221 C400 220 432 194 465 119 C490 62 530 64 558 128 C588 196 622 219 705 220" fill="none" stroke="#9e2f2b" stroke-width="3" stroke-dasharray="7 5"/>
        <path d="M485 221 C570 220 605 203 641 166 C682 123 720 128 756 173 C786 210 814 219 845 220" fill="none" stroke="#257052" stroke-width="3" stroke-dasharray="7 5"/>
        <text x="285" y="52" class="v-small" fill="#256b8a">componente 1</text>
        <text x="491" y="48" class="v-small" fill="#9e2f2b">componente 2</text>
        <text x="712" y="118" class="v-small" fill="#257052">componente 3</text>
        <rect x="625" y="35" width="205" height="54" rx="12" class="v-violet"/>
        <text x="728" y="57" text-anchor="middle" class="v-label">Línea negra = suma</text><text x="728" y="77" text-anchor="middle" class="v-tiny">Debe seguir la banda observada</text>
        """,
        "Las curvas de color no son especies identificadas: son componentes del modelo. La línea total, los residuos y la estabilidad de los parámetros indican si la descripción matemática es razonable.",
        "0 0 900 270",
    )
    reading_guide(
        "Un intervalo espectral que contenga la banda completa y una línea base razonable.",
        "Ajusta varias funciones cuya suma intenta reproducir la señal observada.",
        "Obtienes centros, anchuras y áreas del modelo; una asignación química exige evidencia adicional.",
    )
    peaks_demo()
    method(
        "La señal se representa como la suma de una línea base y varios perfiles. El programa permite perfiles "
        "Gaussianos, Lorentzianos y pseudo-Voigt, con detección automática por prominencia o centros definidos manualmente."
    )
    st.subheader("Resultados informados")
    st.markdown(
        """
        - Centro, amplitud, anchura y área de cada componente.
        - Incertidumbres de los parámetros cuando la matriz de covarianza puede estimarse.
        - Curva total ajustada, componentes individuales, línea base y residuos.
        - R², RMSE y AIC como indicadores complementarios de calidad.
        """
    )
    caution(
        "Una componente ajustada no equivale automáticamente a una especie química, transición electrónica o mecanismo. "
        "La asignación requiere evidencia experimental independiente y bibliografía."
    )
    guided_example(
        "Una banda con hombro",
        "Un buen ajuste con dos componentes indica que dos perfiles describen mejor la forma observada. No demuestra por sí solo que existan dos especies químicas.",
    )
    st.subheader("Cuándo no se calcula")
    st.markdown(
        """
        El análisis se rechaza si la señal es constante, hay muy pocos puntos, no se detectan máximos bajo los criterios
        seleccionados o el ajuste no converge. En esos casos el programa no crea picos ficticios y conserva el motivo
        del rechazo en el diagnóstico exportado.
        """
    )


elif selected == "Band gap óptico":
    page_header(
        "Band gap óptico",
        "Estimación exploratoria mediante transformaciones de Tauc, condicionada al tipo de señal y a una hipótesis de transición.",
    )
    plain_language(
        "El gráfico cambia porque cada hipótesis transforma la señal de manera diferente. Por eso también cambia la intersección utilizada como Eg."
    )
    learning_goals(
        "declarar correctamente qué representa el eje Y",
        "entender por qué cada hipótesis produce un Eg diferente",
        "revisar el intervalo lineal antes de aceptar el resultado",
    )
    st.subheader("Cómo se construye una estimación")
    step_flow(
        ("Señal", "Declara absorbancia, α o reflectancia difusa."),
        ("Hipótesis", "Elige una transformación de Tauc provisional."),
        ("Intervalo", "Revisa la región aproximadamente lineal."),
        ("Conclusión", "Reporta Eg condicionado a esa hipótesis."),
    )
    st.subheader("Cómo interpretar el gráfico de Tauc")
    reading_guide(
        "Absorbancia, coeficiente de absorción o reflectancia difusa; nunca un espectro de emisión.",
        "Convierte energía, aplica la potencia elegida y ajusta una región aproximadamente lineal.",
        "Obtienes un Eg condicionado a la hipótesis y al intervalo; no una identificación automática de la transición.",
    )
    method(
        "El programa transforma la señal declarada a una magnitud óptica, construye (αhν)ᵖ frente a hν y extrapola "
        "un intervalo aproximadamente lineal. Para reflectancia difusa utiliza la transformación de Kubelka–Munk."
    )
    st.subheader("Antes de calcular")
    st.markdown(
        """
        1. Indica qué representa el eje Y: absorbancia, coeficiente de absorción o reflectancia difusa.
        2. Confirma sus unidades; no se aceptan espectros de emisión o excitación como entrada.
        3. Selecciona una hipótesis de transición. La opción directa permitida es solo el punto de partida editable.
        4. Revisa y confirma manualmente el intervalo usado para la extrapolación.
        """
    )
    st.subheader("Interpretación de las hipótesis")
    st.markdown(
        """
        | Hipótesis | Potencia mostrada por el programa | Qué significa |
        |---|---:|---|
        | Directa permitida | 2 | Modelo de Tauc evaluado bajo esa hipótesis |
        | Indirecta permitida | 1/2 | Modelo alternativo que requiere sustento físico |
        | Directa prohibida | 2/3 | Hipótesis exploratoria, no identificación automática |
        | Indirecta prohibida | 1/3 | Hipótesis exploratoria, no identificación automática |
        """
    )
    caution(
        "El valor mayor o menor obtenido no determina qué transición es correcta. Tampoco basta escoger el R² más alto. "
        "La naturaleza directa, indirecta, permitida o prohibida debe justificarse con el material, su estructura "
        "electrónica, el intervalo físico de ajuste y evidencia externa."
    )
    guided_example(
        "Un mismo espectro, cuatro valores de Eg",
        "No debes escoger automáticamente el mayor, el menor ni el que tenga mejor R². Cada cifra responde a una pregunta distinta: “¿qué Eg resulta si esta hipótesis fuera válida?”.",
    )
    st.subheader("Control de calidad")
    st.markdown(
        "El mismo intervalo energético se conserva al comparar hipótesis. Se muestran residuos, R², incertidumbre "
        "estadística aproximada de Eg y sensibilidad frente a pequeños cambios de los límites. El resultado permanece "
        "como exploratorio hasta confirmar manualmente el intervalo."
    )


elif selected == "Color y colorimetría":
    page_header(
        "Color y colorimetría",
        "Una sola ruta para interpretar emisiones, reflectancia y transmitancia sin mezclar condiciones experimentales.",
    )
    plain_language(
        "Primero identifica qué mediste. Una emisión produce su propia luz y puede ubicarse directamente en CIE 1931; "
        "un objeto necesita además un iluminante y un observador para describir su color."
    )
    learning_goals(
        "elegir la ruta adecuada para emisión, reflectancia o transmitancia",
        "relacionar el espectro con XYZ, xy, Lab y otros espacios derivados",
        "reconocer cuándo la cobertura o las condiciones impiden comparar resultados",
    )
    st.subheader("Diagrama CIE 1931 con coordenadas")
    static_cie_coordinates_figure()
    reading_guide(
        "Espectro identificado como emisión, reflectancia o transmitancia, con unidades y cobertura real.",
        "Selecciona la ruta, combina las distribuciones necesarias e integra los valores triestímulo XYZ.",
        "Obtienes cromaticidad o color bajo condiciones declaradas; no una propiedad universal independiente del experimento.",
    )

    emission_tab, object_tab, quality_tab = st.tabs(
        ["Emisión y CIE 1931", "Color de objetos", "Cobertura y comparación"]
    )
    with emission_tab:
        st.subheader("De un espectro de emisión a CIE 1931")
        method(
            "La emisión se integra con las funciones de igualación de color del observador seleccionado para obtener "
            "X, Y y Z relativos. Después se normalizan para calcular x e y. No se aplica un iluminante externo."
        )
        st.latex(r"X=\int S(\lambda)\,\bar{x}(\lambda)\,d\lambda\quad Y=\int S(\lambda)\,\bar{y}(\lambda)\,d\lambda\quad Z=\int S(\lambda)\,\bar{z}(\lambda)\,d\lambda")
        st.latex(r"x=\frac{X}{X+Y+Z}\qquad y=\frac{Y}{X+Y+Z}")
        st.markdown(
            "Las coordenadas **x,y describen cromaticidad, no intensidad luminosa absoluta**. La longitud dominante "
            "y la pureza se calculan geométricamente solo cuando el punto blanco y la muestra están dentro del locus físico."
        )
        caution(
            "Un color sRGB mostrado en pantalla es una representación aproximada. No sustituye el espectro ni garantiza "
            "que un monitor reproduzca el color medido."
        )
    with object_tab:
        st.subheader("Reflectancia y transmitancia")
        method(
            "El color de un objeto se calcula combinando su reflectancia o transmitancia con la distribución del "
            "iluminante y las funciones del observador. Cambiar cualquiera de esas condiciones puede cambiar XYZ y Lab."
        )
        st.markdown(
            """
            <table class="scope-table">
              <thead><tr><th>Entrada</th><th>Debe declararse</th><th>Resultados</th></tr></thead>
              <tbody>
                <tr><td>Reflectancia</td><td>Iluminante, observador, geometría y escala</td><td>XYZ, Lab, LCh, xy, u′v′, ΔE00</td></tr>
                <tr><td>Transmitancia</td><td>Iluminante, observador, geometría y escala</td><td>XYZ, Lab, LCh, xy, u′v′, ΔE00</td></tr>
                <tr><td>Emisión</td><td>Observador y distribución espectral</td><td>XYZ relativo, xy, u′v′; CCT/Duv si aplica</td></tr>
              </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
        guided_example(
            "El mismo sólido bajo D50 y D65",
            "La reflectancia medida puede ser la misma, pero XYZ y Lab cambian porque representan la apariencia bajo iluminantes diferentes.",
        )
    with quality_tab:
        st.subheader("Cobertura espectral y comparabilidad")
        st.markdown(
            """
            - **Cálculo completo del programa:** datos medidos continuamente entre 380 y 780 nm.
            - **Estimación aproximada:** cobertura mínima continua entre 400 y 700 nm.
            - Para objetos, la aproximación prolonga los valores de borde; para emisión supone intensidad cero fuera del intervalo.
            - ΔE00 solo debe comparar muestras calculadas con el mismo iluminante, observador y condiciones experimentales.
            - La CCT se informa únicamente para emisiones suficientemente próximas al locus de Planck.
            """
        )
        caution(
            "Extender bordes o suponer intensidad cero no crea datos medidos. El informe debe conservar el intervalo real, "
            "el diagnóstico y la hipótesis aplicada."
        )


elif selected == "Rendimiento cuántico":
    page_header(
        "Rendimiento cuántico relativo",
        "Método comparativo exclusivamente para muestras en solución y una referencia fluorescente conocida.",
    )
    plain_language(
        "La muestra no se evalúa sola: se compara con una referencia conocida medida de la misma manera. Si las condiciones no son equivalentes, la comparación pierde validez."
    )
    learning_goals(
        "identificar los datos de muestra y referencia",
        "entender la corrección por fracción absorbida",
        "reconocer los controles experimentales indispensables",
    )
    st.subheader("Comparación paso a paso")
    step_flow(
        ("Referencia", "Escoge un Φ conocido y apropiado."),
        ("Absorción", "Mide A en la misma longitud de excitación."),
        ("Emisión", "Corrige e integra toda la banda comparable."),
        ("Corrección", "Aplica fracción absorbida e índices n²."),
    )
    st.subheader("Comparación entre muestra y referencia")
    reading_guide(
        "Absorbancia y emisión integrada de muestra y referencia, Φref e índices de refracción.",
        "Compara áreas, fracciones absorbidas y la corrección n² bajo condiciones equivalentes.",
        "Obtienes un Φ relativo para la solución; no es aplicable a sólidos ni corrige por sí solo efectos experimentales.",
    )
    quantum_yield_demo()
    method(
        "El rendimiento cuántico de fluorescencia es la relación entre los fotones emitidos y los fotones absorbidos. "
        "El método relativo compara la emisión integrada, la fracción absorbida y el índice de refracción de la muestra "
        "con los de una referencia medida bajo las mismas condiciones."
    )
    st.latex(r"\Phi_x=\Phi_{ref}\left(\frac{I_x}{I_{ref}}\right)\left(\frac{1-10^{-A_{ref}}}{1-10^{-A_x}}\right)\left(\frac{n_x^2}{n_{ref}^2}\right)")
    st.markdown(
        "Donde I es el área de emisión corregida e integrada, A la absorbancia a la longitud de onda de excitación "
        "y n el índice de refracción del solvente."
    )
    st.subheader("Condiciones experimentales")
    st.markdown(
        """
        - Soluciones ópticamente diluidas y absorbancias bajas y semejantes.
        - Misma longitud de onda de excitación, geometría y configuración instrumental.
        - Corrección por la respuesta espectral del equipo.
        - Integración de toda la banda de emisión sobre intervalos comparables.
        - Referencia apropiada para el intervalo espectral, el solvente y las condiciones experimentales.
        - Control de temperatura, oxígeno, filtro interno, reabsorción y dispersión.
        """
    )
    caution(
        "SpectraLab no calcula rendimiento cuántico de sólidos. Ese experimento requiere una esfera integradora, "
        "geometría, calibración y correcciones que no pueden sustituirse usando solamente reflectancia y área de emisión."
    )
    guided_example(
        "Muestra y referencia con absorbancias diferentes",
        "Aunque sus áreas de emisión sean iguales, la que absorbió menos luz puede tener un Φ mayor. Por eso el programa compara fracciones absorbidas y no solo intensidades de emisión.",
    )
    st.subheader("Referencia incluida")
    st.markdown(
        "El programa incluye datos de sulfato de quinina en H₂SO₄ 0,5 M como referencia trazable. El usuario debe "
        "comprobar que el valor de Φ, la longitud de excitación y las condiciones coincidan con su experimento."
    )


elif selected == "Datos y diagnósticos":
    page_header(
        "Datos, formatos y diagnósticos",
        "Requisitos comunes para que un cálculo sea trazable y para comunicar claramente cuándo no puede realizarse.",
        "Aprende a documentar",
    )
    plain_language(
        "Un resultado científico no es solo una cifra: también necesita el archivo de origen, los parámetros usados y una explicación de cualquier limitación."
    )
    learning_goals(
        "preparar un archivo que el programa pueda interpretar",
        "diferenciar los estados VALID, APPROXIMATE e INVALID",
        "conservar un análisis para que otra persona pueda revisarlo",
    )
    concept_figure(
        "Imagen guía · el diagnóstico decide cómo comunicar el resultado",
        """
        <rect x="25" y="101" width="155" height="78" rx="15" class="v-blue"/>
        <text x="102" y="132" text-anchor="middle" class="v-label">Archivo</text><text x="102" y="156" text-anchor="middle" class="v-small">datos + unidades</text>
        <path d="M180 140 H247" class="v-arrow"/><path d="M239 132 L250 140 L239 148" class="v-arrow"/>
        <polygon points="330,59 410,140 330,221 250,140" fill="#f1eefb" stroke="#6453a5" stroke-width="2"/>
        <text x="330" y="128" text-anchor="middle" class="v-label">¿Supera los</text><text x="330" y="151" text-anchor="middle" class="v-label">controles?</text>
        <path d="M410 140 H482" class="v-arrow"/>
        <path d="M330 59 V27 H482" class="v-arrow"/>
        <path d="M330 221 V253 H482" class="v-arrow"/>
        <rect x="485" y="15" width="365" height="66" rx="14" class="v-green"/>
        <text x="512" y="44" class="v-label">VALID</text><text x="590" y="44" class="v-small">Resultado calculable y controles superados.</text><text x="590" y="64" class="v-tiny">Interpretar con las condiciones experimentales.</text>
        <rect x="485" y="107" width="365" height="66" rx="14" class="v-warn"/>
        <text x="512" y="136" class="v-label">APPROXIMATE</text><text x="640" y="136" class="v-small">Existe una limitación declarada.</text><text x="640" y="156" class="v-tiny">Reportar la aproximación junto con la cifra.</text>
        <rect x="485" y="199" width="365" height="66" rx="14" fill="#f8eceb" stroke="#9e2f2b" stroke-width="1.5"/>
        <text x="512" y="228" class="v-label">INVALID</text><text x="590" y="228" class="v-small">No hay resultado defendible.</text><text x="590" y="248" class="v-tiny">Corregir datos o experimento; no forzar una cifra.</text>
        """,
        "El diagnóstico no califica si el resultado es “bonito” o “feo”; indica si las comprobaciones del programa permiten calcularlo, si depende de una aproximación o si debe rechazarse.",
        "0 0 900 285",
    )
    diagnostics_demo()
    st.subheader("Archivos de entrada")
    st.markdown(
        """
        - CSV con separadores habituales o XLSX con una o más hojas.
        - Encabezados identificables y columnas numéricas seleccionables.
        - Longitud de onda en orden ascendente o reordenable sin ambigüedad.
        - Valores faltantes, infinitos o no numéricos se eliminan antes del cálculo y deben quedar documentados.
        """
    )
    st.subheader("Estados de análisis")
    st.markdown(
        """
        <table class="scope-table">
          <thead><tr><th>Estado</th><th>Significado</th><th>Cómo actuar</th></tr></thead>
          <tbody>
            <tr><td class="status-ok">VALID</td><td>El cálculo superó las verificaciones implementadas.</td><td>Interpretar junto con las condiciones experimentales y la incertidumbre.</td></tr>
            <tr><td class="status-approx">APPROXIMATE</td><td>El resultado depende de cobertura incompleta o de una aproximación declarada.</td><td>Reportar la advertencia y, si es posible, repetir la medición.</td></tr>
            <tr><td class="status-invalid">INVALID</td><td>Los datos no permiten un resultado defendible.</td><td>No usar una cifra; corregir los datos o el diseño experimental.</td></tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Paquete reproducible")
    st.markdown(
        "Cada módulo puede exportar un ZIP con resultados, datos realmente procesados cuando existen, metadatos JSON "
        "e informe diagnóstico. Se registran la versión, los parámetros, el nombre y SHA-256 del archivo de origen, "
        "sin incluir rutas personales ni guardar automáticamente la sesión."
    )
    notice(
        "Cuando un análisis se rechaza, el paquete conserva el motivo y los datos procesables, pero deja vacíos los "
        "resultados que no pudieron calcularse. No se fuerza una cifra mediante interpolación o extrapolación oculta."
    )


elif selected == "Validación y referencias":
    page_header(
        "Validación y referencias",
        "Fuentes normativas, pruebas internas y límites de la evidencia proporcionada por el software.",
        "Aprende a evaluar evidencia",
    )
    plain_language(
        "Que el programa funcione correctamente no significa que cualquier dato de entrada ni cualquier interpretación sean correctos. La validación numérica y la validez experimental son preguntas distintas."
    )
    learning_goals(
        "distinguir pruebas de software de validación experimental",
        "identificar las referencias normativas principales",
        "reportar la información mínima para reproducir un análisis",
    )
    concept_figure(
        "Imagen guía · tres niveles de confianza que no deben confundirse",
        """
        <polygon points="450,22 555,94 345,94" fill="#edf8f2" stroke="#257052" stroke-width="2"/>
        <text x="450" y="63" text-anchor="middle" class="v-label">Interpretación científica</text><text x="450" y="82" text-anchor="middle" class="v-tiny">¿la conclusión tiene sentido para el material?</text>
        <polygon points="337,105 564,105 690,191 210,191" fill="#f1eefb" stroke="#6453a5" stroke-width="2"/>
        <text x="450" y="142" text-anchor="middle" class="v-label">Validación experimental</text><text x="450" y="165" text-anchor="middle" class="v-small">patrones, materiales de referencia y comparación entre laboratorios</text>
        <polygon points="198,202 702,202 820,278 80,278" fill="#eaf5fa" stroke="#256b8a" stroke-width="2"/>
        <text x="450" y="235" text-anchor="middle" class="v-label">Verificación del software</text><text x="450" y="258" text-anchor="middle" class="v-small">pruebas unitarias, casos conocidos, límites y manejo de errores</text>
        """,
        "Las pruebas de código sostienen la base, pero no demuestran por sí solas que el método represente una muestra real. La conclusión científica exige además validación experimental y conocimiento del sistema estudiado.",
        "0 0 900 300",
    )
    reading_guide(
        "Datos de prueba, referencias normativas y, cuando sea posible, materiales o resultados independientes.",
        "Compara cálculos conocidos, verifica salvaguardas y documenta la versión y los parámetros.",
        "Puedes defender la trazabilidad del cálculo; la validez de la interpretación requiere evidencia experimental.",
    )
    st.subheader("Validación implementada")
    st.markdown(
        """
        - Pruebas de colorimetría con espectros conocidos y control de cobertura.
        - Pruebas geométricas para cromaticidades válidas e inválidas en el locus CIE.
        - Datos sintéticos de picos para recuperación de centros y evaluación de ajustes.
        - Casos de band gap con hipótesis y parámetros conocidos, además de entradas que deben rechazarse.
        - Pruebas de salvaguardas científicas y del contenido de los paquetes reproducibles.
        """
    )
    caution(
        "Las pruebas de software verifican coherencia numérica e implementación, pero no sustituyen una validación "
        "experimental interlaboratorio ni demuestran que una hipótesis física sea correcta para una muestra concreta."
    )
    st.subheader("Referencias principales")
    st.markdown(
        """
        1. CIE. [CIE 1931 colour-matching functions, 2° observer](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer). DOI: 10.25039/CIE.DS.xvudnb9b.
        2. ISO/CIE 11664-1:2019. [CIE standard colorimetric observers](https://www.cie.co.at/publications/colorimetry-part-1-cie-standard-colorimetric-observers-0).
        3. ISO/CIE 11664-3 / CIE S 014-3. [CIE tristimulus values](https://www.cie.co.at/publications/colorimetry-part-3-cie-tristimulus-values-1).
        4. IUPAC Gold Book. [Quantum yield](https://goldbook.iupac.org/terms/view/Q04991). DOI: 10.1351/goldbook.Q04991.
        5. Brouwer, A. M. *et al.* [Standards for photoluminescence quantum yield measurements in solution](https://publications.iupac.org/pac/83/12/2213/). *Pure Appl. Chem.* **2011**, 83, 2213–2228.
        6. Eaton, D. F. Reference materials for fluorescence measurement. *Pure Appl. Chem.* **1988**, 60, 1107–1114.
        """
    )
    st.subheader("Qué debe reportarse en un trabajo")
    st.markdown(
        "Versión del programa, módulo, archivo original, parámetros, procesamiento, intervalo analizado, modelo o "
        "hipótesis, diagnóstico, incertidumbre disponible y referencia bibliográfica del método."
    )


elif selected == "Historial de cambios":
    page_header(
        "Historial de cambios",
        "Resumen de las modificaciones que afectan la interpretación científica y la reproducibilidad.",
        "Versiones",
    )
    with st.expander("Beta 1.8 — simplificación visual y figura CIE oficial", expanded=True):
        st.markdown(
            """
            - Se eliminan las guías visuales de Band gap, CIE y Rendimiento cuántico.
            - Se retiran las pruebas interactivas de Band gap y CIE.
            - CIE incorpora un diagrama estático construido con las coordenadas oficiales del locus espectral.
            - Autores e identidad institucional aparecen primero en el inicio y se añade el logo de Uniandes.
            """
        )
    with st.expander("Beta 1.7 — rediseño pedagógico e integración del color"):
        st.markdown(
            """
            - CIE 1931 y Colorimetría se integran en el módulo **Color y colorimetría**.
            - Nueva ruta visual que separa emisiones de reflectancia y transmitancia.
            - Demostraciones interactivas reorganizadas con gráficos y resultados más claros.
            - Nueva guía experimental y análisis de sensibilidad para rendimiento cuántico relativo.
            """
        )
    with st.expander("Beta 1.6 — reproducibilidad y diagnósticos"):
        st.markdown(
            """
            - Paquetes reproducibles por módulo con resultados, datos procesados, metadatos e informe.
            - Registro de versión, parámetros y SHA-256 del archivo fuente.
            - Estados `VALID`, `APPROXIMATE` e `INVALID` con códigos diagnósticos estables.
            - Los análisis rechazados se documentan sin inventar resultados.
            """
        )
    with st.expander("Beta 1.5 — FWHM y band gap"):
        st.markdown(
            """
            - La FWHM global se omite cuando hay varios picos significativos.
            - Band gap incorpora residuos, incertidumbre, intervalo de confianza y sensibilidad a los límites.
            - El resultado de Tauc permanece exploratorio hasta confirmar manualmente el intervalo.
            """
        )
    with st.expander("Beta 1.4 — validación geométrica CIE"):
        st.markdown(
            """
            - Validación de muestra y punto blanco dentro del locus físico.
            - Rechazo de cromaticidades imposibles y purezas fuera del intervalo físico.
            """
        )
    with st.expander("Beta 1.3 — rendimiento cuántico y salvaguardas"):
        st.markdown(
            """
            - Rendimiento cuántico limitado exclusivamente a soluciones.
            - Corrección mediante fracciones absorbidas `(1 − 10⁻ᴬ)`.
            - Advertencias metodológicas IUPAC y rechazo de extrapolaciones no justificadas.
            """
        )
    with st.expander("Beta 1.2 a Beta 0.1"):
        st.markdown(
            "Las versiones anteriores incorporaron Colorimetría, Band gap, Análisis de picos, mejoras del visor, "
            "personalización CIE y la interfaz portable. El historial completo se distribuye con el programa."
        )


st.markdown(
    f"""
    <div class="institutional">
      <span class="label">SpectraLab Toolkit</span><br>
      <span class="small">Documentación correspondiente a {APP_VERSION}.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
