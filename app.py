import base64
from pathlib import Path

import streamlit as st


APP_NAME = "SpectraLab Toolkit"
APP_VERSION = "Beta 1.6"
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logog.jpg"


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
    "CIE 1931",
    "Colorimetría",
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
    "CIE 1931": "🎨",
    "Colorimetría": "🌈",
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

    notice(
        "Los cálculos se realizan en el programa portable para Windows o macOS. "
        "Aquí encontrarás una ruta de aprendizaje, ejemplos y criterios para saber cuándo un resultado es confiable."
    )

    st.subheader("Empieza por tu pregunta")
    st.markdown(
        """
        <div class="question-strip">
          <div><b>¿Quiero observar o comparar?</b><span>Comienza por el Visor. Desde allí puedes enviar la muestra a otros análisis.</span></div>
          <div><b>¿Quiero obtener un parámetro?</b><span>Usa Picos, Band gap, CIE, Colorimetría o Rendimiento cuántico según tu medición.</span></div>
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
    row_2 = st.columns(3)
    with row_2[0]:
        tool_card("🎨", "CIE 1931", "Ubica el color de una emisión en el diagrama xy.", "interpretar cromaticidad, longitud y pureza")
    with row_2[1]:
        tool_card("🌈", "Colorimetría", "Describe objetos y emisiones bajo condiciones declaradas.", "escoger iluminante, observador y espacio de color")
    with row_2[2]:
        tool_card("💡", "Rendimiento cuántico", "Compara una solución con una referencia conocida.", "reconocer las condiciones del método relativo")

    st.subheader("Ruta recomendada")
    step_flow(
        ("Pregunta", "Define qué propiedad quieres observar o estimar."),
        ("Datos", "Comprueba tipo de espectro, unidades y cobertura."),
        ("Análisis", "Aplica el mínimo procesamiento necesario."),
        ("Interpretación", "Revisa diagnóstico, incertidumbre y límites."),
    )

    st.subheader("Identificación")
    identity_left, identity_right = st.columns([1.15, 1], gap="large")
    with identity_left:
        st.markdown("**Autores:** Camilo Serrano y Mario Macias")
        st.markdown(
            "Grupo de Investigación en Cristalografía y Química de Materiales — "
            "CrisQuimMat — Universidad de los Andes."
        )
        st.markdown("#### Cítanos")
        st.markdown("_Publicación en preparación._")
        st.markdown("#### Contacto")
        st.markdown("[ma.maciasl@uniandes.edu.co](mailto:ma.maciasl@uniandes.edu.co)")
        st.markdown("[camiloserrano02@hotmail.com](mailto:camiloserrano02@hotmail.com)")
    with identity_right:
        st.markdown("#### Logos")
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width="stretch")
        else:
            st.warning("No se encontró el recurso gráfico institucional.")


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
    concept_figure(
        "Imagen guía · de la señal óptica a la intersección Eg",
        """
        <rect x="28" y="38" width="210" height="200" rx="16" class="v-box"/>
        <text x="133" y="68" text-anchor="middle" class="v-label">1 · Señal declarada</text>
        <line x1="62" y1="201" x2="208" y2="201" class="v-axis"/><line x1="62" y1="201" x2="62" y2="92" class="v-axis"/>
        <path d="M62 196 C95 195 116 190 132 176 C151 159 158 125 174 102 C184 88 195 82 208 78" fill="none" stroke="#256b8a" stroke-width="4"/>
        <text x="133" y="222" text-anchor="middle" class="v-tiny">hν</text>
        <path d="M245 138 H310" class="v-arrow"/><path d="M302 130 L313 138 L302 146" class="v-arrow"/>
        <rect x="315" y="38" width="235" height="200" rx="16" class="v-violet"/>
        <text x="432" y="68" text-anchor="middle" class="v-label">2 · Hipótesis de Tauc</text>
        <text x="342" y="105" class="v-small">Directa permitida</text><text x="504" y="105" class="v-label">p = 2</text>
        <text x="342" y="133" class="v-small">Indirecta permitida</text><text x="504" y="133" class="v-label">p = ½</text>
        <text x="342" y="161" class="v-small">Directa prohibida</text><text x="504" y="161" class="v-label">p = ⅔</text>
        <text x="342" y="189" class="v-small">Indirecta prohibida</text><text x="504" y="189" class="v-label">p = ⅓</text>
        <text x="432" y="219" text-anchor="middle" class="v-tiny">Cada p genera un gráfico distinto</text>
        <path d="M557 138 H622" class="v-arrow"/><path d="M614 130 L625 138 L614 146" class="v-arrow"/>
        <rect x="627" y="38" width="245" height="200" rx="16" class="v-green"/>
        <text x="750" y="68" text-anchor="middle" class="v-label">3 · Ajuste e intersección</text>
        <line x1="660" y1="201" x2="843" y2="201" class="v-axis"/><line x1="660" y1="201" x2="660" y2="88" class="v-axis"/>
        <path d="M660 198 C700 198 714 192 728 176 C748 153 762 115 786 89 C804 70 823 68 843 67" fill="none" stroke="#257052" stroke-width="4"/>
        <line x1="709" y1="201" x2="806" y2="73" stroke="#9e2f2b" stroke-width="3"/>
        <circle cx="709" cy="201" r="5" fill="#9e2f2b"/><text x="709" y="222" text-anchor="middle" class="v-small">Eg</text>
        """,
        "Eg es la intersección extrapolada de una región elegida en un gráfico que ya depende de la hipótesis. Por eso cuatro hipótesis pueden producir cuatro valores sin que el programa sepa cuál describe al material.",
        "0 0 900 275",
    )
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


elif selected == "CIE 1931":
    page_header(
        "CIE 1931",
        "Coordenadas de cromaticidad para espectros de emisión mediante el observador colorimétrico estándar CIE 1931 de 2°.",
    )
    plain_language(
        "Convierte la forma completa de un espectro de emisión en un punto x,y que resume su cromaticidad, es decir, su color sin expresar cuánto ilumina."
    )
    learning_goals(
        "relacionar el espectro con X, Y, Z y las coordenadas x,y",
        "distinguir cromaticidad de intensidad",
        "reconocer por qué la cobertura espectral importa",
    )
    st.subheader("Del espectro al punto de color")
    step_flow(
        ("Espectro", "Usa intensidades medidas en el visible."),
        ("Observador", "Combina la señal con x̄, ȳ y z̄."),
        ("Integración", "Obtén los valores relativos X, Y y Z."),
        ("Cromaticidad", "Normaliza y ubica el punto x,y."),
    )
    concept_figure(
        "Imagen guía · cómo un espectro se convierte en un punto x,y",
        """
        <rect x="25" y="35" width="235" height="205" rx="16" class="v-box"/>
        <text x="143" y="65" text-anchor="middle" class="v-label">Espectro de emisión S(λ)</text>
        <line x1="55" y1="202" x2="230" y2="202" class="v-axis"/><line x1="55" y1="202" x2="55" y2="90" class="v-axis"/>
        <path d="M55 200 C93 199 111 189 129 149 C146 110 161 82 184 104 C207 126 207 181 230 199" fill="none" stroke="#9e2f2b" stroke-width="4"/>
        <text x="143" y="223" text-anchor="middle" class="v-tiny">380 ← longitud de onda → 780 nm</text>
        <path d="M270 138 H322" class="v-arrow"/><path d="M314 130 L325 138 L314 146" class="v-arrow"/>
        <rect x="327" y="35" width="230" height="205" rx="16" class="v-blue"/>
        <text x="442" y="65" text-anchor="middle" class="v-label">Observador CIE 1931</text>
        <path d="M354 189 C382 180 391 112 417 102 C444 91 451 166 480 184" fill="none" stroke="#d64545" stroke-width="3"/>
        <path d="M354 192 C389 186 411 156 434 112 C457 68 481 79 500 133 C516 176 526 188 535 191" fill="none" stroke="#2e8b57" stroke-width="3"/>
        <path d="M354 191 C368 182 376 87 398 77 C419 67 429 156 455 187" fill="none" stroke="#4067d8" stroke-width="3"/>
        <text x="376" y="218" class="v-tiny">x̄</text><text x="430" y="218" class="v-tiny">ȳ</text><text x="482" y="218" class="v-tiny">z̄</text>
        <path d="M567 138 H619" class="v-arrow"/><path d="M611 130 L622 138 L611 146" class="v-arrow"/>
        <rect x="624" y="35" width="251" height="205" rx="16" class="v-green"/>
        <text x="750" y="65" text-anchor="middle" class="v-label">Diagrama de cromaticidad</text>
        <line x1="656" y1="205" x2="846" y2="205" class="v-axis"/><line x1="656" y1="205" x2="656" y2="82" class="v-axis"/>
        <path d="M677 198 C658 170 660 104 687 83 C726 54 797 108 829 154 L677 198 Z" fill="#edf3ff" stroke="#17212b" stroke-width="2"/>
        <circle cx="754" cy="132" r="8" fill="#9e2f2b"/><line x1="754" y1="132" x2="814" y2="104" stroke="#9e2f2b" stroke-dasharray="5 4"/>
        <text x="817" y="102" class="v-small">(x,y)</text><text x="750" y="224" text-anchor="middle" class="v-tiny">cromaticidad, no brillo</text>
        """,
        "El punto depende de toda la forma espectral ponderada por el observador estándar. Al normalizar X, Y y Z se conserva la cromaticidad, pero se elimina la información de intensidad absoluta.",
        "0 0 900 275",
    )
    reading_guide(
        "Una emisión con cobertura visible suficiente e intensidades no negativas.",
        "Integra el espectro con x̄, ȳ y z̄ para calcular X, Y, Z y normalizarlos a x,y.",
        "Puedes describir cromaticidad y, si la geometría lo permite, longitud dominante y pureza; no brillo absoluto.",
    )
    method(
        "El espectro de emisión se integra con las funciones de igualación de color x̄(λ), ȳ(λ) y z̄(λ) para obtener "
        "los valores triestímulo relativos X, Y y Z. Después se calculan x = X/(X+Y+Z) e y = Y/(X+Y+Z)."
    )
    st.latex(r"X=\int S(\lambda)\,\bar{x}(\lambda)\,d\lambda\qquad Y=\int S(\lambda)\,\bar{y}(\lambda)\,d\lambda\qquad Z=\int S(\lambda)\,\bar{z}(\lambda)\,d\lambda")
    st.latex(r"x=\frac{X}{X+Y+Z}\qquad y=\frac{Y}{X+Y+Z}")
    st.subheader("Cobertura y muestreo")
    st.markdown(
        "La recomendación normalizada para valores triestímulo usa 360–830 nm con paso de 1 nm. CIE también define "
        "métodos abreviados, incluidos intervalos de 380–780 nm y pasos de hasta 5 nm, cuando son apropiados y se ha "
        "revisado su efecto sobre el resultado. SpectraLab no extrapola las funciones CIE fuera de su dominio."
    )
    caution(
        "No deben inventarse intensidades en longitudes de onda no medidas. Un espectro incompleto puede sesgar x, y, "
        "la longitud dominante y la pureza, especialmente si falta una banda o un extremo significativo."
    )
    guided_example(
        "Dos espectros con el mismo color aparente",
        "Si tienen formas relativas semejantes, uno puede ser diez veces más intenso y aun así producir coordenadas x,y casi iguales. La diferencia de brillo no queda representada por x,y.",
    )
    st.subheader("Longitud dominante y pureza")
    st.markdown(
        "Se calculan geométricamente desde el punto blanco hacia la cromaticidad de la muestra. Solo se informan si "
        "la muestra y el punto blanco están dentro del locus espectral cerrado por la línea púrpura. Los puntos "
        "físicamente imposibles se rechazan y la pureza se limita al intervalo 0–100 %."
    )
    st.subheader("Qué no representa")
    st.markdown(
        "Las coordenadas x,y describen cromaticidad, no intensidad luminosa absoluta. Dos espectros con potencias "
        "distintas pueden producir coordenadas semejantes. El color sRGB en pantalla es únicamente una representación aproximada."
    )


elif selected == "Colorimetría":
    page_header(
        "Colorimetría",
        "Análisis de reflectancia, transmitancia y emisión con condiciones colorimétricas declaradas.",
    )
    plain_language(
        "Para un objeto, el color depende tanto de la muestra como de la luz que la ilumina y del observador estándar elegido. Para una emisión, la luz proviene de la propia muestra."
    )
    learning_goals(
        "escoger el modo correcto para reflectancia, transmitancia o emisión",
        "entender el papel del iluminante y del observador",
        "diferenciar un cálculo completo de una estimación aproximada",
    )
    st.subheader("Tres preguntas antes de calcular")
    step_flow(
        ("¿Qué medí?", "Reflectancia, transmitancia o emisión."),
        ("¿Con qué unidades?", "Porcentaje, fracción o intensidad relativa."),
        ("¿Bajo qué condiciones?", "Iluminante y observador cuando aplican."),
        ("¿Qué cobertura?", "Completa 380–780 o aproximada 400–700 nm."),
    )
    concept_figure(
        "Imagen guía · por qué el color de un objeto necesita tres elementos",
        """
        <circle cx="115" cy="132" r="70" fill="#fff4c9" stroke="#bc7b00" stroke-width="2"/>
        <circle cx="115" cy="132" r="28" fill="#ffd34d"/><path d="M115 47 V21 M115 243 V217 M30 132 H4 M226 132 H200 M55 72 L36 53 M194 211 L175 192 M55 192 L36 211 M194 53 L175 72" stroke="#bc7b00" stroke-width="5"/>
        <text x="115" y="266" text-anchor="middle" class="v-label">Iluminante</text>
        <text x="115" y="286" text-anchor="middle" class="v-tiny">A, D50 o D65</text>
        <path d="M203 132 H298" class="v-arrow"/><path d="M290 124 L301 132 L290 140" class="v-arrow"/>
        <rect x="303" y="61" width="190" height="142" rx="22" fill="#77b8d2" stroke="#256b8a" stroke-width="3"/>
        <path d="M328 171 C355 134 383 151 407 112 C433 70 461 103 477 81" fill="none" stroke="#fff" stroke-width="5" opacity=".8"/>
        <text x="398" y="231" text-anchor="middle" class="v-label">Muestra</text>
        <text x="398" y="252" text-anchor="middle" class="v-tiny">R(λ) o T(λ)</text>
        <path d="M500 132 H592" class="v-arrow"/><path d="M584 124 L595 132 L584 140" class="v-arrow"/>
        <path d="M630 132 C650 92 703 92 723 132 C703 172 650 172 630 132 Z" fill="#fff" stroke="#17212b" stroke-width="3"/>
        <circle cx="677" cy="132" r="19" fill="#6453a5"/><circle cx="677" cy="132" r="8" fill="#17212b"/>
        <text x="677" y="206" text-anchor="middle" class="v-label">Observador</text>
        <text x="677" y="227" text-anchor="middle" class="v-tiny">2° o 10°</text>
        <path d="M733 132 H786" class="v-arrow"/><path d="M778 124 L789 132 L778 140" class="v-arrow"/>
        <rect x="791" y="88" width="88" height="88" rx="16" class="v-green"/>
        <text x="835" y="123" text-anchor="middle" class="v-label">XYZ</text><text x="835" y="148" text-anchor="middle" class="v-small">Lab · xy</text>
        """,
        "En reflectancia o transmitancia, el color calculado no pertenece únicamente a la muestra: también depende del iluminante y del observador. En emisión se omite el iluminante porque la muestra produce su propia distribución espectral.",
        "0 0 900 305",
    )
    reading_guide(
        "R(λ), T(λ) o emisión, con unidades, cobertura, iluminante y observador cuando correspondan.",
        "Combina las distribuciones espectrales, integra XYZ y convierte a espacios derivados.",
        "Obtienes valores comparables solo si mantienes las mismas condiciones colorimétricas.",
    )
    st.subheader("Configuración según la medición")
    st.markdown(
        """
        <table class="scope-table">
          <thead><tr><th>Entrada</th><th>Condición necesaria</th><th>Resultados principales</th></tr></thead>
          <tbody>
            <tr><td>Reflectancia</td><td>Iluminante A, D50 o D65; observador 2° o 10°</td><td>XYZ, Lab, LCh, xy, u′v′, ΔE00 y sRGB aproximado</td></tr>
            <tr><td>Transmitancia</td><td>Iluminante A, D50 o D65; observador 2° o 10°</td><td>XYZ, Lab, LCh, xy, u′v′, ΔE00 y sRGB aproximado</td></tr>
            <tr><td>Emisión</td><td>Intensidad espectral relativa; no usa iluminante externo</td><td>XYZ relativo, xy, u′v′ y CCT/Duv cuando son aplicables</td></tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    method(
        "Los objetos reflectantes o transmisivos se evalúan bajo un iluminante y un observador especificados. "
        "La emisión se trata como una distribución espectral propia y no se multiplica por un iluminante externo."
    )
    st.subheader("Cobertura espectral")
    st.markdown(
        """
        - **Cálculo completo del programa:** datos medidos entre 380 y 780 nm.
        - **Estimación aproximada:** cobertura continua mínima entre 400 y 700 nm.
        - En reflectancia o transmitancia, la estimación prolonga los valores de borde hasta 380 y 780 nm.
        - En emisión, la estimación supone intensidad cero fuera del intervalo medido.
        """
    )
    caution(
        "La extensión de bordes es una hipótesis numérica, no una medición. El resultado aproximado debe reportar "
        "el intervalo real, el porcentaje de cobertura y el método aplicado."
    )
    guided_example(
        "El mismo sólido bajo D50 y D65",
        "La reflectancia medida no cambia, pero los valores XYZ y Lab sí pueden cambiar porque el cálculo representa cómo se observaría el objeto bajo dos iluminantes distintos.",
    )
    st.subheader("Comparación y temperatura correlacionada")
    st.markdown(
        "ΔE00 compara muestras bajo las mismas condiciones colorimétricas y respecto de una referencia seleccionada. "
        "La CCT solo se informa para emisiones suficientemente próximas al locus de Planck; no describe cualquier color."
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
    concept_figure(
        "Imagen guía · la muestra se compara con una referencia",
        """
        <rect x="35" y="41" width="290" height="205" rx="17" class="v-blue"/>
        <text x="180" y="72" text-anchor="middle" class="v-label">Referencia conocida</text>
        <rect x="72" y="92" width="72" height="100" rx="8" fill="#d9f0ff" stroke="#256b8a" stroke-width="2"/>
        <path d="M74 150 H142 V190 H74 Z" fill="#78c8ef" opacity=".7"/>
        <path d="M170 190 C192 188 199 168 211 137 C222 108 238 106 250 136 C264 171 273 187 298 190" fill="none" stroke="#256b8a" stroke-width="4"/>
        <text x="108" y="215" text-anchor="middle" class="v-tiny">Aref</text><text x="237" y="215" text-anchor="middle" class="v-tiny">Iref integrada</text>
        <rect x="575" y="41" width="290" height="205" rx="17" class="v-green"/>
        <text x="720" y="72" text-anchor="middle" class="v-label">Muestra problema</text>
        <rect x="612" y="92" width="72" height="100" rx="8" fill="#ecdcff" stroke="#6453a5" stroke-width="2"/>
        <path d="M614 163 H682 V190 H614 Z" fill="#b899df" opacity=".8"/>
        <path d="M710 190 C732 188 739 176 751 151 C763 126 780 122 792 149 C806 177 814 188 838 190" fill="none" stroke="#6453a5" stroke-width="4"/>
        <text x="648" y="215" text-anchor="middle" class="v-tiny">Ax</text><text x="778" y="215" text-anchor="middle" class="v-tiny">Ix integrada</text>
        <path d="M333 140 H420" class="v-arrow"/><path d="M567 140 H480" class="v-arrow"/>
        <circle cx="450" cy="140" r="62" fill="#fff" stroke="#9e2f2b" stroke-width="3"/>
        <text x="450" y="126" text-anchor="middle" class="v-label">Comparación</text><text x="450" y="149" text-anchor="middle" class="v-small">área · absorción</text><text x="450" y="170" text-anchor="middle" class="v-small">índice n²</text>
        <text x="450" y="277" text-anchor="middle" class="v-label">Φx depende de Φref y de condiciones equivalentes</text>
        """,
        "El área de emisión no se interpreta sola. Se corrige por cuánto absorbió cada solución y por el índice de refracción; por eso muestra y referencia deben medirse con la misma excitación y configuración.",
        "0 0 900 300",
    )
    reading_guide(
        "Absorbancia y emisión integrada de muestra y referencia, Φref e índices de refracción.",
        "Compara áreas, fracciones absorbidas y la corrección n² bajo condiciones equivalentes.",
        "Obtienes un Φ relativo para la solución; no es aplicable a sólidos ni corrige por sí solo efectos experimentales.",
    )
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
    with st.expander("Beta 1.6 — reproducibilidad y diagnósticos", expanded=True):
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
    """
    <div class="institutional">
      <span class="label">SpectraLab Toolkit</span><br>
      <span class="small">Documentación correspondiente a Beta 1.6.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
