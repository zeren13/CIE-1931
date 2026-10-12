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
        .step-flow, .question-strip { grid-template-columns: 1fr; }
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
