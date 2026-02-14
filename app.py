import base64
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import streamlit as st
import streamlit.components.v1 as components


# =========================
# CONFIGURACIÓN RÁPIDA
# =========================
TITULO_INICIAL = "Holaaa"
MENSAJE_INICIAL = """Tengo algo que decirte...

Me encanta compartir tiempo contigo, y me haces muy feliz.
Hoy quería preguntarte algo especial."""
PREGUNTA = "¿Quieres ser mi San Valentín?"

# Imágenes locales (en la misma carpeta que app.py)
HEADER_IMAGE_FILE = "portada.jpg"  # primera pantalla
HOVER_IMAGE_FILE = "broma.jpg"     # aparece al pasar el cursor sobre "No"

# Fallbacks si no tienes las imágenes aún
HEADER_IMAGE_FALLBACK_URL = "https://images.unsplash.com/photo-1520975958225-1c2d0c8b6a8e?auto=format&fit=crop&w=1200&q=60"
HOVER_IMAGE_FALLBACK_URL = "https://images.unsplash.com/photo-1518199266791-5375a83190b7?auto=format&fit=crop&w=900&q=60"

# Canción (pega aquí tu link de YouTube)
YOUTUBE_LINK = "https://www.youtube.com/watch?v=2Vv-BfVoq4g"


# =========================
# HELPERS
# =========================
def file_to_data_uri(file_path: str, mime: str) -> str | None:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return None
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def guess_mime(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

def youtube_to_embed(url: str) -> str:
    try:
        u = urlparse(url)
        host = u.netloc.lower()

        if "youtu.be" in host:
            video_id = u.path.strip("/").split("/")[0]
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}"

        if "youtube.com" in host:
            if u.path.startswith("/embed/"):
                video_id = u.path.split("/embed/")[1].split("/")[0]
                return f"https://www.youtube.com/embed/{video_id}"

            qs = parse_qs(u.query)
            video_id = (qs.get("v") or [None])[0]
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}"
    except Exception:
        pass

    return "https://www.youtube.com/embed/2Vv-BfVoq4g"


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="San Valentín", page_icon="💚", layout="wide")

# CSS para que el componente ocupe toda la pantalla (ancho y alto)
st.markdown(
    """
    <style>
      /* Ocultar UI de Streamlit */
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}

      /* Quitar márgenes/padding y límites de ancho */
      .block-container {
        max-width: 100% !important;
        padding: 0rem !important;
        margin: 0rem !important;
      }
      [data-testid="stAppViewContainer"] {
        padding: 0rem !important;
      }
      [data-testid="stVerticalBlock"] {
        gap: 0rem !important;
      }

      /* Forzar iframe del componente a 100vh */
      div[data-testid="stIFrame"] {
        height: 100vh !important;
      }
      div[data-testid="stIFrame"] iframe {
        height: 100vh !important;
        width: 100% !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

header_data_uri = file_to_data_uri(HEADER_IMAGE_FILE, guess_mime(HEADER_IMAGE_FILE))
hover_data_uri = file_to_data_uri(HOVER_IMAGE_FILE, guess_mime(HOVER_IMAGE_FILE))

header_src = header_data_uri if header_data_uri else HEADER_IMAGE_FALLBACK_URL
hover_src = hover_data_uri if hover_data_uri else HOVER_IMAGE_FALLBACK_URL

embed_url = youtube_to_embed(YOUTUBE_LINK)

# Fondo (solo corazones rojos y verdes) como SVG repetido (sin emojis)
hearts_svg = (
    "data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%20width%3D%27320%27%20height%3D%27320%27%20viewBox%3D%270%200%20320%20320%27%3E"
    "%3Crect%20width%3D%27100%25%27%20height%3D%27100%25%27%20fill%3D%27none%27%2F%3E"
    "%3Cpath%20d%3D%27M52%2070c0-15%2011-26%2026-26%2010%200%2019%206%2023%2014%204-8%2013-14%2023-14%2015%200%2026%2011%2026%2026%200%2031-49%2057-49%2057S52%20101%2052%2070z%27%20fill%3D%27%23ff3e67%27%20opacity%3D%270.95%27%2F%3E"
    "%3Cpath%20d%3D%27M218%20230c0-14%2010-24%2024-24%209%200%2017%205%2021%2012%204-7%2012-12%2021-12%2014%200%2024%2010%2024%2024%200%2028-45%2052-45%2052s-45-24-45-52z%27%20fill%3D%27%2324b56a%27%20opacity%3D%270.92%27%2F%3E"
    "%3Cpath%20d%3D%27M250%2090c0-12%209-21%2021-21%208%200%2015%204%2018%2010%203-6%2010-10%2018-10%2012%200%2021%209%2021%2021%200%2025-39%2046-39%2046s-39-21-39-46z%27%20fill%3D%27%23ff3e67%27%20opacity%3D%270.70%27%2F%3E"
    "%3Cpath%20d%3D%27M95%20255c0-11%208-19%2019-19%207%200%2013%204%2016%209%203-5%209-9%2016-9%2011%200%2019%208%2019%2019%200%2023-35%2041-35%2041s-35-18-35-41z%27%20fill%3D%27%2324b56a%27%20opacity%3D%270.65%27%2F%3E"
    "%3C%2Fsvg%3E"
)

html = f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{{
      --bg1:#0f1020;
      --bg2:#1c1233;
      --card:#ffffff10;
      --border:#ffffff22;
      --text:#f7f2ff;
      --muted:#d9cfff;
      --shadow: 0 18px 60px rgba(0,0,0,.45);
      --radius: 22px;
    }}

    *{{ box-sizing:border-box; }}

    html, body {{
      width: 100%;
      height: 100%;
    }}

    body{{
      margin:0;
      min-height:100vh;
      width:100%;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color:var(--text);
      background:
        radial-gradient(1200px 700px at 20% 10%, #ff4d8d22, transparent 55%),
        radial-gradient(900px 600px at 90% 30%, #35d07f18, transparent 55%),
        radial-gradient(900px 600px at 70% 90%, #7a5cff22, transparent 55%),
        linear-gradient(160deg, var(--bg1), var(--bg2));
      display:flex;
      align-items:center;
      justify-content:center;
      padding: clamp(14px, 2.2vw, 28px);
      overflow:hidden;
    }}

    .pattern{{
      position:fixed;
      inset:0;
      pointer-events:none;
      opacity:.22;
      background-image: url("{hearts_svg}");
      background-size: 320px 320px;
      background-repeat: repeat;
      filter: blur(.25px);
      z-index:0;
    }}

    .card{{
      width:min(780px, 96vw);
      background:var(--card);
      border:1px solid var(--border);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
      padding: clamp(18px, 2.2vw, 28px);
      backdrop-filter: blur(10px);
      position:relative;
      z-index:2;
      text-align:center;
    }}

    .header-img{{
      width: 190px;
      max-width: 75%;
      height: auto;
      display:block;
      margin: 0 auto 14px auto;
      border-radius: 18px;
      border: 1px solid #ffffff33;
      box-shadow: 0 14px 40px rgba(0,0,0,.35);
    }}

    .title{{
      font-size: clamp(22px, 2.2vw, 30px);
      letter-spacing: .2px;
      margin: 0 0 10px 0;
    }}

    .text{{
      font-size: clamp(15px, 1.25vw, 17px);
      line-height: 1.55;
      color: var(--muted);
      margin: 0;
      white-space: pre-wrap;
    }}

    .divider{{
      height:1px;
      background: linear-gradient(90deg, transparent, #ffffff33, transparent);
      margin:18px 0;
    }}

    .row{{
      display:flex;
      gap:12px;
      flex-wrap:wrap;
      align-items:center;
      justify-content:center;
    }}

    button{{
      border:0;
      padding:12px 18px;
      border-radius: 14px;
      font-size: 16px;
      cursor:pointer;
      user-select:none;
      min-width: 120px;
      transition: transform .08s ease, filter .2s ease;
    }}
    button:active{{ transform: scale(.98); }}

    .btn-primary{{
      background: linear-gradient(180deg, #ffffff22, #ffffff0f);
      color: var(--text);
      border:1px solid #ffffff33;
    }}

    .btn-yes{{
      background: linear-gradient(180deg, #42e690, #24b56a);
      color:#082114;
      font-weight:600;
    }}

    .btn-no{{
      background: linear-gradient(180deg, #ff6b8a, #ff3e67);
      color:#2b0010;
      font-weight:600;
    }}

    .hidden{{ display:none !important; }}

    .hover-img{{
      position: fixed;
      width: 240px;
      max-width: 60vw;
      border-radius: 16px;
      border: 1px solid #ffffff33;
      box-shadow: var(--shadow);
      transform: translate(14px, 14px);
      z-index: 50;
      display:none;
    }}

    .modal{{
      position:fixed;
      inset:0;
      background: rgba(0,0,0,.6);
      display:none;
      align-items:center;
      justify-content:center;
      padding:18px;
      z-index: 100;
    }}
    .modal.open{{ display:flex; }}

    .modal-content{{
      width:min(900px, 100%);
      background:#0b0a14;
      border:1px solid #ffffff1f;
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow:hidden;
    }}

    .modal-top{{
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:12px 14px;
      border-bottom:1px solid #ffffff1a;
      gap:10px;
    }}

    .modal-top span{{
      color:#efe8ff;
      font-size: 14px;
      opacity:.9;
    }}

    .close{{
      background:#ffffff12;
      color:#fff;
      border:1px solid #ffffff2a;
      padding:8px 12px;
      border-radius: 12px;
      cursor:pointer;
    }}

    .video-wrap{{
      aspect-ratio: 16 / 9;
      width:100%;
      background:#000;
    }}
    iframe{{
      width:100%;
      height:100%;
      border:0;
      display:block;
    }}

    @media (max-width: 480px){{
      button{{ width:100%; }}
    }}
  </style>
</head>

<body>
  <div class="pattern"></div>

  <img id="hoverFloat" class="hover-img" alt="Sorpresa" src="{hover_src}" />

  <div class="card">
    <img id="headerImg" class="header-img" alt="Portada" src="{header_src}" />

    <h1 class="title" id="h1">{TITULO_INICIAL}</h1>
    <p class="text" id="message">{MENSAJE_INICIAL}</p>

    <div class="divider"></div>

    <div class="row" id="step1">
      <button class="btn-primary" id="btnContinue">Continuar</button>
    </div>

    <div class="row hidden" id="step2">
      <button class="btn-yes" id="btnYes">Sí</button>
      <button class="btn-no" id="btnNo">No</button>
    </div>
  </div>

  <div class="modal" id="modal">
    <div class="modal-content">
      <div class="modal-top">
        <span>Te la dedico</span>
        <button class="close" id="btnClose">Cerrar</button>
      </div>
      <div class="video-wrap">
        <iframe id="ytFrame" title="Video" allow="autoplay; encrypted-media" allowfullscreen></iframe>
      </div>
    </div>
  </div>

  <script>
    const pregunta = {PREGUNTA!r};
    const embedUrl = {embed_url!r};

    const step1 = document.getElementById("step1");
    const step2 = document.getElementById("step2");
    const btnContinue = document.getElementById("btnContinue");
    const btnYes = document.getElementById("btnYes");
    const btnNo = document.getElementById("btnNo");
    const hoverFloat = document.getElementById("hoverFloat");
    const headerImg = document.getElementById("headerImg");

    const modal = document.getElementById("modal");
    const btnClose = document.getElementById("btnClose");
    const ytFrame = document.getElementById("ytFrame");

    btnContinue.addEventListener("click", () => {{
      document.getElementById("h1").textContent = pregunta;
      step1.classList.add("hidden");
      step2.classList.remove("hidden");
      headerImg.style.display = "none";
    }});

    btnNo.addEventListener("mouseenter", () => {{
      hoverFloat.style.display = "block";
    }});
    btnNo.addEventListener("mouseleave", () => {{
      hoverFloat.style.display = "none";
    }});

    window.addEventListener("mousemove", (e) => {{
      if (hoverFloat.style.display === "block") {{
        const padding = 18;
        let x = e.clientX + padding;
        let y = e.clientY + padding;

        const rectW = hoverFloat.offsetWidth || 240;
        const rectH = hoverFloat.offsetHeight || 160;
        const maxX = window.innerWidth - rectW - 10;
        const maxY = window.innerHeight - rectH - 10;

        if (x > maxX) x = maxX;
        if (y > maxY) y = maxY;

        hoverFloat.style.left = x + "px";
        hoverFloat.style.top = y + "px";
      }}
    }});

    btnYes.addEventListener("click", () => {{
      ytFrame.src = embedUrl + (embedUrl.includes("?") ? "&" : "?") + "autoplay=1";
      modal.classList.add("open");
    }});

    function closeModal(){{
      modal.classList.remove("open");
      ytFrame.src = "";
    }}
    btnClose.addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => {{
      if (e.target === modal) closeModal();
    }});
  </script>
</body>
</html>
"""

# Nota: el height lo dejamos pequeño porque lo fuerza a 100vh con CSS (!important)
components.html(html, height=10, scrolling=False)
