import streamlit as st
import base64
import os
from pathlib import Path

from main import main 


st.set_page_config(page_title="Proyecto IA - Analizador", layout="centered")

#  Estilo Retorika (azul/blanco) 
RETORIKA_BLUE = "#1f4fa3"

st.markdown(
    f"""
    <style>
    :root {{
        --retorika-blue: {RETORIKA_BLUE};
    }}

    /* Botón principal */
    .stButton > button {{
        background-color: var(--retorika-blue);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }}

    /* Títulos */
    h1, h2, h3 {{
        color: var(--retorika-blue);
    }}

    /* Cabecera */
    header[data-testid="stHeader"] {{
        background-color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Logo fijo arriba derecha
def add_logo_top_right(path: str, width_px: int = 140):
    if not Path(path).exists():
        st.warning(f"No encuentro el logo: {path}")
        return

    img_bytes = Path(path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .retorika-logo {{
            position: fixed;
            top: 12px;
            right: 16px;
            z-index: 999999;
        }}
        </style>
        <div class="retorika-logo">
            <img src="data:image/png;base64,{encoded}" width="{width_px}">
        </div>
        """,
        unsafe_allow_html=True
    )

add_logo_top_right("retorika_logo.png", width_px=140)

# UI
st.title("Analizador de Comunicación con IA")
st.write("Sube un video (.mp4/.mov/.avi) y genera un informe PDF con voz + lenguaje corporal.")

uploaded = st.file_uploader("Sube tu video", type=["mp4", "mov", "avi"])

if uploaded is not None:
    # Guardar con el nombre que main.py espera
    with open("video.mp4", "wb") as f:
        f.write(uploaded.read())

    st.success("Video cargado como video.mp4")

    if st.button("Analizar"):
        # borrar outputs anteriores (si existen)
        for fn in ["audio.wav", "informe.pdf", "radar_body.png", "radar_voice.png"]:
            if os.path.exists(fn):
                try:
                    os.remove(fn)
                except PermissionError:
                    st.warning(f"Cierra {fn} si está abierto y vuelve a intentar.")
                    st.stop()

        with st.spinner("Analizando..."):
            main()

        # Mostrar PDF
        if os.path.exists("informe.pdf"):
            st.success("Informe generado correctamente")

            with open("informe.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                label=" Descargar informe.pdf",
                data=pdf_bytes,
                file_name="informe.pdf",
                mime="application/pdf"
            )

           
    
        