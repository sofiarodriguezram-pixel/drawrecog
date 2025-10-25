import os
import streamlit as st
import base64
from openai import OpenAI
import openai
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# ==================== CONFIGURACIÓN DE PÁGINA ====================
st.set_page_config(page_title='Tablero Inteligente', layout='centered')

# Fondo azul pastel con nubes difuminadas
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #cae9ff 0%, #bde0fe 50%, #a2d2ff 100%);
    background-attachment: fixed;
    position: relative;
}

/* Nubes suaves */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: -100px;
    left: -100px;
    width: 300%;
    height: 300%;
    background: radial-gradient(ellipse at 10% 20%, rgba(255,255,255,0.7) 0%, transparent 70%),
                radial-gradient(ellipse at 70% 10%, rgba(255,255,255,0.6) 0%, transparent 60%),
                radial-gradient(ellipse at 90% 60%, rgba(255,255,255,0.7) 0%, transparent 70%),
                radial-gradient(ellipse at 40% 80%, rgba(255,255,255,0.6) 0%, transparent 70%);
    background-repeat: no-repeat;
    z-index: 0;
}

/* Asegura que el contenido esté por encima del fondo */
[data-testid="stAppViewContainer"] > div {
    position: relative;
    z-index: 1;
}

/* Estilos generales */
h1, h2, h3, h4, h5, h6, p, label {
    font-family: 'Poppins', sans-serif;
    color: #002855;
}
.stButton>button {
    background-color: #f8f9fa;
    color: #003566;
    border-radius: 12px;
    padding: 0.5em 1.5em;
    border: 1px solid #89c2ff;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #a2d2ff;
    color: #001d3d;
}
[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==================== INTERFAZ ====================
st.title('☁️ Tablero Inteligente')
st.subheader("Dibuja el boceto en el panel y presiona el botón para analizarlo")

with st.sidebar:
    st.subheader("🧠 Acerca de:")
    st.write("En esta aplicación se evalúa la capacidad de una máquina para interpretar un boceto.")

# ==================== CANVAS ====================
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider('Selecciona el ancho de línea', 1, 30, 5)
stroke_color = "#000000" 
bg_color = '#FFFFFF'

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=400,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas",
)

# ==================== API KEY Y CLIENTE ====================
ke = st.text_input('🔑 Ingresa tu Clave')
os.environ['OPENAI_API_KEY'] = ke
api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

# ==================== BOTÓN DE ANÁLISIS ====================
analyze_button = st.button("✨ Analiza la imagen", type="secondary")

# ==================== PROCESO DE ANÁLISIS ====================
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None

if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando ..."):
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('img.png')

        base64_image = encode_image_to_base64("img.png")
        prompt_text = "Describe en español brevemente lo que se observa en la imagen"

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

            if response.choices[0].message.content:
                st.markdown("### 🖼️ Resultado del análisis:")
                st.markdown(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

else:
    if not api_key:
        st.warning("Por favor ingresa tu API key.")
