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

# Fondo azul claro con efecto de nubes (sin imagen externa)
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #cce5ff 0%, #b3daff 50%, #99ccff 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.0);
}
h1, h2, h3, h4, h5, h6, p, label {
    font-family: 'Poppins', sans-serif;
    color: #003566;
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
if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando ..."):
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('img.png')

        def encode_image_to_base64(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            except FileNotFoundError:
                return None

        base64_image = encode_image_to_base64("img.png")
        prompt_text = "Describe in spanish briefly the image"

        try:
            full_response = ""
            message_placeholder = st.empty()
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

            if response.choices[0].message.content is not None:
                full_response += response.choices[0].message.content
                message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

else:
    if not api_key:
        st.warning("Por favor ingresa tu API key.")
