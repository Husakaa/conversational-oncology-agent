import streamlit as st
import requests

# Rutas URL de la API separadas
URL_EXTRACT = "http://127.0.0.1:8000/api/v1/extract"
URL_SYNTHESIZE = "http://127.0.0.1:8000/api/v1/synthesize"

st.title("Agente Conversacional Oncologico")

# Inicializacion del historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = [] # Creamos una lista como "base de datos" no persistente

# Barra lateral para gestion de archivos y herramientas
with st.sidebar:
    st.header("Documentos")
    archivo = st.file_uploader("Subir Analitica txt", type=["txt"])
    
    if archivo:
        texto_analitica = archivo.getvalue().decode("utf-8")
        st.write("Archivo cargado")
        
        st.header("Herramientas")
        
        if st.button("Extraer entidades"):
            payload = {"texto_clinico": texto_analitica} 
            # Llamamos solo al motor Regex
            response = requests.post(URL_EXTRACT, json=payload).json()
            resultado = str(response.get("datos_estructurados", "Error en extracción"))
            st.session_state.messages.append({"role": "assistant", "content": resultado})

        if st.button("Resumen rápido"):
            payload = {"texto_clinico": texto_analitica, "metodologia_prompt": "Zero-Shot"}
            # Llamamos a Ollama
            response = requests.post(URL_SYNTHESIZE, json=payload).json()
            resultado = response.get("contexto_generado", "").split("\n\n")[0]
            st.session_state.messages.append({"role": "assistant", "content": resultado})

        if st.button("Informe completo"):
            payload = {"texto_clinico": texto_analitica, "metodologia_prompt": "CoT"}
            # Llamamos a Ollama
            response = requests.post(URL_SYNTHESIZE, json=payload).json()
            resultado = response.get("sintesis_clinica", "Error en síntesis")
            st.session_state.messages.append({"role": "assistant", "content": resultado})

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada de chat para lenguaje natural
if prompt := st.chat_input("Escriba su consulta"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Respuesta del modelo (simulada en este prototipo)
    with st.chat_message("assistant"):
        respuesta = "Respuesta del modelo basada en su consulta: " + prompt
        st.write(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})