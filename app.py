import streamlit as st
import requests
import time

from src.ui.formatters import ner_to_markdown, quick_summary, clinical_context


# Rutas URL de la API separadas
URL_EXTRACT = "http://127.0.0.1:8000/api/v1/extract"
URL_SYNTHESIZE = "http://127.0.0.1:8000/api/v1/synthesize"
URL_CONSULT = "http://127.0.0.1:8000/api/v1/consult"

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
            response = requests.post(URL_EXTRACT, json=payload)
            response.raise_for_status()
            datos_json = response.json().get("datos_estructurados", {})
            # Usamos el formateador 
            resultado_limpio = ner_to_markdown(datos_json)
            st.session_state.messages.append({"role": "assistant", "content": resultado_limpio})

        if st.button("Resumen rápido"):
            payload = {"texto_clinico": texto_analitica}
            # Llamamos al motor Regex 
            response = requests.post(URL_EXTRACT, json=payload)
            response.raise_for_status()    
            # Extraemos el JSON
            datos_json = response.json().get("datos_estructurados", {})
            # Generamos la línea clínica 
            resultado_resumen = quick_summary(datos_json)
            st.session_state.messages.append({"role": "assistant", "content": resultado_resumen})

        if st.button("Informe completo"):
            payload = {"texto_clinico": texto_analitica, "metodologia_prompt": "CoT+FS"}
            
            try:
                # Feedback visual porque Ollama tarda en responder
                with st.spinner("🧠 Generando síntesis clínica con Qwen3.5..."):
                    # Iniciamos cronómetro justo antes de llamar a la API
                    inicio = time.time()
                    # Llamamos a Ollama
                    response = requests.post(URL_SYNTHESIZE, json=payload)
                    
                    if not response.ok:
                        st.error(f"Error del backend: {response.text}")
                        response.raise_for_status()
                    
                    # Paramos el cronómetro al recibir la respuesta
                    fin = time.time()
                    tiempo_total = fin - inicio

                    # Extraemos síntesis
                    resultado = response.json().get("sintesis_clinica", "Error leyendo la síntesis.")

                    # Formateamos el resultado añadiendo el tiempo
                    resultado_con_tiempo = f"{resultado}\n\n---\n*Pensó durante {tiempo_total:.2f}s*"
                    
                    st.session_state.messages.append({"role": "assistant", "content": resultado_con_tiempo})
            
            except requests.exceptions.ConnectionError:
                st.error("Error de conexión: El servidor FastAPI no está corriendo.")
            except Exception as e:
                st.error(f"Error en la petición: {e}")

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada de chat para lenguaje natural
if consult := st.chat_input("Escriba su consulta"):
    st.session_state.messages.append({"role": "user", "content": consult})
    with st.chat_message("user"):
        st.write(consult)

    # Respuesta del modelo 
    with st.chat_message("assistant"):
        payload = {"consulta": consult} 
        
        try:
            # Llamamos a Ollama
            response = requests.post(URL_CONSULT, json=payload)
            response.raise_for_status() 
            
            respuesta = response.json().get("respuesta", "Error leyendo respuesta del backend.")
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            st.write(respuesta)
            
        except requests.exceptions.ConnectionError:
            st.error("Error de conexión: El servidor FastAPI no está corriendo.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error en la petición: {e}")
        