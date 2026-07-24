import streamlit as st
import requests
import time
import os

from src.ui.formatters import ner_to_dataframe, aplicar_estilo, quick_summary, clinical_context

# Analítica de ejemplo
ANALITICA_EJEMPLO = """
HEMATOLOGÍA:
Neutrófilos abs: 0.45 x10^3/µL [1.5 - 7.5]
Hemoglobina: 8.2 g/dL [13.0 - 17.0]
Plaquetas: 85 x10^3/µL [150 - 450]

BIOQUÍMICA:
Creatinina: 2.1 mg/dL [0.7 - 1.2]
LDH: 450 U/L [135 - 225]
Bilirrubina total: 1.1 mg/dL [0.1 - 1.2]
"""

# Rutas URL de la API separadas
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
URL_EXTRACT = f"{BACKEND_URL}/api/v1/extract"
URL_SYNTHESIZE = f"{BACKEND_URL}/api/v1/synthesize"
URL_CONSULT = f"{BACKEND_URL}/api/v1/consult"


st.title("Agente Conversacional Oncológico")

# Inicializacion del historial del chat y texto de la analítica
if "messages" not in st.session_state:
    st.session_state.messages = [] # Creamos una lista como base de datos no persistente
if "texto_analitica" not in st.session_state:
    st.session_state.texto_analitica = ""

# Barra lateral para gestion de archivos y herramientas
with st.sidebar:
    st.header("Documentos")
    
    # Botón para cargar el ejemplo
    if st.button("Cargar Analítica de Ejemplo"):
        st.session_state.texto_analitica = ANALITICA_EJEMPLO
        
    archivo = st.file_uploader("Subir Analitica txt", type=["txt"])
    
    # Si se sube un archivo, sobrescribe el estado
    if archivo:
        if not archivo.name.lower().endswith(".txt"):
            st.error("El archivo debe tener extensión .txt para ser procesado.")
        else:
            try:
                st.session_state.texto_analitica = archivo.getvalue().decode("utf-8")
                st.success("Archivo cargado correctamente.")
            except Exception:
                st.error("No se pudo leer el archivo. Asegúrese de que sea un .txt válido.")
        
    # Solo mostramos el área de edición y las herramientas si hay texto cargado
    if st.session_state.texto_analitica:
        # text_area permite previsualizar y editar la analítica
        texto_analitica = st.text_area(
            "Contenido a procesar:", 
            value=st.session_state.texto_analitica, 
            height=200
        )
        st.session_state.texto_analitica = texto_analitica
        
        st.header("Herramientas")
        
        def get_error_detail(response):
            try:
                return response.json().get("detail", response.text)
            except ValueError:
                return response.text

        if st.button("Extraer entidades"):
            payload = {"texto_clinico": texto_analitica}
            try:
                response = requests.post(URL_EXTRACT, json=payload, timeout=20)
                if not response.ok:
                    detalle = get_error_detail(response)
                    st.error(f"No se pudo extraer la analítica: {detalle}")
                else:
                    datos_json = response.json().get("datos_estructurados", {})
                    df_resultados = ner_to_dataframe(datos_json)
                    if not df_resultados.empty:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "type": "dataframe", 
                            "content": df_resultados
                        })
                    else:
                        st.warning("No se detectaron biomarcadores válidos en la analítica.")
            except requests.exceptions.ConnectionError:
                st.error("Error de conexión: el servidor backend no está disponible.")
            except requests.exceptions.Timeout:
                st.error("La petición de extracción tardó demasiado. Intente de nuevo.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error en la petición de extracción: {e}")

        if st.button("Resumen rápido"):
            payload = {"texto_clinico": texto_analitica}
            try:
                response = requests.post(URL_EXTRACT, json=payload, timeout=20)
                if not response.ok:
                    detalle = get_error_detail(response)
                    st.error(f"No se pudo generar el resumen: {detalle}")
                else:
                    datos_json = response.json().get("datos_estructurados", {})
                    resultado_resumen = quick_summary(datos_json)
                    st.session_state.messages.append({"role": "assistant", "content": resultado_resumen})
            except requests.exceptions.ConnectionError:
                st.error("Error de conexión: el servidor backend no está disponible.")
            except requests.exceptions.Timeout:
                st.error("La petición de extracción tardó demasiado. Intente de nuevo.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error en la petición de resumen: {e}")

        if st.button("Informe completo"):
            payload = {"texto_clinico": texto_analitica, "metodologia_prompt": "CoT+FS"}
            
            try:
                # Feedback visual porque Ollama tarda en responder
                with st.spinner("🧠 Generando síntesis clínica con Qwen2.5-7B..."):
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
        # Si el mensaje es una tabla, la renderizamos a pantalla completa
        if message.get("type") == "dataframe":
            st.write("**Resultados de Laboratorio Extraídos:**")
            st.dataframe(
                aplicar_estilo(message["content"]), 
                use_container_width=True, 
                hide_index=True
            )
        # Si no, renderizamos el texto normal 
        else:
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
            # Llamamos a Ollama (BioMistral)
            response = requests.post(URL_CONSULT, json=payload)
            response.raise_for_status() 
            
            respuesta = response.json().get("respuesta", "Error leyendo respuesta del backend.")
            
            # Guardamos la respuesta normal de texto en el historial
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            st.write(respuesta)
            
        except requests.exceptions.ConnectionError:
            st.error("Error de conexión: El servidor FastAPI no está corriendo.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error en la petición: {e}")