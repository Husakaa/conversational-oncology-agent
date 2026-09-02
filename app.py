import streamlit as st
import requests
import time
import os

from src.ui.formatters import ner_to_dataframe, aplicar_estilo, quick_summary

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
URL_DEV_OPTIONS = f"{BACKEND_URL}/api/v1/dev/options"

# Herramientas de desarrollador (alternar SLM, metodología de prompting e
# hiperparámetros en tiempo de ejecución). Desactivadas por defecto: para
# producción basta con no definir DEV_MODE (o ponerla a "false").
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"


def get_error_detail(response: requests.Response) -> str:
    """Extrae el mensaje de error de una respuesta HTTP no-OK (JSON 'detail' o texto plano)."""
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def post_api(url: str, payload: dict, timeout: int = 20):
    """POST a la API, clasificando errores de red/HTTP en un mensaje de usuario legible.

    Nunca lanza: devuelve (response, None) en éxito o (None, mensaje_error) en fallo,
    para que cada llamada solo tenga que comprobar `error` en vez de repetir los mismos
    try/except de ConnectionError/Timeout/RequestException en cada botón.
    """
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return None, "Error de conexión: el servidor backend no está disponible."
    except requests.exceptions.Timeout:
        return None, "La petición tardó demasiado. Inténtelo de nuevo."
    except requests.exceptions.RequestException as e:
        return None, f"Error en la petición: {e}"

    if not response.ok:
        return None, get_error_detail(response)
    return response, None


st.title("Agente Conversacional Oncológico")

# Inicializacion del historial del chat y texto de la analítica
if "messages" not in st.session_state:
    st.session_state.messages = [] # Creamos una lista como base de datos no persistente
if "texto_analitica" not in st.session_state:
    st.session_state.texto_analitica = ""
if "dev_config" not in st.session_state:
    st.session_state.dev_config = {}

# Barra lateral para gestion de archivos y herramientas
with st.sidebar:
    # Herramientas de desarrollador: alternar SLM, metodología de prompting e
    # hiperparámetros de la síntesis narrativa en tiempo de ejecución, sin
    # tocar código. Se pide una sola vez por sesión al backend (DEV_MODE=true)
    # qué modelos/metodologías/hiperparámetros por defecto hay disponibles.
    if DEV_MODE:
        if "dev_options" not in st.session_state:
            try:
                resp = requests.get(URL_DEV_OPTIONS, timeout=5)
                st.session_state.dev_options = resp.json() if resp.ok else None
            except requests.exceptions.RequestException:
                st.session_state.dev_options = None

        st.header("🛠️ Herramientas de Desarrollador")
        opts = st.session_state.dev_options
        if not opts:
            st.warning("No se pudieron cargar las opciones de desarrollador. ¿Backend arrancado con DEV_MODE=true?")
        else:
            model_key = st.selectbox(
                "Modelo (SLM)", opts["models"],
                index=opts["models"].index("qwen2.5-7B") if "qwen2.5-7B" in opts["models"] else 0
            )
            methodology = st.selectbox(
                "Metodología de prompting", opts["methodologies"],
                index=opts["methodologies"].index("CoT+FS") if "CoT+FS" in opts["methodologies"] else 0
            )

            hyperparams = {}
            default_hp = opts["default_hyperparams"].get(model_key, {})
            with st.expander("Hiperparámetros", expanded=False):
                for key, value in default_hp.items():
                    if isinstance(value, bool):
                        hyperparams[key] = st.checkbox(key, value=value)
                    elif isinstance(value, int):
                        hyperparams[key] = st.number_input(key, value=value, step=1)
                    elif isinstance(value, float):
                        hyperparams[key] = st.number_input(key, value=value, step=0.05, format="%.2f")
                    else:
                        hyperparams[key] = value

            st.session_state.dev_config = {
                "model_key": model_key,
                "metodologia_prompt": methodology,
                "hyperparams": hyperparams
            }
        st.divider()
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

        if st.button("Extraer entidades"):
            response, error = post_api(URL_EXTRACT, {"texto_clinico": texto_analitica})
            if error:
                st.error(f"No se pudo extraer la analítica: {error}")
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

        if st.button("Resumen rápido"):
            response, error = post_api(URL_EXTRACT, {"texto_clinico": texto_analitica})
            if error:
                st.error(f"No se pudo generar el resumen: {error}")
            else:
                datos_json = response.json().get("datos_estructurados", {})
                resultado_resumen = quick_summary(datos_json)
                st.session_state.messages.append({"role": "assistant", "content": resultado_resumen})

        if st.button("Informe completo"):
            payload = {"texto_clinico": texto_analitica, "metodologia_prompt": "CoT+FS"}
            if DEV_MODE and st.session_state.dev_config:
                payload.update(st.session_state.dev_config)
            modelo_usado = payload.get("model_key", "Qwen2.5-7B")

            # Feedback visual + cronómetro, porque la síntesis con Ollama puede tardar
            with st.spinner(f"🧠 Generando síntesis clínica con {modelo_usado}..."):
                inicio = time.time()
                response, error = post_api(URL_SYNTHESIZE, payload, timeout=120)
                tiempo_total = time.time() - inicio

            if error:
                st.error(f"Error del backend: {error}")
            else:
                resultado = response.json().get("sintesis_clinica", "Error leyendo la síntesis.")
                resultado_con_tiempo = f"{resultado}\n\n---\n*Pensó durante {tiempo_total:.2f}s*"
                st.session_state.messages.append({"role": "assistant", "content": resultado_con_tiempo})

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

    # Respuesta del modelo (BioMistral)
    with st.chat_message("assistant"):
        response, error = post_api(URL_CONSULT, {"consulta": consult}, timeout=120)
        if error:
            st.error(error)
        else:
            respuesta = response.json().get("respuesta", "Error leyendo respuesta del backend.")
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            st.write(respuesta)