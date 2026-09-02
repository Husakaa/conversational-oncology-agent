from fastapi import APIRouter, HTTPException
import logging

from .models import ExtractRequest, SynthesisRequest, ExtractResponse, SynthesisResponse, ConsultRequest, ConsultResponse, DevOptionsResponse
from src.ner.extractor import InvalidDocumentError, MedicalExtractor
from src.ollama.llm_service import LLMService
from src.config import DEV_MODE, OLLAMA_MODELS, METODOLOGIAS_PROMPT, MODEL_OPTIONS_MAP

# Creamos un enrutador
router = APIRouter()

# Instanciamos los motores para no cargarlos con cada petición
extractor = MedicalExtractor()
llm_service = LLMService()


@router.post("/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    if not request.texto_clinico.strip():
        raise HTTPException(status_code=400, detail="El texto clínico no puede estar vacío.")

    try:
        logging.info("Iniciando procesamiento de nueva analítica...")
        # Solo usa el motor Regex
        datos = extractor.analizar_texto(request.texto_clinico)
        if not datos:
            raise HTTPException(status_code=404, detail="No se detectaron biomarcadores.")
        logging.info("Procesamiento completado con éxito.")
        return ExtractResponse(
            datos_estructurados= datos
        )

    except InvalidDocumentError as e:
        logging.warning(f"Documento inválido recibido en /extract: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.error(f"Error en endpoint /analizar: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


@router.post("/synthesize", response_model=SynthesisResponse)
async def generate_report(request: SynthesisRequest):
    # Asignamos los datos estructurados que vengan en la peticion
    datos = request.datos_estructurados

    # Si no vienen datos estructurados, usamos el texto crudo
    if not datos:
        if not request.texto_clinico:
            raise HTTPException(status_code=400, detail="Debe proporcionar 'datos_estructurados (json)' o 'texto_clinico (txt)'")

        # Extraemos los datos
        try:
            datos = extractor.analizar_texto(request.texto_clinico)
        except InvalidDocumentError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if not datos:
            raise HTTPException(status_code=404, detail="No se detectaron biomarcadores.")
        logging.info("Procesamiento completado con éxito.")

    # Una vez garantizado que tenemos los datos, llamamos al LLM.
    # model_key/hyperparams son herramientas de desarrollador: se ignoran fuera de DEV_MODE
    # para que la API en producción siempre use la configuración por defecto de src/config.py.
    kwargs = {}
    if DEV_MODE:
        if request.model_key:
            kwargs["model_key"] = request.model_key
        if request.hyperparams:
            kwargs["hyperparams"] = request.hyperparams

    contexto, sintesis = llm_service.generate_synthesis(
        extracted_data=datos,
        method=request.metodologia_prompt,
        **kwargs
    )

    return SynthesisResponse(
        contexto_generado= contexto,
        sintesis_clinica= sintesis
    )

@router.post("/consult", response_model=ConsultResponse)
async def generate_response(request: ConsultRequest):
    kwargs = {}
    if DEV_MODE:
        if request.model_key:
            kwargs["model_key"] = request.model_key
        if request.hyperparams:
            kwargs["hyperparams"] = request.hyperparams

    respuesta = llm_service.generate_response(consult=request.consulta, method=request.metodologia_prompt, **kwargs)
    return ConsultResponse(
        respuesta=respuesta
    )


@router.get("/dev/options", response_model=DevOptionsResponse)
async def get_dev_options():
    """Modelos, metodologías de prompting e hiperparámetros por defecto para las
    herramientas de desarrollador del frontend. Solo disponible con DEV_MODE=true."""
    if not DEV_MODE:
        raise HTTPException(status_code=404, detail="Herramientas de desarrollador desactivadas (DEV_MODE=false).")

    return DevOptionsResponse(
        dev_mode=DEV_MODE,
        models=list(OLLAMA_MODELS.keys()),
        methodologies=METODOLOGIAS_PROMPT,
        default_hyperparams=MODEL_OPTIONS_MAP
    )
