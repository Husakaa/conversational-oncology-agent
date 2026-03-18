from fastapi import APIRouter, HTTPException
import logging

from .models import ExtractRequest, SynthesisRequest, ExtractResponse, SynthesisResponse
from src.ner.extractor import MedicalExtractor
from src.ollama.llm_service import LLMService

# Creamos un enrutador 
router = APIRouter()

# Instanciamos los motores para no cargarlos con cada petición
extractor = MedicalExtractor()
llm_service = LLMService(model_name="qwen-oncologo") 


@router.post("/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    if not request.texto_clinico.strip():
        raise HTTPException(status_code=400, detail="El texto clínico no puede estar vacío.")

    try:
        logging.info("Iniciando procesamiento de nueva analítica...")
        # Solo usa el motor Regex
        datos = extractor.analizar_texto(request.texto_clinico)
        if not datos:
            raise HTTPException(status_code=404, detail="No se detectaron biomarcadores")
        logging.info("Procesamiento completado con éxito.")
        return ExtractResponse(
            datos_estructurados= datos
        )

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
        datos = extractor.analizar_texto(request.texto_clinico)
        if not datos:
            raise HTTPException(status_code=404, detail="No se detectaron biomarcadores")
        logging.info("Procesamiento completado con éxito.")

    # Una vez garantizado que tenemos los datos, llamamos al LLM
    contexto, sintesis = llm_service.generate_synthesis(
        extracted_data=datos,
        method=request.metodologia_prompt
    )
    
    return SynthesisResponse(
        contexto_generado= contexto,
        sintesis_clinica= sintesis
    )
    
