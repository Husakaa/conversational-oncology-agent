from fastapi import APIRouter, HTTPException
import logging

from .models import AnaliticaRequest, AnaliticaResponse
from src.ner.extractor import MedicalExtractor
from src.ollama.llm_service import LLMService

# Creamos un enrutador 
router = APIRouter()

# Instanciamos los motores para no cargarlos con cada petición
extractor = MedicalExtractor()
llm_service = LLMService(model_name="qwen-oncologo") 

@router.post("/analizar", response_model=AnaliticaResponse, tags=["Análisis Clínico"])
async def analizar_informe(request: AnaliticaRequest):
    """
    Endpoint principal: Extrae biomarcadores, calcula CTCAE y genera síntesis médica.
    """
    if not request.texto_clinico.strip():
        raise HTTPException(status_code=400, detail="El texto clínico no puede estar vacío.")

    try:
        logging.info("Iniciando procesamiento de nueva analítica...")
        
        # Capa Determinista 
        datos_extraidos = extractor.analizar_texto(request.texto_clinico)
        if not datos_extraidos:
            raise HTTPException(status_code=422, detail="No se reconocieron biomarcadores.")

        # Capa Generativa 
        contexto, sintesis = llm_service.generate_synthesis(
            extracted_data=datos_extraidos,
            method=request.metodologia_prompt
        )
        
        logging.info("Procesamiento completado con éxito.")
        
        return AnaliticaResponse(
            status="success",
            datos_estructurados=datos_extraidos,
            contexto_generado=contexto,
            sintesis_clinica=sintesis
        )

    except Exception as e:
        logging.error(f"Error en endpoint /analizar: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")