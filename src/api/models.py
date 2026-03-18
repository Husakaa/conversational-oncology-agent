from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ExtractRequest(BaseModel):
    """Esquema de validación para el motor de extracción."""
    texto_clinico: str = Field(
        ..., 
        description="Texto bruto de la analítica de laboratorio."
    )

class SynthesisRequest(BaseModel):
    """Esquema de validación para Ollama"""
    texto_clinico: Optional[str] = None
    datos_estructurados: Optional[Dict[str, Any]] = None
    metodologia_prompt: str = Field(
        default="CoT", 
        description="Estrategia de inferencia: 'Zero-Shot', 'Few-Shot' o 'CoT'."
    )

class ExtractResponse(BaseModel):
    """Esquema de respuesta para el motor de extracción."""
    status: str = "success"
    datos_estructurados: Dict[str, Any]

class SynthesisResponse(BaseModel):
    """Esquema de respuesta para Ollama."""
    status: str = "success"
    contexto_generado: str
    sintesis_clinica: str
