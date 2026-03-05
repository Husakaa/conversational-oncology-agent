from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class AnaliticaRequest(BaseModel):
    """Esquema de validación para la petición entrante."""
    texto_clinico: str = Field(
        ..., 
        description="Texto bruto de la analítica de laboratorio."
    )
    metodologia_prompt: Optional[str] = Field(
        default="CoT", 
        description="Estrategia de inferencia: 'Zero-Shot', 'Few-Shot' o 'CoT'."
    )

class AnaliticaResponse(BaseModel):
    """Esquema de validación para la respuesta de la API."""
    status: str
    datos_estructurados: Dict[str, Any]
    contexto_generado: str
    sintesis_clinica: str