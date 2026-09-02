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
    model_key: Optional[str] = Field(
        default=None,
        description="[Herramientas de desarrollador, requiere DEV_MODE] Clave de OLLAMA_MODELS a usar en vez del modelo por defecto."
    )
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="[Herramientas de desarrollador, requiere DEV_MODE] Overrides puntuales de las opciones de inferencia (temperature, top_k, top_p, num_ctx, num_predict, repeat_penalty...)."
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

class ConsultRequest(BaseModel):
    """Esquema de validación para consulta a Ollama"""
    consulta: str
    metodologia_prompt: str = Field(
        default="CoT+FS",
        description="Estrategia de inferencia: 'Zero-Shot', 'Few-Shot', 'CoT' o 'CoT+FS'."
    )
    model_key: Optional[str] = Field(
        default=None,
        description="[Herramientas de desarrollador, requiere DEV_MODE] Clave de OLLAMA_MODELS a usar en vez del modelo por defecto."
    )
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="[Herramientas de desarrollador, requiere DEV_MODE] Overrides puntuales de las opciones de inferencia."
    )

class ConsultResponse(BaseModel):
    """Esquema de respuesta para respuesta de Ollama"""
    respuesta: str

class DevOptionsResponse(BaseModel):
    """Esquema de respuesta para las opciones de desarrollador (solo con DEV_MODE activo)."""
    dev_mode: bool
    models: list[str]
    methodologies: list[str]
    default_hyperparams: Dict[str, Dict[str, Any]]
