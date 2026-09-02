import logging
from typing import Dict, Any, Optional, Tuple
import ollama

from src.ui.formatters import clinical_context, biomarkers_to_markdown
from src.config import OLLAMA_MODELS, QWEN35_OPTIONS, MODEL_OPTIONS_MAP, MODEL_PROMPTS, THINKING_MODELS, THINK_BY_METHOD, CONSULT_METHODOLOGY_MODELS

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LLMService:
    def __init__(self):
        self.models = OLLAMA_MODELS
        self.model_options_map = MODEL_OPTIONS_MAP
        self.default_options = QWEN35_OPTIONS  # fallback para model_key no mapeado
        self.last_biomarkers = None

    def _resolve_options(self, model_key: str, hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Opciones base del modelo, con overrides puntuales (herramientas de desarrollador)."""
        options = dict(self.model_options_map.get(model_key, self.default_options))
        if hyperparams:
            options.update(hyperparams)
        return options

    def _resolve_think(self, model_key: str, method: str) -> Optional[bool]:
        """True/False para modelos con canal de razonamiento nativo (ver THINKING_MODELS);
        None si el modelo no soporta 'thinking' (no se pasa el parámetro a ollama.chat)."""
        if model_key not in THINKING_MODELS:
            return None
        per_method = THINK_BY_METHOD.get(model_key)
        if per_method is not None:
            return per_method.get(method, True)
        return True

    def _build_prompt(self, model_id: str, methodology: str, context: str, consult: bool = False) -> str:
        model_prompts = MODEL_PROMPTS.get(model_id, MODEL_PROMPTS["default"])
        if consult:
            # biomistral-7B define su plantilla de consulta por metodología (ver
            # CONSULT_METHODOLOGY_MODELS); el resto usa siempre la "consult" genérica, aunque
            # su propio dict tenga claves Zero-Shot/CoT/... — esas son de síntesis, no de consulta.
            if model_id in CONSULT_METHODOLOGY_MODELS and methodology in model_prompts:
                prompt_template = model_prompts[methodology]
            else:
                prompt_template = model_prompts.get("consult", MODEL_PROMPTS["default"]["consult"])
            context_block = ""
            if self.last_biomarkers:
                context_block = f"\n\nRESULTADOS DE LABORATORIO DEL PACIENTE ACTUAL:\n{self.last_biomarkers}\n"
            return prompt_template.format(context_block=context_block, context=context)
        else:
            prompt_template = model_prompts.get(methodology)
            if not prompt_template:
                raise ValueError(f"Metodología '{methodology}' no soportada para el modelo '{model_id}'.")
            return prompt_template.format(context=context)

    def generate_synthesis(self, extracted_data: Dict[str, Any], method: str = "CoT+FS", model_key: str = "qwen2.5-7B", hyperparams: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        context = clinical_context(extracted_data)
        model_id = self.models.get(model_key, model_key)
        prompt_final = self._build_prompt(model_id, method, context)
        options = self._resolve_options(model_key, hyperparams)
        think = self._resolve_think(model_key, method)
        logging.info(f"Lanzando inferencia a Ollama (Modelo: {model_id} | Método: {method} | Think: {think} | Opciones: {options})")

        chat_kwargs = {"model": model_id, "messages": [{'role': 'user', 'content': prompt_final}], "options": options}
        if think is not None:
            chat_kwargs["think"] = think

        try:
            response = ollama.chat(**chat_kwargs)
            synthesis = response['message']['content']
            self.last_biomarkers = biomarkers_to_markdown(extracted_data)
            logging.info("Tabla de biomarcadores guardada como contexto para BioMistral.")
            return context, synthesis
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return context, f"Error generando la síntesis clínica: Asegúrate de que Ollama está ejecutándose y el modelo '{model_id}' está descargado."

    def generate_response(self, consult: str, method: str = "CoT+FS", model_key: str = "biomistral-7B", hyperparams: Optional[Dict[str, Any]] = None) -> str:
        model_id = self.models.get(model_key, "biomistral-7B")
        prompt = self._build_prompt(model_id, methodology=method, context=consult, consult=True)
        options = self._resolve_options(model_key, hyperparams)
        think = self._resolve_think(model_key, method)
        logging.info(f"Lanzando consulta a Ollama (Modelo: {model_id} | Think: {think} | Opciones: {options})")

        chat_kwargs = {"model": model_id, "messages": [{'role': 'user', 'content': prompt}], "options": options}
        if think is not None:
            chat_kwargs["think"] = think

        try:
            response = ollama.chat(**chat_kwargs)
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error en la comunicación con Ollama: {str(e)}")
            return f"Error generando la respuesta: Asegúrate de que Ollama está ejecutándose y el modelo '{model_id}' está descargado."