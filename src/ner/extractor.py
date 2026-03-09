import re
import logging
from typing import Dict, Any, Optional

# Importamos los diccionarios de patrones y reglas 
from .patterns import PATRONES, REGLAS_CTCAE

# Configuración básica de logging 
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MedicalExtractor:
    """
    Motor determinista para la extracción de biomarcadores mediante Regex
    y evaluación de toxicidad clínica según estándares CTCAE v5.0.
    """

    def __init__(self):
        self.patrones = PATRONES
        self.reglas_ctcae = REGLAS_CTCAE
        
        # Mapea la clave del diccionario 'PATRONES' a la clave del diccionario 'REGLAS_CTCAE'
        self.mapa_nombres = {
            'Hb': 'Hemoglobina',
            'Plaquetas': 'Plaquetas',
            'Neutrófilos': 'Neutrófilos',
            'Linfocitos': 'Linfocitos',
            'Creatinina': 'Creatinina',
            'Calcio': 'Calcio',
            'Potasio': 'Potasio',
            'Sodio': 'Sodio',
            'Magnesio': 'Magnesio',
            'ALT': 'ALT',
            'AST': 'AST',
            'LDH': 'LDH',
            'GGT': 'GGT',
            'FA': 'FA',
            'Bilirrubina': 'Bilirrubina',
            'Albumina': 'Albumina',
            'Proteinas': 'Proteinas',
            'Glucosa': 'Glucosa',
            'Colesterol': 'Colesterol',
            'HDL': 'HDL',
            'Trigliceridos': 'Trigliceridos',
            'Lipasa': 'Lipasa',
            'Amilasa': 'Amilasa',
            'Creatinquinasa': 'Creatinquinasa',
            'TTPA': 'TTPA',
            'Fibrinogeno': 'Fibrinogeno',
            'INR': 'INR',
            'PH': 'pH'
        }

    def _limpiar_valor(self, valor_str: str) -> float:
        """
        Convierte strings numéricos del formato español al estándar internacional.
        """
        if not valor_str:
            return 0.0
        try:
            return float(valor_str.replace(',', '.'))
        except ValueError:
            logging.warning(f"No se pudo convertir a float el valor: {valor_str}")
            return 0.0

    def _determinar_toxicidad(self, nombre_patron: str, valor: float) -> Dict[str, Any]:
        """
        Evalúa el valor extraído contra las reglas bidireccionales (High/Low) del CTCAE.
        """
        nombre_clinico = self.mapa_nombres.get(nombre_patron)
        config_marcador = self.reglas_ctcae.get(nombre_clinico)

        if not config_marcador:
            return {
                "grado": 0, 
                "etiqueta": "N/A", 
                "descripcion": "Sin criterio CTCAE definido"
            }

        # Iteramos por las posibles direcciones clínicas (ej: High para Hipercalcemia, Low para Hipocalcemia)
        for direccion, info in config_marcador.items():
            desc_base = info['desc_base']
            for grado, condicion, desc_detallada in info['reglas']:
                if condicion(valor):
                    return {
                        "grado": grado,
                        "etiqueta": direccion,
                        "descripcion": f"{desc_base}: {desc_detallada}"
                    }

        # Si no cumple ninguna condición de las reglas, se asume normalidad
        return {
            "grado": 0, 
            "etiqueta": "Normal", 
            "descripcion": "Normal / Sin toxicidad detectada"
        }

    def analizar_texto(self, texto: str) -> Dict[str, Any]:
        """
        Función principal expuesta a la API. 
        Ingiere el texto bruto de la analítica y devuelve un JSON estructurado.
        """
        resultados = {}

        if not texto or not isinstance(texto, str):
            logging.error("Se recibió un texto vacío o inválido.")
            return resultados

        for nombre, regex in self.patrones.items():
            match = regex.search(texto)
            if match:
                # Extracción de grupos por nombre desde la expresión regular
                val_raw = match.group("valor")
                unidad = match.group("unidad") if "unidad" in regex.groupindex else "n/a"
                inf = match.group("inferior") if "inferior" in regex.groupindex else None
                sup = match.group("superior") if "superior" in regex.groupindex else None
                
                # Limpieza de espacios o caracteres raros en la unidad
                unidad = unidad.strip() if unidad else "n/a"
                
                # Normalización numérica
                val_num = self._limpiar_valor(val_raw)
                
                # Cálculo de toxicidad determinista
                toxicidad = self._determinar_toxicidad(nombre, val_num)
                
                # Estructuración del output final
                resultados[nombre] = {
                    "valor": val_num,
                    "unidad": unidad,
                    "rango_referencia": {
                        "inf": self._limpiar_valor(inf) if inf else None,
                        "sup": self._limpiar_valor(sup) if sup else None
                    },
                    "ctcae": toxicidad
                }

        return resultados
