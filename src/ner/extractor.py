import re
import logging
from typing import Dict, Any, Optional

# Importamos los diccionarios de patrones y reglas 
from .patterns import PATRONES, REGLAS_CTCAE

# Configuración básica de logging 
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Rangos biológicos plausibles (valor_min, valor_max)
# Valores fuera de estos rangos se consideran errores de extracción o unidades incorrectas.
# Los límites son generosos para no descartar valores extremos pero reales.
RANGOS_BIOLOGICOS = {
    'Hb':              (1.0,    25.0),     # g/dL
    'Plaquetas':       (1.0,    1500.0),   # ×10⁹/L (o ×10³/µL)
    'Neutrófilos':     (0.0,    50.0),     # ×10⁹/L
    'Linfocitos':      (0.0,    30.0),     # ×10⁹/L
    'Creatinina':      (0.1,    30.0),     # mg/dL
    'Calcio':          (2.0,    20.0),     # mg/dL
    'Potasio':         (1.0,    10.0),     # mEq/L
    'Sodio':           (90.0,   190.0),    # mEq/L
    'Magnesio':        (0.1,    15.0),     # mg/dL
    'ALT':             (1.0,    10000.0),  # U/L
    'AST':             (1.0,    10000.0),  # U/L
    'LDH':             (10.0,   15000.0),  # U/L
    'GGT':             (1.0,    10000.0),  # U/L
    'FA':              (5.0,    10000.0),  # U/L
    'Bilirrubina':     (0.05,   50.0),     # mg/dL
    'Albumina':        (0.5,    7.0),      # g/dL
    'Proteinas':       (1.0,    15.0),     # g/dL
    'Glucosa':         (10.0,   2000.0),   # mg/dL
    'Colesterol':      (30.0,   1500.0),   # mg/dL
    'HDL':             (1.0,    200.0),    # mg/dL
    'Trigliceridos':   (10.0,   5000.0),   # mg/dL
    'Lipasa':          (1.0,    5000.0),   # U/L
    'Amilasa':         (1.0,    5000.0),   # U/L
    'Creatinquinasa':  (5.0,    50000.0),  # U/L
    'TTPA':            (5.0,    200.0),    # segundos
    'Fibrinogeno':     (10.0,   3000.0),   # mg/dL
    'INR':             (0.5,    15.0),     # ratio (adimensional)
    'PH':              (6.5,    8.0),      # pH
}

# Unidades válidas por biomarcador
# Si la unidad extraída no coincide con ninguna de estas Y no hay rango de referencia,
# se considera que la extracción regex fue incorrecta.
UNIDADES_VALIDAS = {
    'Hb':              {'g/dL', 'g/dl', 'g/L', 'g/l', 'gr/dL', 'gr/dl'},
    'Plaquetas':       {'10^3/uL', '10^3/µL', '10^3/μL', 'x10^3/uL', 'x10^3/µL',
                        '10^9/L', '10^9/l', 'x10^9/L', '10e3/uL', '10e3/µL',
                        'mil/mm3', 'mil/µL', '10³/µL', '10⁹/L'},
    'Neutrófilos':     {'10^3/uL', '10^3/µL', '10^3/μL', 'x10^3/uL', 'x10^3/µL',
                        '10^9/L', '10^9/l', 'x10^9/L', '10e3/uL', '10e3/µL',
                        'mil/mm3', '10³/µL', '10⁹/L'},
    'Linfocitos':      {'10^3/uL', '10^3/µL', '10^3/μL', 'x10^3/uL', 'x10^3/µL',
                        '10^9/L', '10^9/l', 'x10^9/L', '10e3/uL', '10e3/µL',
                        'mil/mm3', '10³/µL', '10⁹/L'},
    'Creatinina':      {'mg/dL', 'mg/dl', 'mg/100ml', 'µmol/L', 'umol/L'},
    'Calcio':          {'mg/dL', 'mg/dl', 'mmol/L', 'mmol/l', 'mEq/L', 'mEq/l'},
    'Potasio':         {'mEq/L', 'mEq/l', 'meq/L', 'meq/l', 'mmol/L', 'mmol/l'},
    'Sodio':           {'mEq/L', 'mEq/l', 'meq/L', 'meq/l', 'mmol/L', 'mmol/l'},
    'Magnesio':        {'mg/dL', 'mg/dl', 'mEq/L', 'mEq/l', 'mmol/L', 'mmol/l'},
    'ALT':             {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'AST':             {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'LDH':             {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'GGT':             {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'FA':              {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'Bilirrubina':     {'mg/dL', 'mg/dl', 'µmol/L', 'umol/L'},
    'Albumina':        {'g/dL', 'g/dl', 'g/L', 'g/l'},
    'Proteinas':       {'g/dL', 'g/dl', 'g/L', 'g/l'},
    'Glucosa':         {'mg/dL', 'mg/dl', 'mmol/L', 'mmol/l'},
    'Colesterol':      {'mg/dL', 'mg/dl', 'mmol/L', 'mmol/l'},
    'HDL':             {'mg/dL', 'mg/dl', 'mmol/L', 'mmol/l'},
    'Trigliceridos':   {'mg/dL', 'mg/dl', 'mmol/L', 'mmol/l'},
    'Lipasa':          {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'Amilasa':         {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'Creatinquinasa':  {'U/L', 'u/l', 'U/l', 'UI/L', 'IU/L'},
    'TTPA':            {'seg', 's', 'segundos', 'sec'},
    'Fibrinogeno':     {'mg/dL', 'mg/dl', 'g/L', 'g/l'},
    'INR':             set(),  # Adimensional
    'PH':              set(),  # Adimensional
}

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

    def _validar_valor_biologico(self, nombre_patron: str, valor: float) -> bool:
        """
        Verifica que el valor extraído esté dentro de un rango biológicamente plausible.
        Descarta valores que son físicamente imposibles o indican errores de extracción/unidades.
        """
        rango = RANGOS_BIOLOGICOS.get(nombre_patron)
        if rango is None:
            return True  # Sin rango definido → se acepta por defecto
        
        val_min, val_max = rango
        if valor < val_min or valor > val_max:
            logging.warning(
                f"⚠️ Valor descartado por implausibilidad biológica: "
                f"{nombre_patron} = {valor} (rango aceptable: {val_min} - {val_max})"
            )
            return False
        return True

    def _validar_unidad(self, nombre_patron: str, unidad: str, tiene_rango: bool) -> bool:
        """
        Verifica que la unidad extraída sea coherente con el biomarcador.
        Si la unidad no es válida Y no hay rango de referencia, la extracción
        se considera un falso positivo del motor regex.
        
        Si hay rango de referencia presente, se relaja la validación porque
        la coincidencia contextual del regex es más fiable.
        """
        unidades_ok = UNIDADES_VALIDAS.get(nombre_patron)
        
        # Sin lista de unidades definida
        if unidades_ok is None:
            return True
        
        # Biomarcadores adimensionales 
        if len(unidades_ok) == 0:
            return True
        
        # Comprobar si la unidad extraída es válida
        unidad_limpia = unidad.strip() if unidad else "n/a"
        unidad_valida = unidad_limpia in unidades_ok or unidad_limpia.lower() in {u.lower() for u in unidades_ok}
        
        if not unidad_valida and not tiene_rango:
            logging.warning(
                f"Extracción descartada por unidad inválida sin rango de referencia: "
                f"{nombre_patron} - unidad='{unidad_limpia}' "
                f"(esperadas: {', '.join(sorted(unidades_ok)[:5])}...)"
            )
            return False
        
        return True

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
                
                # Filtro de plausibilidad biológica
                if not self._validar_valor_biologico(nombre, val_num):
                    continue
                
                # Filtro de unidad + rango de referencia
                tiene_rango = inf is not None and sup is not None
                if not self._validar_unidad(nombre, unidad, tiene_rango):
                    continue
                
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
