import re

# --- Patrones Base --- #

# Series hematológicas 
#pat_hb = re.compile(r"^-?(Hemoglobina|HB)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:[\s\S]*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.S | re.M | re.I)
# Patrón hemoglobina más flexible
pat_hb = re.compile(r"-?(?:Hemoglobina|HB)\D*(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?[([]\s*(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\s*[)\]])?", re.S | re.M | re.I)
#pat_plaquetas = re.compile(r"^-?Plaquetas[\s\S]*?(?P<valor>\d+)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.S | re.M | re.I)
pat_plaquetas = re.compile(r"-?Plaquetas\D*(?P<valor>\d+)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.S | re.M | re.I)

pat_neutrofilos = re.compile(r"(:?Neutr[oó]fi\s?los|Neu)(?:(?!%).)*?(?P<valor>\d+(?:[.,]\d+)?)(?![.,\d])(?!\s*%)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?[\s\S]*?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.S | re.M | re.I)
pat_linfocitos = re.compile(r"Linfocitos(?:(?!%).)*?(?P<valor>\d+(?:[.,]\d+)?)(?![.,\d])(?!\s*%)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?[\s\S]*?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.S | re.M | re.I)

# Función renal e iones 
pat_creatinina = re.compile(r"Creati\s?nina[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:[\s\S]*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_calcio = re.compile(r"Calcio.*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_potasio = re.compile(r"Potasio.*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_sodio = re.compile(r"Sodio.*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_magnesio = re.compile(r"Magnesio.*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)

# Perfil hepático y enzimas 
pat_alt = re.compile(r"(:?Alanina(amino)?\s?(transferasa|transaminasa)|ALA?T|GPT)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_ast = re.compile(r"(:?Aspartato(amino)?\s?(transferasa|transaminasa)|ASA?T|GOT)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_ldh = re.compile(r"(:?Lactato\s?deshidrogenasa|LDH)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_ggt = re.compile(r"(:?Gamma[- ]?(:?glutamiltransferasa|GT)|GGT)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_fa = re.compile(r"Fosfatasa alcalina[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_bilirrubina = re.compile(r"Bilirrubina total[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_albumina = re.compile(r"Alb[úu]mina[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_proteinas = re.compile(r"Prote[ií]nas? total(:?es)?[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)

# Metabolismo, lípidos y páncreas 
pat_glu = re.compile(r"Glucosa.*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:[\s\S]*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_colesterol = re.compile(r"Colesterol[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_hdl = re.compile(r"(:?HDL|DHL)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_trigliceridos = re.compile(r"Tri[\s-]?glic[eé]ridos?[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_lipasa = re.compile(r"Lipasa[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I)
pat_amilasa = re.compile(r"Amilasa[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_creatinquinasa = re.compile(r"(:?[cC]reatin[\s-][qQ]uinasa|\(CP?K\))[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.M | re.S)

# Coagulación y estado ácido-base 
pat_ttpa = re.compile(r"(:?Tromboplastina|TTPA|APTT|TPTA)[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_fibrinogeno = re.compile(r"Fibrin[oó]geno[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.I | re.M | re.S)
pat_inr = re.compile(r"INR[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.M | re.S)
pat_ph = re.compile(r"[pP]H[\s\S]*?(?P<valor>\d+(?:[.,]\d+)?)(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?(?:.*?\[?(?P<inferior>\d+(?:[.,]\d+)?)\s*-\s*(?P<superior>\d+(?:[.,]\d+)?)\]?)?", re.M | re.S)

pat_creatinina = re.compile(
    r"Creati\s?nina[\s\S]*?" # 1. Identificador del biomarcador
    r"(?P<valor>\d+(?:[.,]\d+)?)" # 2. Grupo de captura: Valor numérico
    r"(?:.*?(?P<unidad>[a-zA-Z\d\^/µμ%]+))?" # 3. Grupo opcional: Unidad
    r"(?:[\s\S]*?\[?(?P<inferior>\d+(?:[.,]\d+)?)" # 4a. Rango Ref: Límite inferior
    r"\s*-\s*" # Separador de rango
    r"(?P<superior>\d+(?:[.,]\d+)?)\]?)?", # 4b. Rango Ref: Límite superior
    re.I
)

# --- Diccionario estructurado de patrones --- #
PATRONES = {
    # Hematología
    'Hb': pat_hb,
    'Plaquetas': pat_plaquetas,
    'Neutrófilos': pat_neutrofilos,
    'Linfocitos': pat_linfocitos,
    
    # Renal e Iones
    'Creatinina': pat_creatinina,
    'Calcio': pat_calcio,
    'Potasio': pat_potasio,
    'Sodio': pat_sodio,
    'Magnesio': pat_magnesio,
    
    # Hepático / Enzimas
    'ALT': pat_alt,
    'AST': pat_ast,
    'LDH': pat_ldh,
    'GGT': pat_ggt,
    'FA': pat_fa,
    'Bilirrubina': pat_bilirrubina,
    'Albumina': pat_albumina,
    'Proteinas': pat_proteinas,
    
    # Metabolismo / Lípidos / Páncreas
    'Glucosa': pat_glu,
    'Colesterol': pat_colesterol,
    'HDL': pat_hdl,
    'Trigliceridos': pat_trigliceridos,
    'Lipasa': pat_lipasa,
    'Amilasa': pat_amilasa,
    'Creatinquinasa': pat_creatinquinasa,
    
    # Coagulación y pH
    'TTPA': pat_ttpa,
    'Fibrinogeno': pat_fibrinogeno,
    'INR': pat_inr,
    'PH': pat_ph
}

# --- Reglas CTCAE --- #
REGLAS_CTCAE = {
    
    # Series hematológicas
    'Hemoglobina': {
        'Low': {
            'desc_base': 'Anemia',
            'reglas': [
                (4, lambda x: x < 6.0, "Amenaza vital (<6.0)"),
                (3, lambda x: x < 8.0, "Severa (<8.0 g/dL)"),
                (2, lambda x: 8.0 <= x < 10.0, "Moderada (8.0 - 10.0 g/dL)"),
                (1, lambda x: 10.0 <= x < 12.0, "Leve (10.0 - 12.0 g/dL)")
            ]
        }
    },
    'Plaquetas': {
        'Low': {
            'desc_base': 'Trombocitopenia',
            'reglas': [
                (4, lambda x: x < 25.0, "Amenaza vital (<25.0 x 10⁹/L)"),
                (3, lambda x: 25.0 <= x < 50.0, "Severa (25.0 - 50.0 x 10⁹/L)"),
                (2, lambda x: 50.0 <= x < 75.0, "Moderada (50.0 - 75.0 x 10⁹/L)"),
                (1, lambda x: 75.0 <= x < 140.0, "Leve (75.0 - 140.0 x 10⁹/L)")
            ]
        }
    },
    'Neutrófilos': {
        'Low': {
            'desc_base': 'Neutropenia',
            'reglas': [
                (4, lambda x: x < 0.5, "Amenaza vital (<0.5)"),
                (3, lambda x: 0.5 <= x < 1.0, "Severa (0.5 - 0.9)"),
                (2, lambda x: 1.0 <= x < 1.5, "Moderada (1.0 - 1.4)"),
                (1, lambda x: 1.5 <= x < 1.8, "Leve (1.5 - 1.7)")
            ]
        }
    },
    'Linfocitos': {
        'Low': {
            'desc_base': 'Linfopenia',
            'reglas': [
                # Escala en 10³/μL (o 10⁹/L). Normal suele ser > 1.0
                (4, lambda x: x < 0.2, "Amenaza vital (<0.2 x 10³/μL)"),
                (3, lambda x: 0.2 <= x < 0.5, "Severa (0.2 - 0.5 x 10³/μL)"),
                (2, lambda x: 0.5 <= x < 0.8, "Moderada (0.5 - 0.8 x 10³/μL)"),
                (1, lambda x: 0.8 <= x < 1.0, "Leve (0.8 - 1.0 x 10³/μL)")
            ]
        }
    },

    # Función renal e iones
    'Creatinina': {
        'High': {
            'desc_base': 'Creatinina elevada',
            'reglas': [
                (4, lambda x: x > 6.12, "Amenaza vital (>6.12 mg/dL)"),
                (3, lambda x: 3.07 <= x <= 6.12, "Severa (3.07 - 6.12 mg/dL)"),
                (2, lambda x: 1.54 <= x < 3.07, "Moderada (1.54 - 3.06 mg/dL)"),
                (1, lambda x: 1.03 <= x < 1.54, "Leve (1.03 - 1.53 mg/dL)")
            ]
        }
    },
    'Calcio': {
        'High': { 'desc_base': 'Hipercalcemia', 'reglas': [
            (4, lambda x: x > 13.5, "Amenaza vital (>13.5 mg/dL)"),
            (3, lambda x: 12.6 <= x <= 13.5, "Severa (12.6 - 13.5 mg/dL)"),
            (2, lambda x: 11.6 <= x < 12.6, "Moderada (11.6 - 12.5 mg/dL)"),
            (1, lambda x: 10.5 <= x < 11.6, "Leve (10.5 - 11.5 mg/dL)")]},
        'Low': { 'desc_base': 'Hipocalcemia', 'reglas': [
            (4, lambda x: x < 6.0, "Amenaza vital (<6.0 mg/dL)"),
            (3, lambda x: 6.0 <= x < 7.0, "Severa (6.0 - 7.0 mg/dL)"),
            (2, lambda x: 7.0 <= x < 8.0, "Moderada (7.0 - 8.0 mg/dL)"),
            (1, lambda x: 8.0 <= x < 8.7, "Leve (8.0 - 8.7 mg/dL)")]}
    },
    'Potasio': {
        'High': { 'desc_base': 'Hiperpotasemia', 'reglas': [
            (4, lambda x: x > 7.0, "Amenaza vital (>7.0 mEq/L)"),
            (3, lambda x: 6.1 <= x <= 7.0, "Severa (6.1 - 7.0 mEq/L)"),
            (2, lambda x: 5.6 <= x < 6.1, "Moderada (5.6 - 6.0 mEq/L)"),
            (1, lambda x: 5.0 <= x < 5.6, "Leve (5.0 - 5.5 mEq/L)")]},
        'Low': { 'desc_base': 'Hipopotasemia', 'reglas': [
            (4, lambda x: x < 2.5, "Amenaza vital (<2.5 mEq/L)"),
            (3, lambda x: 2.5 <= x < 3.0, "Severa (2.5 - 2.9 mEq/L)"),
            (1, lambda x: 3.0 <= x < 3.5, "Leve (3.0 - 3.5 mEq/L)")]}
    },
    'Sodio': {
        'High': { 'desc_base': 'Hipernatremia', 'reglas': [
            (4, lambda x: x > 160, "Amenaza vital (>160)"),
            (3, lambda x: 156 <= x <= 160, "Severa (156 - 160)"),
            (2, lambda x: 151 <= x < 156, "Moderada (151 - 155)"),
            (1, lambda x: 147 <= x < 151, "Leve (147 - 150)")]},
        'Low': { 'desc_base': 'Hiponatremia', 'reglas': [
            (4, lambda x: x < 120, "Amenaza vital (<120)"),
            (3, lambda x: 120 <= x < 125, "Severa (120 - 124)"),
            (2, lambda x: 125 <= x < 130, "Moderada (125 - 129)"),
            (1, lambda x: 130 <= x < 132, "Leve (130 - 131)")]}
    },
    'Magnesio': {
        'High': { 'desc_base': 'Hipermagnesemia', 'reglas': [
            (4, lambda x: x > 8.0, "Amenaza vital (>8.0 mg/dL)"),
            (3, lambda x: 5.1 <= x <= 8.0, "Severa (5.1 - 8.0 mg/dL)"),
            (1, lambda x: 2.5 <= x < 3.0, "Leve (2.5 - 3.0 mg/dL)")]},
        'Low': { 'desc_base': 'Hipomagnesemia', 'reglas': [
            (4, lambda x: x < 0.7, "Amenaza vital (<0.7 mg/dL)"),
            (3, lambda x: 0.7 <= x < 0.9, "Severa (0.7 - 0.9 mg/dL)"),
            (2, lambda x: 0.9 <= x < 1.2, "Moderada (0.9 - 1.2 mg/dL)")]}
    },

    # Perfil hepático y enzimas
    'ALT': {
        'High': {
            'desc_base': 'Elevación ALT (GPT)',
            'reglas': [
                (4, lambda x: x > 800, "Amenaza vital (>800 U/L)"),
                (3, lambda x: 201 <= x <= 800, "Severa (201 - 800 U/L)"),
                (2, lambda x: 121 <= x < 201, "Moderada (121 - 200 U/L)"),
                (1, lambda x: 41 <= x < 121, "Leve (41 - 120 U/L)")
            ]
        }
    },
    'AST': {
        'High': {
            'desc_base': 'Elevación AST (GOT)',
            'reglas': [
                (4, lambda x: x > 800, "Amenaza vital (>800 U/L)"),
                (3, lambda x: 201 <= x <= 800, "Severa (201 - 800 U/L)"),
                (2, lambda x: 121 <= x < 201, "Moderada (121 - 200 U/L)"),
                (1, lambda x: 41 <= x < 121, "Leve (41 - 120 U/L)")
            ]
        }
    },
    'LDH': {
        'High': {
            'desc_base': 'Elevación LDH',
            'reglas': [
                (4, lambda x: x > 2460, "Amenaza vital (>2460 U/L)"),
                (3, lambda x: 1231 <= x <= 2460, "Severa (1231 - 2460 U/L)"),
                (2, lambda x: 616 <= x < 1231, "Moderada (616 - 1230 U/L)"),
                (1, lambda x: 247 <= x < 616, "Leve (247 - 615 U/L)")
            ]
        }
    },
    'GGT': {
        'High': {
            'desc_base': 'Elevación GGT',
            'reglas': [
                (4, lambda x: x > 1700, "Amenaza vital (>1700 U/L)"),
                (3, lambda x: 426 <= x <= 1700, "Severa (426 - 1700 U/L)"),
                (2, lambda x: 214 <= x < 426, "Moderada (214 - 425 U/L)"),
                (1, lambda x: 86 <= x < 214, "Leve (86 - 213 U/L)")
            ]
        }
    },
    'FA': {
        'High': {
            'desc_base': 'Elevación Fosfatasa Alc.',
            'reglas': [
                (4, lambda x: x > 2340, "Amenaza vital (>2340 U/L)"),
                (3, lambda x: 585 <= x <= 2340, "Severa (585 - 2340 U/L)"),
                (2, lambda x: 292 <= x < 585, "Moderada (292 - 584 U/L)"),
                (1, lambda x: 117 <= x < 292, "Leve (117 - 291 U/L)")
            ]
        }
    },
    'Bilirrubina': {
        'High': {
            'desc_base': 'Hiperbilirrubinemia',
            'reglas': [
                (4, lambda x: x > 10.0, "Amenaza vital (>10.0 mg/dL)"),
                (3, lambda x: 3.1 <= x <= 10.0, "Severa (3.1 - 10.0 mg/dL)"),
                (2, lambda x: 1.6 <= x < 3.1, "Moderada (1.6 - 3.0 mg/dL)"),
                (1, lambda x: 1.1 <= x < 1.6, "Leve (1.1 - 1.5 mg/dL)")
            ]
        }
    },
    'Albumina': {
        'Low': {
            'desc_base': 'Hipoalbuminemia',
            'reglas': [
                (3, lambda x: x < 2.0, "Severa (<2.0 g/dL)"),
                (2, lambda x: 2.0 <= x < 3.0, "Moderada (2.0 - 2.99 g/dL)"),
                (1, lambda x: 3.0 <= x < 3.4, "Leve (3.0 - 3.39 g/dL)")
            ]
        }
    },
    'Proteinas': {
        'Low': {
            'desc_base': 'Hipoproteinemia',
            'reglas': [
                (3, lambda x: x < 4.0, "Severa (<4.0 g/dL)"),
                (2, lambda x: 4.0 <= x < 5.0, "Moderada (4.0 - 4.9 g/dL)"),
                (1, lambda x: 5.0 <= x < 5.7, "Leve (5.0 - 5.69 g/dL)")
            ]
        }
    },

    # Metabolismo, lípidos y páncreas
    'Glucosa': {
        'High': { 'desc_base': 'Hiperglucemia', 'reglas': [
            (4, lambda x: x > 500, "Amenaza vital (>500 mg/dL)"),
            (3, lambda x: 251 <= x <= 500, "Severa (251 - 500 mg/dL)"),
            (2, lambda x: 161 <= x < 251, "Moderada (161 - 250 mg/dL)"),
            (1, lambda x: 111 <= x < 161, "Leve (111 - 160 mg/dL)")]},
        'Low': { 'desc_base': 'Hipoglucemia', 'reglas': [
            (4, lambda x: x < 30, "Amenaza vital (<30 mg/dL)"),
            (3, lambda x: 30 <= x < 40, "Severa (30 - 39 mg/dL)"),
            (2, lambda x: 40 <= x < 55, "Moderada (40 - 54 mg/dL)"),
            (1, lambda x: 55 <= x < 69, "Leve (55 - 69 mg/dL)")]}
    },
    'Colesterol': {
        'High': {
            'desc_base': 'Hipercolesterolemia',
            'reglas': [
                (4, lambda x: x > 1000, "Amenaza vital (>1000 mg/dL)"),
                (3, lambda x: 401 <= x <= 1000, "Severa (401 - 1000 mg/dL)"),
                (2, lambda x: 301 <= x < 401, "Moderada (301 - 400 mg/dL)"),
                (1, lambda x: 200 <= x < 301, "Leve (200 - 300 mg/dL)")
            ]
        }
    },
    'HDL': {
        'Low': {
            'desc_base': 'HDL bajo',
            'reglas': [
                (3, lambda x: x < 20, "Severo (<20 mg/dL)"),
                (2, lambda x: 20 <= x < 30, "Moderado (20 - 29 mg/dL)"),
                (1, lambda x: 30 <= x < 40, "Leve (30 - 39 mg/dL)")
            ]
        }
    },
    'Trigliceridos': {
        'High': {
            'desc_base': 'Hipertrigliceridemia',
            'reglas': [
                (4, lambda x: x > 1000, "Amenaza vital (>1000 mg/dL)"),
                (3, lambda x: 501 <= x <= 1000, "Severa (501 - 1000 mg/dL)"),
                (2, lambda x: 301 <= x < 501, "Moderada (301 - 500 mg/dL)"),
                (1, lambda x: 150 <= x < 301, "Leve (150 - 300 mg/dL)")
            ]
        }
    },
    'Lipasa': {
        'High': {
            'desc_base': 'Elevación Lipasa',
            'reglas': [
                (4, lambda x: x > 265, "Amenaza vital (>265 U/L)"),
                (3, lambda x: 106 <= x <= 265, "Severa (106 - 265 U/L)"),
                (2, lambda x: 79.5 <= x < 106, "Moderada (79.5 - 106 U/L)"),
                (1, lambda x: 54 <= x < 79.5, "Leve (54 - 79.5 U/L)")
            ]
        }
    },
    'Amilasa': {
        'High': {
            'desc_base': 'Elevación Amilasa',
            'reglas': [
                (4, lambda x: x > 1150, "Amenaza vital (>1150 U/L)"),
                (3, lambda x: 576 <= x <= 1150, "Severa (576 - 1150 U/L)"),
                (2, lambda x: 173 <= x < 576, "Moderada (173 - 575 U/L)"),
                (1, lambda x: 116 <= x < 173, "Leve (116 - 172 U/L)")
            ]
        }
    },
    'Creatinquinasa': {
        'High': {
            'desc_base': 'Elevación CK',
            'reglas': [
                (4, lambda x: x > 3080, "Amenaza vital (>3080 U/L)"),
                (3, lambda x: 1541 <= x <= 3080, "Severa (1541 - 3080 U/L)"),
                (2, lambda x: 771 <= x < 1541, "Moderada (771 - 1540 U/L)"),
                (1, lambda x: 309 <= x < 771, "Leve (309 - 770 U/L)")
            ]
        }
    },

    # Coagulación y estado ácido-base
    'TTPA': {
        'High': {
            'desc_base': 'TTPA prolongado',
            'reglas': [
                (3, lambda x: x > 77.5, "Severo (>77.6 s)"),
                (2, lambda x: 46.5 <= x <= 77.5, "Moderado (46.6 - 77.5 s)"),
                (1, lambda x: 32.0 <= x < 46.5, "Leve (32.0 - 46.5 s)")
            ]
        }
    },
    'Fibrinogeno': {
        'Low': { 'desc_base': 'Hipofibrinogenemia', 'reglas': [
            (4, lambda x: x < 50, "Amenaza vital (<50 mg/dL)"),
            (3, lambda x: 50 <= x < 100, "Severa (50 - 99 mg/dL)"),
            (2, lambda x: 100 <= x < 150, "Moderada (100 - 149 mg/dL)"),
            (1, lambda x: 150 <= x < 180, "Leve (150 - 179 mg/dL)")]},
        'High': { 'desc_base': 'Hiperfibrinogenemia', 'reglas': [
            (3, lambda x: x > 1000, "Severa (>1000 mg/dL)"),
            (2, lambda x: 601 <= x <= 1000, "Moderada (601 - 1000 mg/dL)"),
            (1, lambda x: 451 <= x <= 600, "Leve (451 - 600 mg/dL)")]}
    },
    'INR': {
        'High': {
            'desc_base': 'INR elevado',
            'reglas': [
                (3, lambda x: x > 2.5, "Severo (>2.5)"),
                (2, lambda x: 1.5 <= x <= 2.5, "Moderado (1.5 - 2.5)"),
                (1, lambda x: 1.2 <= x < 1.5, "Leve (1.2 - 1.5)")
            ]
        }
    },
    'PH': {
        'Low': { 'desc_base': 'Acidemia', 'reglas': [
            (4, lambda x: x < 7.30, "Acidosis severa (<7.30)"),
            (1, lambda x: 7.30 <= x < 7.31, "Leve (7.30 - 7.31)")]},
        'High': { 'desc_base': 'Alcalemia', 'reglas': [
            (4, lambda x: x > 7.50, "Alcalosis severa (>7.50)"),
            (1, lambda x: 7.48 <= x <= 7.50, "Leve (7.48 - 7.50)")]}
    }
}