from typing import Dict, Any
from datetime import datetime

def ner_to_markdown(datos_estructurados: Dict[str, Any]) -> str:
    """
    Convierte el diccionario del motor NER en una tabla Markdown estructurada.
    """
    if not datos_estructurados or not isinstance(datos_estructurados, dict):
        return "No se detectaron biomarcadores válidos."

    # Cabecera de la tabla
    tabla = "### Resultados de Laboratorio Extraídos\n\n"
    tabla += "| Biomarcador | Valor | Rango Ref. | CTCAE | Descripción |\n"
    tabla += "|---|---|---|---|---|\n"

    for bio, data in datos_estructurados.items():
        valor = f"{data.get('valor', '-')} {data.get('unidad', '')}"
        
        rango = data.get('rango_referencia', {})
        inf = rango.get('inf')
        sup = rango.get('sup')
        rango_str = f"{inf if inf is not None else '-'} / {sup if sup is not None else '-'}"

        ctcae = data.get('ctcae', {})
        grado = ctcae.get('grado', 0)
        desc = ctcae.get('descripcion', 'Sin datos')

        # Semáforo de toxicidad
        if grado == 0:
            semaforo = "🟢 G0"
        elif grado == 1:
            semaforo = "🟡 G1"
            valor = f"**{valor}**"
        elif grado == 2:
            semaforo = "🟠 G2"
            valor = f"**{valor}**"
        elif grado >= 3:
            semaforo = "🔴 G3+"
            valor = f"**{valor}**"
        else:
            semaforo = "⚪ N/A"

        tabla += f"| **{bio}** | {valor} | {rango_str} | {semaforo} | {desc} |\n"

    return tabla

def quick_summary(datos_estructurados: Dict[str, Any], fecha: str = None) -> str:
    """
    Genera una línea de resumen rápido a partir de los datos estructurados del NER.
    """
    if not datos_estructurados or not isinstance(datos_estructurados, dict):
        return "No se detectaron datos suficientes."

    # Si no viene fecha, usamos la fecha de procesamiento 
    if not fecha:
        fecha = datetime.now().strftime("%d/%m/%Y")

    claves_busqueda = {
        'Hb': ['Hb', 'Hemoglobina'],
        'Plaquetas': ['Plaquetas', 'Plaqueta'],
        'Neutrófilos': ['Neutrófilos', 'Neutrofilos'],
        'Creatinina': ['Creatinina']
    }

    fijos_encontrados = {}
    claves_usadas = []

    for etiqueta, posibles_claves in claves_busqueda.items():
        valor = "N/A"
        for clave in posibles_claves:
            if clave in datos_estructurados:
                valor = datos_estructurados[clave].get('valor', 'N/A')
                claves_usadas.append(clave)
                break
        fijos_encontrados[etiqueta] = valor

    criticos = []
    for bio, data in datos_estructurados.items():
        if bio not in claves_usadas:
            grado = data.get('ctcae', {}).get('grado', 0)
            if grado >= 1:
                criticos.append((bio, data.get('valor', 'N/A'), grado))

    criticos.sort(key=lambda x: x[2], reverse=True)
    criticos_seleccionados = criticos[:2] # Dejamos 2 para mantener el formato original 

    linea_resumen = [
        f"Hb: {fijos_encontrados['Hb']}",
        f"Plaquetas: {fijos_encontrados['Plaquetas']}",
        f"Neutrófilos: {fijos_encontrados['Neutrófilos']}",
        f"Creatinina: {fijos_encontrados['Creatinina']}"
    ]

    for crit in criticos_seleccionados:
        linea_resumen.append(f"{crit[0]}: {crit[1]}")

    linea_resumen.append("resto bien.")

    cadena_final = " | ".join(linea_resumen)
    texto_base = f"[{fecha}]: {cadena_final}"
    
    return texto_base

def clinical_context(extracted_data: Dict[str, Any]) -> str:
    """
    Transforma el diccionario NER en el bloque de texto (Secciones A y C) 
    utilizado tanto para el prompt del LLM como para visualización detallada.
    """
    # Sección A
    linea_a = quick_summary(extracted_data)

    # Sección C
    seccion_c = "SECCIÓN C: ESTRATIFICACIÓN DE TOXICIDAD (CTCAE v5.0) ⚠️\n"
    all_tox = [(k, v) for k, v in extracted_data.items() if v.get("ctcae", {}).get("grado", 0) >= 1]
    all_tox.sort(key=lambda x: x[1]["ctcae"]["grado"], reverse=True)
    
    if not all_tox:
        seccion_c += "    - No se observan toxicidades significativas.\n"
    else:
        for k, v in all_tox:
            ctcae = v["ctcae"]
            ref = v.get("rango_referencia", {})
            ref_str = f"{ref.get('inf', '?')} - {ref.get('sup', '?')}"
            seccion_c += f"    - {k} ({v['valor']} {v['unidad']}): Grado {ctcae['grado']}. {ctcae['descripcion']}. (Ref Lab: {ref_str}).\n"

    separador = "\n" + "-"*100 + "\n"
    return f"{linea_a}{separador}{seccion_c}"