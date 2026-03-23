from typing import Dict, Any

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