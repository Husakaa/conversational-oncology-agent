import json
import os
import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ner.extractor import InvalidDocumentError, MedicalExtractor


def ordenar_nombre_archivo(nombre: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', nombre)]


def cargar_analiticas(directorio: str) -> pd.DataFrame:
    datos = []
    if not os.path.exists(directorio):
        raise FileNotFoundError(f"Directorio no encontrado: {directorio}")

    for nombre_archivo in sorted(os.listdir(directorio), key=ordenar_nombre_archivo):
        ruta_completa = os.path.join(directorio, nombre_archivo)
        if not nombre_archivo.lower().endswith('.txt') or not os.path.isfile(ruta_completa):
            continue

        contenido = None
        for encoding in ('utf-8', 'latin-1'):
            try:
                with open(ruta_completa, 'r', encoding=encoding) as f:
                    contenido = f.read()
                break
            except UnicodeDecodeError:
                continue

        if contenido is None:
            raise UnicodeDecodeError(f"No se pudo decodificar el archivo: {ruta_completa}")

        datos.append({'document_id': nombre_archivo.replace('.txt', ''), 'texto': contenido})

    return pd.DataFrame(datos)


def extraer_valores(df_textos: pd.DataFrame, extractor: MedicalExtractor, lote: str) -> pd.DataFrame:
    registros = []
    patrones = list(extractor.patrones.keys())

    for _, fila in df_textos.iterrows():
        resultado = {'document_id': fila['document_id'], 'lote': lote}
        texto = fila['texto']

        try:
            datos = extractor.analizar_texto(texto)
            resultado['valid'] = True
            resultado['biomarker_count'] = len(datos)
            resultado['capture_rate'] = len(datos) / len(patrones) if patrones else 0.0
        except InvalidDocumentError:
            datos = {}
            resultado['valid'] = False
            resultado['biomarker_count'] = 0
            resultado['capture_rate'] = 0.0

        for biomarcador in patrones:
            resultado[biomarcador] = np.nan

        for biomarcador, info in datos.items():
            resultado[biomarcador] = info.get('valor', np.nan)

        registros.append(resultado)

    return pd.DataFrame(registros)


def crear_plot_histogramas(df_long: pd.DataFrame, lote: str, salida: str):
    sns.set_style('whitegrid')
    biomarkers = df_long['biomarker'].unique()
    n_biom = len(biomarkers)
    cols = 4
    filas = int(np.ceil(n_biom / cols))

    plt.figure(figsize=(cols * 4.5, filas * 3.5))
    for idx, biomarcador in enumerate(biomarkers, start=1):
        subset = df_long[df_long['biomarker'] == biomarcador]
        plt.subplot(filas, cols, idx)
        sns.histplot(subset['value'], kde=False, bins=20, color='#4c72b0')
        plt.title(biomarcador)
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')

    plt.suptitle(f'Distribución numérica por biomarcador - {lote}', y=1.02, fontsize=16)
    plt.tight_layout()
    ruta = os.path.join(salida, f'distribucion_biomarcadores_{lote}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Guardado: {ruta}')


def crear_plot_boxplots(df: pd.DataFrame, lote: str, salida: str):
    sns.set_style('whitegrid')
    numeric = df.select_dtypes(include=[np.number]).drop(columns=['capture_rate', 'biomarker_count'])
    numeric = numeric.loc[:, numeric.notna().sum() >= 2]

    if numeric.shape[1] == 0:
        print(f'No hay suficientes biomarcadores numéricos para generar boxplots en {lote}.')
        return

    plt.figure(figsize=(12, max(8, numeric.shape[1] * 0.4)))
    sns.boxplot(data=numeric, orient='h', palette='Set3', showfliers=True)
    plt.title(f'Diagrama de bigotes por biomarcador - {lote}')
    plt.xlabel('Valor numérico')
    plt.ylabel('Biomarcador')
    plt.tight_layout()
    ruta = os.path.join(salida, f'boxplot_biomarcadores_{lote}.png')
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Guardado: {ruta}')


def guardar_resumen_estadistico(df: pd.DataFrame, lote: str, salida: str):
    numeric = df.select_dtypes(include=[np.number]).drop(columns=['capture_rate', 'biomarker_count'])
    ruta = os.path.join(salida, f'resumen_estadistico_{lote}.csv')
    numeric.describe(include='all').transpose().to_csv(ruta, index=True)
    print(f'Guardado: {ruta}')


def _estadisticos_basicos(valores: List[float], usar_mediana: bool = False, excluir_outliers: bool = False) -> Dict[str, float]:
    if not valores:
        return {'media': 0.0, 'desviacion_estandar': 0.0, 'min': 0.0, 'max': 0.0, 'rango': [0.0, 0.0]}

    arr = np.asarray(valores, dtype=float)
    if excluir_outliers and len(arr) > 2:
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        arr = arr[(arr >= lower) & (arr <= upper)]

    if arr.size == 0:
        return {'media': 0.0, 'desviacion_estandar': 0.0, 'min': 0.0, 'max': 0.0, 'rango': [0.0, 0.0]}

    medida_central = float(np.median(arr)) if usar_mediana else float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    minimo = float(np.min(arr))
    maximo = float(np.max(arr))

    resultado = {
        'media': round(medida_central, 4),
        'desviacion_estandar': round(std, 4),
        'min': round(minimo, 4),
        'max': round(maximo, 4),
        'rango': [round(minimo, 4), round(maximo, 4)]
    }

    if usar_mediana:
        resultado['mediana'] = round(medida_central, 4)

    return resultado


def calcular_metricas_recomendadas(df_textos: pd.DataFrame, lote: str, extractor: MedicalExtractor = None) -> Dict[str, object]:
    """Calcula las métricas recomendadas para un lote de analíticas.

    Métricas incluidas:
    - N total de analíticas.
    - Longitud del documento en palabras y tokens aproximados.
    - Número de biomarcadores detectables por informe.
    - Tasa de escasez del catálogo por analítica.

    Se evita cualquier indicador no explicativo para informes tabulados, como nubes
    de palabras, conteo de caracteres por frase o índices de legibilidad.
    """
    if isinstance(df_textos, list):
        df_textos = pd.DataFrame(df_textos)
    elif isinstance(df_textos, dict):
        df_textos = pd.DataFrame([df_textos])

    if extractor is None:
        extractor = MedicalExtractor()

    catalogo = list(extractor.patrones.keys())
    total_catalogo = len(catalogo)

    longitudes_palabras = []
    longitudes_tokens = []
    biomarcadores_detectados = []
    tasa_escasez_catalogo = []

    for _, fila in df_textos.iterrows():
        texto = str(fila.get('texto', ''))
        palabras = re.findall(r"\b\w+\b", texto)
        n_palabras = len(palabras)
        n_tokens = max(1, round(n_palabras * 1.33)) if texto.strip() else 0

        longitudes_palabras.append(float(n_palabras))
        longitudes_tokens.append(float(n_tokens))

        try:
            resultados = extractor.analizar_texto(texto)
        except InvalidDocumentError:
            resultados = {}

        n_biomed = len(resultados)
        biomarcadores_detectados.append(float(n_biomed))

        if total_catalogo > 0:
            porcentaje_ausente = ((total_catalogo - n_biomed) / total_catalogo) * 100.0
        else:
            porcentaje_ausente = 0.0
        tasa_escasez_catalogo.append(float(porcentaje_ausente))

    metricas = {
        'lote': lote,
        'n_analiticas': int(len(df_textos)),
        'longitud_documentos_palabras': _estadisticos_basicos(longitudes_palabras, usar_mediana=True, excluir_outliers=False),
        'longitud_documentos_tokens_aproximados': _estadisticos_basicos(longitudes_tokens, usar_mediana=True, excluir_outliers=False),
        'biomarcadores_detectables_por_informe': _estadisticos_basicos(biomarcadores_detectados),
        'tasa_escasez_catalogo': {
            **_estadisticos_basicos(tasa_escasez_catalogo),
            'unidad': '%'
        }
    }
    return metricas


def calcular_metricas(df_textos: pd.DataFrame, lote: str, extractor: MedicalExtractor = None) -> Dict[str, object]:
    return calcular_metricas_recomendadas(df_textos, lote, extractor)


def generar_estudio(distribution_dir: str = None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    analiticas_dir = os.path.join(base_dir, 'analiticas')
    salida = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(salida, exist_ok=True)

    extractor = MedicalExtractor()
    lotes = {'lote1': os.path.join(analiticas_dir, 'lote1'), 'lote2': os.path.join(analiticas_dir, 'lote2')}
    df_resultados = []

    for nombre_lote, carpeta in lotes.items():
        df_textos = cargar_analiticas(carpeta)
        if df_textos.empty:
            print(f'No se encontraron textos para {nombre_lote}.')
            continue

        print(f'Procesando {len(df_textos)} documentos de {nombre_lote}...')
        df_lote = extraer_valores(df_textos, extractor, nombre_lote)
        df_resultados.append(df_lote)

        # Gráficas por lote
        df_valores = df_lote.melt(
            id_vars=['document_id', 'lote', 'valid', 'biomarker_count', 'capture_rate'],
            var_name='biomarker',
            value_name='value'
        )
        df_valores = df_valores.dropna(subset=['value']).query('biomarker not in ["capture_rate", "biomarker_count"]')

        if df_valores.empty:
            print(f'No hay valores numéricos extraídos para {nombre_lote}.')
            continue

        crear_plot_histogramas(df_valores, nombre_lote, salida)
        crear_plot_boxplots(df_lote, nombre_lote, salida)
        guardar_resumen_estadistico(df_lote, nombre_lote, salida)

        metricas = calcular_metricas(df_textos, nombre_lote, extractor)
        ruta_metricas = os.path.join(os.path.dirname(base_dir), 'output', f'metricas_{nombre_lote}.json')
        os.makedirs(os.path.dirname(ruta_metricas), exist_ok=True)
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, ensure_ascii=False, indent=4)
        print(f'Métricas guardadas en {ruta_metricas}')

        invalidos = df_lote[~df_lote['valid']]
        if not invalidos.empty:
            ruta_invalidos = os.path.join(salida, f'documentos_invalidos_{nombre_lote}.csv')
            invalidos.to_csv(ruta_invalidos, index=False)
            print(f'Documentos inválidos guardados en {ruta_invalidos}')

    if df_resultados:
        df_completo = pd.concat(df_resultados, ignore_index=True)
        resumen_general = os.path.join(salida, 'resumen_general.csv')
        df_completo.to_csv(resumen_general, index=False)
        print(f'Resumen general guardado en {resumen_general}')


if __name__ == '__main__':
    generar_estudio()
