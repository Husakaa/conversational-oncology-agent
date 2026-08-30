import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.test.analyze_corpus_distributions import calcular_metricas_recomendadas


def test_calcular_metricas_recomendadas():
    textos = [
        {
            'document_id': 'doc_1',
            'texto': 'Hb 12.5 g/dL. Glucosa 96 mg/dL. Creatinina 0.8 mg/dL.'
        },
        {
            'document_id': 'doc_2',
            'texto': 'Hb 10.0 g/dL. Creatinina 0.7 mg/dL. ALT 20 U/L. AST 18 U/L.'
        }
    ]

    metricas = calcular_metricas_recomendadas(textos, lote='lote_test')

    assert metricas['n_analiticas'] == 2
    assert 'longitud_documentos_palabras' in metricas
    assert 'biomarcadores_detectables_por_informe' in metricas
    assert 'tasa_escasez_catalogo' in metricas
    assert metricas['longitud_documentos_palabras']['media'] > 0
    assert metricas['biomarcadores_detectables_por_informe']['media'] > 0
    assert 0 <= metricas['tasa_escasez_catalogo']['media'] <= 100


def test_descarte_outliers_iqr():
    # Crear conjunto con un outlier claro en longitud
    textos = [
        {'document_id': f'doc_{i}', 'texto': 'Hb 12.5 g/dL. Glucosa 96 mg/dL.'}
        for i in range(10)
    ]
    # Documento anormalmente largo (outlier)
    textos.append({
        'document_id': 'doc_outlier',
        'texto': 'Hb 12.5 g/dL. ' + 'palabra ' * 500
    })

    metricas = calcular_metricas_recomendadas(textos, lote='lote_outlier_test')
    descartadas_palabras = metricas['longitud_documentos_palabras']['analiticas_descartadas_por_iqr']
    descartadas_tokens = metricas['longitud_documentos_tokens_aproximados']['analiticas_descartadas_por_iqr']

    assert len(descartadas_palabras) == 1
    assert descartadas_palabras[0]['document_id'] == 'doc_outlier'
    assert descartadas_palabras[0]['metrica'] == 'palabras'
    assert descartadas_palabras[0]['motivo'] == 'Outlier superior'

    assert len(descartadas_tokens) == 1
    assert descartadas_tokens[0]['document_id'] == 'doc_outlier'
    assert descartadas_tokens[0]['metrica'] == 'tokens'
    assert descartadas_tokens[0]['motivo'] == 'Outlier superior'

