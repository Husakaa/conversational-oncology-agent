import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.test.analyze_corpus_distributions import calcular_metricas


def test_calcular_metricas():
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

    metricas = calcular_metricas(textos, lote='lote_test')

    assert metricas['n_analiticas'] == 2
    assert 'longitud_documentos_palabras' in metricas
    assert 'biomarcadores_detectables_por_informe' in metricas
    assert 'tasa_escasez_catalogo' in metricas
    assert metricas['longitud_documentos_palabras']['media'] > 0
    assert metricas['biomarcadores_detectables_por_informe']['media'] > 0
    assert 0 <= metricas['tasa_escasez_catalogo']['media'] <= 100
