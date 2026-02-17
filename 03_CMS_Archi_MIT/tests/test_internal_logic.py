import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_server.physics_mcp_server import cms_data_analyzer, AnalyzerRequest


def test_physics():
    print("--- TEST DE LÓGICA DE FÍSICA ---")

    try:
        print("\n1. Probando validación negativa (debe fallar):")
        req_fail = AnalyzerRequest(min_pt=-10)
    except Exception as e:
        print(f" Validación Pydantic capturada correctamente: {e}")

    print("\n2. Ejecutando análisis sobre MuRun2010B.csv (pT > 40):")
    req_success = AnalyzerRequest(min_pt=40.0)
    result = cms_data_analyzer(req_success)
    print(result)


if __name__ == "__main__":
    test_physics()
