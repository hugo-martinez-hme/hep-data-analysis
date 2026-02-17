import asyncio
import os
import sys
import pandas as pd
import numpy as np
from types import SimpleNamespace
from langchain_core.tools import tool

sys.path.append("/app/03_CMS_Archi_MIT")
from agents.physics_agent import CMSPhysicsAgent
from middleware.physics_guardrails import validate_physics_response
import src.utils.config_access as config_access


@tool
def cms_data_analyzer(min_pt: float) -> str:
    """Extrae datos reales del dataset CMS."""
    try:
        df = pd.read_csv("/app/MuRun2010B.csv")
        df.columns = df.columns.str.strip()
        mask = (df["pt1"] > min_pt) | (df["pt2"] > min_pt)
        filtered = df[mask].copy()
        d_eta, d_phi = (
            filtered["eta1"] - filtered["eta2"],
            filtered["phi1"] - filtered["phi2"],
        )
        filtered["M"] = np.sqrt(
            2 * filtered["pt1"] * filtered["pt2"] * (np.cosh(d_eta) - np.cos(d_phi))
        )
        res = (
            filtered[["Run", "Event", "pt1", "pt2", "M"]]
            .sort_values(by="M", ascending=False)
            .head(5)
        )
        return res.to_string(index=False)
    except Exception as e:
        return f"Error: {str(e)}"


async def run_cms_analysis():
    config = {
        "archi": {
            "pipeline_map": {
                "CMSPhysicsAgent": {
                    "models": {
                        "required": {"chat_model": "openai/llama-3.3-70b-versatile"}
                    },
                    "tools": ["native"],
                }
            }
        },
        "services": {
            "chat_app": {
                "providers": {
                    "openai": {
                        "base_url": "https://api.groq.com/openai/v1",
                        "default_model": "llama-3.3-70b-versatile",
                    }
                }
            }
        },
    }

    # Mock de Archi Config
    mock_static = SimpleNamespace(
        archi_config=config["archi"],
        services_config=config["services"],
        global_config={},
        data_manager_config={},
        deployment_name="dev",
        config_version="1",
        sources_config={},
        available_pipelines=[],
        available_models=[],
        available_providers=[],
    )
    config_access.get_static_config = lambda: mock_static
    config_access.get_full_config = lambda **kwargs: config

    # 1. OBTENER DATOS REALES PRIMERO
    print("🧪 [DATA] Extrayendo datos del CSV...")
    tabla_real = cms_data_analyzer.invoke({"min_pt": 45.0})

    # 2. INYECTAR DATOS EN EL PROMPT DEL AGENTE
    agent = CMSPhysicsAgent(config=config)

    system_context = (
        "Eres un físico del CERN. NO eres un modelo de lenguaje genérico. "
        "Tienes prohibido decir que no tienes acceso a herramientas. "
        f"Aquí están los DATOS REALES del CMS para pT > 45 GeV:\n{tabla_real}\n"
        "Analiza estos datos específicamente. Si la masa M no es 91 GeV, dilo."
    )

    query = "Analiza los datos proporcionados, muestra la tabla y dime si son candidatos a Bosón Z."
    print(f"\n🧠 [AGENT] Analizando con Llama-3.3...")

    # Pasamos el contexto de sistema forzado
    response = agent.invoke(history=[("system", system_context), ("human", query)])

    print("\n--- 🏁 RESPUESTA FINAL ---")
    print(validate_physics_response(response.answer))


if __name__ == "__main__":
    asyncio.run(run_cms_analysis())
