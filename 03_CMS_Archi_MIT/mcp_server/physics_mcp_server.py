import os
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np

mcp = FastMCP("CMS-Physics-Engine")


@mcp.tool()
def cms_data_analyzer(min_pt: float) -> str:
    """
    Analiza datos reales de muones del CMS. Busca eventos que superen un umbral de pT
    y calcula la Masa Invariante (M). Entrada: pT mínimo (float).
    """
    try:
        csv_path = "/app/MuRun2010B.csv"
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        mask = (df["pt1"] > min_pt) | (df["pt2"] > min_pt)
        filtered = df[mask].copy()

        if filtered.empty:
            return f"No se encontraron eventos con pT > {min_pt} GeV."

        filtered["M"] = np.sqrt(
            2
            * filtered["pt1"]
            * filtered["pt2"]
            * (
                np.cosh(filtered["eta1"] - filtered["eta2"])
                - np.cos(filtered["phi1"] - filtered["phi2"])
            )
        )

        top_5 = (
            filtered[["Run", "Event", "pt1", "pt2", "M"]]
            .sort_values(by="M", ascending=False)
            .head(5)
        )
        return "DATOS REALES CMS (Top 5 Candidatos):\n" + top_5.to_string(index=False)

    except Exception as e:
        return f"Error en el análisis de física: {str(e)}"


@mcp.tool()
def physics_calculator(pulse_data: str) -> str:
    """Calcula el área bajo un pulso de detector (Regla del trapecio). Entrada: '1,10,20,10,1'."""
    try:
        data = np.array([float(x) for x in pulse_data.split(",")])
        area = np.trapezoid(data) if hasattr(np, "trapezoid") else np.trapz(data)
        return f"Integrated Pulse Area: {area:.4f}"
    except Exception as e:
        return f"Error de formato o cálculo: {str(e)}"


# Validate tool


@mcp.tool()
def validate_physics_request(pt_threshold: float, etas: list[float]) -> str:
    """
    Valida si una petición de análisis tiene sentido físico antes de procesarla.
    """

    if pt_threshold < 0:
        return "ERROR: El pT (momento transversal) no puede ser negativo. Revisa la petición."

    if any(abs(eta) > 2.5 for eta in etas):
        return "ADVERTENCIA: Algunos muones están fuera de la aceptación del detector CMS (|eta| > 2.5)."

    return "VALID: Parámetros físicos dentro de los límites operacionales del CMS."


if __name__ == "__main__":
    mcp.run()
