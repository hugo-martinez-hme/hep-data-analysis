import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

mcp = FastMCP("CMS-Physics-Engine")


class AnalyzerRequest(BaseModel):
    min_pt: float = Field(..., description="pT mínimo")


@mcp.tool()
def cms_data_analyzer(request: AnalyzerRequest) -> str:
    try:
        path = "/app/MuRun2010B.csv"
        if not os.path.exists(path):
            return "Error: Dataset no encontrado"
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        mask = (df["pt1"] > request.min_pt) | (df["pt2"] > request.min_pt)
        filtered = df[mask].copy()
        if filtered.empty:
            return "Sin eventos encontrados"

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
        return f"--- RESULTADOS CMS ---\n{res.to_string(index=False)}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
