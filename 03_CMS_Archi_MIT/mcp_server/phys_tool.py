from mcp.server.fastmcp import FastMCP
import numpy as np

# Servidor MCP
mcp = FastMCP("HEP-Lab-Assistant")


@mcp.tool()
def calculate_pulse_area(amplitude: float, t_max: float = 25.0) -> str:
    """Calcula el área bajo el pulso (energía total)"""
    t = np.linspace(0, 150, 1000)
    pulse = amplitude * (t / t_max) * np.exp(1 - t / t_max)
    area = np.trapezoid(pulse, t)

    return f"La energía total depositada (área) es: {area:.4f} GeV*ns"


if __name__ == "__main__":
    mcp.run()
