import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CMS-Detector-Monitor")


@mcp.tool()
def get_muon_chamber_status(chamber_id: str) -> str:
    """
    Checks the operational status of a specific CMS Muon Chamber (DT/CSC/RPC).
    Returns HV (High Voltage) status and gas pressure stability.
    """

    chambers = {
        "DT+1/1/1": {"status": "Operational", "HV": "3600V", "gas": "Stable"},
        "DT-2/1/1": {"status": "Warning", "HV": "3200V", "gas": "Low Pressure"},
        "CSC+1/1": {"status": "Operational", "HV": "3500V", "gas": "Stable"},
    }

    data = chambers.get(chamber_id, {"status": "Unknown", "HV": "0V", "gas": "N/A"})
    return f"Chamber {chamber_id} Report: Status={data['status']}, HighVoltage={data['HV']}, Gas={data['gas']}"


@mcp.resource("cms://detector/summary")
def get_detector_summary() -> str:
    """Returns a general summary of the CMS detector status."""
    return "All systems nominal. Magnet at 3.8T. Muon systems armed."


if __name__ == "__main__":
    mcp.run()
