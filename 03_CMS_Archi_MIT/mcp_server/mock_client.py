import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_test():
    # Configuración para conectar con el servidor
    server_params = StdioServerParameters(
        command="python3",
        args=["03_CMS_Archi_MIT/mcp_server/phys_tool.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"✅ Herramientas detectadas: {[t.name for t in tools.tools]}")

            # Simulación de llamada
            print("\n IA: 'Calculando energía para una amplitud de 5.0...'")
            result = await session.call_tool("calculate_pulse_area", {"amplitude": 5.0})
            print(f" Resultado del servidor: {result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(run_test())
