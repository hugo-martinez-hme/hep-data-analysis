import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from pysqlite3 import dbapi2 as sqlite3

sys.modules["sqlite3"] = sqlite3

env_path = Path("/app/.env")
load_dotenv(dotenv_path=env_path)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate


async def run_archi_agent():
    server_params = StdioServerParameters(
        command="python3",
        args=["/app/03_CMS_Archi_MIT/mcp_server/physics_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()

            langchain_tools = []
            for t in mcp_tools.tools:

                async def create_mcp_tool(input_arg, name=t.name, schema=t.inputSchema):
                    try:
                        val = float(input_arg)
                    except:
                        val = input_arg

                    param_name = schema.get("required", [None])[0]
                    result = await session.call_tool(name, {param_name: val})
                    return result.content[0].text

                langchain_tools.append(
                    Tool(
                        name=t.name,
                        func=None,
                        coroutine=create_mcp_tool,
                        description=t.description,
                    )
                )

            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

            template = """Eres Archi-MIT, el asistente de IA para el experimento CMS del CERN.
            Tu objetivo es analizar colisiones y calcular energías usando herramientas reales.
            Si estas seguro de que ya tienes la respuesta, no uses más herramientas y responde DIRECTAMENTE con Final Answer,
            NO intentes usar una herramienta si no tienes el Action Input.
            TOOLS:
            {tools}

            Formato:
            Thought: ¿Qué necesito hacer?
            Action: [{tool_names}]
            Action Input: entrada para la herramienta
            Observation: resultado
            ... (repetir si es necesario)
            Final Answer: respuesta final técnica.

            Pregunta: {input}
            Thought: {agent_scratchpad}"""

            prompt = PromptTemplate.from_template(template)
            agent = create_react_agent(llm, langchain_tools, prompt)

            agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                verbose=True,
                handle_parsing_errors=True,
            )

            print("\n--- 🚀 ARCHI-MIT: SISTEMA MCP INICIADO ---")

            await agent_executor.ainvoke(
                {
                    "input": "Analiza eventos reales en el CMS con pT > 40. Después, calcula el área del pulso '2,15,30,15,2'."
                }
            )


if __name__ == "__main__":
    try:
        asyncio.run(run_archi_agent())
    except KeyboardInterrupt:
        pass
