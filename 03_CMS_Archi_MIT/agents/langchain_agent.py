import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pysqlite3 import dbapi2 as sqlite3

sys.modules["sqlite3"] = sqlite3

load_dotenv()


api_key = os.getenv("GROQ_API_KEY").strip()
os.environ["GROQ_API_KEY"] = api_key

if not api_key:
    print("❌ No se encontró GROQ_API_KEY")
    sys.exit(1)


from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [
    "Si el servidor de transferencia de datos falla, reinicie el servicio FTS3.",
    "El detector TileCal de ATLAS está dividido en 64 sectores en cada barril.",
    "Para calcular el área de un pulso de física, usa la herramienta physics_calculator.",
]
vectorstore = Chroma.from_texts(texts, embeddings)


def calculate_pulse_area(pulse_data):
    import numpy as np

    try:
        data = np.array([float(x) for x in pulse_data.split(",")])
        if hasattr(np, "trapezoid"):
            area = np.trapezoid(data)
        else:
            area = np.trapz(data)
        return f"El área calculada (método trapezoidal) es: {area}"
    except Exception as e:
        return f"Error en el cálculo: {str(e)}"


def search_manual(query):
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No se encontró información."


tools = [
    Tool(
        name="physics_calculator",
        func=calculate_pulse_area,
        description="Calcula el área de señales. Entrada: números separados por coma.",
    ),
    Tool(
        name="search_manual",
        func=search_manual,
        description="Busca instrucciones en el manual de operaciones del CMS.",
    ),
]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key.strip()
)

template = """Eres un experto en el CMS del CERN. Responde a la pregunta usando tus herramientas.

TOOLS:
------
Tienes acceso a estas herramientas:
{tools}

Para usar una herramienta, usa este formato exacto:
Thought: ¿Necesito usar una herramienta? Sí
Action: la herramienta (una de [{tool_names}])
Action Input: la entrada de la herramienta
Observation: el resultado de la herramienta
... (repite si es necesario)
Thought: ¿Necesito usar una herramienta? No
Final Answer: [tu respuesta final]

Pregunta: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

print("\n--- 🚀 AGENTE ARCHI-MIT INICIADO ---")
agent_executor.invoke(
    {
        "input": "Tengo un pulso 1,2,5,2,1. ¿Qué debo hacer según el manual y cuál es el área?"
    }
)
