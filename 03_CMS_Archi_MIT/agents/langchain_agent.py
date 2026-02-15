import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pysqlite3 import dbapi2 as sqlite3

sys.modules["sqlite3"] = sqlite3
env_path = Path("/app/.env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GROQ_API_KEY")

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [
    "En el dataset Zmumu, pt1 y pt2 son los momentos transversales de los dos muones detectados.",
    "La masa invariante (M) es la masa de la partícula madre que se desintegró en dos muones.",
    "Si la masa invariante (M) está cerca de 91.2 GeV, el evento es un candidato claro a Bosón Z.",
    "Si la masa M está entre 80 y 102 GeV, se considera un evento de alta prioridad para estudio de la fuerza débil.",
    "Un muón con pt > 30 GeV se considera de alta energía y es clave para identificar procesos de física interesantes.",
    "Para calcular el área del pulso de calibración del detector, usa la herramienta physics_calculator.",
]
vectorstore = Chroma.from_texts(texts, embeddings)


def calculate_pulse_area(pulse_data):
    import numpy as np

    try:
        data = np.array([float(x) for x in pulse_data.split(",")])
        area = np.trapezoid(data) if hasattr(np, "trapezoid") else np.trapz(data)
        return f"Área de calibración: {area}"
    except:
        return "Error de formato."


def search_manual(query):
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No hay info."


def cms_real_data_analyzer(min_pt):
    import pandas as pd
    import numpy as np

    try:
        df = pd.read_csv("/app/MuRun2010B.csv")
        df.columns = df.columns.str.strip()

        interesantes = df[
            (df["pt1"] > float(min_pt)) | (df["pt2"] > float(min_pt))
        ].copy()
        interesantes["M"] = np.sqrt(
            2
            * interesantes["pt1"]
            * interesantes["pt2"]
            * (
                np.cosh(interesantes["eta1"] - interesantes["eta2"])
                - np.cos(interesantes["phi1"] - interesantes["phi2"])
            )
        )

        top_events = interesantes.sort_values(by="M", ascending=False).head(5)

        if top_events.empty:
            return f"No encontré eventos con pt > {min_pt} GeV."

        return f"DATOS REALES CMS (Candidatos Z):\n{top_events[['Run', 'Event', 'pt1', 'pt2', 'M']].to_string(index=False)}"
    except Exception as e:
        return f"Error analizando dimuones: {str(e)}"


tools = [
    Tool(
        name="physics_calculator",
        func=calculate_pulse_area,
        description="Calcula áreas de pulsos. Entrada: '1,2,3'.",
    ),
    Tool(
        name="search_manual",
        func=search_manual,
        description="Busca reglas de física en el manual.",
    ),
    Tool(
        name="cms_data_analyzer",
        func=cms_real_data_analyzer,
        description="Busca eventos en el dataset real de CMS. Entrada: un número de pt mínimo.",
    ),
]

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

template = """Eres Archi-MIT, el agente de IA para el CMS del CERN. 
Responde a la pregunta usando tus herramientas sobre los datos reales.

TOOLS:
{tools}

Formato de respuesta:
Thought: ¿Qué debo hacer?
Action: una de [{tool_names}]
Action Input: la entrada de la herramienta
Observation: resultado de la acción
... (repetir si es necesario)
Final Answer: respuesta final para el investigador.

Pregunta: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

print("\n--- 🚀 AGENTE ARCHI (DATOS REALES) INICIADO ---")
agent_executor.invoke(
    {
        "input": "Busca eventos reales en el CMS con pt superior a 30 GeV. ¿Son de alta prioridad según el manual? También calcula el área del pulso 1,10,20,10,1."
    }
)
