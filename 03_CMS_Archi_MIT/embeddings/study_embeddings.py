import numpy as np
from sentence_transformers import SentenceTransformer

# Modelo que traduce frases a números (vectores)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Base de Conocimientos
knowledge = [
    "El calorímetro TileCal de ATLAS mide la energía de los hadrones.",
    "Si el servidor de transferencia de datos falla, reinicie el servicio FTS3.",
    "El bosón de Higgs fue descubierto en 2012 por ATLAS y CMS.",
]

embeddings = model.encode(knowledge)

query = "¿Qué hago si fallan las transferencias de datos?"
query_vec = model.encode([query])

similarities = np.dot(embeddings, query_vec.T)
best_idx = np.argmax(similarities)

print(f" Pregunta: {query}")
print(f" Respuesta recuperada: {knowledge[best_idx]}")
