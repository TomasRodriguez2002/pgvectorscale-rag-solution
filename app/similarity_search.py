from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Devolution question
# --------------------------------------------------------------

relevant_question = "Que productos puedo devolver?"
results = vec.search(relevant_question, limit=3)

response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "Cuantos goles hizo Messi en el partido de la semana?"

results = vec.search(irrelevant_question, limit=3)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------

relevant_question = "Cuantos goles hizo Messi en el partido de la semana?"

metadata_filter = {"category": "Compras"}

results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Advanced filtering using Predicates
# --------------------------------------------------------------

predicates = client.Predicates("category", "==", "Devoluciones")
results = vec.search(relevant_question, limit=3, predicates=predicates)

predicates = client.Predicates("category", "==", "Compras") | client.Predicates(
    "category", "==", "Devoluciones") | client.Predicates("category", "==", "Cancelaciones")
results = vec.search(relevant_question, limit=3, predicates=predicates)

predicates = client.Predicates("category", "==", "Shipping") & client.Predicates(
    "created_at", ">", "2025-05-01"
)
results = vec.search(relevant_question, limit=3, predicates=predicates)

# --------------------------------------------------------------
# Time-based filtering
# --------------------------------------------------------------

# September — Returning results
time_range = (datetime(2025, 5, 1), datetime(2025, 5, 7))
results = vec.search(relevant_question, limit=3, time_range=time_range)
