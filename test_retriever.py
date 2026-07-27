from src.agents.retriever import retriever_agent
from src.state import initial_state

state = initial_state("What happens when the model has low confidence in a merchant prediction?")
result = retriever_agent(state)
print("steps:", result["steps"])
print("\nretrieved chunks:")
for i, doc in enumerate(result["documents"], 1):
    print(f"\n--- chunk {i} ---\n{doc[:300]}")
