from src.agents.critic import critic
from src.state import initial_state

state = initial_state("How many transactions were categorized as Dining & Coffee?")
state["sql_result"] = "SELECT COUNT(*) FROM transactions WHERE category='Dining & Coffee';\n\u2192 [(120,)]"
state["answer"] = "There were 9999 transactions categorized as Dining & Coffee."  # deliberately wrong

result = critic(state)
print("revisions:", result["revisions"])
print("steps:", result["steps"])
