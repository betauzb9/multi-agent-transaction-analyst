from src.agents.code_agent import code_agent
from src.state import initial_state

if __name__ == "__main__":
    state = initial_state("What is 15% of 840, rounded to two decimal places?")
    result = code_agent(state)
    print("code_result:", result["code_result"])
    print("steps:", result["steps"])
