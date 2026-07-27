from src.memory import remember_turn, recall_relevant_turns

remember_turn(
    "How many transactions were categorized as Dining & Coffee?",
    "There are 142 transactions categorized as Dining & Coffee.",
)

recalled = recall_relevant_turns("and what about last quarter, same category?")

print("recalled turns:")
for i, turn in enumerate(recalled, 1):
    print("\n--- turn " + str(i) + " ---\n" + turn)

assert recalled, "F10 FAILED: nothing recalled - memory store is empty or not persisting."
assert "Dining" in recalled[0], "F10 FAILED: recalled turn doesn't match the follow-up topic."
print("\nF10 OK: earlier turn was recalled for the follow-up question.")
