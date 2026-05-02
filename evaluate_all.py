import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.rl_task import main as rl_main
from experiments.nlp_task import main as nlp_main

def main():
    print("==================================================")
    print("ReMoDA Unified Evaluation Framework")
    print("==================================================\n")

    print("--- TASK 1: Reinforcement Learning (Decision Transformer on CartPole) ---")
    try:
        rl_main()
        print("\n[RL Task Completed Successfully]\n")
    except Exception as e:
        print(f"\n[RL Task Failed]: {e}\n")

    print("==================================================\n")

    print("--- TASK 2: Natural Language Processing (Sequence Classification on IMDb) ---")
    try:
        nlp_main()
        print("\n[NLP Task Completed Successfully]\n")
    except Exception as e:
        print(f"\n[NLP Task Failed]: {e}\n")

    print("==================================================")
    print("All Evaluations Completed.")
    print("==================================================")

if __name__ == "__main__":
    main()
