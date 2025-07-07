import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.full_agent_trainer import FullAgentTrainer

def main():
    trainer = FullAgentTrainer()
    trainer.validate()

if __name__ == "__main__":
    main()
