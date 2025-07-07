import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.controller_trainer import ControllerTrainer

def main():
    trainer = ControllerTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
