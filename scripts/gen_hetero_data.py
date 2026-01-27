from src.data_generation import generate_dataset

if __name__ == "__main__":
    generate_dataset("data/sim_hetero_001.jsonl", num_frames=300)
