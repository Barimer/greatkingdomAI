import os
import numpy as np

def main():
    data_dir = r"C:\Users\User\source\repos\greatkingdomAI\data"
    
    # We will combine multiple selfplay datasets to create a rich and diverse value dataset
    dataset_files = [
        "selfplay_dataset_v1.npz",
        "selfplay_diverse_v3_1000.npz",
        "selfplay_fast_depth3_500.npz"
    ]
    
    all_states = []
    all_targets = []
    
    print("=== GREAT KINGDOM AI - VALUE DATASET CREATOR ===")
    
    for filename in dataset_files:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: File {filename} not found, skipping...")
            continue
            
        print(f"Loading {filename}...")
        data = np.load(path)
        states = data["states"]
        players = data["players"]
        results = data["results"]
        
        # Calculate targets: +1 if current player == winner, else -1
        targets = np.where(players == results, 1.0, -1.0)
        
        all_states.append(states)
        all_targets.append(targets)
        
    if not all_states:
        raise ValueError("No datasets were successfully loaded!")
        
    combined_states = np.concatenate(all_states, axis=0)
    combined_targets = np.concatenate(all_targets, axis=0)
    
    output_path = os.path.join(data_dir, "value_dataset.npz")
    print(f"\nSaving combined dataset...")
    np.savez_compressed(
        output_path,
        states=combined_states,
        targets=combined_targets
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    plus_ones = np.sum(combined_targets == 1.0)
    minus_ones = np.sum(combined_targets == -1.0)
    
    print("-" * 50)
    print(f"Value Dataset saved to: {output_path}")
    print(f"Total States: {len(combined_states)}")
    print(f"Target Distribution: +1 (Win): {plus_ones} ({plus_ones/len(combined_targets)*100:.1f}%) | -1 (Loss): {minus_ones} ({minus_ones/len(combined_targets)*100:.1f}%)")
    print(f"File Size: {file_size_mb:.2f} MB")
    print("=" * 50)

if __name__ == "__main__":
    main()
