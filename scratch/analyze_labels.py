import numpy as np

def main():
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_dataset_v1.npz"
    data = np.load(npz_path)
    states = data["states"]      # shape (N, 4, 9, 9)
    actions = data["actions"]    # shape (N, 2)
    
    # Flatten states to 1D to hash them easily
    num_samples = len(states)
    flat_states = states.reshape(num_samples, -1)
    
    # Group actions by state
    state_to_actions = {}
    for i in range(num_samples):
        state_bytes = flat_states[i].tobytes()
        act = tuple(actions[i])
        if state_bytes not in state_to_actions:
            state_to_actions[state_bytes] = []
        state_to_actions[state_bytes].append(act)
        
    # Stats
    total_unique_states = len(state_to_actions)
    inconsistent_states_count = 0
    max_labels = 0
    inconsistent_samples_count = 0
    
    for state_bytes, acts in state_to_actions.items():
        unique_acts = set(acts)
        if len(unique_acts) > 1:
            inconsistent_states_count += 1
            inconsistent_samples_count += len(acts)
            if len(unique_acts) > max_labels:
                max_labels = len(unique_acts)
                
    print("=== DATASET LABEL INCONSISTENCY ANALYSIS ===")
    print(f"Total samples in dataset      : {num_samples}")
    print(f"Total unique states           : {total_unique_states}")
    print(f"Inconsistent states (multi-act): {inconsistent_states_count} ({inconsistent_states_count / total_unique_states * 100:.2f}%)")
    print(f"Samples with inconsistent labels: {inconsistent_samples_count} ({inconsistent_samples_count / num_samples * 100:.2f}%)")
    print(f"Maximum labels for a single state: {max_labels}")
    
    # Let's count how many samples have states that appear only once
    single_appearance_states = sum(1 for acts in state_to_actions.values() if len(acts) == 1)
    print(f"States appearing exactly once  : {single_appearance_states} ({single_appearance_states / total_unique_states * 100:.2f}%)")

if __name__ == "__main__":
    main()
