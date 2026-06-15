import os
import numpy as np
import torch
from torch.utils.data import Dataset

def move_to_index(move):
    """
    Converts a move coordinate to a flat index.
    - If move is a pass ([-1, -1] or (-1, -1)), returns 81.
    - Otherwise, returns row * 9 + col.
    """
    if move[0] == -1 and move[1] == -1:
        return 81
    return int(move[0] * 9 + move[1])

def index_to_move(index):
    """
    Converts a flat index to a move coordinate.
    - 81 -> [-1, -1] (PASS)
    - [0, 80] -> [row, col]
    """
    if index == 81:
        return [-1, -1]
    return [index // 9, index % 9]

class GreatKingdomDataset(Dataset):
    """
    Dataset class for Great Kingdom AI self-play game data.
    Provides (state, action_index) pairs.
    """
    def __init__(self, npz_path, mode="train", split_ratio=0.9, seed=42):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found at {npz_path}")
            
        # Load the compressed dataset
        data = np.load(npz_path)
        states = data["states"]      # shape (N, 4, 9, 9)
        actions = data["actions"]    # shape (N, 2)
        
        # Convert actions to indices
        action_indices = np.array([move_to_index(act) for act in actions], dtype=np.int64)
        
        # Determine the total number of samples and split
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        # Shuffle deterministically to ensure reproducibility of split
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * split_ratio)
        
        if mode == "train":
            self.subset_indices = indices[:split_idx]
        elif mode == "val":
            self.subset_indices = indices[split_idx:]
        else:
            self.subset_indices = indices
            
        self.states = states
        self.action_indices = action_indices
        self.mode = mode
        
    def __len__(self):
        return len(self.subset_indices)
        
    def __getitem__(self, idx):
        real_idx = self.subset_indices[idx]
        state = self.states[real_idx]  # numpy array of shape (4, 9, 9)
        action_idx = self.action_indices[real_idx]
        
        # Apply random D4 symmetry transformations in training mode to prevent overfitting
        if self.subset_indices is not None and len(self.subset_indices) > 0:
            # We determine if we are in training mode based on the mode passed in constructor
            # (which we can save to self.mode)
            if hasattr(self, 'mode') and self.mode == "train":
                rot = np.random.randint(0, 4)
                flip = np.random.randint(0, 2)
                
                # Rotate state tensor channels
                state = np.rot90(state, rot, axes=(1, 2))
                
                # Rotate action coordinate
                if action_idx != 81:
                    r, c = action_idx // 9, action_idx % 9
                    r_temp, c_temp = r, c
                    for _ in range(rot):
                        r_temp, c_temp = 8 - c_temp, r_temp
                    r, c = r_temp, c_temp
                    
                    # Flip horizontally
                    if flip:
                        state = np.flip(state, axis=2)
                        c = 8 - c
                        
                    action_idx = r * 9 + c
                else:
                    if flip:
                        state = np.flip(state, axis=2)
                
                # Copy array to resolve negative strides from rot90/flip
                state = state.copy()
                
        # Convert to torch tensors
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action_idx, dtype=torch.long)
        return state_tensor, action_tensor
