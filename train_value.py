import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from value_model import ValueNetwork

class ValueDataset(Dataset):
    """
    Dataset class for Great Kingdom AI value network training.
    Provides (state, target) pairs.
    """
    def __init__(self, npz_path, mode="train", split_ratio=0.9, seed=42):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found at {npz_path}")
            
        data = np.load(npz_path)
        states = data["states"]      # shape (N, 4, 9, 9)
        targets = data["targets"]    # shape (N,)
        
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        # Shuffle deterministically
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
        self.targets = targets
        self.mode = mode
        
    def __len__(self):
        return len(self.subset_indices)
        
    def __getitem__(self, idx):
        real_idx = self.subset_indices[idx]
        state = self.states[real_idx]  # numpy array of shape (4, 9, 9)
        target = self.targets[real_idx]
        
        # Data augmentation (D4 symmetries) during training
        if self.mode == "train":
            rot = np.random.randint(0, 4)
            flip = np.random.randint(0, 2)
            
            # Rotate state tensor channels
            state = np.rot90(state, rot, axes=(1, 2))
            
            # Flip horizontally
            if flip:
                state = np.flip(state, axis=2)
                
            state = state.copy()
            
        state_tensor = torch.tensor(state, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return state_tensor, target_tensor

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    abs_errors = []
    sq_errors = []
    
    with torch.no_grad():
        for states, targets in val_loader:
            states = states.to(device)
            targets = targets.to(device)
            
            outputs = model(states)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * states.size(0)
            
            # Error calculations
            errors = (outputs - targets).cpu().numpy()
            abs_errors.extend(np.abs(errors))
            sq_errors.extend(errors ** 2)
            
    val_loss /= len(val_loader.dataset)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(sq_errors))
    
    return val_loss, mae, rmse

def main():
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\value_dataset.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== GREAT KINGDOM AI - VALUE NETWORK TRAINING ===")
    print(f"Device: {device}")
    
    # Datasets and Loaders
    train_dataset = ValueDataset(npz_path, mode="train", split_ratio=0.9)
    val_dataset = ValueDataset(npz_path, mode="val", split_ratio=0.9)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    model = ValueNetwork().to(device)
    criterion = nn.MSELoss() # For regression task [-1, 1]
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 20 epochs training
    max_epochs = 20
    checkpoints = [5, 10, 20]
    
    start_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for states, targets in train_loader:
            states = states.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * states.size(0)
            
        train_loss /= len(train_dataset)
        
        # Evaluate validation metrics
        val_loss, mae, rmse = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f}")
        
        # Save checkpoints at specified epochs
        if epoch in checkpoints:
            checkpoint_path = f"C:\\Users\\User\\source\\repos\\greatkingdomAI\\value_model_e{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved checkpoint to {checkpoint_path}")
            
    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
