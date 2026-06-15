import os
import numpy as np
import torch
from value_model import ValueNetwork
from train_value import ValueDataset

def main():
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\value_dataset.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== GREAT KINGDOM AI - VALUE NETWORK VERIFICATION ===")
    
    # We will use the validation dataset to ensure generalizability on unseen positions
    val_dataset = ValueDataset(npz_path, mode="val", split_ratio=0.9)
    print(f"Loaded {len(val_dataset)} validation samples.")
    
    # Checkpoints to test
    epochs = [5, 10, 20]
    
    for ep in epochs:
        model_path = r"C:\Users\User\source\repos\greatkingdomAI\value_model_e{ep}.pt".format(ep=ep)
        if not os.path.exists(model_path):
            print(f"Checkpoint {model_path} not found, skipping...")
            continue
            
        print(f"\nEvaluating Checkpoint: epoch {ep}")
        model = ValueNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # We will collect predictions, targets, and stage classifications for all validation samples
        all_preds = []
        all_targets = []
        all_stages = [] # 'early', 'mid', 'late'
        all_stone_counts = []
        
        # Load all validation samples
        loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        with torch.no_grad():
            for states, targets in loader:
                states = states.to(device)
                outputs = model(states)
                
                preds = outputs.cpu().numpy()
                targs = targets.numpy()
                
                all_preds.extend(preds)
                all_targets.extend(targs)
                
                # Classify stage based on stone count
                # state channel 0: current player, channel 1: opponent
                # shape: (B, 4, 9, 9)
                states_np = states.cpu().numpy()
                for i in range(len(states_np)):
                    # sum of stones on board
                    stones = np.sum(states_np[i, 0] == 1.0) + np.sum(states_np[i, 1] == 1.0)
                    all_stone_counts.append(stones)
                    
                    if stones < 10:
                        all_stages.append('early')
                    elif stones < 25:
                        all_stages.append('mid')
                    else:
                        all_stages.append('late')
                        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_stages = np.array(all_stages)
        all_stone_counts = np.array(all_stone_counts)
        
        # Calculate accuracy: sign(pred) == sign(target)
        # since target is +1 or -1, correctness is whether pred has the correct sign
        correct = (all_preds * all_targets) > 0
        overall_acc = np.mean(correct) * 100
        overall_mae = np.mean(np.abs(all_preds - all_targets))
        overall_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        
        print(f"Overall Accuracy (Sign Match): {overall_acc:.2f}% | MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f}")
        
        # Breakdown by stages
        stages_list = ['early', 'mid', 'late']
        for stage in stages_list:
            mask = (all_stages == stage)
            stage_count = np.sum(mask)
            if stage_count == 0:
                print(f"  Stage {stage.upper()}: No samples.")
                continue
                
            stage_acc = np.mean(correct[mask]) * 100
            stage_mae = np.mean(np.abs(all_preds[mask] - all_targets[mask]))
            stage_rmse = np.sqrt(np.mean((all_preds[mask] - all_targets[mask]) ** 2))
            avg_stones = np.mean(all_stone_counts[mask])
            
            print(f"  Stage {stage.upper():5s} (Stones ~{avg_stones:.1f}, count={stage_count:5d}): Acc {stage_acc:5.2f}% | MAE {stage_mae:.4f} | RMSE {stage_rmse:.4f}")
            
        # Draw 100 random samples evaluation for Task 4
        print(f"\n--- Extracting 100 Random Samples detailed evaluation ---")
        np.random.seed(42)
        sample_indices = np.random.choice(len(all_preds), 100, replace=False)
        
        correct_samples = 0
        stages_samples = {'early': [0, 0], 'mid': [0, 0], 'late': [0, 0]} # [correct, total]
        
        print(f"Index | Stones | Stage | Actual Winner | Prediction Value | Sign Match")
        print("-" * 70)
        
        for idx in sample_indices:
            pred_val = all_preds[idx]
            target_val = all_targets[idx]
            stone_cnt = all_stone_counts[idx]
            stage = all_stages[idx]
            
            sign_match = "PASS" if (pred_val * target_val) > 0 else "FAIL"
            if sign_match == "PASS":
                correct_samples += 1
                stages_samples[stage][0] += 1
            stages_samples[stage][1] += 1
            
            actual_str = "CURRENT" if target_val == 1.0 else "OPPONENT"
            
            # Print first 10 for log brevity
            if len(sample_indices) <= 10 or idx in sample_indices[:10]:
                print(f"{idx:5d} | {stone_cnt:6d} | {stage:5s} | {actual_str:13s} | {pred_val:16.4f} | {sign_match}")
                
        print(f"... (showing 10/100 samples) ...")
        print("-" * 70)
        print(f"Sample Accuracy: {correct_samples}/100 ({correct_samples:.1f}%)")
        for stage, (corr, tot) in stages_samples.items():
            stage_acc = (corr / tot) * 100 if tot > 0 else 0.0
            print(f"  {stage.upper():5s} Sample Accuracy: {corr}/{tot} ({stage_acc:.1f}%)")

if __name__ == "__main__":
    main()
