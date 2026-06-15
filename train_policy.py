import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GreatKingdomDataset, move_to_index, index_to_move
from model import PolicyNetwork

def get_topk_correct(outputs, targets, k):
    """
    Computes the number of correct predictions in the top-k choices.
    """
    _, topk_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.any(dim=1).sum().item()

def index_to_coord_str(idx):
    if idx == 81:
        return "PASS"
    return f"({idx // 9}, {idx % 9})"

def main():
    print("=== GREAT KINGDOM AI - POLICY NETWORK V1 TRAINING ===")
    
    # Set paths
    npz_path = r"C:\Users\User\source\repos\greatkingdomAI\data\selfplay_dataset_v1.npz"
    model_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_model_v1.pth"
    report_path = r"C:\Users\User\source\repos\greatkingdomAI\policy_training_report.md"
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Datasets
    print("Loading datasets (90% Train / 10% Validation split)...")
    train_dataset = GreatKingdomDataset(npz_path, mode="train", split_ratio=0.9)
    val_dataset = GreatKingdomDataset(npz_path, mode="val", split_ratio=0.9)
    
    print(f"Train samples     : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Dataloaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model Setup
    model = PolicyNetwork().to(device)
    
    # Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.15)
    
    # Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    # Training Loop
    epochs = 20
    history = []
    best_val_acc1 = 0.0
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
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
            scheduler.step()
            
            train_loss += loss.item() * states.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct1 = 0
        val_correct3 = 0
        val_correct5 = 0
        
        with torch.no_grad():
            for states, targets in val_loader:
                states = states.to(device)
                targets = targets.to(device)
                
                outputs = model(states)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * states.size(0)
                
                val_correct1 += get_topk_correct(outputs, targets, 1)
                val_correct3 += get_topk_correct(outputs, targets, 3)
                val_correct5 += get_topk_correct(outputs, targets, 5)
                
        val_loss /= len(val_dataset)
        val_acc1 = (val_correct1 / len(val_dataset)) * 100
        val_acc3 = (val_correct3 / len(val_dataset)) * 100
        val_acc5 = (val_correct5 / len(val_dataset)) * 100
        
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Top1: {val_acc1:.2f}% | Top3: {val_acc3:.2f}% | Top5: {val_acc5:.2f}% | Time: {epoch_elapsed:.1f}s")
        
        # Checkpoint the best model
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            torch.save(model.state_dict(), model_path)
            print(f"  => New best Val Top-1 Accuracy: {val_acc1:.2f}%. Checkpoint saved.")
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "top1": val_acc1,
            "top3": val_acc3,
            "top5": val_acc5
        })
        
    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time:.1f} seconds ({total_train_time/60:.2f} minutes).")
    
    # Load the best model for final evaluation
    print(f"Loading best model weights from {model_path} (Best Val Top-1: {best_val_acc1:.2f}%) for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    
    # Model statistics
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Final Evaluation & Analysis for the Report
    print("Performing final evaluation and confusion analysis...")
    model.eval()
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for states, targets in val_loader:
            states = states.to(device)
            outputs = model(states)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_targets.extend(targets.numpy())
            all_preds.extend(preds)
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    # Confusion analysis
    errors = {}
    for t, p in zip(all_targets, all_preds):
        if t != p:
            errors[(t, p)] = errors.get((t, p), 0) + 1
            
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    
    # Find 3 specific prediction examples based on game stage
    # 1. Early game (0-2 stones)
    # 2. Mid game (10-20 stones)
    # 3. Late game (>=25 stones)
    example_indices = []
    for idx in range(len(val_dataset)):
        state, target = val_dataset[idx]
        num_stones = torch.sum(state[0]) + torch.sum(state[1])
        
        if len(example_indices) == 0 and num_stones <= 2:
            example_indices.append(idx)
        elif len(example_indices) == 1 and 10 <= num_stones <= 18:
            example_indices.append(idx)
        elif len(example_indices) == 2 and num_stones >= 25:
            example_indices.append(idx)
            
        if len(example_indices) == 3:
            break
            
    # Fallback to random if not found
    while len(example_indices) < 3:
        example_indices.append(len(example_indices))
        
    examples_data = []
    for i, val_idx in enumerate(example_indices):
        state, target = val_dataset[val_idx]
        state_batch = state.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(state_batch)
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]
        
        my_stones = int(torch.sum(state[0]).item())
        opp_stones = int(torch.sum(state[1]).item())
        
        preds_str = []
        for idx, pb in zip(top5_idx, top5_probs):
            preds_str.append(f"`{index_to_coord_str(idx)}` ({pb*100:.2f}%)")
            
        examples_data.append({
            "stage": "Early Game (0-2 stones)" if i == 0 else "Mid Game (10-20 stones)" if i == 1 else "Late Game (>=25 stones)",
            "my_stones": my_stones,
            "opp_stones": opp_stones,
            "actual": index_to_coord_str(target.item()),
            "top5": ", ".join(preds_str)
        })
        
    # Generate report markdown
    md = []
    md.append("# Great Kingdom AI - Policy Network v1 Training Report\n")
    md.append(f"* **측정 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"* **훈련 디바이스**: {device}")
    md.append(f"* **총 훈련 시간**: {total_train_time:.1f}초 ({total_train_time/60:.2f}분)\n")
    
    # 1. Final Accuracy (from the best checkpoint)
    best_epoch_idx = np.argmax([row["top1"] for row in history])
    best_epoch_data = history[best_epoch_idx]
    md.append("## 1. 최종 검증 정확도 (Final Validation Accuracy - Best Checkpoint)")
    md.append(f"* **최적 에포크 (Best Epoch)**: {best_epoch_data['epoch']} Epoch")
    md.append("| Metric | Result | Target (Success Criteria) | Status |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Top-1 Accuracy** | **{best_epoch_data['top1']:.2f}%** | 35.0% 이상 | {'PASS' if best_epoch_data['top1'] >= 35.0 else 'FAIL'} |")
    md.append(f"| **Top-3 Accuracy** | **{best_epoch_data['top3']:.2f}%** | 70.0% 이상 | {'PASS' if best_epoch_data['top3'] >= 70.0 else 'FAIL'} |")
    md.append(f"| **Top-5 Accuracy** | **{best_epoch_data['top5']:.2f}%** | 85.0% 이상 | {'PASS' if best_epoch_data['top5'] >= 85.0 else 'FAIL'} |")
    md.append("")
    
    # Training Curve Table
    md.append("### Epoch별 학습 추이")
    md.append("| Epoch | Train Loss | Val Loss | Top-1 Acc | Top-3 Acc | Top-5 Acc |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in history:
        md.append(f"| {row['epoch']:02d} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['top1']:.2f}% | {row['top3']:.2f}% | {row['top5']:.2f}% |")
    md.append("\n")
    
    # 2. Confusion Analysis
    md.append("## 2. 오차 분석 (Confusion Analysis)")
    md.append("모델이 실제 Depth-2 AI의 착수와 다르게 예측한 오답 사례 중 가장 빈번하게 발생한 탑 5 조합입니다:\n")
    md.append("| 순위 | 실제 착수 (Target) | 모델 예측 (Predicted) | 발생 횟수 |")
    md.append("| :---: | :---: | :---: | :---: |")
    for rank, ((tgt, prd), count) in enumerate(sorted_errors[:5], 1):
        md.append(f"| {rank} | `{index_to_coord_str(tgt)}` | `{index_to_coord_str(prd)}` | {count}회 |")
    md.append("\n")
    
    # 3. Prediction Examples
    md.append("## 3. 착수 예측 예시 (Prediction Examples)")
    md.append("| 대국 단계 | 아군 돌 | 적군 돌 | 실제 착수 (Target) | 모델 예측 Top-5 (probability) |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    for ex in examples_data:
        md.append(f"| {ex['stage']} | {ex['my_stones']} | {ex['opp_stones']} | `{ex['actual']}` | {ex['top5']} |")
    md.append("\n")
    
    # 4. Model size
    md.append("## 4. 모델 스펙 (Model Specifications)")
    md.append(f"* **총 파라미터 수**: {num_params:,} 개 (약 {num_params/1000000:.2f}M)")
    md.append(f"* **모델 저장 파일 용량**: {model_size_mb:.2f} MB")
    md.append(f"* **저장 경로**: `policy_model_v1.pth`\n")
    
    # 5. Conclusion
    overall_pass = best_epoch_data['top1'] >= 35.0 and best_epoch_data['top3'] >= 70.0 and best_epoch_data['top5'] >= 85.0
    md.append("## 5. 최종 판정 (Overall Judgment)")
    md.append(f"### 최종 결과: **{'PASS' if overall_pass else 'FAIL'}**")
    if overall_pass:
        md.append("- 신경망이 기존 Depth-2 AI의 휴리스틱 미니맥스 정책을 성공적으로 복제하였으며, 학습 파이프라인의 유효성이 검증되었습니다.")
    else:
        md.append("- 일부 정확도 기준을 만족하지 못해 검증 실패하였습니다. 하이퍼파라미터 튜닝 또는 추가 학습이 필요합니다.")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Policy Network training report generated successfully at: {report_path}")

if __name__ == "__main__":
    main()
