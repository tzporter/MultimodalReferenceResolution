import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from src.models import ClassificationHead
from src.config import (
    DEVICE, REDUCED_DIMENSION, NUM_CLASSES, FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH_SIZE
)

def train_and_evaluate_classifiers(X_train_full, y_train, X_val_full, y_val):
    """
    Performs PCA, trains all classifiers, and returns a dictionary of their results.
    """
    # --- Step 1: Dimensionality Reduction (PCA) ---
    print("\n--- Applying PCA ---", flush=True)
    if np.any(np.isnan(X_train_full)) or np.any(np.isinf(X_train_full)):
        print("Warning: NaNs or Infs detected in training embeddings. Replacing with 0.")
        X_train_full = np.nan_to_num(X_train_full)
    if np.any(np.isnan(X_val_full)) or np.any(np.isinf(X_val_full)):
        print("Warning: NaNs or Infs detected in validation embeddings. Replacing with 0.")
        X_val_full = np.nan_to_num(X_val_full)

    pca = PCA(n_components=REDUCED_DIMENSION, random_state=42)
    X_train_red = pca.fit_transform(X_train_full)
    X_val_red = pca.transform(X_val_full)
    
    print(f"Reduced shape: {X_train_red.shape} | Variance Retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # --- Step 2: Prepare Fast DataLoaders for NN ---
    train_ds_opt = TensorDataset(torch.tensor(X_train_red, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds_opt = TensorDataset(torch.tensor(X_val_red, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_ds_opt, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds_opt, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # --- Step 3: Train and Evaluate Models ---
    print("\n--- Training Classifiers ---", flush=True)
    results = {}

    # 3a. Neural Network Head
    print("Training Neural Network...", flush=True)
    nn_preds, nn_train_preds = train_nn_head(train_loader, val_loader)
    results["Neural Network"] = (nn_train_preds, nn_preds)

    # 3b. SVM
    print("Training SVM...", flush=True)
    # find best penalty
    # Cs = np.logspace(-3, 3, 7)
    Cs = [1]
    best_svc = None
    best_score = 0
    for C in Cs:
        svc = SVC(kernel='rbf', gamma='scale', C=C, random_state=42)
        svc.fit(X_train_red, y_train)
        score = svc.score(X_val_red, y_val)
        if score > best_score:
            best_score = score
            best_svc = svc
    print(f"Best C: {best_svc.C}")
    results["SVM"] = (best_svc.predict(X_train_red), best_svc.predict(X_val_red))

    # 3c. Random Forest
    print("Training Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train_red, y_train)
    results["Random Forest"] = (rf.predict(X_train_red), rf.predict(X_val_red))

    return results

def train_nn_head(train_loader, val_loader):
    """Trains and evaluates the simple neural network classifier."""
    clf = ClassificationHead(REDUCED_DIMENSION, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=FINETUNE_LR)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(1, FINETUNE_EPOCHS + 1):
        clf.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(clf(bx), by)
            loss.backward()
            opt.step()
        
        if ep % 20 == 0 or ep == FINETUNE_EPOCHS:
             # Minimal logging to avoid clutter during CV
            print(f"  NN Epoch {ep}/{FINETUNE_EPOCHS}")

    # Evaluate NN
    clf.eval()
    nn_preds, nn_train_preds = [], []
    with torch.no_grad():
        for bx, _ in train_loader:
            logits = clf(bx.to(DEVICE))
            nn_train_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        for vx, _ in val_loader:
            logits = clf(vx.to(DEVICE))
            nn_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            
    return np.concatenate(nn_preds), np.concatenate(nn_train_preds)

def report_results(results, y_train, y_val):
    """Prints a summary table and detailed report for the best model."""
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS FOR THIS FOLD")
    print("="*80)
    print(f"{'Model':<20} {'Train Acc':<15} {'Val Acc':<15} {'Precision':<15} {'Recall':<15}")
    print("-"*80)
    
    fold_results = {}
    for name, (train_preds, val_preds) in results.items():
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, average='weighted', zero_division=0)
        rec = recall_score(y_val, val_preds, average='weighted', zero_division=0)
        print(f"{name:<20} {train_acc:.4f} ({train_acc*100:5.2f}%)  {val_acc:.4f} ({val_acc*100:5.2f}%)  {prec:.4f}           {rec:.4f}")
        
        fold_results[name] = {
            'train_accuracy': train_acc, 'val_accuracy': val_acc,
            'precision': prec, 'recall': rec,
            'confusion_matrix': confusion_matrix(y_val, val_preds).tolist()
        }
    return fold_results