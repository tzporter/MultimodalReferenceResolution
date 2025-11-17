import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def fix_seeds(seed=42):
    """Fixes random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BinaryDataset(Dataset):
    """
    Holds (embedding_pair, binary_label) for training or testing.
    """
    def __init__(self, X_pairs, y_labels, device='cpu'):
        """
        X_pairs: 2D numpy array of shape [num_samples, embedding_dim * 2]
        y_labels: 1D numpy array of shape [num_samples]
        """
        self.X = torch.tensor(X_pairs, dtype=torch.float32).to(device)
        self.y = torch.tensor(y_labels, dtype=torch.float32).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[100]):
        """
        input_dim: size of concatenated (embedding_1, embedding_2)
        hidden_dims: list of hidden layer sizes, e.g. [100, 50]
        """
        super(BinaryMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        
        layers.append(nn.Linear(prev_dim, 1)) # Output is a single logit
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: shape (batch_size, input_dim)
        returns: shape (batch_size, 1) (logits for binary classification)
        """
        return self.net(x)


def run_binary_classification(
    X_train, X_val, X_test, y_train, y_val, y_test,
    hidden_dims=[100],
    epochs=10,
    batch_size=32,
    lr=1e-3,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # split train into train and val

    train_dataset = BinaryDataset(X_train, y_train, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = BinaryDataset(X_val, y_val, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    test_dataset = BinaryDataset(X_test, y_test, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = BinaryMLP(input_dim, hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track best model
    best_val_loss = float('inf')
    best_model_path = 'best_binary_model.pt'

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        epoch_loss = total_loss / len(train_dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_dataset)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f} (Best model saved)")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    # Evaluation
    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).long()
            predictions.extend(preds.cpu().numpy())
            references.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(references, predictions)
    precision = precision_score(references, predictions)
    recall = recall_score(references, predictions)
    f1 = f1_score(references, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(confusion_matrix(references, predictions))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == '__main__':
    fix_seeds(seed=42)

    # --- 1. Load your data ---
    # This is a placeholder. You should load your actual data here.
    # For example, load a pandas DataFrame with your embedding pairs and labels.
    # df = pd.read_pickle('path/to/your/binary_classification_data.pkl')
    # X = np.stack(df['embedding_pairs'].values) # Assuming pairs are concatenated
    # y = df['labels'].values

    # Example placeholder data:
    df = pd.read_pickle("data/parent_gesture_embeddings.pkl")
    
    gestures_info_exploded = df.dropna(subset=['multimodal-x-skeleton-semantic', 'is_present']).sample(frac=1, random_state=42).reset_index(drop=True)
    X = gestures_info_exploded['multimodal-x-skeleton-semantic'].to_numpy()
    # iter = 0
    # for row in range(X.shape[0]):
        # if X[row].shape[0] != 256:
            # iter+=1
    # print(f"Number of incorrect embedding sizes: {iter}")
    # Filter out incorrect sizes
    # mesh = np.array([X[row].shape[0] == 256 for row in range(X.shape[0])])
    # shuffle the train mesh
    # X = X[mesh]
    train_chs = pd.read_csv("speaker_segments_train.csv")["pair_speaker"].to_numpy()
    test_chs = pd.read_csv("speaker_segments_test.csv")["pair_speaker"].to_numpy()
    test_chs, val_chs = train_test_split(test_chs, test_size=0.2, random_state=42) # 0.1667 * 0.85 = 0.14285
    X = np.stack(X, axis=0).astype(np.float32).squeeze()
    y = gestures_info_exploded['is_present'].to_numpy() # [mesh]
    # Split into train and test sets
    chapters = gestures_info_exploded['pair_speaker'].to_numpy() # [mesh]
    train_mesh = np.isin(chapters, train_chs)
    val_mesh = np.isin(chapters, val_chs)
    test_mesh = np.isin(chapters, test_chs)
    print(train_mesh)
    X_train, X_val, X_test = X[train_mesh], X[val_mesh], X[test_mesh]
    y_train, y_val, y_test = y[train_mesh], y[val_mesh], y[test_mesh]

    print(f"Data shape: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    # print(f"Labels shape: y_train: {y_train.shape}, y_test: {y_test.shape}")
    # print(gestures_info_exploded['is_present'][test_mesh].value_counts()) # [mesh]
    # num_samples = 1000
    # embedding_dim = 256 # Should match your model's output
    # X = np.random.rand(num_samples, embedding_dim * 2).astype(np.float32)
    # y = np.random.randint(0, 2, num_samples).astype(np.float32)

    # --- 2. Run classification ---
    run_binary_classification(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        hidden_dims=[50, 25, 10],
        epochs=50,
        batch_size=32,
        lr=0.001,
    )
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    test_preds = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    print("------------")
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(confusion_matrix(y_test, test_preds))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    test_preds = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    print("------------")
    print("Random Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(confusion_matrix(y_test, test_preds))

    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    test_preds = svc_model.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    print("------------")
    print("SVC Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(confusion_matrix(y_test, test_preds))