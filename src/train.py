import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from dataset import BinaryDataset
from model import BinaryClassifier
from metrics import compute_metrics

def train_model(epochs=300, lr=0.01, save_dir="saved_models/"):

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")

    # Dataset + Loader
    train_ds = BinaryDataset(X_train, y_train)
    test_ds = BinaryDataset(X_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)


    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = BinaryClassifier(in_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_true = []
            
            for X_batch, y_batch in test_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                
                all_preds.append(preds)
                all_true.append(y_batch)
            all_preds = torch.cat(all_preds)
            all_true = torch.cat(all_true)
            
            metrics = compute_metrics(all_true, all_preds)
        
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Acc={metrics['accuracy']:.4f}")


    # Save model
    torch.save(model.state_dict(), f"{save_dir}/model.pth")

    print("Training complete!")
    print("Model and scaler saved!")


if __name__ == "__main__":
    train_model()
