import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# Configuration MLflow
mlflow.set_tracking_uri("file:./mlruns")  # Stockage local, ou URI distant
mlflow.set_experiment("gpu_training_experiment")

# Définition d'un modèle simple
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SimpleNN, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def prepare_data(n_samples=10000, n_features=20):
    """Prépare des données synthétiques pour la démonstration"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_model(params):
    """Fonction d'entraînement avec tracking MLflow"""
    
    # Démarrage d'un run MLflow
    with mlflow.start_run(run_name=f"run_{params['hidden_sizes']}_{params['lr']}"):
        
        # Log des hyperparamètres
        mlflow.log_params(params)
        
        # Configuration du device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de: {device}")
        mlflow.log_param("device", str(device))
        
        # Préparation des données
        X_train, X_test, y_train, y_test = prepare_data(
            n_samples=params.get('n_samples', 10000),
            n_features=params.get('n_features', 20)
        )
        
        # Conversion en tenseurs PyTorch
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)
        
        # Création des DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        
        # Initialisation du modèle
        model = SimpleNN(
            input_size=params['n_features'],
            hidden_sizes=params['hidden_sizes'],
            num_classes=2
        ).to(device)
        
        # Critère et optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        
        # Entraînement
        print("Début de l'entraînement...")
        start_time = time.time()
        
        for epoch in range(params['epochs']):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            # Log des métriques
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{params['epochs']}], "
                      f"Loss: {epoch_loss:.4f}, "
                      f"Accuracy: {epoch_acc:.2f}%")
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Évaluation sur le test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = 100 * (test_predicted == y_test_t).sum().item() / y_test_t.size(0)
            test_loss = criterion(test_outputs, y_test_t).item()
        
        print(f"\nTest Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Training time: {training_time:.2f}s")
        
        # Log des métriques finales
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        
        # Sauvegarde du modèle
        mlflow.pytorch.log_model(model, "model")
        
        # Sauvegarde d'artefacts supplémentaires
        with open("model_summary.txt", "w") as f:
            f.write(f"Model Architecture:\n{model}\n")
            f.write(f"\nTest Accuracy: {test_accuracy:.2f}%\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Training Time: {training_time:.2f}s\n")
        mlflow.log_artifact("model_summary.txt")
        
        return test_accuracy

if __name__ == "__main__":
    # Configurations à tester
    configurations = [
        {
            "hidden_sizes": [128, 64],
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "n_samples": 10000,
            "n_features": 20
        },
        {
            "hidden_sizes": [256, 128, 64],
            "lr": 0.0005,
            "batch_size": 128,
            "epochs": 50,
            "n_samples": 10000,
            "n_features": 20
        },
        {
            "hidden_sizes": [512, 256],
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "n_samples": 10000,
            "n_features": 20
        }
    ]
    
    print("="*60)
    print("PIPELINE D'ENTRAÎNEMENT AVEC MLFLOW")
    print("="*60)
    
    results = []
    for i, config in enumerate(configurations, 1):
        print(f"\n{'='*60}")
        print(f"Configuration {i}/{len(configurations)}")
        print(f"{'='*60}")
        accuracy = train_model(config)
        results.append((config, accuracy))
    
    print("\n" + "="*60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    for config, acc in results:
        print(f"Hidden layers: {config['hidden_sizes']}, "
              f"LR: {config['lr']}, "
              f"Accuracy: {acc:.2f}%")