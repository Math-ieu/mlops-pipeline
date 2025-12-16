import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("cifar10_cnn_experiment")

# Architecture CNN
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size, data_dir='./data'):
    """Charge CIFAR-10 avec augmentation de données"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return trainloader, testloader

def train_epoch(model, trainloader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, testloader, criterion, device):
    """Évalue le modèle sur le test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def train_cnn(params):
    """Pipeline d'entraînement complet avec MLflow"""
    
    with mlflow.start_run(run_name=f"cnn_lr{params['lr']}_bs{params['batch_size']}"):
        
        # Log des paramètres
        mlflow.log_params(params)
        
        # Configuration device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_param("device", str(device))
        
        # Chargement des données
        print("Chargement des données CIFAR-10...")
        trainloader, testloader = get_data_loaders(params['batch_size'])
        
        # Initialisation du modèle
        model = ConvNet(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Entraînement
        print(f"\nDébut de l'entraînement pour {params['epochs']} epochs...")
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(params['epochs']):
            train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, testloader, criterion, device)
            
            # Ajustement du learning rate
            scheduler.step(test_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log des métriques
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Sauvegarde du meilleur modèle
            if test_acc > best_acc:
                best_acc = test_acc
                mlflow.log_metric("best_test_accuracy", best_acc)
                torch.save(model.state_dict(), 'best_model.pth')
            
            print(f"Epoch [{epoch+1}/{params['epochs']}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
                  f"LR: {current_lr:.6f}")
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("final_best_accuracy", best_acc)
        
        # Chargement et log du meilleur modèle
        model.load_state_dict(torch.load('best_model.pth'))
        mlflow.pytorch.log_model(model, "best_model")
        mlflow.log_artifact('best_model.pth')
        
        # Sauvegarde du résumé
        with open("training_summary.txt", "w") as f:
            f.write(f"CIFAR-10 CNN Training Summary\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Model: ConvNet\n")
            f.write(f"Device: {device}\n")
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"\nHyperparameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Best Test Accuracy: {best_acc:.2f}%\n")
            f.write(f"  Training Time: {training_time:.2f}s ({training_time/60:.2f} min)\n")
            f.write(f"  Time per Epoch: {training_time/params['epochs']:.2f}s\n")
        
        mlflow.log_artifact("training_summary.txt")
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        print(f"Total Training Time: {training_time:.2f}s ({training_time/60:.2f} min)")
        print(f"{'='*60}\n")
        
        return best_acc

if __name__ == "__main__":
    
    # Configuration d'entraînement
    config = {
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "model": "ConvNet"
    }
    
    print("="*60)
    print("CIFAR-10 CNN Training Pipeline with MLflow")
    print("="*60)
    
    # Entraînement
    best_accuracy = train_cnn(config)
    
    print(f"\nFinal Best Accuracy: {best_accuracy:.2f}%")
    print("Consultez MLflow UI pour voir tous les détails!")