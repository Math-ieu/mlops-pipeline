"""
Exemple complet intégrant:
- Optuna pour l'hyperparameter tuning
- Monitoring GPU avancé
- MLflow Model Registry
- Best practices
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import optuna
from optuna.integration.mlflow import MLflowCallback
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import pynvml
import numpy as np
from datetime import datetime
import threading

# Configuration
mlflow.set_tracking_uri("file:./mlruns")
EXPERIMENT_NAME = "complete_production_pipeline"
MODEL_NAME = "cifar10-production-cnn"

mlflow.set_experiment(EXPERIMENT_NAME)

# GPU Monitor (version simplifiée intégrée)
class SimpleGPUMonitor:
    def __init__(self, device_id=0, interval=1.0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.interval = interval
        self.running = False
        self.metrics = {'util': [], 'mem': [], 'temp': [], 'power': []}
        
    def _monitor(self):
        while self.running:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, 0)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            
            self.metrics['util'].append(util.gpu)
            self.metrics['mem'].append(mem.used / 1024**2)
            self.metrics['temp'].append(temp)
            self.metrics['power'].append(power)
            
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
    
    def get_stats(self):
        if not self.metrics['util']:
            return {}
        return {
            'gpu_util_mean': np.mean(self.metrics['util']),
            'gpu_util_max': np.max(self.metrics['util']),
            'gpu_mem_mean_mb': np.mean(self.metrics['mem']),
            'gpu_mem_max_mb': np.max(self.metrics['mem']),
            'gpu_temp_mean': np.mean(self.metrics['temp']),
            'gpu_temp_max': np.max(self.metrics['temp']),
            'gpu_power_mean_w': np.mean(self.metrics['power']),
            'gpu_power_max_w': np.max(self.metrics['power']),
        }

class CNN(nn.Module):
    def __init__(self, num_filters, dropout):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters, num_filters*2, 3, padding=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters*2, num_filters*4, 3, padding=1),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_filters*4*4*4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, 
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return loss_sum / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return loss_sum / len(loader), 100. * correct / total

def objective(trial):
    """Fonction objectif Optuna avec monitoring GPU et MLflow"""
    
    # Hyperparamètres à optimiser
    params = {
        'num_filters': trial.suggest_categorical('num_filters', [32, 64]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.4),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
        'epochs': 25
    }
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)
        
        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU monitoring
        gpu_monitor = None
        if torch.cuda.is_available():
            gpu_monitor = SimpleGPUMonitor(device_id=0, interval=0.5)
            gpu_monitor.start()
            mlflow.log_param("gpu", torch.cuda.get_device_name(0))
        
        try:
            # Data
            train_loader, test_loader = get_dataloaders(params['batch_size'])
            
            # Model
            model = CNN(params['num_filters'], params['dropout']).to(device)
            num_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("num_parameters", num_params)
            
            criterion = nn.CrossEntropyLoss()
            if params['optimizer'] == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
            
            # Training
            best_acc = 0.0
            start_time = time.time()
            
            for epoch in range(params['epochs']):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc
                }, step=epoch)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                
                # Pruning
                trial.report(test_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_sec", training_time)
            mlflow.log_metric("best_test_acc", best_acc)
            
            # GPU stats
            if gpu_monitor:
                gpu_monitor.stop()
                stats = gpu_monitor.get_stats()
                mlflow.log_metrics(stats)
            
            return best_acc
            
        except Exception as e:
            if gpu_monitor:
                gpu_monitor.stop()
            raise e

def run_hyperparameter_search(n_trials=20):
    """Phase 1: Recherche d'hyperparamètres"""
    
    print("\n" + "="*70)
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
    print("="*70 + "\n")
    
    study = optuna.create_study(
        direction="maximize",
        study_name="cifar10_search",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    with mlflow.start_run(run_name="hyperparameter_search"):
        mlflow.log_param("n_trials", n_trials)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✓ Optimization complete!")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best accuracy: {study.best_value:.2f}%")
        print(f"\n  Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
            mlflow.log_param(f"best_{key}", value)
        
        mlflow.log_metric("best_accuracy", study.best_value)
        
        # Save Optuna visualizations
        try:
            import plotly
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html("optimization_history.html")
            mlflow.log_artifact("optimization_history.html")
            
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html("param_importances.html")
            mlflow.log_artifact("param_importances.html")
        except:
            pass
    
    return study.best_params

def train_final_model(params):
    """Phase 2: Entraînement du modèle final avec meilleurs params"""
    
    print("\n" + "="*70)
    print("PHASE 2: FINAL MODEL TRAINING")
    print("="*70 + "\n")
    
    client = MlflowClient()
    
    with mlflow.start_run(run_name=f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Extended training for final model
        params['epochs'] = 50
        mlflow.log_params(params)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")
        
        # GPU monitoring
        gpu_monitor = None
        if torch.cuda.is_available():
            gpu_monitor = SimpleGPUMonitor(device_id=0, interval=1.0)
            gpu_monitor.start()
        
        # Data
        train_loader, test_loader = get_dataloaders(params['batch_size'])
        
        # Model
        model = CNN(params['num_filters'], params['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        
        if params['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
        
        # Training
        best_acc = 0.0
        best_state = None
        start_time = time.time()
        
        print("\nTraining...")
        for epoch in range(params['epochs']):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{params['epochs']}: "
                      f"Train {train_acc:.2f}%, Test {test_acc:.2f}%, Best {best_acc:.2f}%")
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_sec", training_time)
        mlflow.log_metric("final_best_acc", best_acc)
        
        # GPU stats
        if gpu_monitor:
            gpu_monitor.stop()
            stats = gpu_monitor.get_stats()
            mlflow.log_metrics(stats)
            print(f"\nGPU Stats:")
            print(f"  Utilization: {stats['gpu_util_mean']:.1f}% (max: {stats['gpu_util_max']:.1f}%)")
            print(f"  Memory: {stats['gpu_mem_mean_mb']:.0f}MB (max: {stats['gpu_mem_max_mb']:.0f}MB)")
            print(f"  Temperature: {stats['gpu_temp_mean']:.1f}°C (max: {stats['gpu_temp_max']:.1f}°C)")
        
        # Load best model
        model.load_state_dict(best_state)
        
        # Infer signature
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        signature = infer_signature(dummy_input.cpu().numpy(), dummy_output.cpu().numpy())
        
        # Register model
        print(f"\n✓ Registering model in Model Registry...")
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            input_example=dummy_input.cpu().numpy(),
            registered_model_name=MODEL_NAME
        )
        
        run_id = mlflow.active_run().info.run_id
        
        # Get version number
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max([int(v.version) for v in versions])
        
        # Update model version metadata
        client.update_model_version(
            name=MODEL_NAME,
            version=latest_version,
            description=f"CNN trained with Optuna. Test Accuracy: {best_acc:.2f}%"
        )
        
        # Add tags
        client.set_model_version_tag(MODEL_NAME, latest_version, "accuracy", f"{best_acc:.2f}")
        client.set_model_version_tag(MODEL_NAME, latest_version, "framework", "pytorch")
        client.set_model_version_tag(MODEL_NAME, latest_version, "optimized", "true")
        
        print(f"✓ Model registered: {MODEL_NAME} v{latest_version}")
        print(f"  Accuracy: {best_acc:.2f}%")
        print(f"  Run ID: {run_id}")
        
        return MODEL_NAME, latest_version, best_acc

def promote_model_to_production(model_name, version):
    """Phase 3: Promotion du modèle en production"""
    
    print("\n" + "="*70)
    print("PHASE 3: MODEL PROMOTION TO PRODUCTION")
    print("="*70 + "\n")
    
    client = MlflowClient()
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Model {model_name} v{version} promoted to Production")
    
    # List all versions
    versions = client.search_model_versions(f"name='{model_name}'")
    print(f"\nModel Registry Status:")
    print(f"{'Version':<10} {'Stage':<15} {'Accuracy'}")
    print("-" * 40)
    
    for v in sorted(versions, key=lambda x: int(x.version)):
        run = client.get_run(v.run_id)
        acc = run.data.metrics.get('final_best_acc', 0)
        print(f"{v.version:<10} {v.current_stage:<15} {acc:.2f}%")

def complete_pipeline():
    """Pipeline complet de bout en bout"""
    
    print("\n" + "="*80)
    print(" "*25 + "COMPLETE PRODUCTION PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Optimize hyperparameters with Optuna (GPU monitored)")
    print("  2. Train final model with best hyperparameters")
    print("  3. Register model in MLflow Model Registry")
    print("  4. Promote model to Production stage")
    print("\n" + "="*80 + "\n")
    
    # Phase 1: Hyperparameter optimization
    best_params = run_hyperparameter_search(n_trials=15)
    
    # Phase 2: Train final model
    model_name, version, accuracy = train_final_model(best_params)
    
    # Phase 3: Promote to production
    promote_model_to_production(model_name, version)
    
    # Summary
    print("\n" + "="*80)
    print(" "*30 + "PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n✓ Model: {model_name} v{version}")
    print(f"✓ Accuracy: {accuracy:.2f}%")
    print(f"✓ Stage: Production")
    print(f"\n✓ Access MLflow UI: http://YOUR_IP:5000")
    print(f"✓ Load model: mlflow.pytorch.load_model('models:/{model_name}/Production')")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}\n")
    else:
        print("\n⚠ Warning: No GPU detected. Training will be slow.\n")
    
    # Run complete pipeline
    complete_pipeline()