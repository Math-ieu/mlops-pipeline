import mlflow
import mlflow.pytorch
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
import psutil
import threading
import numpy as np

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("optuna_hyperparameter_tuning")

# Initialisation pour le monitoring GPU
pynvml.nvmlInit()

class GPUMonitor:
    """Classe pour monitorer l'utilisation du GPU en temps réel"""
    def __init__(self, device_id=0, interval=1.0):
        self.device_id = device_id
        self.interval = interval
        self.running = False
        self.thread = None
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.metrics = {
            'gpu_utilization': [],
            'memory_used_mb': [],
            'memory_total_mb': None,
            'temperature': [],
            'power_watts': []
        }
    
    def _monitor_loop(self):
        """Boucle de monitoring"""
        while self.running:
            try:
                # Utilisation GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.metrics['gpu_utilization'].append(util.gpu)
                
                # Mémoire
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.metrics['memory_used_mb'].append(mem_info.used / 1024**2)
                if self.metrics['memory_total_mb'] is None:
                    self.metrics['memory_total_mb'] = mem_info.total / 1024**2
                
                # Température
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                self.metrics['temperature'].append(temp)
                
                # Puissance
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
                self.metrics['power_watts'].append(power)
                
            except Exception as e:
                print(f"Error monitoring GPU: {e}")
            
            time.sleep(self.interval)
    
    def start(self):
        """Démarre le monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Arrête le monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def get_stats(self):
        """Retourne les statistiques agrégées"""
        if not self.metrics['gpu_utilization']:
            return {}
        
        return {
            'gpu_utilization_mean': np.mean(self.metrics['gpu_utilization']),
            'gpu_utilization_max': np.max(self.metrics['gpu_utilization']),
            'memory_used_mean_mb': np.mean(self.metrics['memory_used_mb']),
            'memory_used_max_mb': np.max(self.metrics['memory_used_mb']),
            'memory_total_mb': self.metrics['memory_total_mb'],
            'memory_utilization_mean': np.mean(self.metrics['memory_used_mb']) / self.metrics['memory_total_mb'] * 100,
            'temperature_mean': np.mean(self.metrics['temperature']),
            'temperature_max': np.max(self.metrics['temperature']),
            'power_mean_watts': np.mean(self.metrics['power_watts']),
            'power_max_watts': np.max(self.metrics['power_watts'])
        }

class ConvNet(nn.Module):
    """Architecture CNN flexible"""
    def __init__(self, num_filters, num_layers, dropout_rate, num_classes=10):
        super(ConvNet, self).__init__()
        
        layers = []
        in_channels = 3
        
        for i in range(num_layers):
            out_channels = num_filters * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculer la taille après convolutions
        feature_size = out_channels * (32 // (2 ** num_layers)) ** 2
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_data_loaders(batch_size, data_dir='./data'):
    """Charge CIFAR-10"""
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

def objective(trial):
    """Fonction objectif pour Optuna"""
    
    # Définition de l'espace de recherche
    params = {
        'num_filters': trial.suggest_categorical('num_filters', [16, 32, 64]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW']),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
        'epochs': 20  # Réduit pour l'optimisation
    }
    
    # Démarrage du run MLflow
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        
        # Log des paramètres
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)
        
        # Configuration device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Démarrage du monitoring GPU
        gpu_monitor = None
        if torch.cuda.is_available():
            gpu_monitor = GPUMonitor(device_id=0, interval=0.5)
            gpu_monitor.start()
            gpu_name = torch.cuda.get_device_name(0)
            mlflow.log_param("gpu_name", gpu_name)
        
        try:
            # Chargement des données
            trainloader, testloader = get_data_loaders(params['batch_size'])
            
            # Création du modèle
            model = ConvNet(
                num_filters=params['num_filters'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate'],
                num_classes=10
            ).to(device)
            
            # Log du nombre de paramètres
            num_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("num_parameters", num_params)
            
            criterion = nn.CrossEntropyLoss()
            
            # Sélection de l'optimiseur
            if params['optimizer'] == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=params['lr'], 
                    weight_decay=params['weight_decay']
                )
            elif params['optimizer'] == 'SGD':
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=params['lr'], 
                    momentum=0.9,
                    weight_decay=params['weight_decay']
                )
            else:  # AdamW
                optimizer = optim.AdamW(
                    model.parameters(), 
                    lr=params['lr'], 
                    weight_decay=params['weight_decay']
                )
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params['epochs']
            )
            
            # Entraînement
            best_acc = 0.0
            start_time = time.time()
            
            for epoch in range(params['epochs']):
                # Training
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                
                train_loss = train_loss / len(trainloader)
                train_acc = 100. * train_correct / train_total
                
                # Evaluation
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()
                
                test_loss = test_loss / len(testloader)
                test_acc = 100. * test_correct / test_total
                
                # Update learning rate
                scheduler.step()
                
                # Log des métriques
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("test_loss", test_loss, step=epoch)
                mlflow.log_metric("test_accuracy", test_acc, step=epoch)
                mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
                
                # Sauvegarde du meilleur modèle
                if test_acc > best_acc:
                    best_acc = test_acc
                    mlflow.log_metric("best_test_accuracy", best_acc)
                
                # Pruning avec Optuna
                trial.report(test_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                print(f"Trial {trial.number} - Epoch {epoch+1}/{params['epochs']}: "
                      f"Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%")
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Arrêt du monitoring GPU et log des statistiques
            if gpu_monitor:
                gpu_monitor.stop()
                gpu_stats = gpu_monitor.get_stats()
                for key, value in gpu_stats.items():
                    mlflow.log_metric(f"gpu_{key}", value)
                
                print(f"GPU Stats - Util: {gpu_stats['gpu_utilization_mean']:.1f}%, "
                      f"Mem: {gpu_stats['memory_used_mean_mb']:.0f}MB, "
                      f"Temp: {gpu_stats['temperature_mean']:.1f}°C")
            
            # Log du modèle final
            mlflow.pytorch.log_model(model, "model")
            
            return best_acc
            
        except Exception as e:
            if gpu_monitor:
                gpu_monitor.stop()
            raise e

def run_optimization(n_trials=50, timeout=3600):
    """Lance l'optimisation avec Optuna"""
    
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA + MLFLOW")
    print("="*60)
    
    # Création de l'étude Optuna
    study = optuna.create_study(
        direction="maximize",
        study_name="cifar10_optimization",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Callback MLflow pour Optuna
    mlflc = MLflowCallback(
        tracking_uri="file:./mlruns",
        metric_name="best_test_accuracy"
    )
    
    # Lancement de l'optimisation
    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("timeout", timeout)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflc],
            show_progress_bar=True
        )
        
        # Log des meilleurs paramètres
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.2f}%")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            mlflow.log_param(f"best_{key}", value)
        
        mlflow.log_metric("best_accuracy", study.best_value)
        
        # Visualisations Optuna
        try:
            import optuna.visualization as vis
            import plotly
            
            # Importance des paramètres
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html("param_importances.html")
            mlflow.log_artifact("param_importances.html")
            
            # Historique d'optimisation
            fig_history = vis.plot_optimization_history(study)
            fig_history.write_html("optimization_history.html")
            mlflow.log_artifact("optimization_history.html")
            
            # Parallel coordinate plot
            fig_parallel = vis.plot_parallel_coordinate(study)
            fig_parallel.write_html("parallel_coordinate.html")
            mlflow.log_artifact("parallel_coordinate.html")
            
            print("\nVisualizations saved!")
            
        except ImportError:
            print("\nInstall plotly for visualizations: pip install plotly kaleido")
        
        # Sauvegarde des résultats
        results_df = study.trials_dataframe()
        results_df.to_csv("optimization_results.csv", index=False)
        mlflow.log_artifact("optimization_results.csv")
    
    return study

if __name__ == "__main__":
    # Vérification GPU
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Attention: Aucun GPU détecté!")
    
    # Lancement de l'optimisation
    # Ajustez n_trials et timeout selon vos besoins
    study = run_optimization(n_trials=30, timeout=7200)  # 2 heures max
    
    print("\n" + "="*60)
    print("Optimization completed! Check MLflow UI for detailed results.")
    print("="*60)