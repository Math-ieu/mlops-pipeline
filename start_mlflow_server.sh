#!/bin/bash

# Script pour démarrer le serveur MLflow

# Configuration
HOST="0.0.0.0"  # Accessible depuis l'extérieur
PORT="5000"
BACKEND_STORE="./mlruns"  # Stockage local des méta-données
ARTIFACT_ROOT="./mlartifacts"  # Stockage local des artefacts

echo "=================================="
echo "Démarrage du serveur MLflow"
echo "=================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Backend store: $BACKEND_STORE"
echo "Artifact root: $ARTIFACT_ROOT"
echo "=================================="
echo ""
echo "Accès: http://51.44.19.71:$PORT"
echo ""
echo "Assurez-vous que le port $PORT est ouvert dans votre Security Group AWS"
echo ""

# Création des dossiers si nécessaire
mkdir -p $BACKEND_STORE
mkdir -p $ARTIFACT_ROOT

# Démarrage du serveur
mlflow server \
    --host $HOST \
    --port $PORT \
    --backend-store-uri $BACKEND_STORE \
    --default-artifact-root $ARTIFACT_ROOT