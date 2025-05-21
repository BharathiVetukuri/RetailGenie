#!/bin/bash

# Azure Resource Names
RESOURCE_GROUP="retailgenie-rg"
LOCATION="eastus"
WORKSPACE_NAME="retailgenie-workspace"
COMPUTE_NAME="retailgenie-compute"
ACR_NAME="retailgenieacr"

# Create Resource Group
echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo "Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic

# Create Azure ML Workspace
echo "Creating Azure ML Workspace..."
az ml workspace create --resource-group $RESOURCE_GROUP --name $WORKSPACE_NAME

# Create Compute Instance
echo "Creating Compute Instance..."
az ml compute create --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --name $COMPUTE_NAME --type ComputeInstance --size Standard_DS3_v2

# Enable admin access to ACR
echo "Enabling admin access to ACR..."
az acr update -n $ACR_NAME --admin-enabled true

# Get ACR credentials
echo "Getting ACR credentials..."
ACR_USERNAME=$(az acr credential show -n $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

echo "Azure setup completed!"
echo "ACR Username: $ACR_USERNAME"
echo "ACR Password: $ACR_PASSWORD"

# Install Azure CLI and login
az login

# Set environment variables
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export AZURE_TENANT_ID="your-tenant-id"
export SMTP_USERNAME="your-email"
export SMTP_PASSWORD="your-email-password"
export ALERT_RECIPIENTS="email1@example.com,email2@example.com"

# Run the setup script
python mlops/setup_azure_ml.py

# After logging in with 'az login', run:
az account show --query id -o tsv 