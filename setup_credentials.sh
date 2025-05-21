#!/bin/bash

echo "Setting up Azure credentials..."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Azure CLI not found. Installing..."
    # For Windows
    if [[ "$OSTYPE" == "msys" ]]; then
        winget install Microsoft.AzureCLI
    # For Linux
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    # For macOS
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install azure-cli
    fi
fi

# Login to Azure
echo "Logging in to Azure..."
az login

# Get subscription ID
echo "Getting subscription ID..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"

# Create service principal
echo "Creating service principal..."
SP_OUTPUT=$(az ad sp create-for-rbac --name "RetailGenieServicePrincipal" --role contributor --scopes /subscriptions/$SUBSCRIPTION_ID)

# Extract credentials
CLIENT_ID=$(echo $SP_OUTPUT | jq -r '.clientId')
CLIENT_SECRET=$(echo $SP_OUTPUT | jq -r '.clientSecret')
TENANT_ID=$(echo $SP_OUTPUT | jq -r '.tenantId')

echo "Service Principal created successfully!"

# Create .env file
echo "Creating .env file..."
cat > .env << EOL
AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID
AZURE_CLIENT_ID=$CLIENT_ID
AZURE_CLIENT_SECRET=$CLIENT_SECRET
AZURE_TENANT_ID=$TENANT_ID
EOL

echo "Credentials have been saved to .env file"
echo "Please add your email settings manually to the .env file:"
echo "SMTP_USERNAME=your.email@gmail.com"
echo "SMTP_PASSWORD=your-app-password"
echo "ALERT_RECIPIENTS=your.email@gmail.com"

echo "Setup completed!" 