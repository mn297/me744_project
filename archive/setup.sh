#!/bin/bash

# Cross-platform setup script for Linux and Mac

# Detect the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux - using wget"
    # Use wget on Linux
    wget -O pointvessel_data.zip "https://nextcloud.in.tum.de/index.php/s/7ooyYxoP6HyPXQK/download?path=/&files=pointvessel_data.zip&downloadStartSecret=i46y2qmnsg"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - using curl"
    # Use curl on Mac (pre-installed)
    curl -L -o pointvessel_data.zip "https://nextcloud.in.tum.de/index.php/s/7ooyYxoP6HyPXQK/download?path=/&files=pointvessel_data.zip&downloadStartSecret=i46y2qmnsg"
else
    echo "Unsupported operating system: $OSTYPE"
    echo "This script supports Linux and macOS only."
    exit 1
fi

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download pointvessel_data.zip"
    exit 1
fi

echo "Download completed successfully"

# Extract the zip file (unzip is available on both platforms)
echo "Extracting pointvessel_data.zip..."
unzip pointvessel_data.zip -d pointvessel_data

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract pointvessel_data.zip"
    exit 1
fi

echo "Setup completed successfully!"