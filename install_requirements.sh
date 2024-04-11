#!/bin/bash

echo "Checking if Python3 version 3.6-3.9.16 is installed..."

# Function to compare versions
version_lte() {
    [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

# Function to check and perform operations based on Python version
check_and_install() {
    local py_executable=$1
    # Extracting the Python version
    local version=$($py_executable -c 'import platform; print(platform.python_version())')
    
    # Define the version limits
    local min_version="3.6"
    local max_version="3.9.16"
    
    # Check if the Python version is within the specified range
    if version_lte $min_version $version && version_lte $version $max_version; then
        echo "Version of $py_executable is within the range ($min_version to $max_version). Proceeding with pip, wheel, and setuptools installation/upgrade."
        
        # Upgrade pip
        $py_executable -m pip install --upgrade pip
        
        if grep -s -q "Raspberry Pi" /sys/firmware/devicetree/base/model; then
            echo "Raspberry Pi detected. Installing Google Coral TPU dependencies."
            echo "Installing wheel and setuptools for Raspberry Pi."
            $py_executable -m pip install wheel==0.42.0 setuptools==58.0.0
            $py_executable -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
            $py_executable -m pip install svgwrite
            $py_executable -m pip install -r requirements.txt
        fi
    else
        echo "Version of $py_executable ($version) is outside the range. Skipping."
    fi
}

# Attempt to use python3 as the primary executable
if command -v python3 &>/dev/null; then
    check_and_install python3
else
    echo "Python3 command is not available."
fi
