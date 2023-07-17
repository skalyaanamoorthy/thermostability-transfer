# Download the CUDA 11.4 Pin file
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin

# Move the CUDA 11.4 Pin file to the appropriate location
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download the CUDA 11.4 Debian package
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb

# Install the CUDA 11.4 Debian package
sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb

# Copy the keyring
sudo cp /var/cuda-repo-wsl-ubuntu-11-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update the system's package index
sudo apt-get update

# Install CUDA 11.4
sudo apt-get -y install cuda
