#!/bin/bash
# mount_block_storage.sh - Mount block storage from main node to GPU node

# Exit script if any command fails
set -e

# Default values
MAIN_NODE_IP=${MAIN_NODE_IP:-"129.114.25.37"}
BLOCK_PATH=${BLOCK_PATH:-"/mnt/block"}
SSH_KEY=${SSH_KEY:-"~/.ssh/project_key"}
SSH_USER=${SSH_USER:-"cc"}
MOUNT_POINT=${MOUNT_POINT:-"/mnt/block"}

# Print usage information
function show_usage {
    echo "Usage: $0 [OPTIONS]"
    echo "Mount block storage from main node to GPU node"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -i, --ip IP               Main node IP address (default: $MAIN_NODE_IP)"
    echo "  -p, --path BLOCK_PATH     Path to block storage on main node (default: $BLOCK_PATH)"
    echo "  -k, --key SSH_KEY         SSH key path (default: $SSH_KEY)"
    echo "  -u, --user SSH_USER       SSH user (default: $SSH_USER)"
    echo "  -m, --mount MOUNT_POINT   Local mount point (default: $MOUNT_POINT)"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -i|--ip)
            MAIN_NODE_IP="$2"
            shift 2
            ;;
        -p|--path)
            BLOCK_PATH="$2"
            shift 2
            ;;
        -k|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        -u|--user)
            SSH_USER="$2"
            shift 2
            ;;
        -m|--mount)
            MOUNT_POINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Display configuration
echo "Mounting block storage with the following configuration:"
echo "  Main Node IP:      $MAIN_NODE_IP"
echo "  Block Storage Path: $BLOCK_PATH"
echo "  SSH Key:           $SSH_KEY"
echo "  SSH User:          $SSH_USER"
echo "  Local Mount Point: $MOUNT_POINT"
echo ""

# Check if sshfs is installed
if ! command -v sshfs &> /dev/null; then
    echo "sshfs is not installed. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y sshfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y fuse-sshfs
    else
        echo "Error: Could not install sshfs. Please install it manually."
        exit 1
    fi
fi

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point directory: $MOUNT_POINT"
    sudo mkdir -p "$MOUNT_POINT"
fi

# Check if already mounted
if mount | grep -q "$MOUNT_POINT"; then
    echo "Block storage is already mounted at $MOUNT_POINT"
    echo "To unmount, run: sudo umount $MOUNT_POINT"
else
    # Mount the block storage using sshfs
    echo "Mounting block storage from $MAIN_NODE_IP:$BLOCK_PATH to $MOUNT_POINT..."
    sudo sshfs -o allow_other,IdentityFile="$SSH_KEY" "${SSH_USER}@${MAIN_NODE_IP}:${BLOCK_PATH}" "$MOUNT_POINT"
    
    if [ $? -eq 0 ]; then
        echo "Block storage mounted successfully!"
        echo "To unmount, run: sudo umount $MOUNT_POINT"
        
        # Set environment variable
        echo "export BLOCK_STORAGE_MOUNT=$MOUNT_POINT" >> ~/.bashrc
        echo "export BLOCK_STORAGE_MOUNT=$MOUNT_POINT" >> ~/.profile
        echo ""
        echo "Added BLOCK_STORAGE_MOUNT=$MOUNT_POINT to environment variables"
        echo "To use it in current shell, run: export BLOCK_STORAGE_MOUNT=$MOUNT_POINT"
    else
        echo "Failed to mount block storage."
        exit 1
    fi
fi

# Check disk space
echo ""
echo "Checking disk space on mounted block storage:"
df -h "$MOUNT_POINT"

# Test read/write
echo ""
echo "Testing read/write access to mounted block storage..."
TEST_FILE="$MOUNT_POINT/.mount_test_$(date +%s)"
if sudo touch "$TEST_FILE" && sudo rm "$TEST_FILE"; then
    echo "Read/write access confirmed."
else
    echo "Warning: Could not write to mounted block storage. Check permissions."
fi

echo ""
echo "Block storage setup complete!" 