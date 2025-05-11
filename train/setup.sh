#!/bin/bash

# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Enable user_allow_other in fuse.conf
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Create rclone config directory
mkdir -p ~/.config/rclone

# Copy rclone config file
sudo cp rclone.conf ~/.config/rclone/

# Create and set permissions for mount point
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object

# Mount the remote storage
rclone mount chi_tacc:mlops_project9_persistant /mnt/object --read-only --allow-other --daemon

# Test MinIO connection
echo "Testing MinIO connection..."
python3 - <<EOF
import boto3
from botocore.client import Config
import sys

try:
    # Create S3 client
    s3 = boto3.client('s3',
        endpoint_url='http://129.114.25.37:9000',
        aws_access_key_id='your-acccess-key',
        aws_secret_access_key='hrwbqzUS85G253yKi43T',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    # Try to list buckets
    response = s3.list_buckets()
    print("Successfully connected to MinIO!")
    print("Available buckets:", [bucket['Name'] for bucket in response['Buckets']])
    
except Exception as e:
    print("Error connecting to MinIO:", str(e))
    sys.exit(1)
EOF

# Check if MinIO test was successful
if [ $? -eq 0 ]; then
    echo "MinIO connection test passed!"
else
    echo "MinIO connection test failed!"
    exit 1
fi


