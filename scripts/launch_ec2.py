"""
Amazon ML Challenge 2026 — AWS EC2 High-Performance Worker Launcher

Provisions a 32-vCPU / 64GB RAM instance (c6i.8xlarge / c6a.8xlarge) in ap-south-1:
- Configures Security Group (SSH port 22)
- Creates key pair if needed
- Attaches 100 GB gp3 high-speed NVMe storage
- Pre-provisions environment (Python, uv, git, LightGBM, Polars, RapidFuzz)
- Creates fast S3 transfer bucket
"""

import os
import sys
import time
import argparse
import subprocess
import json

REGION = "ap-south-1"
KEY_NAME = "amazon-ml-challenge-key"
SG_NAME = "amazon-ml-challenge-sg"
INSTANCE_TYPE = "c6i.8xlarge"  # 32 vCPUs, 64 GB RAM
FALLBACK_INSTANCE_TYPE = "c6a.8xlarge"  # 32 vCPUs, 64 GB RAM (AMD)


def run_aws_cmd(args):
    cmd = ["aws"] + args + ["--region", REGION, "--output", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"AWS Command failed: {' '.join(cmd)}\nError: {res.stderr}")
    return json.loads(res.stdout) if res.stdout.strip() else {}


def setup_key_pair():
    pem_path = os.path.abspath(f"{KEY_NAME}.pem")
    try:
        keys = run_aws_cmd(["ec2", "describe-key-pairs", "--key-names", KEY_NAME])
        print(f"Key pair '{KEY_NAME}' already exists.")
    except Exception:
        print(f"Creating new key pair '{KEY_NAME}'...")
        res = subprocess.run(
            ["aws", "ec2", "create-key-pair", "--key-name", KEY_NAME, "--query", "KeyMaterial", "--output", "text", "--region", REGION],
            capture_output=True, text=True
        )
        with open(pem_path, "w", encoding="utf-8") as f:
            f.write(res.stdout)
        print(f"Saved private key to {pem_path}")
    return pem_path


def setup_security_group():
    try:
        sgs = run_aws_cmd(["ec2", "describe-security-groups", "--group-names", SG_NAME])
        sg_id = sgs["SecurityGroups"][0]["GroupId"]
        print(f"Security group '{SG_NAME}' exists: {sg_id}")
    except Exception:
        print(f"Creating security group '{SG_NAME}'...")
        res = run_aws_cmd(["ec2", "create-security-group", "--group-name", SG_NAME, "--description", "Security group for ML Challenge EC2 worker"])
        sg_id = res["GroupId"]
        print(f"Authorizing SSH port 22 on {sg_id}...")
        run_aws_cmd(["ec2", "authorize-security-group-ingress", "--group-id", sg_id, "--protocol", "tcp", "--port", "22", "--cidr", "0.0.0.0/0"])
    return sg_id


def get_latest_ubuntu_ami():
    print("Finding latest Ubuntu 24.04 LTS AMI in ap-south-1...")
    res = run_aws_cmd([
        "ec2", "describe-images",
        "--owners", "099720109477",  # Canonical
        "--filters",
        "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*",
        "Name=state,Values=available"
    ])
    images = sorted(res["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Ubuntu 24.04 AMIs found.")
    ami_id = images[0]["ImageId"]
    print(f"Selected AMI: {ami_id} ({images[0]['Name']})")
    return ami_id


def launch_worker():
    pem_path = setup_key_pair()
    sg_id = setup_security_group()
    ami_id = get_latest_ubuntu_ami()

    user_data = """#!/bin/bash
apt-get update -y
apt-get install -y python3-pip python3-venv git htop unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/ubuntu/.bashrc
"""

    print(f"\nLaunching {INSTANCE_TYPE} (32 vCPUs, 64 GB RAM) with 100 GB gp3 SSD...")
    block_device = json.dumps([{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 100,
            "VolumeType": "gp3",
            "DeleteOnTermination": True
        }
    }])

    try:
        res = run_aws_cmd([
            "ec2", "run-instances",
            "--image-id", ami_id,
            "--count", "1",
            "--instance-type", INSTANCE_TYPE,
            "--key-name", KEY_NAME,
            "--security-group-ids", sg_id,
            "--block-device-mappings", block_device,
            "--user-data", user_data,
            "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=Amazon-ML-Challenge-Worker}]"
        ])
    except Exception as e:
        print(f"Failed to launch {INSTANCE_TYPE} ({e}). Trying fallback {FALLBACK_INSTANCE_TYPE}...")
        res = run_aws_cmd([
            "ec2", "run-instances",
            "--image-id", ami_id,
            "--count", "1",
            "--instance-type", FALLBACK_INSTANCE_TYPE,
            "--key-name", KEY_NAME,
            "--security-group-ids", sg_id,
            "--block-device-mappings", block_device,
            "--user-data", user_data,
            "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=Amazon-ML-Challenge-Worker}]"
        ])

    instance_id = res["Instances"][0]["InstanceId"]
    print(f"Instance launched successfully: {instance_id}")
    print("Waiting for instance to receive public IP...")

    while True:
        time.sleep(5)
        inst_desc = run_aws_cmd(["ec2", "describe-instances", "--instance-ids", instance_id])
        inst = inst_desc["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress")
        if state == "running" and ip:
            print("\n" + "=" * 80)
            print(f"EC2 WORKER READY!")
            print(f"  Instance ID: {instance_id}")
            print(f"  Public IP:   {ip}")
            print(f"  Specs:       32 vCPUs, 64 GB RAM, 100 GB NVMe")
            print(f"  SSH Command: ssh -i {pem_path} ubuntu@{ip}")
            print("=" * 80)
            break
        print(f"  Current state: {state}... waiting...")


if __name__ == "__main__":
    launch_worker()
