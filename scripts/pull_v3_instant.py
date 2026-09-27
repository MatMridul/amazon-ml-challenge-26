import subprocess
import time
import os

print("Instant V3 watcher started. Monitoring EC2...")
while True:
    cmd = 'ssh -o StrictHostKeyChecking=no -i amazon-ml-challenge-key.pem ubuntu@16.4.44.107 "ls -lh ~/amazon-ml-challenge-26/output/matching_results_v3_stricter.tsv 2>/dev/null"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if "matching_results_v3_stricter.tsv" in res.stdout:
        print("V3 file detected on EC2! Downloading immediately...")
        scp_cmd = "scp -o StrictHostKeyChecking=no -i amazon-ml-challenge-key.pem ubuntu@16.4.44.107:~/amazon-ml-challenge-26/output/matching_results_v3_stricter.tsv output/matching_results.tsv"
        subprocess.run(scp_cmd, shell=True, check=True)
        print("SUCCESS: V3 has been placed at output/matching_results.tsv!")
        break
    time.sleep(2)
