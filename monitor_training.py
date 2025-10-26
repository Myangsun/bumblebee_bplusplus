#!/usr/bin/env python3
import subprocess
import time
import re

print("🔄 Training Monitor (updates every 10 seconds)\n")

while True:
    # Get the process output
    try:
        result = subprocess.run(
            ['ps aux | grep "pipeline_train_baseline.py" | grep -v grep'],
            shell=True, capture_output=True, text=True
        )
        
        if not result.stdout:
            print("✅ Training completed!")
            break
            
        # Check model directory for latest checkpoint
        result = subprocess.run(
            ['ls -lht RESULTS/baseline_gbif/*.pt 2>/dev/null | head -1'],
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout:
            print(f"Latest checkpoint: {result.stdout.strip()}")
        
        # Check for epoch completion
        result = subprocess.run(
            ['tail -5 RESULTS/baseline_gbif/* 2>/dev/null | grep -i "epoch\\|validation"'],
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout:
            print("Latest updates:")
            print(result.stdout)
        
        print(f"Last check: {time.strftime('%H:%M:%S')}\n")
        time.sleep(10)
        
    except Exception as e:
        print(f"Monitor error: {e}")
        time.sleep(10)
