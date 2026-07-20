import numpy as np
import sys
import os
import asyncio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for d in [CURRENT_DIR, PARENT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from manager import run_pipeline

async def test_chained_pipeline():
    print("=== Chained Pipeline Test: Bandpass Filter -> Quality Check ===")
    
    # 1. Pipeline configuration with registered nodes
    pipeline_config = {
        "nodes": [
            {
                "type": "bandpass filter", 
                "config": {"sfreq": 250.0, "l_freq": 1.0, "h_freq": 40.0}
            },
            {
                "type": "signal quality check", 
                "config": {"sfreq": 250.0}
            }
        ]
    }

    # 2. Synthetic 4-channel EEG (250 Hz, 5 sec = 1250 samples)
    raw_eeg = np.random.randn(1250, 4).astype(np.float32)
    raw_eeg[:, 0] = 0.0       # Channel 0: Flatline
    raw_eeg[:, 1] = 50000.0   # Channel 1: Railing / DC offset

    print(f"Input shape: {raw_eeg.shape}")
    print("Running sequence: [bandpass filter] ➔ [signal quality check]...\n")

    # 3. Run pipeline
    output = await run_pipeline(pipeline_config, raw_eeg)

    print("✅ Pipeline Execution Succeeded!\n")
    if isinstance(output, dict):
        if "processed_eeg" in output:
            processed = np.asarray(output['processed_eeg'])
            print(f"Filtered EEG Shape: {processed.shape}")
        if "classifier_output" in output:
            print(f"Quality Check Results:\n{output['classifier_output']}")
    else:
        print(f"Output: {output}")

if __name__ == "__main__":
    asyncio.run(test_chained_pipeline())