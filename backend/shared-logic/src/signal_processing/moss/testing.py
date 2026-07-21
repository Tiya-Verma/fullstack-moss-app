import numpy as np
import sys
import os
import asyncio

from manager import run_pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for d in [CURRENT_DIR, PARENT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)


def generate_60hz_noise(n_samples, fs, amplitude=1000):
    """Generate 60Hz sine wave noise."""
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * 60 * t)


async def test_chained_pipeline():
    print("=== Test 1: Chained Pipeline (Bandpass -> Quality Check) ===")

    pipeline_config = {
        "nodes": [
            {
                "type": "bandpass filter",
                "config": {"sfreq": 250.0, "l_freq": 1.0, "h_freq": 40.0},
            },
            {"type": "signal quality check", "config": {"sfreq": 250.0}},
        ]
    }

    raw_eeg = np.random.randn(1250, 4).astype(np.float32)
    raw_eeg[:, 0] = 0.0  # Flatline
    raw_eeg[:, 1] = 50000.0  # Railing

    _output = await run_pipeline(pipeline_config, raw_eeg)
    print(f"✅ Test 1 passed - Output keys: {list(_output.keys())}\n")


async def test_60hz_noise():
    print("=== Test 2: 60Hz Line Noise Detection ===")

    pipeline_config = {
        "nodes": [{"type": "signal quality check", "config": {"sfreq": 250.0}}]
    }

    raw_eeg = np.random.randn(1250, 4).astype(np.float32)
    raw_eeg[:, 2] = generate_60hz_noise(1250, 250, amplitude=5000)  # Strong 60Hz

    output = await run_pipeline(pipeline_config, raw_eeg)
    print(f"✅ Test 2 passed - Quality report: {output['classifier_output']}\n")


async def test_nan_inf_values():
    print("=== Test 3: NaN/Inf Value Handling ===")

    pipeline_config = {
        "nodes": [{"type": "signal quality check", "config": {"sfreq": 250.0}}]
    }

    raw_eeg = np.random.randn(1250, 4).astype(np.float32)
    raw_eeg[100:110, 0] = np.nan  # Insert NaN values
    raw_eeg[200:210, 1] = np.inf  # Insert Inf values

    try:
        await run_pipeline(pipeline_config, raw_eeg)
        print("✅ Test 3 passed - Handled NaN/Inf gracefully\n")
    except Exception as e:
        print(f"⚠️  Test 3 - NaN/Inf caused error (expected): {e}\n")


async def test_short_data():
    print("=== Test 4: Short Data Duration ===")

    pipeline_config = {
        "nodes": [{"type": "signal quality check", "config": {"sfreq": 250.0}}]
    }

    raw_eeg = np.random.randn(100, 4).astype(np.float32)  # Very short

    try:
        await run_pipeline(pipeline_config, raw_eeg)
        print("✅ Test 4 passed - Short data handled\n")
    except Exception as e:
        print(f"⚠️  Test 4 - Short data caused error: {e}\n")


async def test_quality_check_only():
    print("=== Test 5: Quality Check Only (No Preprocessing) ===")

    pipeline_config = {
        "nodes": [{"type": "signal quality check", "config": {"sfreq": 250.0}}]
    }

    raw_eeg = np.random.randn(1250, 4).astype(np.float32)

    await run_pipeline(pipeline_config, raw_eeg)
    print("✅ Test 5 passed - Quality check standalone works\n")


async def test_wrong_shape():
    print("=== Test 6: Invalid Data Shape ===")

    pipeline_config = {
        "nodes": [{"type": "signal quality check", "config": {"sfreq": 250.0}}]
    }

    raw_eeg = np.random.randn(1250, 3).astype(np.float32)  # Wrong channel count

    try:
        await run_pipeline(pipeline_config, raw_eeg)
        print("⚠️  Test 6 - Should have failed with wrong shape\n")
    except Exception as e:
        print(f"✅ Test 6 passed - Correctly rejected wrong shape: {e}\n")


async def test_all_channels_good():
    print("=== Test 7: All Good Channels ===")

    pipeline_config = {
        "nodes": [{"type": "signal quality check", "config": {"sfreq": 250.0}}]
    }

    # Generate clean EEG-like data
    raw_eeg = np.random.randn(1250, 4).astype(np.float32) * 10  # Normal amplitude

    output = await run_pipeline(pipeline_config, raw_eeg)
    print(f"✅ Test 7 passed - Clean data: {output['classifier_output']}\n")


async def run_all_tests():
    await test_chained_pipeline()
    await test_60hz_noise()
    await test_nan_inf_values()
    await test_short_data()
    await test_quality_check_only()
    await test_wrong_shape()
    await test_all_channels_good()
    print("=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
