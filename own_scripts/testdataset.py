#!/usr/bin/env python3

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def test_samples(repo_id, max_samples=None):
    """Test samples to find where corruption occurs."""
    
    print(f"Testing dataset: {repo_id}")
    
    dataset = LeRobotDataset(
        repo_id=repo_id,
        video_backend="pyav"
    )
    
    total_samples = len(dataset)
    test_samples = total_samples if max_samples is None else min(max_samples, total_samples)
    
    print(f"Dataset has {total_samples} total samples")
    print(f"Testing all {test_samples} samples...")
    
    corrupted_samples = []
    
    for i in range(test_samples):
        try:
            sample = dataset[i]
            if i % 100 == 0:  # Progress update every 100 samples
                print(f"✓ Sample {i}: OK")
            
        except Exception as e:
            print(f"✗ Sample {i}: FAILED - {type(e).__name__}: {e}")
            corrupted_samples.append(i)
            
            # Try to get more info about this sample
            try:
                if hasattr(dataset, 'hf_dataset'):
                    row = dataset.hf_dataset[i]
                    print(f"  Sample {i} metadata: episode_index={row.get('episode_index', 'N/A')}, "
                          f"frame_index={row.get('frame_index', 'N/A')}")
            except:
                pass
    
    print(f"\nTesting complete!")
    print(f"Total corrupted samples: {len(corrupted_samples)}")
    if corrupted_samples:
        print(f"Corrupted sample indices: {corrupted_samples}")
        print(f"First corrupted sample: {corrupted_samples[0]}")
        return corrupted_samples[0]
    else:
        print("No corrupted samples found!")
        return None

def check_episodes(repo_id):
    """Check which episodes exist and their basic info."""
    
    dataset = LeRobotDataset(
        repo_id=repo_id,
        video_backend="pyav"
    )
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")
    
    # Try to get episode info
    try:
        if hasattr(dataset, 'hf_dataset'):
            episodes = set()
            for i in range(min(10, len(dataset))):  # Check first 10 samples
                row = dataset.hf_dataset[i]
                episodes.add(row.get('episode_index', 'unknown'))
            print(f"  Episode indices found: {sorted(episodes)}")
    except Exception as e:
        print(f"  Could not get episode info: {e}")

if __name__ == "__main__":
    repo_id = "Flimdejong/touch_black_chip"
    
    # First check basic dataset info
    check_episodes(repo_id)
    
    print("\n" + "="*50)
    
    # Then test all samples
    first_failed = test_samples(repo_id)  # This will now test ALL samples