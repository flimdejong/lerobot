#!/usr/bin/env python3

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def create_clean_dataset(source_repo, target_repo, exclude_episodes):
    """Create a new dataset excluding corrupted episodes."""
    
    print(f"Loading source dataset: {source_repo}")
    source_dataset = LeRobotDataset(
        repo_id=source_repo,
        video_backend="pyav"
    )
    
    print(f"Creating clean dataset: {target_repo}")
    print(f"Excluding episodes: {exclude_episodes}")
    
    # Create new dataset
    clean_dataset = LeRobotDataset.create(
        repo_id=target_repo,
        fps=source_dataset.fps,
        features=source_dataset.features,
        use_videos=True
    )
    
    current_episode = None
    samples_copied = 0
    episodes_copied = 0
    
    for i in range(len(source_dataset)):
        try:
            # Get episode info
            if hasattr(source_dataset, 'hf_dataset'):
                row = source_dataset.hf_dataset[i]
                episode_idx = row.get('episode_index')
            else:
                # Fallback: try to get sample to extract episode info
                sample = source_dataset[i]
                episode_idx = sample.get('episode_index', torch.tensor(-1)).item()
            
            # Skip corrupted episodes
            if episode_idx in exclude_episodes:
                continue
            
            # Load the sample (this will fail if corrupted)
            sample = source_dataset[i]
            
            # Start new episode if needed
            if current_episode != episode_idx:
                if current_episode is not None:
                    clean_dataset.save_episode()
                    print(f"Saved episode {current_episode}")
                    episodes_copied += 1
                current_episode = episode_idx
            
            # Prepare frame data
            frame = {'task': sample['task']}
            for key, value in sample.items():
                if key not in ['episode_index', 'frame_index', 'index', 'task_index', 'task', 'dataset_index', 'timestamp']:
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                        # Fix image format: convert (C,H,W) to (H,W,C)
                        if 'image' in key and len(value.shape) == 3 and value.shape[0] == 3:
                            value = value.transpose(1, 2, 0)
                    frame[key] = value
            
            clean_dataset.add_frame(frame)
            samples_copied += 1
            
            if samples_copied % 100 == 0:
                print(f"Copied {samples_copied} samples...")
                
        except Exception as e:
            print(f"Skipping corrupted sample {i}: {e}")
            continue
    
    # Save final episode
    if current_episode is not None:
        clean_dataset.save_episode()
        print(f"Saved final episode {current_episode}")
        episodes_copied += 1
    
    print(f"\nCleaning complete!")
    print(f"Samples copied: {samples_copied}")
    print(f"Episodes copied: {episodes_copied}")
    
    print("Uploading clean dataset...")
    clean_dataset.push_to_hub()
    print(f"Clean dataset uploaded: {target_repo}")

if __name__ == "__main__":
    source_repo = "Flimdejong/touch_black_chip"
    target_repo = "Flimdejong/touch_black_chip_clean"
    exclude_episodes = [49]
    
    create_clean_dataset(source_repo, target_repo, exclude_episodes)