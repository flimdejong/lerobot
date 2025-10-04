#!/usr/bin/env python3

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset, LeRobotDataset

# Dataset settings
DATASET_1 = "Flimdejong/left_1"
DATASET_2 = "Flimdejong/right_1" 
MERGED_NAME = "Flimdejong/merged_left_right"

def main():
    print("Step 1: Creating virtual merge...")
    virtual_merge = MultiLeRobotDataset(
        repo_ids=[DATASET_1, DATASET_2],
        video_backend="pyav"
    )
    
    print(f"Virtual merge: {len(virtual_merge)} frames, {virtual_merge.num_episodes} episodes")
    
    print("Step 2: Getting dataset info...")
    ds1 = LeRobotDataset(DATASET_1, video_backend="pyav")
    
    print("Step 3: Creating new physical dataset...")
    new_dataset = LeRobotDataset.create(
        repo_id=MERGED_NAME,
        fps=ds1.fps,
        features=ds1.features,
        use_videos=True
    )
    
    print("Step 4: Copying data...")
    current_episode = None
    
    for i, sample in enumerate(virtual_merge):
        episode_idx = sample['episode_index'].item()
        
        # Start new episode if needed
        if current_episode != episode_idx:
            if current_episode is not None:
                new_dataset.save_episode()
                print(f"Saved episode {current_episode}")
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
        
        new_dataset.add_frame(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(virtual_merge)} frames")
    
    # Save final episode
    new_dataset.save_episode()
    print(f"Saved final episode {current_episode}")
    
    print("Step 5: Uploading to hub...")
    new_dataset.push_to_hub()
    print(f"Success! Merged dataset uploaded: {MERGED_NAME}")

if __name__ == "__main__":
    main()