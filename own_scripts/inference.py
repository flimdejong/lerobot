from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.record import record_loop
from lerobot.policies.factory import make_processor

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Grab the red cube and place it in the basket"
HF_MODEL_ID = "Flimdejong/smolvla_pick_place_red_cube_20000_ckpt"
# HF_DATASET_ID = "<hf_username>/<eval_dataset_repo_id>"

# Create the robot configuration
camera_config = {
    "gripper": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS),
    "top": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
}
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM1",
    id="follower_1",
    cameras=camera_config,
)

# Initialize the robot
robot = SO100Follower(robot_config)

# Initialize the policy
policy = SmolVLA.from_pretrained(HF_MODEL_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Map the stats from full keys to short names
mapped_stats = {}
for key in dataset.meta.stats:
    if key.startswith("observation.images."):
        short_key = key.replace("observation.images.", "")
        mapped_stats[short_key] = dataset.meta.stats[key]
    else:
        mapped_stats[key] = dataset.meta.stats[key]

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot
robot.connect()

# Use mapped_stats in the preprocessor
preprocessor, postprocessor = make_processor(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=mapped_stats,  # Use mapped stats here
)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")
    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )
    dataset.save_episode()

# Clean up
robot.disconnect()
dataset.push_to_hub()
