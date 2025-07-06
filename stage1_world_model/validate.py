import torch
import numpy as np
import os
from src.components.world_model import WorldModel
import yaml
import imageio

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(checkpoint_path):
    # --- Load Configurations ---
    global_config = load_config('configs/global.yaml')
    wm_config = load_config('configs/world_model.yaml')
    
    # --- Device Setup ---
    device = torch.device(global_config['device'])

    # --- World Model Initialization ---
    world_model = WorldModel(
        observation_shape=tuple(global_config['observation_shape']),
        action_dim=global_config['action_dim'],
        latent_dim=wm_config['latent_dim'],
        reward_dim=global_config['reward_dim'],
        xlstm_config=wm_config['xlstm_config']
    ).to(device)

    # --- Load Checkpoint ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # --- Generate Dream Sequence ---
    dream_length = 100
    batch_size = 1 # Generate one dream at a time
    
    # Start with a random latent state
    previous_latent_state = torch.randn(batch_size, wm_config['latent_dim']).to(device)
    hidden_state = None
    
    dream_frames = []

    with torch.no_grad():
        for _ in range(dream_length):
            # Use a zero action for dreaming (or could be a learned action policy)
            action = torch.zeros(batch_size, global_config['action_dim']).to(device)

            # The "observation" input to the forward pass is not used for dreaming,
            # as we are generating everything from the dynamics model.
            # We only need its shape for the function signature.
            dummy_obs = torch.zeros(batch_size, *global_config['observation_shape']).to(device)

            (
                pred_latent_mean, _, _, recon_obs,
                _, _, _, hidden_state
            ) = world_model(
                observation=dummy_obs, # This input is ignored in the dream loop
                action=action,
                previous_latent_state=previous_latent_state,
                hidden_state=hidden_state
            )
            
            # Use the mean of the predicted latent state for the next step
            previous_latent_state = pred_latent_mean

            # Convert the reconstructed observation to a displayable format (e.g., numpy array)
            # Assuming the observation is an image (C, H, W)
            frame = recon_obs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Normalize from [-1, 1] or [0, 1] to [0, 255]
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = (frame * 255).astype(np.uint8)
            dream_frames.append(frame)

    # --- Save Dream as a GIF ---
    dream_dir = os.path.join(os.path.dirname(checkpoint_path), 'dreams')
    os.makedirs(dream_dir, exist_ok=True)
    gif_path = os.path.join(dream_dir, f'dream_{os.path.basename(checkpoint_path)}.gif')
    imageio.mimsave(gif_path, dream_frames, fps=10)
    print(f"Saved dream sequence to {gif_path}")


if __name__ == '__main__':
    # This is an example of how to run it.
    # You would typically pass the path to a specific checkpoint.
    # For example: python stage1_world_model/validate.py checkpoints/stage1_world_model/YYYY-MM-DD_HH-MM-SS/wm_epoch_10.pt
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python stage1_world_model/validate.py <path_to_checkpoint>")

