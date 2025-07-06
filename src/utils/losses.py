import torch
import torch.nn.functional as F

def world_model_loss(
    reconstructed_observation,
    target_observation,
    predicted_reward,
    target_reward,
    predicted_latent_mean,
    predicted_latent_logvar,
    encoded_latent_mean,
    encoded_latent_logvar,
    kl_beta=1.0
):
    """
    Calculates the full loss for the World Model.

    Args:
        reconstructed_observation (torch.Tensor): The observation reconstructed by the decoder.
        target_observation (torch.Tensor): The ground truth observation.
        predicted_reward (torch.Tensor): The reward predicted by the reward predictor.
        target_reward (torch.Tensor): The ground truth reward.
        predicted_latent_mean (torch.Tensor): The mean of the latent state from the dynamics model.
        predicted_latent_logvar (torch.Tensor): The log variance of the latent state from the dynamics model.
        encoded_latent_mean (torch.Tensor): The mean of the latent state from the encoder.
        encoded_latent_logvar (torch.Tensor): The log variance of the latent state from the encoder.
        kl_beta (float): A scaling factor for the KL divergence loss.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the total loss
        and a dictionary of the individual loss components.
    """
    # Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(reconstructed_observation, target_observation)

    # Reward Prediction Loss (MSE)
    reward_loss = F.mse_loss(predicted_reward, target_reward)

    # KL Divergence Loss
    # This loss pushes the dynamics model's predictions (prior) to match the
    # encoder's output (posterior) from the actual observation.
    kl_div_loss = -0.5 * torch.sum(
        1 + predicted_latent_logvar - encoded_latent_mean.pow(2) - encoded_latent_logvar.exp(),
        dim=-1
    ).mean()

    # Total Loss
    total_loss = recon_loss + reward_loss + kl_beta * kl_div_loss

    loss_components = {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'reward_loss': reward_loss.item(),
        'kl_div_loss': kl_div_loss.item()
    }

    return total_loss, loss_components

def controller_loss(predicted_value, target_value):
    """
    Calculates the loss for the Controller and the Intrinsic Critic.

    Args:
        predicted_value (torch.Tensor): The value predicted by the Intrinsic Critic.
        target_value (torch.Tensor): The target value (bootstrapped from the next state).

    Returns:
        torch.Tensor: The mean squared error loss.
    """
    return F.mse_loss(predicted_value, target_value)
