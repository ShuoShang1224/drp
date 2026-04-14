import numpy as np
import torch

# Franka joint limits
FRANKA_LOWER_LIMITS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FRANKA_UPPER_LIMITS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

def clamp_to_franka_limits(joint_angles):
    """
    Clamp joint angles to Franka joint limits and check if any values were out of bounds.
    
    Args:
        joint_angles (np.ndarray or torch.Tensor): Joint angles in radians, can be batched with shape (..., 7)
        
    Returns:
        tuple: (clamped_joint_angles, out_of_bounds)
            - clamped_joint_angles: Joint angles clamped to Franka limits
            - out_of_bounds: Boolean indicating if any values were out of bounds
    """
    if torch.is_tensor(joint_angles):
        lower_limits = torch.tensor(FRANKA_LOWER_LIMITS).to(joint_angles.device).float()
        upper_limits = torch.tensor(FRANKA_UPPER_LIMITS).to(joint_angles.device).float()
        
        # Reshape limits to match joint_angles dimensions for broadcasting
        for _ in range(len(joint_angles.shape) - 1):
            lower_limits = lower_limits.unsqueeze(0)
            upper_limits = upper_limits.unsqueeze(0)
        
        # Check if any values are out of bounds
        out_of_bounds = torch.any((joint_angles < lower_limits) | (joint_angles > upper_limits))
        
        # Clamp values to limits
        clamped_joint_angles = torch.clamp(joint_angles, lower_limits, upper_limits)
    else:
        # Reshape limits to match joint_angles dimensions for broadcasting
        lower_limits = FRANKA_LOWER_LIMITS
        upper_limits = FRANKA_UPPER_LIMITS
        
        for _ in range(len(joint_angles.shape) - 1):
            lower_limits = np.expand_dims(lower_limits, 0)
            upper_limits = np.expand_dims(upper_limits, 0)
        
        # Check if any values are out of bounds
        out_of_bounds = np.any((joint_angles < lower_limits) | (joint_angles > upper_limits))
        
        # Clamp values to limits
        clamped_joint_angles = np.clip(joint_angles, lower_limits, upper_limits)
    
    return clamped_joint_angles, out_of_bounds


def normalize_franka_joints(joint_angles):
    """
    Normalize joint angles to [-1, 1] range.
    
    Args:
        joint_angles (np.ndarray): Joint angles in radians
        
    Returns:
        np.ndarray: Normalized joint angles in [-1, 1] range
    """
    # Reshape if needed
    if len(joint_angles.shape) == 1:
        joint_angles = joint_angles.reshape(1, -1)
    
    # Normalize to [-1, 1]
    if (torch.is_tensor(joint_angles)):
        lower_limits = torch.tensor(FRANKA_LOWER_LIMITS).to(joint_angles.device).float()
        upper_limits = torch.tensor(FRANKA_UPPER_LIMITS).to(joint_angles.device).float()
        normalized = 2.0 * (joint_angles - lower_limits) / (upper_limits - lower_limits) - 1.0
    else:
        normalized = 2.0 * (joint_angles - FRANKA_LOWER_LIMITS) / (FRANKA_UPPER_LIMITS - FRANKA_LOWER_LIMITS) - 1.0
    
    return normalized 

def unnormalize_franka_joints(normalized_joint_angles):
    """
    Unnormalize joint angles from [-1, 1] range to radians.
    
    Args:
        normalized_joint_angles (np.ndarray): Normalized joint angles in [-1, 1] range
        
    Returns:
        np.ndarray: Unnormalized joint angles in radians
    """
    if len(normalized_joint_angles.shape) == 1:
        normalized_joint_angles = normalized_joint_angles.reshape(1, -1)
    if (torch.is_tensor(normalized_joint_angles)):
        lower_limits = torch.tensor(FRANKA_LOWER_LIMITS).to(normalized_joint_angles.device).float()
        upper_limits = torch.tensor(FRANKA_UPPER_LIMITS).to(normalized_joint_angles.device).float()
        unnormalized = normalized_joint_angles * (upper_limits - lower_limits) / 2.0 + (upper_limits + lower_limits) / 2.0
    else:
        
        unnormalized = normalized_joint_angles * (FRANKA_UPPER_LIMITS - FRANKA_LOWER_LIMITS) / 2.0 + (FRANKA_UPPER_LIMITS + FRANKA_LOWER_LIMITS) / 2.0
    return unnormalized

if __name__ == "__main__":
    """
    Test function to verify that normalize and unnormalize functions work correctly
    with different shapes and both numpy arrays and torch tensors.
    """
    import torch
    import numpy as np
    
    # Test with numpy arrays
    print("Testing with numpy arrays...")
    
    # Test with 1D array
    test_angles_1d = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 2.0, -1.0])
    normalized_1d = normalize_franka_joints(test_angles_1d)
    unnormalized_1d = unnormalize_franka_joints(normalized_1d)
    print(f"Original 1D: {test_angles_1d}")
    print(f"Normalized 1D: {normalized_1d}")
    print(f"Unnormalized 1D: {unnormalized_1d}")
    print(f"Max error 1D: {np.max(np.abs(test_angles_1d - unnormalized_1d.squeeze()))}")
    
    # Test with 2D array (batch)
    test_angles_2d = np.array([
        [0.0, 0.5, 1.0, -0.5, 0.0, 2.0, -1.0],
        [1.0, 0.0, -1.0, -2.0, 2.0, 1.0, 0.0]
    ])
    normalized_2d = normalize_franka_joints(test_angles_2d)
    unnormalized_2d = unnormalize_franka_joints(normalized_2d)
    print(f"Original 2D shape: {test_angles_2d.shape}")
    print(f"Normalized 2D shape: {normalized_2d.shape}")
    print(f"Unnormalized 2D shape: {unnormalized_2d.shape}")
    print(f"Max error 2D: {np.max(np.abs(test_angles_2d - unnormalized_2d))}")
    
    # Test with torch tensors
    print("\nTesting with torch tensors...")
    
    # Test with 1D tensor
    test_tensor_1d = torch.tensor([0.0, 0.5, 1.0, -0.5, 0.0, 2.0, -1.0])
    normalized_tensor_1d = normalize_franka_joints(test_tensor_1d)
    unnormalized_tensor_1d = unnormalize_franka_joints(normalized_tensor_1d)
    print(f"Original tensor 1D: {test_tensor_1d}")
    print(f"Normalized tensor 1D: {normalized_tensor_1d}")
    print(f"Unnormalized tensor 1D: {unnormalized_tensor_1d}")
    print(f"Max error tensor 1D: {torch.max(torch.abs(test_tensor_1d - unnormalized_tensor_1d.squeeze())).item()}")
    
    # Test with 2D tensor (batch)
    test_tensor_2d = torch.tensor([
        [0.0, 0.5, 1.0, -0.5, 0.0, 2.0, -1.0],
        [1.0, 0.0, -1.0, -2.0, 2.0, 1.0, 0.0]
    ])
    normalized_tensor_2d = normalize_franka_joints(test_tensor_2d)
    unnormalized_tensor_2d = unnormalize_franka_joints(normalized_tensor_2d)
    print(f"Original tensor 2D shape: {test_tensor_2d.shape}")
    print(f"Normalized tensor 2D shape: {normalized_tensor_2d.shape}")
    print(f"Unnormalized tensor 2D shape: {unnormalized_tensor_2d.shape}")
    print(f"Max error tensor 2D: {torch.max(torch.abs(test_tensor_2d - unnormalized_tensor_2d)).item()}")
    
    # Test with 3D tensor (batch, sequence, joints)
    test_tensor_3d = torch.tensor([
        [
            [0.0, 0.5, 1.0, -0.5, 0.0, 2.0, -1.0],
            [1.0, 0.0, -1.0, -2.0, 2.0, 1.0, 0.0]
        ],
        [
            [-1.0, 1.5, 0.0, -1.5, 1.0, 3.0, -2.0],
            [2.0, -1.0, 2.0, -3.0, 1.0, 2.0, -1.0]
        ]
    ])
    
    # Reshape to 2D for processing
    batch, seq, joints = test_tensor_3d.shape
    reshaped_tensor = test_tensor_3d.reshape(batch * seq, joints)
    
    normalized_reshaped = normalize_franka_joints(reshaped_tensor)
    unnormalized_reshaped = unnormalize_franka_joints(normalized_reshaped)
    
    # Reshape back to 3D
    normalized_tensor_3d = normalized_reshaped.reshape(batch, seq, joints)
    unnormalized_tensor_3d = unnormalized_reshaped.reshape(batch, seq, joints)
    
    print(f"\nOriginal tensor 3D shape: {test_tensor_3d.shape}")
    print(f"Normalized tensor 3D shape: {normalized_tensor_3d.shape}")
    print(f"Unnormalized tensor 3D shape: {unnormalized_tensor_3d.shape}")
    print(f"Max error tensor 3D: {torch.max(torch.abs(test_tensor_3d - unnormalized_tensor_3d)).item()}")
    
    print("\nAll tests completed!")
