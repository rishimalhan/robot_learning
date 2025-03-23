#!/usr/bin/env python3

import rospy
import torch
import numpy as np
from neural_engine.srv import SampleConfig, SampleConfigResponse

# Check MPS availability
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def sample_configs(req):
    try:
        # Convert current joints to tensor on MPS
        current_joints = torch.tensor(req.current_joints, device=DEVICE)
        num_joints = len(current_joints)
        
        # Generate multiple random configurations
        num_samples = 100  # Generate more samples to increase chance of finding valid config
        random_configs = torch.rand((num_samples, num_joints), device=DEVICE)
        
        # Get joint limits (assuming standard joint limits for ABB IRB2400)
        min_limits = torch.tensor([-3.1416, -1.7453, -1.0472, -3.49, -2.0944, -6.9813], device=DEVICE)
        max_limits = torch.tensor([3.1416, 1.9199, 1.1345, 3.49, 2.0944, 6.9813], device=DEVICE)
        
        # Scale random values to joint limits
        random_configs = min_limits + (max_limits - min_limits) * random_configs
        
        # Calculate absolute differences from current position
        joint_diffs = torch.abs(random_configs - current_joints)
        
        # Convert 90 degrees to radians
        max_diff = torch.tensor(1.5708, device=DEVICE)  # 90 degrees in radians
        
        # Find configurations where all joints are within 90 degrees
        valid_configs = torch.all(joint_diffs <= max_diff, dim=1)
        
        if torch.any(valid_configs):
            # Get the first valid configuration
            valid_idx = torch.where(valid_configs)[0][0]
            sampled_config = random_configs[valid_idx]
            
            # Convert back to CPU and then to list for ROS
            sampled_joints = sampled_config.cpu().numpy().tolist()
            
            rospy.loginfo(f"Found valid configuration: {sampled_joints}")
            return SampleConfigResponse(sampled_joints=sampled_joints, success=True)
        else:
            rospy.logwarn("No valid configuration found within 90 degrees")
            return SampleConfigResponse(sampled_joints=[], success=False)
            
    except Exception as e:
        rospy.logerr(f"Error in sample_configs: {str(e)}")
        return SampleConfigResponse(sampled_joints=[], success=False)

def main():
    rospy.init_node('sample_config_service')
    
    # Create service
    s = rospy.Service('sample_config', SampleConfig, sample_configs)
    rospy.loginfo("Sample Config Service Ready")
    rospy.loginfo(f"Using device: {DEVICE}")
    
    # Keep the node running
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 