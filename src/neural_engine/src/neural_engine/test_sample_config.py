#!/usr/bin/env python3

import rospy
from neural_engine.srv import SampleConfig

def test_sample_config():
    rospy.init_node('test_sample_config')
    
    # Wait for the service to be available
    rospy.wait_for_service('sample_config')
    
    try:
        # Create service proxy
        sample_config = rospy.ServiceProxy('sample_config', SampleConfig)
        
        # Test with current joint positions
        current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Call the service
        response = sample_config(current_joints)
        
        if response.success:
            rospy.loginfo(f"Successfully sampled configuration: {response.sampled_joints}")
        else:
            rospy.logwarn("Failed to sample configuration")
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

if __name__ == "__main__":
    try:
        test_sample_config()
    except rospy.ROSInterruptException:
        pass 