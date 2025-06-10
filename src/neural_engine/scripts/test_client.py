#!/usr/bin/env python3

import rospy
import actionlib
from neural_engine_msgs.msg import GenerateAction, GenerateGoal


def feedback_cb(feedback):
    """Callback for streaming feedback."""
    print(feedback.token, end="", flush=True)


def generate_text(prompt: str, stream: bool = False) -> str:
    """Generate text using the inference server.

    Args:
        prompt: The input prompt
        stream: Whether to stream the response

    Returns:
        The generated text
    """
    # Create action client
    client = actionlib.SimpleActionClient("generate_text", GenerateAction)

    # Wait for server
    rospy.loginfo("Waiting for server...")
    client.wait_for_server()

    # Create goal
    goal = GenerateGoal(
        prompt=prompt,
        stream=stream,
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        repetition_penalty=1.1,
        do_sample=True,
    )

    # Send goal
    if stream:
        client.send_goal(goal, feedback_cb=feedback_cb)
    else:
        client.send_goal(goal)

    # Wait for result
    client.wait_for_result()
    result = client.get_result()

    if result.success:
        return result.complete_response
    else:
        raise Exception(f"Generation failed: {result.message}")


def main():
    rospy.init_node("test_client")

    try:
        # Test non-streaming generation
        print("\nTesting non-streaming generation:")
        response = generate_text("Write a haiku about robots:")
        print(f"Response: {response}")

        # Test streaming generation
        print("\nTesting streaming generation:")
        response = generate_text(
            "Write a short story about a robot learning to paint:", stream=True
        )
        print("\nComplete!")

    except Exception as e:
        rospy.logerr(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
