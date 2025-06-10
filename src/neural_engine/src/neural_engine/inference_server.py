#!/usr/bin/env python3

import rospy
import actionlib
from dataclasses import dataclass
from typing import Optional
from neural_engine.inference_engine import InferenceEngine, GenerationConfig
from neural_engine_msgs.msg import (
    GenerateAction,
    GenerateGoal,
    GenerateResult,
    GenerateFeedback,
)


class InferenceServer:
    """ROS Action Server wrapper for the inference engine."""

    def __init__(self):
        # Initialize node
        rospy.init_node("inference_server")

        # Get parameters
        model_name = rospy.get_param("~model_name", None)
        use_fine_tuned = rospy.get_param("~use_fine_tuned", True)

        # Initialize inference engine
        self.engine = InferenceEngine(
            model_name=model_name, use_fine_tuned=use_fine_tuned
        )

        # Initialize action server
        self._action_name = "generate_text"
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            GenerateAction,
            execute_cb=self.execute_cb,
            auto_start=False,
        )

        # Start server
        self._server.start()
        rospy.loginfo("Inference server started")

    def _goal_to_config(self, goal: GenerateGoal) -> GenerationConfig:
        """Convert action goal to generation config."""
        return GenerationConfig(
            max_tokens=goal.max_tokens,
            temperature=goal.temperature,
            top_p=goal.top_p,
            repetition_penalty=goal.repetition_penalty,
            do_sample=goal.do_sample,
        )

    def execute_cb(self, goal: GenerateGoal):
        """Execute callback for the action server."""
        rospy.loginfo(f"Received generation request: {goal.prompt[:50]}...")

        result = GenerateResult()
        feedback = GenerateFeedback()

        try:
            # Convert goal parameters to generation config
            config = self._goal_to_config(goal)

            if goal.stream:
                # Streaming generation
                generated_tokens = []
                for token in self.engine.generate(
                    goal.prompt, stream=True, config=config
                ):
                    # Check for preemption
                    if self._server.is_preempt_requested():
                        rospy.loginfo(f"{self._action_name}: Preempted")
                        self._server.set_preempted()
                        return

                    # Accumulate tokens and send feedback
                    generated_tokens.append(token)
                    feedback.token = token
                    feedback.progress = len(generated_tokens) / config.max_tokens
                    self._server.publish_feedback(feedback)

                result.complete_response = "".join(generated_tokens)

            else:
                # Non-streaming generation
                result.complete_response = self.engine.generate(
                    goal.prompt, stream=False, config=config
                )

            # Set success
            result.success = True
            result.message = "Generation completed successfully"
            self._server.set_succeeded(result)

        except Exception as e:
            rospy.logerr(f"Error during generation: {str(e)}")
            result.success = False
            result.message = str(e)
            self._server.set_aborted(result)


def main():
    try:
        server = InferenceServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
