#! /usr/bin/env python

import time
from typing import Callable, Tuple, Optional, Any, Dict, List
from decorator import decorator
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PlanningStats:
    """Data class to store planning statistics for a single attempt."""

    planner_id: str
    success: bool
    planning_time: float
    error_code: int
    timestamp: float
    waypoints: Optional[int] = None
    path_length: Optional[float] = None


class StatsReporter:
    """Class to manage and persist planning statistics."""

    def __init__(self):
        self._stats: Dict[str, List[PlanningStats]] = defaultdict(list)

    def _compute_path_length(self, plan) -> float:
        """Compute the path length by summing Euclidean distances between consecutive joint configurations."""
        if (
            not plan
            or not hasattr(plan, "joint_trajectory")
            or not plan.joint_trajectory.points
        ):
            return 0.0

        total_length = 0.0
        points = plan.joint_trajectory.points

        for i in range(len(points) - 1):
            # Get joint positions for current and next point
            current = points[i].positions
            next_point = points[i + 1].positions

            # Compute Euclidean distance between joint configurations
            distance = sum((a - b) ** 2 for a, b in zip(current, next_point)) ** 0.5
            total_length += distance

        return total_length

    def add_stats(
        self,
        planner_id: str,
        success: bool,
        planning_time: float,
        error_code: int,
        waypoints: Optional[int] = None,
        plan: Optional[Any] = None,
    ) -> None:
        """Add new planning statistics."""
        path_length = self._compute_path_length(plan) if plan else None

        stats = PlanningStats(
            planner_id=planner_id,
            success=success,
            planning_time=planning_time,
            error_code=error_code,
            timestamp=time.time(),
            waypoints=waypoints,
            path_length=path_length,
        )
        self._stats[planner_id].append(stats)

    def get_stats(self, planner_id: Optional[str] = None) -> List[PlanningStats]:
        """Get statistics for a specific planner or all planners."""
        if planner_id is None:
            return [stat for stats in self._stats.values() for stat in stats]
        return self._stats.get(planner_id, [])

    def get_planner_summary(self, planner_id: str) -> Dict[str, Any]:
        """Get detailed summary for a specific planner."""
        stats = self._stats.get(planner_id, [])
        if not stats:
            return {}

        successful_stats = [stat for stat in stats if stat.success]
        valid_lengths = [
            stat.path_length
            for stat in successful_stats
            if stat.path_length is not None
        ]

        return {
            "successful_attempts": len(successful_stats),
            "failed_attempts": len(stats) - len(successful_stats),
            "success_rate": len(successful_stats) / len(stats) if stats else 0.0,
            "average_planning_time": (
                sum(stat.planning_time for stat in successful_stats)
                / len(successful_stats)
                if successful_stats
                else 0.0
            ),
            "average_path_length": (
                sum(valid_lengths) / len(valid_lengths) if valid_lengths else 0.0
            ),
            "min_path_length": min(valid_lengths) if valid_lengths else 0.0,
            "max_path_length": max(valid_lengths) if valid_lengths else 0.0,
        }

    def clear(self) -> None:
        """Clear all stored statistics."""
        self._stats.clear()


# Create a singleton instance
stats_reporter = StatsReporter()


@decorator
def report_planning_stats(
    func: Callable, *args, **kwargs
) -> Tuple[bool, Optional[Any], float, int]:
    """
    Decorator to report planning statistics.

    Args:
        func: The planning function to decorate
        *args: Positional arguments for the planning function
        **kwargs: Keyword arguments for the planning function

    Returns:
        Tuple containing (success, plan, planning_time, error_code)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time

    # Get waypoints count if available in the result
    waypoints = None
    plan = None
    if (
        isinstance(result, tuple)
        and len(result) > 1
        and hasattr(result[1], "joint_trajectory")
    ):
        plan = result[1]
        waypoints = len(plan.joint_trajectory.points)

    # Add stats to the reporter
    stats_reporter.add_stats(
        planner_id=args[2],  # planner_id is the third argument
        success=result[0],
        planning_time=result[2],
        error_code=result[3],
        waypoints=waypoints,
        plan=plan,
    )

    # Log the planning statistics
    logger.debug(
        f"StatsReporter: "
        f"Planning with {args[2]} took {duration:.3f} seconds. "
        f"Result: success={result[0]}, planning_time={result[2]:.3f}, error_code={result[3]}"
    )

    return result
