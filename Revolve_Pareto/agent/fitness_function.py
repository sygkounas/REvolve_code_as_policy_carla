import math
import numpy as np



def fitness_function_llm():
    """
    Fitness function for a CARLA driving episode.
    Reads variables available in run_episode() scope via `self`:
      - self.positions: list of (x, y) tuples along the trajectory
      - self.steerings: list of steering commands per step in [-1, 1]
      - self.speeds_kmh: list of ego speeds (km/h)
      - self.total_steps: number of steps taken in the episode
      - self.max_steps: maximum allowed steps (timeout)
      - self.collision_flag: dict with key "flag" set to True if any collision occurred
      - self.num_red_violations: number of red-light violations
      - self.done_path: True if loop_complete condition was met
      - self.collision_ids: set/list of actor ids involved in collisions (added to metrics by caller)

    Returns:
      total_fitness ∈ [0,1], metrics dict (interpretable sub-metrics)
    """
    import math
    import numpy as np

    # ---------------------------
    # Configurable targets/limits
    # ---------------------------
    desired_speed_kmh = 25.0          # Target cruise speed
    hard_speed_limit_kmh = 40.0       # Soft rule limit (fractional penalty if exceeded)
    low_speed_kmh = 2.0               # Considered "near-stop" below this speed

    # Smoothness thresholds (per-step changes; simulator ticks are uniform but unknown dt)
    steer_rate_comfort = 0.08         # Comfortable mean abs steering change per step
    speed_dd_comfort = 1.0            # Comfortable mean abs second difference in km/h per step^2

    # Weights for top-level aggregation
    w_safety = 0.5
    w_efficiency = 0.3
    w_smoothness = 0.15
    w_rules = 0.05

    # Helper: clamp to [0,1]
    def _clamp01(x):
        return float(max(0.0, min(1.0, x)))

    # ---------------------------
    # Gather episode data safely
    # ---------------------------
    spd = np.array(getattr(self, "speeds_kmh", []), dtype=np.float32)
    steer = np.array(getattr(self, "steerings", []), dtype=np.float32)

    N = int(spd.shape[0])
    collided = bool(self.collision_flag.get("flag", False))
    num_red = int(getattr(self, "num_red_violations", 0))
    done = bool(getattr(self, "done_path", False))
    steps_used = int(getattr(self, "total_steps", 0))
    max_steps = max(1, int(getattr(self, "max_steps", 1)))

    # Basic speed stats
    if N > 0:
        avg_speed = float(np.mean(spd))
        med_speed = float(np.median(spd))
        low_speed_frac = float(np.mean(spd < low_speed_kmh))
        overspeed_frac = float(np.mean(spd > hard_speed_limit_kmh))
        mae_speed = float(np.mean(np.abs(spd - desired_speed_kmh)))
    else:
        avg_speed = 0.0
        med_speed = 0.0
        low_speed_frac = 1.0
        overspeed_frac = 0.0
        mae_speed = desired_speed_kmh

    # ---------------------------
    # Safety score
    # ---------------------------
    # Any collision zeros safety. Red-light violations reduce safety even without a collision.
    if collided:
        safety_score = 0.0
    else:
        # Each red violation reduces safety by 0.5 (two or more => 0)
        safety_score = _clamp01(1.0 - 0.5 * float(num_red))

    # ---------------------------
    # Efficiency score
    # ---------------------------
    # - Completion is primary
    # - Time-efficiency only counts if the route is completed (finish earlier -> better)
    # - Speed tracking around the target and avoiding getting stuck also matter
    completion_score = 1.0 if done else 0.0
    time_efficiency = (1.0 - float(steps_used) / float(max_steps)) if done else 0.0
    time_efficiency = _clamp01(time_efficiency)

    # Speed target tracking (closer to desired => higher score)
    speed_target_score = 1.0 - min(1.0, (mae_speed / max(desired_speed_kmh, 1e-6)))

    # Anti-stall score (less time near-stop => better)
    stuck_score = _clamp01(1.0 - math.sqrt(max(0.0, low_speed_frac)))

    # Combine into efficiency
    completion_eff = 0.7 * completion_score + 0.3 * time_efficiency
    efficiency_score = (
        0.5 * completion_eff +
        0.35 * speed_target_score +
        0.15 * stuck_score
    )
    efficiency_score = _clamp01(efficiency_score)

    # ---------------------------
    # Smoothness score
    # ---------------------------
    # Steering rate (mean abs diff per step)
    if steer.shape[0] >= 2:
        steer_rate = float(np.mean(np.abs(np.diff(steer))))
    else:
        steer_rate = 0.0

    # Longitudinal jerk proxy from second diff of speeds (km/h units across steps)
    if N >= 3:
        speed_dd = np.diff(spd, n=2)
        jerk_abs = float(np.mean(np.abs(speed_dd)))
    else:
        jerk_abs = 0.0

    # Map to [0,1] via smooth decay (1 / (1 + (x/t)^2))
    steer_rate_score = 1.0 / (1.0 + (steer_rate / max(1e-6, steer_rate_comfort))**2)
    jerk_score = 1.0 / (1.0 + (jerk_abs / max(1e-6, speed_dd_comfort))**2)

    smoothness_score = _clamp01(0.6 * steer_rate_score + 0.4 * jerk_score)

    # ---------------------------
    # Rule compliance score
    # ---------------------------
    # Currently accounts for overspeeding fraction. TL violations covered in safety.
    rule_speed_score = _clamp01(1.0 - overspeed_frac)
    rule_score = rule_speed_score

    # ---------------------------
    # Aggregate total fitness
    # ---------------------------
    total = (
        w_safety * safety_score +
        w_efficiency * efficiency_score +
        w_smoothness * smoothness_score +
        w_rules * rule_score
    )

    # Hard safety gate: if collision occurred, fitness is 0 (safety-first objective)
    if collided:
        total = 0.0

    total_fitness = _clamp01(total)

    # ---------------------------
    # Metrics for interpretability
    # ---------------------------
    metrics = {
        # Goals
        "total_fitness": float(total_fitness),

        # Safety
        "safety_score": float(safety_score),
        "collided": bool(collided),
        "num_red_violations": int(num_red),

        # Efficiency
        "efficiency_score": float(efficiency_score),
        "completion": bool(done),
        "completion_score": float(completion_score),
        "time_efficiency": float(time_efficiency),
        "speed_target_score": float(speed_target_score),
        "stuck_score": float(stuck_score),

        # Smoothness
        "smoothness_score": float(smoothness_score),
        "steer_rate_mean_abs": float(steer_rate),
        "steer_rate_score": float(steer_rate_score),
        "longitudinal_jerk_abs": float(jerk_abs),
        "jerk_score": float(jerk_score),

        # Rules
        "rule_score": float(rule_score),
        "overspeed_fraction": float(overspeed_frac),

        # Episode stats
        "avg_speed_kmh": float(avg_speed),
        "median_speed_kmh": float(med_speed),
        "mae_speed_to_target_kmh": float(mae_speed),
        "low_speed_fraction": float(low_speed_frac),
        "steps_used": int(steps_used),
        "max_steps": int(max_steps),

        # Configuration echoes
        "desired_speed_kmh": float(desired_speed_kmh),
        "hard_speed_limit_kmh": float(hard_speed_limit_kmh),
    }

    # Note: metrics["end_reason"] and metrics["collided_with_ids"] are appended by the caller after this function.
    return float(total_fitness), metrics

# # Optional alias to match the call-site name in the provided snippet.
# # If the simulator calls `fitness_score()`, this binds it to the same implementation.
# def fitness_score():
#     return fitness_function()





def fitness_score(path, positions, steerings, speeds,
                  total_steps, done_path,
                  collision, num_red_violations=0,
                  ref_steps=5500, min_moving_speed=2):
    ref_steps=5500
    delta_dev = 0.35
    margin=1.0  
    delta_osc = 0.05
    if len(positions) == 0:
        return 0.0, {"error": "no positions"}

    # --- Path length ---
    diffs = np.diff(path, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # --- Progress (arc-length projection) ---
    def project_to_path(pt, path):
        dists = np.linalg.norm(path - pt, axis=1)
        idx = np.argmin(dists)
        arc_len = np.sum(np.linalg.norm(np.diff(path[:idx+1], axis=0), axis=1))
        if idx < len(path)-1:
            seg = path[idx+1] - path[idx]
            seg_len = np.linalg.norm(seg)
            if seg_len > 1e-6:
                proj = np.dot(pt - path[idx], seg) / seg_len
                arc_len += max(0, min(seg_len, proj))
        return arc_len

    s_start = project_to_path(positions[0], path)
    s_end   = project_to_path(positions[-1], path)
    progress = min(1.0, max(0.0, (s_end - s_start) / max(1e-6, path_length)))
    if progress >=0.99:
        progress=1

    # --- Lane deviation score ---
    deviations = []
    for step, pos in enumerate(positions):
        dists = np.linalg.norm(path - pos, axis=1)
        min_dev = np.min(dists)
        deviations.append(min_dev)

        # DEBUG: print every 50 steps to avoid spam
        # if step % 50 == 0 or step == len(positions)-1:
        #     print(f"[LANE DEV] step={step:4d} pos=({pos[0]:.2f},{pos[1]:.2f}) "
        #           f"dev={min_dev:.3f} m")

    mean_dev = float(np.mean(deviations))
    lane_score = math.exp(-2.3 * max(0.0, mean_dev - delta_dev))

    # print(f"[LANE SUMMARY] mean_dev={mean_dev:.3f} m  "
    #       f"lane_score={lane_score:.3f}  "
    #       f"max_dev={float(np.max(deviations)):.3f} m "
    #       f"at step {int(np.argmax(deviations))}")

    # --- Smoothness score ---
    if len(steerings) > 2:
        steer_rate = np.diff(steerings)
        osc = float(np.std(steer_rate))
        smooth_score = math.exp(-5.0 * max(0.0, osc - delta_osc))
    else:
        smooth_score = 1.0

    k=2.0
    # --- Efficiency ---
    if done_path and ref_steps > 0:
        ratio = total_steps / float(ref_steps)
        eff_score = math.exp(-k * max(0.0, (ratio - margin)))
    else:
        eff_score = 0.0

    # --- Base weighted score ---
    fitness = (0.40 * progress +
               0.30 * lane_score +
               0.15 * smooth_score +
               0.15 * eff_score)

    # --- Penalties ---
    collision_pen = 0.5 if collision else 0.0
    tl_pen = 0.40 if num_red_violations > 0 else 0.0

    if len(speeds) > 0:
        avg_speed = float(np.mean(speeds))
        N = max(10, len(speeds) // 5)
        recent_avg = float(np.mean(speeds[-N:]))
    else:
        avg_speed, recent_avg = 0.0, 0.0

    

    # --- Final fitness ---
    fitness = max(0.0, fitness - collision_pen - tl_pen )


    metrics = {
        "progress": progress,
        "lane_score": lane_score,
        "smooth_score": smooth_score,
        "eff_score": eff_score,
        "collision": collision,
        "num_red_violations": num_red_violations,
      #  "avg_speed": avg_speed,
        "collision_pen": collision_pen,
        "tl_pen": tl_pen,
        "fitness": fitness,
        # add full deviation trace for analysis
     #   "lane_deviations": deviations,
    }

    return fitness, metrics
