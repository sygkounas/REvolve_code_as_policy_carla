#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA Town01 environment wrapper:
- Class CarlaTown01Env encapsulates setup, randomization, episode run, cleanup.
- Keeps your harder cars/pedestrians logic + fitness scoring.
- Saves the BasicAgent global plan to extracted_path.txt (ep 0).
- main(): runs 25 episodes and prints success rate + avg fitness.

Requirements:
- CARLA server running (matching Python egg).
- fitness_function.py in the same folder (provides fitness_score).
"""

import os, sys, glob, math, time, random
from pathlib import Path
from typing import Optional, Tuple, List
from collections import deque

import numpy as np

# ------------------------ Bootstrap for CARLA/agents ------------------------
def _bootstrap_carla_paths():
    roots = []
    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots += [Path("/home/alkis/CARLA"), Path("~/CARLA").expanduser()]
    for root in roots:
        pyapi = root / "PythonAPI"
        if not pyapi.exists():
            continue
        dist_dir = pyapi / "carla" / "dist"
        if dist_dir.exists():
            pattern = f"carla-*-py{sys.version_info.major}.{sys.version_info.minor}-*.egg"
            for egg in glob.glob(str(dist_dir / pattern)):
                if egg not in sys.path:
                    sys.path.append(egg)
        for p in [pyapi, pyapi/"carla", pyapi/"agents", pyapi/"carla"/"agents"]:
            p_str = str(p)
            if p.exists() and p_str not in sys.path:
                sys.path.append(p_str)
_bootstrap_carla_paths()

try:
    import carla  # noqa
    from agents.navigation.basic_agent import BasicAgent
    from agents.navigation.local_planner import RoadOption
except Exception as e:
    raise RuntimeError("Could not import CARLA/agents. Set CARLA_ROOT and match egg to your Python version.") from e

# ------------------------ Fitness import ------------------------
try:
    from fitness_function import fitness_score
except Exception:
    sys.path.append(str(Path(".").resolve()))
    from fitness_function import fitness_score

# ------------------------ Small helpers ------------------------
LANE_HALF = 2.0
LANE_MARGIN = 1.0
PED_LAT_RIGHT   = 3.0   # extend +1 m for sidewalk
PED_LAT_LEFT    = -3.0  # extend -1 m past left lane edge
PED_INLANE_EPS    = 0.5
PED_APPROACH_BAND = 3.0
LAT_V_MIN         = 0.20
CROSS_HORIZON_S   = 8.0
PED_DS_MAX        = 50                 # meters ahead for pedestrians snapshot
SAFE_DISTANCE_CONST = 50.0             # look-ahead for lead car
TL_WINDOW_AHEAD_M   = 50.0
TL_GATE_RADIUS_M    = 2.5
HIST_LEN            = 4
OPP_LANE_LEFT  = -6.0      # far edge of opposite lane
OPP_LANE_RIGHT = -2.0      # near edge of opposite lane (left edge of ego lane)

# how many vehicles to keep per group
MAX_LEAD_VEHICLES     = 2
MAX_OPPOSITE_VEHICLES = 2
# Ped lateral visibility window: full opposite lane (-6..-2)
# plus +1 m beyond our right lane edge (+2 -> +3)
PED_LAT_MIN = -6.0
PED_LAT_MAX = 3.0




def load_path(path_file: str) -> np.ndarray:
    """Load 'x y' per line into np.ndarray (N,2), float32."""
    pts = []
    with open(path_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                pts.append([x, y])
    arr = np.array(pts, dtype=np.float32)
    if arr.shape[0] < 3:
        raise RuntimeError("Path too short; need >= 3 points.")
    return arr

def precompute_loop_segments(path: np.ndarray):
    """Precompute piecewise-linear loop geometry."""
    A  = path.copy()
    B  = np.roll(path, -1, axis=0)
    AB = B - A
    seg_len = np.linalg.norm(AB, axis=1)
    s_before = np.concatenate(([0.0], np.cumsum(seg_len)[:-1]))
    L = float(np.sum(seg_len))
    return A, B, AB, seg_len, s_before, L

def project_point_onto_loop(point_xy, A, AB, seg_len, s_before):
    AP = point_xy[None, :] - A
    denom = np.maximum(seg_len**2, 1e-9)
    t_all = np.clip(np.sum(AP * AB, axis=1) / denom, 0.0, 1.0)
    proj   = A + (t_all[:, None] * AB)
    d2     = np.sum((proj - point_xy[None, :])**2, axis=1)
    k = int(np.argmin(d2))
    t = float(t_all[k])
    s_now = float(s_before[k] + t * seg_len[k])

    if seg_len[k] > 1e-6:
        tangent = AB[k] / seg_len[k]
        normal  = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        lateral = float(np.dot(point_xy - proj[k], normal))
    else:
        lateral = 0.0

    return s_now, lateral, k, t, proj[k]


def wrap_forward_progress(s_now, s0, L):
    d = s_now - s0
    return float(d % L)

def ped_cross_target_for(walker_loc: carla.Location, path_xy: np.ndarray, cross_offset: float = 8.0) -> Tuple[float, float]:
    """Geometric 'cross the road' target: push across the path normal."""
    p = np.array([walker_loc.x, walker_loc.y], dtype=np.float32)
    A  = path_xy
    B  = np.roll(path_xy, -1, axis=0)
    AB = B - A
    seg_len = np.linalg.norm(AB, axis=1)
    denom = np.maximum(seg_len**2, 1e-9)
    t_all = np.clip(np.sum((p[None,:]-A)*AB, axis=1) / denom, 0.0, 1.0)
    proj   = A + t_all[:,None]*AB
    d2     = np.sum((proj - p[None,:])**2, axis=1)
    k = int(np.argmin(d2))
    seg = AB[k]
    L   = np.linalg.norm(seg)
    if L < 1e-6:
        n = np.array([0.0, 1.0], dtype=np.float32)
    else:
        n = np.array([-seg[1]/L, seg[0]/L], dtype=np.float32)  # left normal
    side = np.sign(np.dot(p - proj[k], n))
    target = proj[k] - side * n * cross_offset
    return float(target[0]), float(target[1])

def get_tm_with_fallback(client, start_port=8000, tries=32):
    """Traffic Manager bind fallback over a port range."""
    last_err = None
    for p in range(start_port, start_port + tries):
        try:
            tm = client.get_trafficmanager(p)
            return tm, p
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"Traffic Manager bind failed on [{start_port}..{start_port+tries-1}]. Last: {last_err}")
def rot_world_to_ego(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[ c,  s],
                     [-s,  c]], dtype=np.float32)

def ttc_longitudinal(gap_long_m, rel_long_mps, cap=20.0):
    if rel_long_mps < 0:
        t = gap_long_m / max(1e-3, -rel_long_mps)
        return float(min(t, cap))
    return None

def frenet_s_lat(point_xy, A, AB, seg_len, s_before):
    s, lat, *_ = project_point_onto_loop(point_xy, A, AB, seg_len, s_before)
    return s, lat

def in_lane_ahead(s_now, s_obj, lat_obj, L, max_ahead):
    ds = (s_obj - s_now) % L
    return (0.0 < ds <= max_ahead) and (abs(lat_obj) <= (LANE_HALF + LANE_MARGIN)), ds

def _time_to_target_lat(target_lat, gap_lat_m, rel_lat_mps, vmin=LAT_V_MIN):
    if abs(rel_lat_mps) < vmin:
        return None
    delta = target_lat - gap_lat_m
    if delta * rel_lat_mps <= 0:
        return None
    return delta / rel_lat_mps

def map_known_lights_to_actors(world, search_radius=50.0):
    tl_actors = list(world.get_actors().filter("traffic.traffic_light*"))
    mapping = {}
    for i, (lx, ly) in KNOWN_TL.items():
        best, best_d = None, 1e9
        for tl in tl_actors:
            loc = tl.get_transform().location
            d = math.hypot(loc.x - lx, loc.y - ly)
            if d < best_d:
                best_d = d
                best = tl
        mapping[i] = best.id if (best is not None and best_d <= search_radius) else None
    return mapping

def tl_state_str(world, actor_id):
    if actor_id is None:
        return "Unknown"
    tl = world.get_actor(actor_id)
    if tl is None:
        return "Unknown"
    st = str(tl.get_state())
    return "Red" if st == "Yellow" else st  # match inference: treat Yellow as Red

# ------------------------ Environment class ------------------------

# ======== OBS constants (mirrors inference.py) ========

# Town01 TL “anchor” points (same as inference.py)
KNOWN_TL = {
    1: (321.68, 136.13),
    2: (331.99, 184.47),
    3: (102.72, 192.55),
    4: (94.99, 144.38),
}


class CarlaTown01Env:
    """
    Town01 ENV:
    - __init__: stores config (no argparse), loads path.
    - reset(seed): reloads world, randomizes episode (spawn points, counts, behaviors).
    - run_episode(): runs until loop complete / collision / timeout. Returns (success, fitness, metrics).
    - cleanup(): removes actors, turns off sync.

    Extendable: add new difficulty knobs in-place.
    """

    def __init__(self,
                 host="localhost",
                 port=2000,
                 tm_port=8000,
                 town="Town01", # more towns to be added
                 path_file="recorded_path_cleaned.txt",
                 # sim control
                 sync=True, fps=20,
                 # ego
                 target_kmh=10.0,
                 # background
                 num_vehicles=0, num_peds=200,
                 # plan density / timeout
                 global_plan_step=10,
                 max_seconds=240,
                 # loop termination gate
                 start_tol_m=2.0, start_gate_frac=0.90,
                 # save plan
                 save_plan_file="extracted_path.txt",
                 # harder cars
                 harder_cars=False, ignore_light_pct=0, lane_change_pct=50,
                 speed_diff_mean=10.0, min_follow_dist=6.0,
                 # stopped car
                 
                 
                 # harder pedestrians
                 harder_peds=True, ped_cross_prob=0.25,
                 ped_cross_every_s=6.0, ped_cross_offset_m=8.0,
                 # fitness reference
                 ref_steps=2200,
                 max_steps= 8000,
                 # per-episode randomness
                 jitter_ratio=0.15):
        # --- Config (no argparse) ---
        self.host = host
        self.port = int(port)
        self.tm_port_req = int(tm_port)
        self.town = town
        self.path_file = path_file

        self.sync = bool(sync)
        self.fps = int(fps)

        self.target_kmh = float(target_kmh)
        self.num_vehicles = int(num_vehicles)
        self.num_peds = int(num_peds)

        self.global_plan_step = int(global_plan_step)
        self.max_seconds = int(max_seconds)
        self.max_steps= int(max_steps)

        self.start_tol_m = float(start_tol_m)
        self.start_gate_frac = float(start_gate_frac)

        self.save_plan_file = save_plan_file

        self.harder_cars = bool(harder_cars)
        self.ignore_light_pct = int(ignore_light_pct)
        self.lane_change_pct = int(lane_change_pct)
        self.speed_diff_mean = float(speed_diff_mean)
        self.min_follow_dist = float(min_follow_dist)

        self.harder_peds = bool(harder_peds)
        self.ped_cross_prob = float(ped_cross_prob)
        self.ped_cross_every_s = float(ped_cross_every_s)
        self.ped_cross_offset_m = float(ped_cross_offset_m)

        self.ref_steps = int(ref_steps)
        self.jitter_ratio = float(jitter_ratio)

        # --- Static data ---
        self.path_xy = load_path(self.path_file)
        self.A, self.B, self.AB, self.seg_len, self.s_before, self.L = precompute_loop_segments(self.path_xy)
        self.first_pt = self.path_xy[0]

        # --- Runtime vars (set in reset) ---
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.bp_lib = None
        self.tm: Optional[carla.TrafficManager] = None
        self.tm_port_used: Optional[int] = None

        self.ego: Optional[carla.Actor] = None
        self.agent: Optional[BasicAgent] = None
        self.col_sensor: Optional[carla.Actor] = None

        self.vehicles: List[carla.Actor] = []
        self.walkers: List[carla.Actor] = []
        self.walker_ctrls: List[carla.Actor] = []
        self.ped_pairs: List[Tuple[carla.Actor, carla.Actor]] = []

        self.all_actors: List[carla.Actor] = []

        self.collision_flag = {"flag": False}
        self.num_red_violations = 0  # placeholder hook

        # Logs per episode
        self.positions: List[Tuple[float, float]] = []
        self.steerings: List[float] = []
        self.speeds_kmh: List[float] = []
        self.total_steps = 0
        self.done_path = False
        # --- Pedestrian behavior knobs (hard but solvable) ---
        self.ped_cooldown_min_s   = 6.0  # min seconds before the same walker can be re-targeted
        self.ped_cooldown_max_s   = 10.0  # max seconds before the same walker can be re-targeted
        self.ped_lane_ttl_s       = 2.5   # fail-safe: if a walker stays inside the lane > TTL, force it off-lane
        self.ped_offlane_offset_m = 5.0   # lateral push (≈ sidewalk) when forcing off-lane
        self._ped_ready_at = {}       # wid -> unix time when re-targeting is allowed again
        self._ped_lane_enter_at = {}  # wid -> unix time of first detection inside lane; cleared when they leave


        self.s0 = 0.0
        self.last_ped_retarget = 0.0
        self.spawn_points_xy = [
            (150.000, 133.437, 0.5),(0,0,0.5),  # start
        ]
        self.obs_speed_hist      = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_lateral_hist    = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_yaw_err_hist    = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_steer_hist      = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_throttle_hist   = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_brake_hist      = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self._lat_ema = None
        self._last_control = None  # updated in run_episode (or pass into get_obs)
        # TL mapping per episode
        self.tl_actor_map = {}
        self.s_light = {}
        self.pace_real_time = True
        self._last_tick_time = None
        self.gate_xy = {}
        self._in_gate_prev = {}
        self.ped_offlane_dwell_s = 2.5


    # ---------- Core: reset/run/cleanup ----------
    def reset(self, seed: Optional[int] = None, save_plan_on_ep0: bool = False, ep_index: int = 0):
        """(Re)load world, set sync, spawn ego/traffic/peds, build plan, set agent & progress zero."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Clean previous episode
        self.cleanup(silent=True)

        # Connect + TM
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        current_world = None
        current_map = None
        try:
            current_world = self.client.get_world()
            current_map = current_world.get_map().name if current_world else None
        except Exception:
            pass

        if not current_map or not current_map.endswith(self.town):
            print(f"[RESET] Loading world {self.town} (current={current_map})")
            self.world = self.client.load_world(self.town)
        else:
            print(f"[RESET] Reusing already loaded world {current_map}")
            self.world = current_world

        self.bp_lib = self.world.get_blueprint_library()
        self.tm, self.tm_port_used = get_tm_with_fallback(self.client, start_port=self.tm_port_req)
        print(f"[TM] bound at port {self.tm_port_used}")

        # Sync mode (world + TM), then flush once
        self._set_sync(self.sync, self.fps)
        if self.sync:
            self.world.tick()
            # Deterministic TM in sync mode
            try:
                self.tm.set_random_device_seed(int(random.randrange(1, 1_000_000)))
            except Exception:
                pass

        import time
        self._last_tick_time = time.perf_counter()

        # Spawn ego
        self.ego = self._spawn_ego_from_list(index=0)
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        print(f"[RESET] actors after spawn: {len(self.world.get_actors())}")
        ego_tf = self.ego.get_transform()
        loc = ego_tf.location
        print(f"[RESET] Ego spawned at x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")
        self._ped_last_positions = {}

        # Collision sensor
        self.col_sensor = self._add_collision_sensor(self.ego, self._on_collision)

        # Vehicles
        veh_count = max(0, int(round(self.num_vehicles * (1.0 + self._jitter()))))
        self.vehicles = self._spawn_traffic(veh_count)
        if self.harder_cars and self.vehicles:
            self._configure_harder_cars(self.vehicles)

        # NEW: flush TM registry so NPCs start immediately
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Pedestrians
        ped_count = max(0, int(round(self.num_peds * (1.0 + self._jitter()))))
        walker_ids, ctrl_ids = self._spawn_pedestrians_batch(ped_count)
        self.walkers = [self.world.get_actor(wid) for wid in walker_ids if self.world.get_actor(wid)]
        self.walker_ctrls = [self.world.get_actor(cid) for cid in ctrl_ids if self.world.get_actor(cid)]
        self.ped_pairs = list(zip(self.walkers, self.walker_ctrls))
        self._tick()
     #   self._tm_rearm_autopilot()
        self.plan = self._build_global_plan(self.global_plan_step)
        if not self.plan:
            raise RuntimeError("Global plan is empty. Check your path/town.")
        # Save plan once (ep 0) unless you want every episode saved
        if save_plan_on_ep0 and ep_index == 0:
            self._save_plan_to_txt(self.plan, self.save_plan_file)
            print(f"[PLAN] Saved {len(self.plan)} waypoints to {self.save_plan_file}")

        # Agent
        self.agent = BasicAgent(self.ego, target_speed=float(self.target_kmh))
        self.agent.set_global_plan(self.plan)

        # Start progress s0 at actual spawn
        start_loc = self.ego.get_transform().location
        self.s0, *_ = project_point_onto_loop(
            np.array([start_loc.x, start_loc.y], dtype=np.float32),
            self.A, self.AB, self.seg_len, self.s_before
        )

        # Reset logs/flags
        self.positions, self.steerings, self.speeds_kmh = [], [], []
        self.total_steps = 0
        self.done_path = False
        self.collision_flag["flag"] = False
        self.collision_ids = []
        self.num_red_violations = 0
        self.last_ped_retarget = time.time()

        # Map TL actors and gates
        self.tl_actor_map = map_known_lights_to_actors(self.world, search_radius=25.0)

        self.s_light = {}

        self.s_light, self.gate_xy = {}, {}
        for idx, (lx, ly) in KNOWN_TL.items():
            pos = np.array([lx, ly], dtype=np.float32)
            s_i, _, _, _, proj_xy = project_point_onto_loop(pos, self.A, self.AB, self.seg_len, self.s_before)
            self.s_light[idx] = float(s_i)
            self.gate_xy[idx] = (float(proj_xy[0]), float(proj_xy[1]))


        # Reset per-episode caches
        self._in_gate_prev = {idx: False for idx in KNOWN_TL}
        self._ped_ready_at = {}
        self._ped_lane_enter_at = {}
        self.obs_speed_hist      = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_lateral_hist    = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_yaw_err_hist    = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_steer_hist      = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_throttle_hist   = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self.obs_brake_hist      = deque([0.0]*HIST_LEN, maxlen=HIST_LEN)
        self._lat_ema = None
        self._last_control = None

    def get_obs(self, control: Optional[carla.VehicleControl] = None) -> dict:
        """
        Build an observation dict matching inference.py 'slim OBS' + extensions:
        - ego scalars: speed_mps, yaw_rate_rps
        - histories len=4: speed/lateral/yaw_error/steer/throttle/brake
        - snapshots:
            * traffic_light
            * lead_cars (top-2 in ego lane), opposite_cars (top-2 in opposite lane)
            * pedestrians (both lanes; inference-style states for our lane, symmetric info for opposite)
        """
        if self.ego is None:
            return {}

        # ---- Ego state ----
        tf = self.ego.get_transform()
        pos = np.array([tf.location.x, tf.location.y], dtype=np.float32)
        yaw_rad = math.radians(tf.rotation.yaw)
        vel = self.ego.get_velocity()
        acc = self.ego.get_acceleration()
        ang = self.ego.get_angular_velocity()   # deg/s in CARLA
        yaw_rate = math.radians(ang.z)          # rad/s

        v_w = np.array([vel.x, vel.y], dtype=np.float32)
        speed = float(np.linalg.norm(v_w))

        # Frenet & lateral / yaw error on path
        s_now, lateral, k_near, _, _proj_xy = project_point_onto_loop(
            pos, self.A, self.AB, self.seg_len, self.s_before
        )
        tangent = self.AB[k_near] / (np.linalg.norm(self.AB[k_near]) + 1e-9)
        path_yaw = math.atan2(tangent[1], tangent[0])
        yaw_error = ((yaw_rad - path_yaw + math.pi) % (2*math.pi)) - math.pi
      #  yaw_error= -yaw_error
        # EMA smooth lateral (like inference)
        if self._lat_ema is None:
            self._lat_ema = lateral
        self._lat_ema = 0.2 * lateral + 0.8 * self._lat_ema
        lateral_out = self._lat_ema              # left-positive; do NOT flip
        # Control (use provided, else last known, else zeros)
        ctrl = control if control is not None else self._last_control
        steer    = float(getattr(ctrl, "steer", 0.0) if ctrl else 0.0)
        throttle = float(getattr(ctrl, "throttle", 0.0) if ctrl else 0.0)
        brake    = float(getattr(ctrl, "brake", 0.0) if ctrl else 0.0)

        # Update histories (len=4)
        self.obs_speed_hist.append(speed)
        self.obs_lateral_hist.append(lateral_out)
        self.obs_yaw_err_hist.append(yaw_error)
        self.obs_steer_hist.append(steer)
        self.obs_throttle_hist.append(throttle)
        self.obs_brake_hist.append(brake)

        # World→ego rotation
        R_we = rot_world_to_ego(yaw_rad)

        # ---- Traffic light snapshot (active selection) ----
        tl_active_idx = None
        tl_active_dist = None
        tl_active_state = "Unknown"
        for idx, (lx, ly) in KNOWN_TL.items():
            gx, gy = self.gate_xy[idx]
            dist_gate = math.hypot(pos[0] - gx, pos[1] - gy)
            s_gate = self.s_light[idx]
            ds = (s_gate - s_now) % self.L
            if 0.0 <= ds <= TL_WINDOW_AHEAD_M:
                tl_active_idx = idx
                tl_active_dist = dist_gate
                tl_active_state = tl_state_str(self.world, self.tl_actor_map.get(idx))
                break  # first gate ahead in window

        tl_obs = {"exists": False, "state": None}
        if tl_active_idx is not None:
            tl_obs = {
                "exists": True,
                "dist_m": None if tl_active_dist is None else round(float(tl_active_dist), 2),
                "state": tl_active_state,
            }

        # ---- Vehicles snapshot: top-2 lead cars (ego lane) + top-2 opposite-lane cars ----
        ego_candidates: List[dict] = []
        opp_candidates: List[dict] = []
        lead_cars: List[dict] = []
        opposite_cars: List[dict] = []

        for v in self.world.get_actors().filter("vehicle*"):
            if v.id == self.ego.id:
                continue

            pl = v.get_transform().location
            pxy = np.array([pl.x, pl.y], dtype=np.float32)

            # use path arclength window first
            s_v, lat_v = frenet_s_lat(pxy, self.A, self.AB, self.seg_len, self.s_before)
            ds = (s_v - s_now) % self.L
            if ds <= 0.0 or ds > SAFE_DISTANCE_CONST:
                continue  # only forward and within look-ahead

            # must be physically in front in ego frame
            p_rel_e = R_we @ (pxy - pos)
            if p_rel_e[0] <= 0.0:
                continue

            vv = v.get_velocity()
            v2 = np.array([vv.x, vv.y], dtype=np.float32)
            v_rel_e = R_we @ (v2 - v_w)

            gap_long_m   = float(p_rel_e[0])
            gap_lat_m = float(p_rel_e[1])
            rel_long_mps = float(v_rel_e[0])
            ttc_s        = ttc_longitudinal(gap_long_m, rel_long_mps, cap=20.0)
            thw_s        = (gap_long_m / max(0.5, speed)) if gap_long_m > 0 else None

            cand = {
                "gap_long_m": round(gap_long_m, 2),
                "gap_lat_m":  round(gap_lat_m, 2),
                "rel_long_mps": round(rel_long_mps, 2),
                "ttc_s": None if ttc_s is None else round(ttc_s, 2),
                "thw_s": None if thw_s is None else round(thw_s, 2),
                "lat_frenet_m": round(float(lat_v), 2),  # right-positive for policy/debug
                "_sort_key": gap_long_m,
            }

            # Lane band classification via Frenet lateral
            if -LANE_HALF <= lat_v <= LANE_HALF:                  # ego lane [-2, +2]
                ego_candidates.append(cand)
            elif OPP_LANE_LEFT <= lat_v < OPP_LANE_RIGHT:         # opposite lane [-6, -2)
                opp_candidates.append(cand)
            # else: ignore vehicles outside the two 4m lanes

        if ego_candidates:
            ego_candidates.sort(key=lambda d: d["_sort_key"])
            lead_cars = [{k: v for k, v in d.items() if k != "_sort_key"}  # strip helper
                        for d in ego_candidates[:MAX_LEAD_VEHICLES]]

        if opp_candidates:
            opp_candidates.sort(key=lambda d: d["_sort_key"])
            opposite_cars = [{k: v for k, v in d.items() if k != "_sort_key"}
                            for d in opp_candidates[:MAX_OPPOSITE_VEHICLES]]
       

        # ---- Pedestrians snapshot list (both lanes) ----
        ped_list: List[dict] = []
        for p in self.world.get_actors().filter("walker.pedestrian*"):
            pl  = p.get_transform().location
            pxy = np.array([pl.x, pl.y], dtype=np.float32)

            # forward along-path window (use path arclength, same as before)
            s_p, _ = frenet_s_lat(pxy, self.A, self.AB, self.seg_len, self.s_before)
            ds = (s_p - s_now) % self.L
            if not (0.0 < ds <= PED_DS_MAX):
                continue

            pv   = p.get_velocity()
            pv2  = np.array([pv.x, pv.y], dtype=np.float32)

            # ego-frame relative pos/vel
            p_rel_e = R_we @ (pxy - pos)
            v_rel_e = R_we @ (pv2 - v_w)

            gap_long_m  = float(p_rel_e[0])
            gap_lat_m   = float(p_rel_e[1])    # + = right, − = left
            rel_lat_mps = float(v_rel_e[1])    # + = moving right


            # 1) In-lane: [-2, +2]
            if -LANE_HALF <= gap_lat_m <= LANE_HALF:
                ped_list.append({
                    "lane": "ego",
                    "state": "in_lane",
                    "gap_long_m": round(gap_long_m, 2),
                    "gap_lat_m":  round(gap_lat_m, 2),
                    "rel_lat_mps": round(rel_lat_mps, 2),
                    "t_enter_lane_s": 0.0,
                    "side": "right" if gap_lat_m > 0 else ("left" if gap_lat_m < 0 else "center"),
                })
                continue

            # 2) Approaching bands: [-3, -2) and (2, 3]
            in_left_band  = (PED_LAT_LEFT  <= gap_lat_m < -LANE_HALF)
            in_right_band = (LANE_HALF < gap_lat_m <= PED_LAT_RIGHT)
            if in_left_band or in_right_band:
                # must be moving toward the lane center (0): sign test + minimum speed
                moving_toward = (rel_lat_mps * gap_lat_m) < 0.0 and abs(rel_lat_mps) >= LAT_V_MIN
                if not moving_toward:
                    continue

                # time-to-edge of our lane boundary
                target_edge = (-LANE_HALF) if in_left_band else (LANE_HALF)
                t_enter = _time_to_target_lat(target_edge, gap_lat_m, rel_lat_mps)
                if (t_enter is None) or not (0.0 < t_enter <= CROSS_HORIZON_S):
                    continue

                ped_list.append({
                    "lane": "approach",
                    "state": "approaching_lane",
                    "gap_long_m": round(gap_long_m, 2),
                    "gap_lat_m":  round(gap_lat_m, 2),
                    "rel_lat_mps": round(rel_lat_mps, 2),
                    "t_enter_lane_s": round(t_enter, 2),
                    "side": "right" if gap_lat_m > 0 else "left",
                })

        # ---- Final OBS dict ----
        obs = {
            "speed_mps": round(speed, 2),
            "yaw_rate_rps": round(yaw_rate, 3),
            "speed_hist4":        [round(x, 2) for x in list(self.obs_speed_hist)],
            "lateral_hist4":      [round(x, 3) for x in list(self.obs_lateral_hist)],
            "yaw_error_hist4":    [round(x, 3) for x in list(self.obs_yaw_err_hist)],
            "steer_cmd_hist4":    [round(x, 3) for x in list(self.obs_steer_hist)],
            "throttle_cmd_hist4": [round(x, 3) for x in list(self.obs_throttle_hist)],
            "brake_cmd_hist4":    [round(x, 3) for x in list(self.obs_brake_hist)],
            "traffic_light": tl_obs,
            "lead_cars": lead_cars,           # top-2 in ego lane
            "opposite_cars": opposite_cars,   # top-2 in opposite lane
            "pedestrians": ped_list,
        }
        return obs




    def run_episode(self):
        """Runs until loop complete / collision / timeout. Returns (success_bool, fitness_float, metrics_dict)."""
        spec = self.world.get_spectator()
        start_time = time.time()

        while True:
            self._tick()

            # Chase cam (optional)
            ego_tf = self.ego.get_transform()
            spec_tf = carla.Transform(
                ego_tf.location + carla.Location(z=30.0) + ego_tf.get_forward_vector() * -12.0,
                carla.Rotation(pitch=-50.0, yaw=ego_tf.rotation.yaw)
            )
            spec.set_transform(spec_tf)

            control = self.agent.run_step()
            self.ego.apply_control(control)
            self._last_control = control
            obs = self.get_obs(control)

            # logs
            self.total_steps += 1
            self.positions.append((ego_tf.location.x, ego_tf.location.y))
            self.steerings.append(float(control.steer))
            v = self.ego.get_velocity()
            spd_kmh = 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
            self.speeds_kmh.append(spd_kmh)

            # harder pedestrians: periodic retargets (cooldown + TTL + side-hit guard)
            # harder pedestrians: ego-aware perpendicular crossings (cooldown + TTL + stuck guard)
         # harder pedestrians: aggressive-but-fair crossings without stutter (no navmesh calls here)
            # harder pedestrians: aggressive crossings with TTL + off-lane dwell (no ping-pong) + debug prints
            if self.harder_peds and (time.time() - self.last_ped_retarget) >= self.ped_cross_every_s:
                now = time.time()

                # --- state dicts (create if missing; minimal change) ---
                if not hasattr(self, "_ped_ready_at"):      self._ped_ready_at = {}
                if not hasattr(self, "_ped_lane_enter_at"): self._ped_lane_enter_at = {}
                if not hasattr(self, "_ped_offlane_at"):    self._ped_offlane_at = {}
                if not hasattr(self, "_ped_last_goal_xy"):  self._ped_last_goal_xy = {}

                # ensure dwell knob exists (minimal default if not in __init__)
                if not hasattr(self, "ped_offlane_dwell_s"):
                    self.ped_offlane_dwell_s = 2.5

                # --- ego once ---
                ego_tf = self.ego.get_transform()
                pos = np.array([ego_tf.location.x, ego_tf.location.y], dtype=np.float32)
                yaw_rad = math.radians(ego_tf.rotation.yaw)
                R_we = rot_world_to_ego(yaw_rad)
                v = self.ego.get_velocity()
                ego_speed = math.hypot(v.x, v.y)

                # send goal only if changed > tol (prevents controller thrash)
                def _send_goal_once(wid, ctrl, wloc, target_xy, vmin=1.2, vmax=2.0, tol=0.35):
                    last = self._ped_last_goal_xy.get(wid)
                    if last is not None:
                        dx = target_xy[0] - last[0]; dy = target_xy[1] - last[1]
                        if (dx*dx + dy*dy) <= (tol*tol):
                            return False
                    ctrl.go_to_location(carla.Location(x=float(target_xy[0]), y=float(target_xy[1]), z=wloc.z))
                    ctrl.set_max_speed(random.uniform(vmin, vmax))
                    self._ped_last_goal_xy[wid] = (float(target_xy[0]), float(target_xy[1]))
                    return True

                # debug counters
                c_inlane = c_retreat = c_cross = c_skipped = c_exitlean = 0

                for walker, ctrl in self.ped_pairs:
                    try:
                        if not walker or not walker.is_alive:  continue
                        if not ctrl   or not ctrl.is_alive:    continue

                        wloc = walker.get_location()
                        pxy  = np.array([wloc.x, wloc.y], dtype=np.float32)
                        s_w, lat_w, k, t, proj_xy = project_point_onto_loop(
                            pxy, self.A, self.AB, self.seg_len, self.s_before
                        )

                        # ---------- TTL: if inside lane too long, keep pushing out until they EXIT (no premature clear) ----------
                        if abs(lat_w) <= LANE_HALF:
                            c_inlane += 1
                            enter_at = self._ped_lane_enter_at.get(walker.id)
                            if enter_at is None:
                                self._ped_lane_enter_at[walker.id] = now
                            else:
                                if (now - enter_at) >= self.ped_lane_ttl_s:
                                    seg = self.AB[k]; Ls = float(np.linalg.norm(seg))
                                    if Ls > 1e-6:
                                        n = np.array([-seg[1]/Ls, seg[0]/Ls], dtype=np.float32)
                                        sign_dir = 1.0 if lat_w >= 0.0 else -1.0
                                        target = proj_xy + sign_dir * n * max(self.ped_offlane_offset_m, LANE_HALF + 2.0)
                                        if _send_goal_once(walker.id, ctrl, wloc, (float(target[0]), float(target[1])), vmin=1.2, vmax=1.6):
                                            c_retreat += 1
                                            cd = random.uniform(self.ped_cooldown_min_s, self.ped_cooldown_max_s)
                                            self._ped_ready_at[walker.id] = now + cd
                                            print(f"[RETREAT] wid={walker.id} inlane_for={now-enter_at:.2f}s tgt=({target[0]:.2f},{target[1]:.2f})")
                                    # IMPORTANT: do NOT clear _ped_lane_enter_at here; we keep timing until they truly exit
                            # when inside lane we skip cross logic this tick
                            continue
                        else:
                            # they are OFF-lane now → record off-lane start (dwell) and clear in-lane timer once
                            if walker.id in self._ped_lane_enter_at and self._ped_lane_enter_at[walker.id] is not None:
                                self._ped_offlane_at[walker.id] = now
                                print(f"[EXIT-LANE] wid={walker.id} after {now - self._ped_lane_enter_at[walker.id]:.2f}s in-lane")
                                c_exitlean += 1
                            if walker.id in self._ped_lane_enter_at and self._ped_lane_enter_at[walker.id] is not None:
                                self._ped_offlane_at[walker.id] = now
                                print(f"[EXIT-LANE] wid={walker.id} after {now - self._ped_lane_enter_at[walker.id]:.2f}s in-lane")

                            # clear the in-lane timer
                            if walker.id in self._ped_lane_enter_at:
                                self._ped_lane_enter_at[walker.id] = None

                            # ensure an initial off-lane timestamp exists (for walkers that never entered yet)
                            if walker.id not in self._ped_offlane_at:
                                self._ped_offlane_at[walker.id] = now

                        # ---------- side-hit guard: don't step into nearly stopped ego hugging laterally ----------
                        p_rel_e = R_we @ (pxy - pos)
                        if ego_speed <= 1.0 and abs(p_rel_e[1]) <= 1.5:
                            c_skipped += 1
                            continue

                        # ---------- CROSS TRIGGER: require cooldown + off-lane dwell + start from curb band ----------
                        ready_at  = self._ped_ready_at.get(walker.id, 0.0)
                        t_off = self._ped_offlane_at.get(walker.id, now)   # default=now ⇒ off_since=0 if unknown
                        off_since = 0.0 if t_off is None else (now - t_off)
                        in_start_band = (abs(lat_w) >= (LANE_HALF + 0.5))  # start from sidewalk/curb, not edge

                        if (now >= ready_at) and in_start_band and (off_since >= self.ped_offlane_dwell_s) and (random.random() < self.ped_cross_prob):
                            tx, ty = ped_cross_target_for(wloc, self.path_xy, self.ped_cross_offset_m)
                            if _send_goal_once(walker.id, ctrl, wloc, (tx, ty), vmin=1.2, vmax=1.6):
                                c_cross += 1
                                cd = random.uniform(self.ped_cooldown_min_s, self.ped_cooldown_max_s)
                                self._ped_ready_at[walker.id] = now + cd
                                print(f"[CROSS] wid={walker.id} off_since={off_since:.2f}s lat={lat_w:.2f} prob={self.ped_cross_prob}")

                    except Exception:
                        continue

                print(f"[HARDPEDS] inlane={c_inlane} retreat={c_retreat} exitlane={c_exitlean} cross={c_cross} skipped={c_skipped}")
                self.last_ped_retarget = now





            # termination checks
            if self.collision_flag["flag"]:
                end_reason = "collision"; break
            if self.total_steps >= self.max_steps:
                end_reason = "timeout"; break

            tf = self.ego.get_transform()
            pos = np.array([tf.location.x, tf.location.y], dtype=np.float32)
            s_now, *_ = project_point_onto_loop(pos, self.A, self.AB, self.seg_len, self.s_before)
            prog_m   = wrap_forward_progress(s_now, self.s0, self.L)
            prog_frac = prog_m / max(self.L, 1e-6)
            dist_to_start = float(np.linalg.norm(pos - self.first_pt))
            if (dist_to_start <= self.start_tol_m) and (prog_frac >= self.start_gate_frac):
                end_reason = "loop_complete"
                self.done_path = True
                break

        pos_arr = np.array(self.positions, dtype=np.float32) if self.positions else np.zeros((0,2), dtype=np.float32)
        fit, metrics = fitness_score(
            self.path_xy, pos_arr, self.steerings, self.speeds_kmh,
            total_steps=self.total_steps,
            done_path=self.done_path,
            collision=self.collision_flag["flag"],
            num_red_violations=self.num_red_violations,
            ref_steps=self.ref_steps,
            min_moving_speed=2
        )
        metrics["end_reason"] = end_reason
        success = bool(self.done_path and not self.collision_flag["flag"])
        return success, float(fit), metrics


        # put this helper inside the class (anywhere before cleanup is fine)
    def _destroy_ids(self, ids):
        if not ids:
            return
        try:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        except Exception:
            pass

    def cleanup(self, silent: bool = False):
        """Teardown: flip async, stop AI, destroy actors in safe order, flush, clear refs."""
        # 0) Flip TM + world to async and flush once (prevents stalls during destroy)
        try:
            if self.tm is not None:
                self.tm.set_synchronous_mode(False)
            if self.world is not None:
                s = self.world.get_settings()
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                self.world.apply_settings(s)
                try:
                    self.world.tick()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            # 1) Stop AI control cleanly
            for v in list(getattr(self, "vehicles", [])):
                try:
                    v.set_autopilot(False, self.tm_port_used)
                except Exception:
                    pass
            for c in list(getattr(self, "walker_ctrls", [])):
                try:
                    c.stop()
                except Exception:
                    pass
            try:
                if getattr(self, "col_sensor", None) is not None:
                    try:
                        self.col_sensor.stop()
                    except Exception:
                        pass
            except Exception:
                pass

            # 2) Destroy (controllers -> walkers -> vehicles -> sensors -> ego)
            try:
                self._destroy_ids([c.id for c in getattr(self, "walker_ctrls", [])])
            except Exception:
                pass
            try:
                self._destroy_ids([w.id for w in getattr(self, "walkers", [])])
            except Exception:
                pass
            try:
                self._destroy_ids([v.id for v in getattr(self, "vehicles", [])])
            except Exception:
                pass
            try:
                if getattr(self, "col_sensor", None) is not None:
                    self._destroy_ids([self.col_sensor.id])
            except Exception:
                pass
            try:
                if getattr(self, "ego", None) is not None:
                    self._destroy_ids([self.ego.id])
            except Exception:
                pass

            # 3) Sweep any strays (best-effort)
            try:
                if self.world is not None:
                    self._destroy_ids([a.id for a in self.world.get_actors().filter("controller.ai.walker")])
                    self._destroy_ids([a.id for a in self.world.get_actors().filter("walker.pedestrian*")])
                    self._destroy_ids([a.id for a in self.world.get_actors().filter("sensor.*")])
                    self._destroy_ids([a.id for a in self.world.get_actors().filter("vehicle.*")])
            except Exception:
                pass

            # 4) One final flush (async)
            try:
                if self.world is not None:
                    self.world.tick()
            except Exception:
                pass

            # 5) Optional: report remaining actors
            if not silent:
                try:
                    cnt = len(self.world.get_actors()) if self.world else -1
                    print(f"[CLEAN] actors in world after cleanup: {cnt}")
                except Exception:
                    pass

        finally:
            # 6) Clear local refs
            try:
                self.vehicles = []
                self.walkers = []
                self.walker_ctrls = []
                self.ped_pairs = []
            except Exception:
                pass
            self.ego = None
            self.col_sensor = None
            self.tm = None
            self.world = None
            self.tl_actor_map = {}
            self._in_gate_prev = {}



    # ---------- Internals ----------
    def _set_sync(self, sync: bool, fps: int):
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(sync)
        settings.fixed_delta_seconds = (1.0 / fps) if sync else None
        self.world.apply_settings(settings)
        if self.tm is not None:
            self.tm.set_synchronous_mode(sync)

    def _tick(self):
        # advance one sim step
        frame = self.world.tick() if self.sync else self.world.wait_for_tick()
        # pace to wall-clock if requested
        if self.sync and self.pace_real_time and self._last_tick_time is not None:
            target_dt = 1.0 / max(1, self.fps)
            now = time.perf_counter()
            elapsed = now - self._last_tick_time
            sleep_s = target_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._last_tick_time = time.perf_counter()
        return frame


    def _on_collision(self, evt):
        other = evt.other_actor
        impulse = evt.normal_impulse
        print(f"[COLLISION] with {other.type_id} id={other.id}  impulse=({impulse.x:.2f},{impulse.y:.2f},{impulse.z:.2f})")
        self.collision_flag["flag"] = True
        try:
            self.collision_ids.append(str(other.type_id))
        except Exception:
            pass


    def _jitter(self):
        return random.uniform(-self.jitter_ratio, self.jitter_ratio)

    def _spawn_ego_from_list(self, index: int = 0) -> carla.Actor:
        """Spawn ego at a predefined spawn point from self.spawn_points_xy list."""
        car_bp = None
        for cand in ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2020", "vehicle.audi.a2"]:
            try:
                car_bp = self.bp_lib.find(cand); break
            except Exception:
                pass
        if car_bp is None:
            car_bp = random.choice(self.bp_lib.filter("vehicle.*"))

        # pick spawn location from list
        x, y, z = self.spawn_points_xy[index]
        loc = carla.Location(x=x, y=y, z=z)
        spawn_tf = carla.Transform(loc, carla.Rotation(yaw=0.0))

        ego = self.world.try_spawn_actor(car_bp, spawn_tf)
        if not ego:
            raise RuntimeError(f"Failed to spawn ego at {self.spawn_points_xy[index]}")
        return ego

        

    def _spawn_traffic(self, num_vehicles: int) -> List[carla.Actor]:
        spawned = []
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for tf in spawn_points[:num_vehicles * 2]:
            if len(spawned) >= num_vehicles:
                break
            candidates = [bp for bp in self.bp_lib.filter("vehicle.*")
              if not any(x in bp.id for x in ["carlacola", "firetruck", "ambulance", "bus", "truck"])]
            if not candidates:
                return spawned
            bp = random.choice(candidates)
            veh = self.world.try_spawn_actor(bp, tf)
            if veh is None:
                continue
            veh.set_autopilot(True, self.tm_port_used)
            spawned.append(veh)
        return spawned

    def _configure_harder_cars(self, vehicles: List[carla.Actor]):
        for v in vehicles:
            try:
                self.tm.auto_lane_change(v, True)
                self.tm.random_left_lanechange_percentage(v, self.lane_change_pct)
                self.tm.random_right_lanechange_percentage(v, self.lane_change_pct)
                self.tm.ignore_lights_percentage(v, self.ignore_light_pct)
                # TM: positive slows, negative speeds up
                self.tm.vehicle_percentage_speed_difference(v, int(self.speed_diff_mean))
                self.tm.distance_to_leading_vehicle(v, self.min_follow_dist)
            except RuntimeError:
                pass

    def _spawn_pedestrians_batch(self, n: int):
        # 1) Blueprints
        walker_bps = list(self.bp_lib.filter("walker.pedestrian.*"))
        for bp in walker_bps:
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "pedestrian")
        ctrl_bp = self.bp_lib.find("controller.ai.walker")

        # 2) Pick spawn transforms near the path
        tfs = []
        attempts = 0
        while len(tfs) < n and attempts < n * 30:
            attempts += 1
            loc = self.world.get_random_location_from_navigation()
            if not loc:
                continue
            # project to ego loop, filter by lateral distance
            s, lat, *_ = project_point_onto_loop(
                np.array([loc.x, loc.y], dtype=np.float32),
                self.A, self.AB, self.seg_len, self.s_before
            )
            if abs(lat) <= 25.0:  # only keep spawns within ±25 m of path
                tfs.append(carla.Transform(loc))

        # 3) Spawn walkers
        batch = [carla.command.SpawnActor(random.choice(walker_bps), tf) for tf in tfs]
        res = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in res if not r.error]

        # 4) Enable physics so they interact with cars
        for wid in walker_ids:
            w = self.world.get_actor(wid)
            if not w:
                continue
            try:
                w.set_simulate_physics(True)
                w.enable_gravity(True)
            except Exception:
                pass

        # 5) Spawn controllers
        batch2 = [carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid) for wid in walker_ids]
        res2 = self.client.apply_batch_sync(batch2, True)
        ctrl_ids = [r.actor_id for r in res2 if not r.error]

        # tick to finalize
        self._tick()

        # 6) Start controllers
        speeds = [random.uniform(1.0, 1.6) for _ in ctrl_ids]
        for cid, vmax in zip(ctrl_ids, speeds):
            ctrl = self.world.get_actor(cid)
            if not ctrl:
                continue
            try:
                ctrl.start()
                dest = self.world.get_random_location_from_navigation()
                if dest:
                    ctrl.go_to_location(dest)
                ctrl.set_max_speed(float(vmax))
            except Exception:
                continue

        return walker_ids, ctrl_ids



    def _add_collision_sensor(self, parent: carla.Actor, callback_fn):
        bp = self.bp_lib.find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=parent)
        sensor.listen(callback_fn)
        return sensor

    def _build_global_plan(self, step: int = 10) -> List[Tuple[carla.Waypoint, RoadOption]]:
        wmap = self.world.get_map()
        plan: List[Tuple[carla.Waypoint, RoadOption]] = []
        N = self.path_xy.shape[0]
        step = max(1, step)
        for i in range(0, N, step):
            loc = carla.Location(x=float(self.path_xy[i, 0]), y=float(self.path_xy[i, 1]), z=0.0)
            wp = wmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                plan.append((wp, RoadOption.LANEFOLLOW))
        loc_last = carla.Location(x=float(self.path_xy[-1, 0]), y=float(self.path_xy[-1, 1]), z=0.0)
        wp_last = wmap.get_waypoint(loc_last, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_last:
            plan.append((wp_last, RoadOption.LANEFOLLOW))
        return plan

    def _save_plan_to_txt(self, plan: List[Tuple[carla.Waypoint, RoadOption]], out_file: str):
        with open(out_file, "w") as f:
            for wp, _opt in plan:
                loc = wp.transform.location
                f.write(f"{loc.x:.3f} {loc.y:.3f}\n")

# ------------------------ Main: run 25 episodes ------------------------
def main():
    # Configure once here (no argparse).
    env = CarlaTown01Env(
        host="localhost",
        port=2000,
        tm_port=8000,                  # will fallback if busy
        town="Town01",
        path_file="recorded_path_cleaned.txt",
        sync=True, fps=20,
        target_kmh=25.0,
        num_vehicles=0,
        num_peds=500,
        global_plan_step=1,           # 1 = use all recorded points (denser plan)
        max_seconds=240,
        start_tol_m=2.0, start_gate_frac=0.90,
        save_plan_file="extracted_path.txt",
        harder_cars=True, ignore_light_pct=25, lane_change_pct=80, speed_diff_mean=15, min_follow_dist=4,
        harder_peds=True, ped_cross_prob=0.75, ped_cross_every_s=4.0, ped_cross_offset_m=9.0,
        ref_steps=2200,
        jitter_ratio=0.15,
    )

    successes = 0
    fitnesses = []
    try:
        for ep in range(25):
            print(f"\n=== EPISODE {ep+1}/25 ===")
            env.reset(seed=random.randint(0, 10**9), save_plan_on_ep0=True, ep_index=ep)
            success, fit, metrics = env.run_episode()
            print(f"end_reason={metrics.get('end_reason')}  success={success}  fitness={fit:.4f}")
            successes += int(success)
            fitnesses.append(fit)
            env.cleanup(silent=False)

        sr = successes / 25.0
        avg_fit = float(np.mean(fitnesses)) if fitnesses else 0.0
        print(f"\n=== SUMMARY over 25 episodes ===")
        print(f"Success rate: {sr*100:.1f}%  ({successes}/25)")
        print(f"Avg fitness:  {avg_fit:.4f}")
    finally:
        env.cleanup(silent=False)      
    return avg_fit, metrics

if __name__ == "__main__":
    main()
