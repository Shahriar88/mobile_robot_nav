import math
import heapq
import random
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class MobileRobotNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    __addressid__ = "73686168726961722e666f726861642e65656540676d61696c2e636f6d"
    __deviceid__ = "4d6420536861687269617220466f72686164"
    __portid__ = "53686168726961723838"

    def __init__(
        self,
        render_mode: Optional[str] = None,
        arena_half: float = 6.0,
        obstacle_count: int = 30,
        max_steps: int = 1000,
        dt: float = 1.0 / 30.0,
        max_lin_speed: float = 2.0,
        max_rot_speed: float = 2.0,
        stop_dist: float = 0.40,
        lidar_n_rays: int = 31,
        lidar_fov_deg: float = 120.0,
        lidar_range: float = 2.0,
        robot_length: float = 1.0,
        robot_width: float = 0.5,
        front_clearance: float = 0.15,
        side_clearance: float = 0.10,
        show_safety_outline: bool = True,
        goal_heading_tolerance_deg: float = 12.0,
        clear_front_threshold: float = 0.90,
        grid_resolution: float = 0.10,
        show_astar_path: bool = True,
        planner_mode: str = "astar",          # "astar" or "modified_astar"
        lambda1: float = 1.0,                 # travel cost multiplier
        lambda2: float = 0.0,                 # direction change penalty
        planning_angle_candidates_deg=None,   # e.g. [-90, -60, -30, 0, 30, 60, 90]
        require_rotation_clearance: bool = True,
        window_size: int = 800,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.arena_half = arena_half
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps
        self.dt = dt
        self.max_lin_speed = max_lin_speed
        self.max_rot_speed = max_rot_speed
        self.stop_dist = stop_dist
        self.lidar_n_rays = lidar_n_rays
        self.lidar_fov_deg = lidar_fov_deg
        self.lidar_range = lidar_range

        self.robot_length = robot_length
        self.robot_width = robot_width

        self.front_clearance = front_clearance
        self.side_clearance = side_clearance
        self.show_safety_outline = show_safety_outline

        self.goal_heading_tolerance_deg = goal_heading_tolerance_deg
        self.clear_front_threshold = clear_front_threshold

        self.grid_resolution = grid_resolution
        self.show_astar_path = show_astar_path
        self.planner_mode = planner_mode
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        if planning_angle_candidates_deg is None:
            planning_angle_candidates_deg = [-90, -60, -30, 0, 30, 60, 90]
        self.planning_angle_candidates_deg = planning_angle_candidates_deg
        self.planning_angle_candidates_rad = [math.radians(a) for a in planning_angle_candidates_deg]
        self.require_rotation_clearance = require_rotation_clearance

        self.window_size = window_size
        self.rng = random.Random(seed)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_low = np.array(
            [-arena_half, -arena_half, -math.pi, -2 * arena_half, -2 * arena_half] + [0.0] * lidar_n_rays,
            dtype=np.float32,
        )
        obs_high = np.array(
            [arena_half, arena_half, math.pi, 2 * arena_half, 2 * arena_half] + [1.0] * lidar_n_rays,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.obstacles = []
        self.goal = np.array([4.8, 4.8], dtype=np.float32)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.step_count = 0
        self.prev_dist = None

        self.path_world = []
        self.path_grid = []
        self.current_waypoint_idx = 0

        self.window = None
        self.clock = None

    # ----------------------------
    # Geometry helpers
    # ----------------------------
    def _random_obstacles(self):
        obstacles = []
        start = np.array([0.0, 0.0], dtype=np.float32)
        goal = self.goal.copy()

        for _ in range(self.obstacle_count):
            hx = self.rng.uniform(0.15, 0.55)
            hy = self.rng.uniform(0.15, 0.55)

            while True:
                cx = self.rng.uniform(-self.arena_half + 0.5, self.arena_half - 0.5)
                cy = self.rng.uniform(-self.arena_half + 0.5, self.arena_half - 0.5)

                if (cx - start[0]) ** 2 + (cy - start[1]) ** 2 <= 1.2 ** 2:
                    continue
                if (cx - goal[0]) ** 2 + (cy - goal[1]) ** 2 <= 1.2 ** 2:
                    continue
                break

            obstacles.append((cx, cy, hx, hy))

        return obstacles

    def _robot_corners_world_at(self, x, y, yaw, front_extra=0.0, side_extra=0.0):
        half_l_front = self.robot_length / 2.0 + front_extra
        half_l_rear = self.robot_length / 2.0
        half_w = self.robot_width / 2.0 + side_extra

        local_corners = [
            ( half_l_front,  half_w),
            ( half_l_front, -half_w),
            (-half_l_rear,  -half_w),
            (-half_l_rear,   half_w),
        ]

        c = math.cos(yaw)
        s = math.sin(yaw)

        world_corners = []
        for lx, ly in local_corners:
            wx = x + c * lx - s * ly
            wy = y + s * lx + c * ly
            world_corners.append((wx, wy))

        return world_corners

    def _robot_corners_world(self):
        return self._robot_corners_world_at(self.x, self.y, self.yaw)

    def _robot_safety_corners_world(self):
        return self._robot_corners_world_at(
            self.x,
            self.y,
            self.yaw,
            front_extra=self.front_clearance,
            side_extra=self.side_clearance,
        )

    def _project_polygon(self, axis, points):
        dots = [axis[0] * p[0] + axis[1] * p[1] for p in points]
        return min(dots), max(dots)

    def _overlap_1d(self, a_min, a_max, b_min, b_max):
        return not (a_max < b_min or b_max < a_min)

    def _rect_rect_collision_sat(self, robot_corners, rect):
        cx, cy, hx, hy = rect
        obs_corners = [
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx + hx, cy + hy),
            (cx - hx, cy + hy),
        ]

        axes = []

        for i in range(2):
            p1 = robot_corners[i]
            p2 = robot_corners[(i + 1) % 4]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            axis = (-edge[1], edge[0])
            norm = math.hypot(axis[0], axis[1])
            if norm > 1e-12:
                axes.append((axis[0] / norm, axis[1] / norm))

        axes.append((1.0, 0.0))
        axes.append((0.0, 1.0))

        for axis in axes:
            r_min, r_max = self._project_polygon(axis, robot_corners)
            o_min, o_max = self._project_polygon(axis, obs_corners)
            if not self._overlap_1d(r_min, r_max, o_min, o_max):
                return False

        return True

    def _collision(self, x, y, yaw):
        robot_corners = self._robot_corners_world_at(
            x,
            y,
            yaw,
            front_extra=self.front_clearance,
            side_extra=self.side_clearance,
        )

        for px, py in robot_corners:
            if px < -self.arena_half or px > self.arena_half:
                return True
            if py < -self.arena_half or py > self.arena_half:
                return True

        for rect in self.obstacles:
            if self._rect_rect_collision_sat(robot_corners, rect):
                return True

        return False

    def _rotation_clear_at_cell(self, x, y, yaw_candidates):
        for yaw in yaw_candidates:
            if self._collision(x, y, yaw):
                return False
        return True

    def _cell_has_valid_angle(self, x, y):
        # Standard astar: old behavior
        if self.planner_mode == "astar":
            return not self._collision(x, y, 0.0)

        # Modified astar: try multiple candidate angles
        valid_angles = []
        for yaw in self.planning_angle_candidates_rad:
            if not self._collision(x, y, yaw):
                valid_angles.append(yaw)

        if not valid_angles:
            return False

        if not self.require_rotation_clearance:
            return True

        # Require that robot can rotate through the tested free orientations
        return self._rotation_clear_at_cell(x, y, valid_angles)

    def _ray_aabb_fraction(self, ox, oy, dx, dy, rect, max_range):
        cx, cy, hx, hy = rect
        x_min, x_max = cx - hx, cx + hx
        y_min, y_max = cy - hy, cy + hy

        tmin = -float("inf")
        tmax = float("inf")

        if abs(dx) < 1e-9:
            if ox < x_min or ox > x_max:
                return None
        else:
            tx1 = (x_min - ox) / dx
            tx2 = (x_max - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if abs(dy) < 1e-9:
            if oy < y_min or oy > y_max:
                return None
        else:
            ty1 = (y_min - oy) / dy
            ty2 = (y_max - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < 0 or tmin > tmax:
            return None

        t_hit = tmin if tmin >= 0 else tmax
        if t_hit < 0 or t_hit > max_range:
            return None
        return t_hit / max_range

    def _ray_wall_fraction(self, ox, oy, dx, dy, max_range):
        candidates = []

        if abs(dx) > 1e-9:
            for xw in (-self.arena_half, self.arena_half):
                t = (xw - ox) / dx
                if 0 <= t <= max_range:
                    yw = oy + t * dy
                    if -self.arena_half <= yw <= self.arena_half:
                        candidates.append(t)

        if abs(dy) > 1e-9:
            for yw in (-self.arena_half, self.arena_half):
                t = (yw - oy) / dy
                if 0 <= t <= max_range:
                    xw = ox + t * dx
                    if -self.arena_half <= xw <= self.arena_half:
                        candidates.append(t)

        if not candidates:
            return None
        return min(candidates) / max_range

    def _lidar(self):
        fov = math.radians(self.lidar_fov_deg)
        angles = np.linspace(-fov / 2, fov / 2, self.lidar_n_rays)

        fracs = []
        for a in angles:
            ang = self.yaw + a
            dx = math.cos(ang)
            dy = math.sin(ang)

            best = 1.0

            wall_hit = self._ray_wall_fraction(self.x, self.y, dx, dy, self.lidar_range)
            if wall_hit is not None:
                best = min(best, wall_hit)

            for rect in self.obstacles:
                hit = self._ray_aabb_fraction(self.x, self.y, dx, dy, rect, self.lidar_range)
                if hit is not None:
                    best = min(best, hit)

            fracs.append(best)

        return np.array(fracs, dtype=np.float32)

    # ----------------------------
    # A* helpers
    # ----------------------------
    def _grid_shape(self):
        n = int(round((2 * self.arena_half) / self.grid_resolution)) + 1
        return n, n

    def _world_to_grid(self, x, y):
        gx = int(round((x + self.arena_half) / self.grid_resolution))
        gy = int(round((y + self.arena_half) / self.grid_resolution))
        nrows, ncols = self._grid_shape()
        gx = int(np.clip(gx, 0, ncols - 1))
        gy = int(np.clip(gy, 0, nrows - 1))
        return gx, gy

    def _grid_to_world(self, gx, gy):
        x = gx * self.grid_resolution - self.arena_half
        y = gy * self.grid_resolution - self.arena_half
        return x, y

    def _grid_cell_blocked(self, gx, gy):
        x, y = self._grid_to_world(gx, gy)
        return not self._cell_has_valid_angle(x, y)

    def _build_occupancy_grid(self):
        nrows, ncols = self._grid_shape()
        occ = np.zeros((nrows, ncols), dtype=np.uint8)

        for gy in range(nrows):
            for gx in range(ncols):
                if self._grid_cell_blocked(gx, gy):
                    occ[gy, gx] = 1

        return occ

    def _astar_neighbors(self, gx, gy, occ):
        nrows, ncols = occ.shape

        moves = [
            (-1,  0, 1.0, 0),
            ( 1,  0, 1.0, 1),
            ( 0, -1, 1.0, 2),
            ( 0,  1, 1.0, 3),
            (-1, -1, math.sqrt(2), 4),
            (-1,  1, math.sqrt(2), 5),
            ( 1, -1, math.sqrt(2), 6),
            ( 1,  1, math.sqrt(2), 7),
        ]

        for dx, dy, base_cost, dir_idx in moves:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < ncols and 0 <= ny < nrows and occ[ny, nx] == 0:
                yield nx, ny, base_cost, dir_idx

    def _astar_heuristic(self, a, b):
        d = math.hypot(a[0] - b[0], a[1] - b[1])
        if self.planner_mode == "modified_astar":
            return self.lambda1 * d
        return d

    def _astar_search(self, start_xy, goal_xy):
        occ = self._build_occupancy_grid()

        start_xy_grid = self._world_to_grid(start_xy[0], start_xy[1])
        goal_xy_grid = self._world_to_grid(goal_xy[0], goal_xy[1])

        if occ[start_xy_grid[1], start_xy_grid[0]] == 1 or occ[goal_xy_grid[1], goal_xy_grid[0]] == 1:
            return [], []

        start = (start_xy_grid[0], start_xy_grid[1], -1)

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in visited:
                continue
            visited.add(current)

            cgx, cgy, cdir = current

            if (cgx, cgy) == goal_xy_grid:
                path_states = [current]
                while current in came_from:
                    current = came_from[current]
                    path_states.append(current)
                path_states.reverse()

                path_grid = [(gx, gy) for gx, gy, _ in path_states]
                path_world = [self._grid_to_world(gx, gy) for gx, gy in path_grid]
                return path_grid, path_world

            for nx, ny, base_cost, ndir in self._astar_neighbors(cgx, cgy, occ):
                neighbor = (nx, ny, ndir)

                if self.planner_mode == "modified_astar":
                    turn_penalty = 0.0
                    if cdir != -1 and cdir != ndir:
                        turn_penalty = self.lambda2
                    step_cost = self.lambda1 * base_cost + turn_penalty
                else:
                    step_cost = base_cost

                tentative_g = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._astar_heuristic((nx, ny), goal_xy_grid)
                    heapq.heappush(open_heap, (f, neighbor))

        return [], []

    def plan_astar(self):
        self.path_grid, self.path_world = self._astar_search(
            start_xy=(self.x, self.y),
            goal_xy=(float(self.goal[0]), float(self.goal[1])),
        )
        self.current_waypoint_idx = 0
        return self.path_world

    def astar_action(self):
        if not self.path_world:
            self.plan_astar()

        if not self.path_world:
            return np.array([0.0, 0.0], dtype=np.float32)

        while self.current_waypoint_idx < len(self.path_world):
            wx, wy = self.path_world[self.current_waypoint_idx]
            if math.hypot(wx - self.x, wy - self.y) < max(0.10, 0.5 * self.grid_resolution):
                self.current_waypoint_idx += 1
            else:
                break

        if self.current_waypoint_idx >= len(self.path_world):
            return np.array([0.0, 0.0], dtype=np.float32)

        wx, wy = self.path_world[self.current_waypoint_idx]
        desired_yaw = math.atan2(wy - self.y, wx - self.x)
        heading_err = wrap_to_pi(desired_yaw - self.yaw)

        rel_yaw = clamp(1.5 * (heading_err / math.pi), -1.0, 1.0)

        if abs(heading_err) < math.radians(15):
            rel_forward = 1.0
        elif abs(heading_err) < math.radians(35):
            rel_forward = 0.5
        else:
            rel_forward = 0.0

        return np.array([rel_forward, rel_yaw], dtype=np.float32)

    # ----------------------------
    # Observation / info
    # ----------------------------
    def _get_obs(self):
        goal_dx = self.goal[0] - self.x
        goal_dy = self.goal[1] - self.y
        lidar = self._lidar()
        obs = np.concatenate(
            [np.array([self.x, self.y, self.yaw, goal_dx, goal_dy], dtype=np.float32), lidar]
        )
        return obs.astype(np.float32)

    def _get_info(self):
        dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        return {
            "goal_distance": dist,
            "front_clearance": self.front_clearance,
            "side_clearance": self.side_clearance,
            "path_len": len(self.path_world),
            "waypoint_idx": self.current_waypoint_idx,
            "planner_mode": self.planner_mode,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "planning_angle_candidates_deg": self.planning_angle_candidates_deg,
            "require_rotation_clearance": self.require_rotation_clearance,
        }

    # ----------------------------
    # Micro-step motion
    # ----------------------------
    def _advance_microsteps(self, lin_speed, rot_speed, n_substeps=20):
        sub_dt = self.dt / n_substeps

        x = self.x
        y = self.y
        yaw = self.yaw

        moved = False
        collided = False

        for _ in range(n_substeps):
            trial_yaw = wrap_to_pi(yaw + rot_speed * sub_dt)
            trial_x = x + lin_speed * sub_dt * math.cos(trial_yaw)
            trial_y = y + lin_speed * sub_dt * math.sin(trial_yaw)

            if not self._collision(trial_x, trial_y, trial_yaw):
                x = trial_x
                y = trial_y
                yaw = trial_yaw
                moved = True
                continue

            if not self._collision(x, y, trial_yaw):
                yaw = trial_yaw
                collided = True
                continue

            collided = True
            break

        self.x = x
        self.y = y
        self.yaw = yaw

        return moved, collided

    # ----------------------------
    # Gym API
    # ----------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
    
        if seed is not None:
            self.rng.seed(seed)
    
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.goal = np.array([4.8, 4.8], dtype=np.float32)
    
        manual_obstacles = None
        if options is not None:
            manual_obstacles = options.get("obstacles", None)
    
        if manual_obstacles is not None:
            self.obstacles = manual_obstacles
        else:
            self.obstacles = self._random_obstacles()
    
        self.step_count = 0
        self.prev_dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
    
        self.plan_astar()
    
        obs = self._get_obs()
        info = self._get_info()
    
        if self.render_mode == "human":
            self.render()
    
        return obs, info

    def step(self, action):
        self.step_count += 1

        rel_forward = float(np.clip(action[0], -1.0, 1.0))
        rel_yaw = float(np.clip(action[1], -1.0, 1.0))

        lin_speed = self.max_lin_speed * rel_forward
        rot_speed = self.max_rot_speed * rel_yaw

        moved, collided = self._advance_microsteps(
            lin_speed=lin_speed,
            rot_speed=rot_speed,
            n_substeps=20
        )

        dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        progress = self.prev_dist - dist
        self.prev_dist = dist

        terminated = False
        truncated = self.step_count >= self.max_steps

        reward = 2.0 * progress - 0.01

        if collided and not moved:
            reward -= 1.0
        elif collided:
            reward -= 0.2

        if dist < self.stop_dist:
            reward += 50.0
            terminated = True

        obs = self._get_obs()
        info = self._get_info()
        info["collision"] = collided
        info["moved"] = moved

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ----------------------------
    # Rendering
    # ----------------------------
    def _world_to_screen(self, x, y):
        scale = self.window_size / (2 * self.arena_half)
        sx = (x + self.arena_half) * scale
        sy = (self.arena_half - y) * scale
        return sx, sy

    def _draw_world(self, canvas):
        canvas.fill((25, 25, 25))

        pygame.draw.rect(
            canvas,
            (180, 180, 180),
            pygame.Rect(0, 0, self.window_size, self.window_size),
            4
        )

        scale = self.window_size / (2 * self.arena_half)

        for cx, cy, hx, hy in self.obstacles:
            x0 = (cx - hx + self.arena_half) * scale
            y0 = (self.arena_half - (cy + hy)) * scale
            w = 2 * hx * scale
            h = 2 * hy * scale
            pygame.draw.rect(canvas, (60, 100, 220), pygame.Rect(x0, y0, w, h))

        if self.show_astar_path and len(self.path_world) >= 2:
            pts = [self._world_to_screen(px, py) for px, py in self.path_world]
            pts = [(int(px), int(py)) for px, py in pts]
            pygame.draw.lines(canvas, (255, 120, 0), False, pts, 3)

            if self.current_waypoint_idx < len(self.path_world):
                wx, wy = self.path_world[self.current_waypoint_idx]
                sx, sy = self._world_to_screen(wx, wy)
                pygame.draw.circle(canvas, (255, 220, 120), (int(sx), int(sy)), 6)

        gx, gy = self._world_to_screen(self.goal[0], self.goal[1])
        pygame.draw.circle(canvas, (40, 210, 60), (int(gx), int(gy)), 12)

        lidar = self._lidar()
        fov = math.radians(self.lidar_fov_deg)
        angles = np.linspace(-fov / 2, fov / 2, self.lidar_n_rays)
        rx, ry = self._world_to_screen(self.x, self.y)

        for frac, a in zip(lidar, angles):
            ang = self.yaw + a
            r = frac * self.lidar_range
            ex = self.x + r * math.cos(ang)
            ey = self.y + r * math.sin(ang)
            sx, sy = self._world_to_screen(ex, ey)
            pygame.draw.line(canvas, (90, 90, 90), (int(rx), int(ry)), (int(sx), int(sy)), 1)

        corners_world = self._robot_corners_world()
        corners_screen = [
            (int(px), int(py))
            for px, py in [self._world_to_screen(x, y) for x, y in corners_world]
        ]
        pygame.draw.polygon(canvas, (50, 170, 255), corners_screen)
        pygame.draw.polygon(canvas, (220, 240, 255), corners_screen, 2)

        if self.show_safety_outline:
            safe_corners_world = self._robot_safety_corners_world()
            safe_corners_screen = [
                (int(px), int(py))
                for px, py in [self._world_to_screen(x, y) for x, y in safe_corners_world]
            ]
            pygame.draw.polygon(canvas, (255, 180, 60), safe_corners_screen, 2)

        front_left = corners_screen[0]
        front_right = corners_screen[1]
        pygame.draw.line(canvas, (255, 255, 255), front_left, front_right, 4)

        front_center_x = 0.5 * (corners_world[0][0] + corners_world[1][0])
        front_center_y = 0.5 * (corners_world[0][1] + corners_world[1][1])
        hx, hy = self._world_to_screen(front_center_x, front_center_y)
        pygame.draw.line(canvas, (255, 255, 255), (int(rx), int(ry)), (int(hx), int(hy)), 3)

        font = pygame.font.SysFont("arial", 18)
        dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        text = font.render(
            f"step={self.step_count} dist={dist:.2f} path={len(self.path_world)} "
            f"wp={self.current_waypoint_idx} mode={self.planner_mode} "
            f"l1={self.lambda1:.2f} l2={self.lambda2:.2f}",
            True,
            (240, 240, 240)
        )
        canvas.blit(text, (10, 10))

    def render(self):
        if self.render_mode is None:
            return None

        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("MobileRobotNavEnv")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        self._draw_world(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(canvas)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None