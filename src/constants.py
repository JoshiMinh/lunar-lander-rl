from gymnasium.envs.box2d.lunar_lander import SCALE

# Realistic Moon Constants
CUSTOM_VIEWPORT_W = 1280
CUSTOM_VIEWPORT_H = 720

# Physical Scaling Constraints
WORLD_W = 5000 / SCALE  # ~166 meters wide
WORLD_H = 3000 / SCALE  # ~100 meters tall
MOON_RADIUS = 3000      # Structural curve arc radius

# NASA Lunar Module Collision Geometry
MODERN_LANDER_POLY = [
    (-8, +18), (+8, +18), (+12, +0), (+14, -18), (-14, -18), (-12, +0)
]

# Graphical Visual Components
DESCENT_STAGE_POLY = [(-12, -18), (12, -18), (15, -5), (10, 5), (-10, 5), (-15, -5)]
ASCENT_STAGE_POLY = [(-8, 5), (8, 5), (10, 15), (6, 22), (-6, 22), (-10, 15)]
COCKPIT_WINDOW_POLY = [(-4, 15), (4, 15), (5, 18), (-5, 18)]
