import math
import numpy as np
from src.constants import WORLD_W, WORLD_H, MOON_RADIUS

def generate_cosmos(viewport_w, viewport_h):
    """Generate distant realistic stars into a list of dictionaries."""
    stars = []
    for _ in range(1000): # Wide field
        x = np.random.randint(-viewport_w * 10, viewport_w * 10) 
        y = np.random.randint(-viewport_h * 10, viewport_h * 10)
        brightness = np.random.randint(100, 255)
        stars.append({
            'pos': [x, y], 
            'size': np.random.uniform(0.5, 1.5), 
            'color': (brightness, brightness, brightness)
        })
    return stars

def build_lunar_surface(world, np_random):
    """
    Builds the massive procedural terrain curve, and initializes physical Box2D bounds.
    Returns dictionaries of generated geometry and attributes.
    """
    W = WORLD_W
    H = WORLD_H
    CHUNKS = 100
    height = np_random.uniform(H / 10, H / 4, size=(CHUNKS + 1,))
    chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
    
    # Pad area
    pad_idx = CHUNKS // 2 + np_random.integers(-10, 10)
    helipad_y = height[pad_idx]
    helipad_x1 = chunk_x[pad_idx - 2]
    helipad_x2 = chunk_x[pad_idx + 2]
    moon_center = (W / 2, helipad_y - MOON_RADIUS)

    # Multi-pass smoothing to build cohesive crests and valleys
    smooth_y = list(height)
    for _ in range(3): 
        new_y = [smooth_y[0]]
        for i in range(1, CHUNKS - 1):
            new_y.append(0.33 * (smooth_y[i-1] + smooth_y[i] + smooth_y[i+1]))
        new_y.append(smooth_y[-1])
        smooth_y = new_y
        
    # Flatten the terrain under the pad
    for i in range(max(0, pad_idx - 5), min(CHUNKS, pad_idx + 6)):
        smooth_y[i] = helipad_y

    moon = world.CreateStaticBody(position=(0, 0))
    moon_polys = []

    # Map flat coordinates to spherical arc
    def to_arc(x, h):
        angle = (x - W / 2) / MOON_RADIUS
        px = moon_center[0] + (MOON_RADIUS + h - helipad_y) * math.sin(angle)
        py = moon_center[1] + (MOON_RADIUS + h - helipad_y) * math.cos(angle)
        return px, py

    # Draw continuous terrain mesh chunks
    for i in range(CHUNKS - 1):
        x1, y1 = chunk_x[i], smooth_y[i]
        x2, y2 = chunk_x[i+1], smooth_y[i+1]
        p1 = to_arc(x1, y1)
        p2 = to_arc(x2, y2)
        moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.5)
        moon_polys.append([p1, p2, (p2[0], p2[1] - WORLD_H * 2), (p1[0], p1[1] - WORLD_H * 2)]) 

    # Add unbounded edge chunks so maximum zoom outs never cut off into space
    left_surface = to_arc(-WORLD_W * 10, smooth_y[0])
    left_inner = to_arc(chunk_x[0], smooth_y[0])
    moon_polys.insert(0, [left_surface, left_inner, (left_inner[0], left_inner[1] - WORLD_H * 2), (left_surface[0], left_surface[1] - WORLD_H * 2)])
    
    right_inner = to_arc(chunk_x[-1], smooth_y[-1])
    right_surface = to_arc(WORLD_W * 10, smooth_y[-1])
    moon_polys.append([right_inner, right_surface, (right_surface[0], right_surface[1] - WORLD_H * 2), (right_inner[0], right_inner[1] - WORLD_H * 2)]) 

    return {
        "moon": moon,
        "moon_polys": moon_polys,
        "helipad_y": helipad_y,
        "helipad_x1": helipad_x1,
        "helipad_x2": helipad_x2,
        "moon_center": moon_center
    }
