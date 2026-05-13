import numpy as np
import gymnasium as gym
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
from gymnasium.envs.box2d.lunar_lander import LunarLander, INITIAL_RANDOM, SCALE, FPS

from src.game.constants import WORLD_W, WORLD_H, MODERN_LANDER_POLY
from src.game.terrain import generate_cosmos, build_lunar_surface
from src.game.renderer import Renderer

class VastSpaceLander(LunarLander):
    """
    Ultra-Realistic Scrolling Lunar Descent Simulation.
    Features a modern 'Starship' style lander and a vast surface.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fuel = 300.0 # Tripled fuel
        self.mission_status = None # 'success' or 'failed'
        self.user_quit = False
        self.user_skip = False
        self.step_count = 0
        self.max_episode_steps = 5000 # Allow full mission duration. Matches fuel capacity for hovering.
        self.success_wait_steps = FPS * 10 if self.render_mode == "human" else int(FPS * 0.4)
        self.success_timer_steps = 0
        self.beacon_state = 0 # For flashing effect in renderer
        
        import src.game.constants as const
        self.stars = generate_cosmos(const.CUSTOM_VIEWPORT_W, const.CUSTOM_VIEWPORT_H)
        self.my_renderer = Renderer(self.render_mode)

        # Update Observation Space for fuel (9 elements)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.custom_prev_shaping = None
        self.user_skip = False
        self.step_count = 0
        self.success_timer_steps = 0
        super().reset(seed=seed, options=options)
        self._rebuild_world_for_realism()
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), {}

    def _rebuild_world_for_realism(self):
        """Build a vast, rugged lunar surface."""
        self._destroy()
        self.gravity = -20.0 # Double gravity for faster descent in the large world
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        from gymnasium.envs.box2d.lunar_lander import ContactDetector
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.custom_prev_shaping = None

        surface_data = build_lunar_surface(self.world, self.np_random)
        self.moon = surface_data["moon"]
        self.moon_polys = surface_data["moon_polys"]
        self.moon_center = surface_data["moon_center"]
        self.helipad_y = surface_data["helipad_y"]
        self.helipad_x1 = surface_data["helipad_x1"]
        self.helipad_x2 = surface_data["helipad_x2"]

        # Create Modern Lander at high altitude
        H = WORLD_H
        W = WORLD_W
        initial_y = H * 0.85
        initial_x = W / 2 + self.np_random.uniform(-W/10, W/10)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in MODERN_LANDER_POLY]),
                density=5.0, friction=0.3, categoryBits=0x0010, maskBits=0x001, restitution=0.0,
            ),
        )
        self.fuel = 300.0 # Reset fuel
        self.mission_status = None
        self.lander.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        # Retractable/Modern Legs (Vertical shock absorbers)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x + i * 12/SCALE, initial_y - 20/SCALE),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(3/SCALE, 8/SCALE)),
                    density=1.0, restitution=0.0, categoryBits=0x0020, maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(i * 12/SCALE, -20/SCALE), localAnchorB=(0, 8/SCALE),
                enableLimit=True, lowerAngle=-0.1, upperAngle=0.1,
            )
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

    def _get_observation(self):
        """State is relative to Landing Pad."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        pad_center = (self.helipad_x1 + self.helipad_x2) / 2
        
        # Normalized state
        import math
        normalized_angle = (self.lander.angle + math.pi) % (2 * math.pi) - math.pi
        
        state = [
            (pos.x - pad_center) / (WORLD_W / 2),
            (pos.y - (self.helipad_y + 21/SCALE)) / (WORLD_H / 2),
            vel.x * 0.2, # Standard LunarLander scale
            vel.y * 0.2, 
            normalized_angle,
            self.lander.angularVelocity * 0.05,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.fuel / 300.0, # 9th element: Fuel awareness
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # User can mark the episode as failed manually.
        if self.user_skip:
            self.user_skip = False
            self.mission_status = 'failed'
            state = self._get_observation()
            info = {'mission_status': self.mission_status, 'fuel': self.fuel}
            return state, -100.0, True, False, info

        # During success showcase window, keep engines off and count down.
        if self.mission_status == 'success' and self.success_timer_steps > 0:
            action = 0

        # 0. Fuel Depletion
        if self.fuel <= 0:
            action = 0 
        
        if action == 2: # Main Engine
            self.fuel -= 0.15 # Slower depletion for larger tank
        elif action in [1, 3]: # Side Engines
            self.fuel -= 0.03
        
        self.fuel = max(0, self.fuel)

        # 1. Base physics/particles
        _, _, _, _, info = super().step(action)
        
        # Explicit particle cleanup
        for obj in self.particles:
            if hasattr(obj, 'ttl'):
                obj.ttl -= 0.1 
        
        # 2. Boost Engine Power
        if self.fuel > 0:
            if action == 2: # Main Engine
                f_v = (0, 12.0 * 20.0) 
                self.lander.ApplyForceToCenter(self.lander.GetWorldVector(f_v), True)
            elif action in [1, 3]: # Side Engines
                f_s = (1.0 if action==1 else -1.0) * (2.0 * 20.0) 
                self.lander.ApplyForceToCenter(self.lander.GetWorldVector((f_s, 0)), True)

        state = self._get_observation()
        
        # 3. Custom Reward (Distance to Pad) - STRENGTHENED SIGNALS
        dist_x = abs(state[0])
        dist_y = abs(state[1])
        
        # Much stronger shaping to guide descent
        # Horizontal positioning + velocity control + angle stability
        shaping = -100 * (dist_x ** 1.5) - 50 * (dist_y ** 1.5) \
                  - 40 * (abs(state[2]) ** 1.5) - 50 * (abs(state[3]) ** 1.5) \
                  - 80 * (abs(state[4]) ** 1.5)
        
        if self.custom_prev_shaping is not None:
            reward = shaping - self.custom_prev_shaping
        else:
            reward = 0
        self.custom_prev_shaping = shaping

        # Higher living cost to prevent indefinite hovering (strengthened from 0.008)
        reward -= 0.05
        
        # Penalty for using thrusters (encourages efficient control)
        if action != 0:
            reward -= 0.05

        # BONUS: Reward controlled descent when near pad (below 0.5 height, moving down)
        approaching_pad = (dist_y < 0.8 and state[3] < -0.05)  # descending from higher up
        if approaching_pad:
            # Reward proportional to descent speed to encourage landing faster
            reward += 0.2 * abs(state[3]) 
        
        # Horizontal correction reward
        if dist_y < 0.5 and dist_x > 0.1:
            if (state[0] > 0 and state[2] < 0) or (state[0] < 0 and state[2] > 0):
                reward += 0.05 # Moving towards center

        # STRONG hover penalty: Not moving much vertically but not landed
        no_leg_contact = (state[6] == 0.0 and state[7] == 0.0)
        near_pad = (dist_x < 0.20 and dist_y < 0.40)
        low_vertical_speed = (abs(state[3]) < 0.08)  # Almost stationary
        if self.mission_status != 'success' and no_leg_contact and near_pad and low_vertical_speed:
            reward -= 1.0  # Much stronger penalty for hovering

        # 4. Success/Failure Detection
        terminated = False
        
        # Check if landed on the pad - RELAXED to allow single leg contact
        any_leg_contact = (state[6] > 0 or state[7] > 0)
        on_pad = (dist_x < 0.15)  # Horizontal tolerance
        safe_vel = (abs(state[2]) < 0.5 and abs(state[3]) < 0.5)  # More lenient velocity threshold
        safe_angle = (abs(state[4]) < 0.6)  # ~34 degrees

        if self.mission_status == 'success':
            self.success_timer_steps -= 1
            reward = 0.0
            if self.success_timer_steps <= 0:
                terminated = True
        # Latch success if ANY leg contacts + on pad + safe conditions
        elif any_leg_contact and on_pad and safe_vel and safe_angle:
            self.mission_status = 'success'
            
            # DYNAMIC SUCCESS REWARDS
            base_reward = 200
            fuel_reward = self.fuel * 0.8 # Strong incentive to save fuel
            speed_reward = (self.max_episode_steps - self.step_count) * 0.05 # Incentive to land fast
            
            reward += (base_reward + fuel_reward + speed_reward)
            self.success_timer_steps = self.success_wait_steps
        # Fail only on crash contact.
        elif self.game_over:
            terminated = True
            self.mission_status = 'failed'
            reward -= 120

        truncated = (self.step_count >= self.max_episode_steps) and (not terminated)
        # Preserve success even if truncation happens during the showcase window
        if truncated and (self.mission_status is None or self.mission_status == 'timeout'):
            self.mission_status = 'timeout'
        info['mission_status'] = self.mission_status
        info['fuel'] = self.fuel
        
        return state, float(reward), terminated, truncated, info

    def render(self):
        return self.my_renderer.render(self)

    def close(self):
        super().close()
        import pygame
        pygame.quit()
