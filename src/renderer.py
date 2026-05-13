import pygame
import numpy as np
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE
from src.constants import (
    CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H,
    DESCENT_STAGE_POLY, ASCENT_STAGE_POLY, COCKPIT_WINDOW_POLY
)

class Renderer:
    def __init__(self, render_mode):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.surf = None
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        self.camera_mode = 'focus'
        self.mouse_dragging = False
        self.last_mouse_pos = None

    def render(self, env):
        if self.render_mode is None: 
            return None

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
            pygame.display.set_caption("LunarLanderRL")
        
        if self.clock is None: 
            self.clock = pygame.time.Clock()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.user_quit = True
                env.game_over = True
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom *= (1.1 if event.y > 0 else 0.9)
                self.zoom = max(0.1, min(self.zoom, 10.0))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                    self.camera_mode = 'manual'
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = (event.pos[0] - self.last_mouse_pos[0]) / self.zoom
                    dy = (event.pos[1] - self.last_mouse_pos[1]) / self.zoom
                    self.camera_x -= dx
                    self.camera_y -= dy
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.camera_mode = 'focus'
                if event.key == pygame.K_q:
                    env.user_quit = True
                    env.game_over = True
                if event.key == pygame.K_s:
                    env.user_skip = True
                if event.key in [pygame.K_a, pygame.K_d, pygame.K_w]:
                    self.camera_mode = 'manual'

        # Keyboard Manual Camera
        keys = pygame.key.get_pressed()
        cam_speed = 15 / self.zoom
        if keys[pygame.K_a]: self.camera_x -= cam_speed
        if keys[pygame.K_d]: self.camera_x += cam_speed
        if keys[pygame.K_w]: self.camera_y -= cam_speed
        if keys[pygame.K_s]: self.camera_y += cam_speed

        # Focus Camera Smoothing
        if self.camera_mode == 'focus' and getattr(env, 'lander', None) is not None:
            target_x = env.lander.position.x * SCALE - CUSTOM_VIEWPORT_W // 2
            target_y = CUSTOM_VIEWPORT_H - env.lander.position.y * SCALE - CUSTOM_VIEWPORT_H // 2
            self.camera_x += (target_x - self.camera_x) * 0.1
            self.camera_y += (target_y - self.camera_y) * 0.1

        def to_screen(x, y):
            cx, cy = self.camera_x + CUSTOM_VIEWPORT_W // 2, self.camera_y + CUSTOM_VIEWPORT_H // 2
            screen_x = (x - cx) * self.zoom + CUSTOM_VIEWPORT_W // 2
            screen_y = (y - cy) * self.zoom + CUSTOM_VIEWPORT_H // 2
            return int(screen_x), int(screen_y)

        self.surf = pygame.Surface((CUSTOM_VIEWPORT_W, CUSTOM_VIEWPORT_H))
        self.surf.fill((2, 2, 8)) # Deep space
        
        # 1. Stars
        for star in getattr(env, 'stars', []):
            sx, sy = to_screen(star['pos'][0] - self.camera_x * 0.1, star['pos'][1] - self.camera_y * 0.1)
            if 0 <= sx <= CUSTOM_VIEWPORT_W and 0 <= sy <= CUSTOM_VIEWPORT_H:
                pygame.draw.circle(self.surf, star['color'], (sx, sy), max(1, int(star['size'] * self.zoom)))

        # 2. Lunar Surface Detail (Mountains/Terrain)
        for p in getattr(env, 'moon_polys', []):
            screen_poly = [to_screen(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in p]
            if any(-500 < x < CUSTOM_VIEWPORT_W + 500 for x, _ in screen_poly):
                pygame.draw.polygon(self.surf, (30, 30, 32), screen_poly)
                pygame.draw.line(self.surf, (110, 110, 115), screen_poly[0], screen_poly[1], max(1, int(1 * self.zoom)))

        # 3. Landing Pad with Flashing Beacons
        ps1 = to_screen(env.helipad_x1 * SCALE, CUSTOM_VIEWPORT_H - env.helipad_y * SCALE)
        ps2 = to_screen(env.helipad_x2 * SCALE, CUSTOM_VIEWPORT_H - env.helipad_y * SCALE)
        
        # Draw the main pad line (Strong Green)
        pygame.draw.line(self.surf, (0, 255, 100), ps1, ps2, max(1, int(4 * self.zoom)))
        
        # Draw Beacons (White Pulsing Lights)
        env.beacon_state = (env.beacon_state + 1) % 60
        pulse = 0.5 + 0.5 * np.sin(env.beacon_state * (np.pi / 30)) # Pulsing factor 0 to 1
        
        beacon_color = (255, 255, 255) # White Beacon
        for pos in [ps1, ps2]:
            # Dynamic size relative to zoom and pulse
            glow_radius = int((15 + 10 * pulse) * self.zoom)
            bulb_radius = int((4 + 2 * pulse) * self.zoom)
            
            # Draw light glow
            glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*beacon_color, int(100 * pulse)), (glow_radius, glow_radius), glow_radius)
            self.surf.blit(glow, (pos[0] - glow_radius, pos[1] - glow_radius))
            
            # Draw beacon bulb
            pygame.draw.circle(self.surf, beacon_color, (pos[0], pos[1]), max(1, bulb_radius))

        # 4. Exhaust Particles (High Fidelity)
        for obj in getattr(env, 'particles', []):
            ttl = getattr(obj, 'ttl', 0)
            if ttl <= 0: continue
            
            pos = obj.transform * (0, 0)
            px, py = to_screen(pos[0] * SCALE, CUSTOM_VIEWPORT_H - pos[1] * SCALE)
            
            is_main = False
            c1 = getattr(obj, 'color1', (0,0,0))
            if isinstance(c1, (list, tuple)) and len(c1) > 0:
                if c1[0] > 0.5: is_main = True

            if is_main:
                color = (int(200 * ttl), int(220 + 35 * ttl), 255)
                size = int((2.2 - ttl) * 7 * self.zoom)
            else:
                color = (255, int(180 + 75 * ttl), int(100 * ttl))
                size = int((1.8 - ttl) * 4 * self.zoom)

            if size > 0:
                glow_size = size * 3
                s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, int(80 * ttl)), (glow_size, glow_size), glow_size)
                self.surf.blit(s, (px - glow_size, py - glow_size))
                pygame.draw.circle(self.surf, (255, 255, 255, int(255 * ttl)), (px, py), max(1, size // 2))

        # 5. NASA Lander Rendering
        if getattr(env, 'lander', None) is not None:
            trans = env.lander.transform
            def get_screen_poly(poly):
                world_poly = [trans * (v[0]/SCALE, v[1]/SCALE) for v in poly]
                return [to_screen(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in world_poly]

            # Draw Descent Stage
            pygame.draw.polygon(self.surf, (180, 150, 40), get_screen_poly(DESCENT_STAGE_POLY))
            pygame.draw.aalines(self.surf, (220, 190, 80), True, get_screen_poly(DESCENT_STAGE_POLY))

            # Draw Ascent Stage
            pygame.draw.polygon(self.surf, (130, 135, 140), get_screen_poly(ASCENT_STAGE_POLY))
            pygame.draw.aalines(self.surf, (200, 205, 210), True, get_screen_poly(ASCENT_STAGE_POLY))

            # Draw Cockpit Window
            pygame.draw.polygon(self.surf, (30, 40, 60), get_screen_poly(COCKPIT_WINDOW_POLY))

            # Draw Legs
            for leg in env.legs:
                for f in leg.fixtures:
                    l_trans = f.body.transform
                    path = [l_trans * v for v in f.shape.vertices]
                    screen_path = [to_screen(v[0] * SCALE, CUSTOM_VIEWPORT_H - v[1] * SCALE) for v in path]
                    pygame.draw.polygon(self.surf, (90, 95, 100), screen_path)

        # 6. HUD and Overlays
        ui_font = pygame.font.SysFont('Consolas', 20, bold=True)
        fuel_rect = pygame.Rect(CUSTOM_VIEWPORT_W - 220, 20, 200, 25)
        pygame.draw.rect(self.surf, (40, 40, 45), fuel_rect, border_radius=5)
        
        fuel_pct = getattr(env, 'fuel', 0) / 300.0
        if fuel_pct > 0.5: f_col = (0, 255, 100)
        elif fuel_pct > 0.2: f_col = (255, 200, 0)
        else: f_col = (255, 50, 50)
        
        pygame.draw.rect(self.surf, f_col, (fuel_rect.x, fuel_rect.y, int(fuel_rect.width * fuel_pct), fuel_rect.height), border_radius=5)
        txt_fuel = ui_font.render(f"FUEL: {int(getattr(env, 'fuel', 0))}L", True, (255, 255, 255))
        self.surf.blit(txt_fuel, (fuel_rect.x + 50, fuel_rect.y + 2))

        if getattr(env, 'mission_status', None):
            overlay = pygame.Surface((CUSTOM_VIEWPORT_W, 100), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.surf.blit(overlay, (0, CUSTOM_VIEWPORT_H // 2 - 50))
            
            msg = "TOUCHDOWN CONFIRMED" if env.mission_status == 'success' else "VESSEL DESTROYED"
            col = (0, 255, 150) if env.mission_status == 'success' else (255, 50, 50)
            status_font = pygame.font.SysFont('Arial', 48, bold=True)
            txt_status = status_font.render(msg, True, col)
            self.surf.blit(txt_status, (CUSTOM_VIEWPORT_W // 2 - txt_status.get_width() // 2, CUSTOM_VIEWPORT_H // 2 - 30))

        menu_items = ["F: Focus", "WASD or Drag: Move", "Scroll: Zoom", "Q: Quit", "S: Skip"]
        for i, text in enumerate(menu_items):
            self.surf.blit(ui_font.render(text, True, (150, 150, 160)), (20, 20 + i * 25))

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            self.clock.tick(FPS)
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))
