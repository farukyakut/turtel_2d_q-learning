# -*- coding: utf-8 -*-
"""
TurtleBot3 DQN TEST - Eğitilmiş Modeli Test Et (Hareketli Hedef)
Model: dqn_moving_goal_ep1500.keras
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math
import random
import time
import os
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

print("=" * 60)
print("🤖 TURTLEBOT3 DQN TESTİ - HAREKETLİ HEDEF")
print("=" * 60)

# ==================== ROBOT ORTAMI (DQN İLE UYUMLU) ====================
class RobotEnv:
    def __init__(self, render=True):
        self.render_mode = render
        self.robot_radius = 0.25
        self.goal_radius = 0.25
        self.lidar_range = 3.0
        self.num_lidar = 12
        self.lidar_angles = np.linspace(-np.pi/2, np.pi/2, self.num_lidar)
        
        # Duvarlar (Robot sol tarafta, sağda küçük duvarlar)
        self.walls = [
            # Dış duvarlar
            (-4, -4, 8, 0.4), (-4, 4, 8, 0.4),
            (-4, -4, 0.4, 8), (4, -4, 0.4, 8),
            # Sağ tarafta küçük duvarlar (robotun sağında)
            (2, -1, 0.8, 0.3),   # Sağ alt
            (2.5, 1, 0.8, 0.3),  # Sağ orta
            (2, 2.5, 0.8, 0.3),  # Sağ üst
            # Sol tarafta hafif engel (robotun solunda)
            (-2.5, 0, 0.3, 1.0), # Sol orta
        ]
        
        # Hedef hareket parametreleri
        self.goal_linear_vel = 0.03
        self.goal_theta = 0.0
        self.goal_change_direction_interval = 100
        self.goal_steps_since_last_change = 0
        self.last_dist = 0.0
        
        # Klasör oluştur
        self.screenshot_dir = "dqn_test_screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        
        if self.render_mode:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 8))
            self.ax_sim = self.fig.add_subplot(111)
            self.fig.suptitle('TurtleBot3 DQN Testi - Hareketli Hedef', fontsize=14, fontweight='bold')
        
        self.reset()
    
    def _get_lidar(self):
        readings = []
        for angle in self.lidar_angles:
            beam_angle = self.robot_theta + angle
            hit_dist = self.lidar_range
            
            for wx, wy, ww, wh in self.walls:
                for t in np.linspace(0, self.lidar_range, 30):
                    px = self.robot_x + t * np.cos(beam_angle)
                    py = self.robot_y + t * np.sin(beam_angle)
                    if (wx - ww/2 <= px <= wx + ww/2) and (wy - wh/2 <= py <= wy + wh/2):
                        if t < hit_dist:
                            hit_dist = t
                        break
            readings.append(hit_dist)
        return np.array(readings)
    
    def _move_goal(self):
        self.goal_steps_since_last_change += 1
        
        if self.goal_steps_since_last_change >= self.goal_change_direction_interval:
            self.goal_theta = random.uniform(-np.pi, np.pi)
            self.goal_steps_since_last_change = 0
        
        new_goal_x = self.goal_x + self.goal_linear_vel * np.cos(self.goal_theta) * 0.1
        new_goal_y = self.goal_y + self.goal_linear_vel * np.sin(self.goal_theta) * 0.1
        
        collision = False
        for wx, wy, ww, wh in self.walls:
            if (wx - ww/2 <= new_goal_x <= wx + ww/2) and (wy - wh/2 <= new_goal_y <= wy + wh/2):
                collision = True
                break
        
        if abs(new_goal_x) > 4.5 or abs(new_goal_y) > 4.5:
            collision = True
        
        if collision:
            self.goal_theta = random.uniform(-np.pi, np.pi)
            self.goal_steps_since_last_change = 0
        else:
            self.goal_x = new_goal_x
            self.goal_y = new_goal_y
    
    def reset(self):
        self.robot_x = -3.5
        self.robot_y = -3.5
        self.robot_theta = 0.0
        
        # Hedef sol üst köşede
        self.goal_x = -2.0
        self.goal_y = 2.0
        
        self.goal_theta = random.uniform(-np.pi, np.pi)
        self.goal_steps_since_last_change = 0
        self.steps = 0
        self.max_steps = 2000
        self.done = False
        self.trajectory_x = [self.robot_x]
        self.trajectory_y = [self.robot_y]
        self.goal_trajectory_x = [self.goal_x]
        self.goal_trajectory_y = [self.goal_y]
        self.total_reward = 0
        self.last_dist = 100.0
        return self._get_state()
    
    def _get_state(self):
        lidar = self._get_lidar()
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx) - self.robot_theta
        angle = (angle + np.pi) % (2*np.pi) - np.pi
        
        goal_vx = self.goal_linear_vel * np.cos(self.goal_theta)
        goal_vy = self.goal_linear_vel * np.sin(self.goal_theta)
        
        state = np.concatenate([lidar, [dist/5.0, angle/np.pi, goal_vx, goal_vy]])
        return state.astype(np.float32)
    
    def step(self, action):
        angular_vel = [-1.0, -0.5, 0, 0.5, 1.0][action]
        linear_vel = 0.18
        self.robot_theta += angular_vel * 0.1
        self.robot_x += linear_vel * np.cos(self.robot_theta) * 0.1
        self.robot_y += linear_vel * np.sin(self.robot_theta) * 0.1
        
        self.steps += 1
        self.trajectory_x.append(self.robot_x)
        self.trajectory_y.append(self.robot_y)
        
        self._move_goal()
        self.goal_trajectory_x.append(self.goal_x)
        self.goal_trajectory_y.append(self.goal_y)
        
        collision = False
        for wx, wy, ww, wh in self.walls:
            closest_x = max(wx - ww/2, min(self.robot_x, wx + ww/2))
            closest_y = max(wy - wh/2, min(self.robot_y, wy + wh/2))
            if np.sqrt((self.robot_x - closest_x)**2 + (self.robot_y - closest_y)**2) < self.robot_radius:
                collision = True
                break
        
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if collision:
            reward = -30
            self.done = True
        elif dist < self.goal_radius + self.robot_radius:
            reward = 100
            self.done = True
        elif self.steps >= self.max_steps:
            reward = -20
            self.done = True
        else:
            if dist < self.last_dist:
                approach_reward = 2.0
            else:
                approach_reward = -0.5
            
            dist_reward = 5.0 / (dist + 0.5)
            
            angle_to_goal = math.atan2(dy, dx) - self.robot_theta
            angle_to_goal = (angle_to_goal + np.pi) % (2*np.pi) - np.pi
            angle_reward = max(-0.5, 1 - (abs(angle_to_goal) / (np.pi/2))) * 0.8
            
            lidar = self._get_lidar()
            min_dist = np.min(lidar)
            obstacle_penalty = 0
            if min_dist < 0.4:
                obstacle_penalty = -5 * (0.4 - min_dist) / 0.4
            elif min_dist < 0.8:
                obstacle_penalty = -2 * (0.8 - min_dist) / 0.4
            
            if dist < 0.8:
                proximity_bonus = (0.8 - dist) * 5
            else:
                proximity_bonus = 0
            
            survival_reward = 0.05
            reward = approach_reward + dist_reward * 0.5 + angle_reward * 0.3 + obstacle_penalty + proximity_bonus + survival_reward
        
        self.last_dist = dist
        self.total_reward += reward
        return self._get_state(), reward, self.done, dist
    
    def render(self, action, distance):
        """Canlı render - RAPOR SAĞ ALT KÖŞEDE"""
        if not self.render_mode:
            return
        
        try:
            self.ax_sim.clear()
            self.ax_sim.set_xlim(-5, 5)
            self.ax_sim.set_ylim(-5, 5)
            self.ax_sim.set_aspect('equal')
            self.ax_sim.grid(True, alpha=0.3)
            self.ax_sim.set_title(f'Robot | Adım: {self.steps} | Reward: {self.total_reward:.1f}', fontsize=11)
            
            # Duvarlar
            for wx, wy, ww, wh in self.walls:
                rect = Rectangle((wx - ww/2, wy - wh/2), ww, wh, color='gray', alpha=0.6)
                self.ax_sim.add_patch(rect)
                # Duvar etiketleri (küçük)
                if ww > 1 or wh > 1:
                    self.ax_sim.text(wx, wy, '🧱', fontsize=8, ha='center', va='center', alpha=0.5)
            
            # Hedef izi
            if len(self.goal_trajectory_x) > 1:
                self.ax_sim.plot(self.goal_trajectory_x, self.goal_trajectory_y, 'r--', linewidth=1, alpha=0.3, label='Hedef İzi')
            
            # Robot izi
            if len(self.trajectory_x) > 1:
                self.ax_sim.plot(self.trajectory_x, self.trajectory_y, 'b--', linewidth=1, alpha=0.4, label='Robot İzi')
            
            # LIDAR
            lidar_meters = self._get_lidar()
            for i, angle in enumerate(self.lidar_angles):
                beam_angle = self.robot_theta + angle
                dist = lidar_meters[i]
                color = 'red' if dist < 0.4 else 'orange' if dist < 0.8 else 'green'
                self.ax_sim.plot([self.robot_x, self.robot_x + dist*np.cos(beam_angle)],
                                 [self.robot_y, self.robot_y + dist*np.sin(beam_angle)], 
                                 color, linewidth=0.8, alpha=0.5)
            
            # Robot
            robot = Circle((self.robot_x, self.robot_y), self.robot_radius, color='blue', fill=False, linewidth=2.5)
            self.ax_sim.add_patch(robot)
            self.ax_sim.arrow(self.robot_x, self.robot_y, 0.5*np.cos(self.robot_theta), 
                              0.5*np.sin(self.robot_theta), head_width=0.2, head_length=0.2, 
                              fc='darkblue', ec='darkblue', linewidth=2)
            self.ax_sim.text(self.robot_x, self.robot_y, '🤖', fontsize=10, ha='center', va='center')
            
            # Hedef
            goal = Circle((self.goal_x, self.goal_y), self.goal_radius, color='red', alpha=0.8, facecolor='red')
            self.ax_sim.add_patch(goal)
            self.ax_sim.text(self.goal_x, self.goal_y, '🎯', fontsize=12, ha='center', va='center')
            
            # Hedefe mesafe
            self.ax_sim.text(self.robot_x, self.robot_y - 0.5, f'{distance:.2f}m', 
                            fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Aksiyon bilgisi (sol üstte)
            action_names = ['Sert Sol', 'Hafif Sol', 'Düz', 'Hafif Sağ', 'Sert Sağ']
            self.ax_sim.text(-4.8, 4.8, f'Aksiyon: {action_names[action]}', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            self.ax_sim.legend(loc='lower left', fontsize=7)
            
            # ========== RAPOR SAĞ ALT KÖŞEDE ==========
            info_text = f"""
╔═══════════════════════════╗
║        📊 DURUM           ║
╠═══════════════════════════╣
║ 🤖 ({self.robot_x:5.2f},{self.robot_y:5.2f}) ║
║ 🎯 ({self.goal_x:5.2f},{self.goal_y:5.2f}) ║
║ 📏 {distance:5.2f} m              ║
║ 💰 {self.total_reward:7.2f}         ║
║ 👣 {self.steps:4d}/{self.max_steps}      ║
╚═══════════════════════════╝
"""
            self.ax_sim.text(4.7, -4.5, info_text, 
                           transform=self.ax_sim.transData,
                           fontsize=9, 
                           verticalalignment='bottom', 
                           horizontalalignment='right',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='blue', linewidth=1.5))
            
            plt.tight_layout()
            plt.pause(0.03)
            
        except Exception as e:
            pass
    
    def save_screenshot(self, action, distance, step):
        """50 adımda bir görsel kaydet"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'DQN Test - Adım {step}', fontsize=12, fontweight='bold')
            
            # Duvarlar
            for wx, wy, ww, wh in self.walls:
                rect = Rectangle((wx - ww/2, wy - wh/2), ww, wh, color='gray', alpha=0.6)
                ax.add_patch(rect)
            
            # Hedef izi
            if len(self.goal_trajectory_x) > 1:
                ax.plot(self.goal_trajectory_x, self.goal_trajectory_y, 'r--', linewidth=1, alpha=0.3, label='Hedef')
            
            # Robot izi
            if len(self.trajectory_x) > 1:
                ax.plot(self.trajectory_x, self.trajectory_y, 'b--', linewidth=1, alpha=0.4, label='Robot')
            
            # LIDAR
            lidar_meters = self._get_lidar()
            for i, angle in enumerate(self.lidar_angles):
                beam_angle = self.robot_theta + angle
                dist = lidar_meters[i]
                color = 'red' if dist < 0.4 else 'orange' if dist < 0.8 else 'green'
                ax.plot([self.robot_x, self.robot_x + dist*np.cos(beam_angle)],
                        [self.robot_y, self.robot_y + dist*np.sin(beam_angle)], 
                        color, linewidth=1, alpha=0.5)
            
            # Robot
            robot = Circle((self.robot_x, self.robot_y), self.robot_radius, color='blue', fill=False, linewidth=3)
            ax.add_patch(robot)
            ax.arrow(self.robot_x, self.robot_y, 0.5*np.cos(self.robot_theta), 
                     0.5*np.sin(self.robot_theta), head_width=0.2, head_length=0.2, 
                     fc='darkblue', ec='darkblue', linewidth=2)
            ax.text(self.robot_x, self.robot_y, '🤖', fontsize=12, ha='center', va='center')
            
            # Hedef
            goal = Circle((self.goal_x, self.goal_y), self.goal_radius, color='red', alpha=0.8, facecolor='red')
            ax.add_patch(goal)
            ax.text(self.goal_x, self.goal_y, '🎯', fontsize=14, ha='center', va='center')
            
            # ========== RAPOR SAĞ ALT KÖŞEDE ==========
            action_names = ['Sert Sol', 'Hafif Sol', 'Düz', 'Hafif Sağ', 'Sert Sağ']
            
            info_text = f"""
╔═══════════════════════════════════╗
║         📊 TEST RAPORU            ║
╠═══════════════════════════════════╣
║ 🤖 ({self.robot_x:5.2f}, {self.robot_y:5.2f})   ║
║ 🎯 ({self.goal_x:5.2f}, {self.goal_y:5.2f})   ║
║ 📏 {distance:5.2f} m                     ║
║ 🎮 {action_names[action]:8}           ║
║ 💰 {self.total_reward:7.2f}                ║
║ 👣 {step:4d}/{self.max_steps}                ║
╚═══════════════════════════════════╝
"""
            ax.text(4.7, -4.5, info_text, 
                   transform=ax.transData,
                   fontsize=9, 
                   verticalalignment='bottom', 
                   horizontalalignment='right',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue', linewidth=1.5))
            
            ax.legend(loc='lower left', fontsize=7)
            plt.tight_layout()
            
            filename = f"{self.screenshot_dir}/dqn_test_step_{step:04d}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📸 Görsel kaydedildi: {filename}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Görsel kaydetme hatası: {e}")


# ==================== DQN TEST ====================
def run_dqn_test(model_path="dqn_moving_goal_ep1500.keras"):
    print("=" * 60)
    print("🧪 DQN MODEL TESTİ - HAREKETLİ HEDEF")
    print("=" * 60)
    
    try:
        model = load_model(model_path)
        print(f"✅ DQN Model yüklendi: {model_path}")
        print(f"📊 Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return
    
    env = RobotEnv(render=True)
    
    class DQNTestAgent:
        def __init__(self, model):
            self.model = model
        
        def act(self, state):
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return np.argmax(q_values[0])
    
    agent = DQNTestAgent(model)
    
    state = env.reset()
    done = False
    
    print("\n🚀 Test başlıyor...")
    print("📌 Robot: Sol tarafta (-3.5, -3.5)")
    print("📌 Hedef: Sol üstte (-2.0, 2.0)")
    print("📌 Sağ tarafta küçük duvarlar")
    print("📌 Hedef hareketli (0.03 m/s - ÇOK YAVAŞ)")
    print("📌 Max Adım: 2000")
    print("📌 Her 50 adımda görsel kaydedilecek")
    print("-" * 50)
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, distance = env.step(action)
        state = next_state
        
        env.render(action, distance)
        
        if env.steps % 50 == 0 and env.steps > 0:
            env.save_screenshot(action, distance, env.steps)
            
            if reward == 100:
                print(f"✅ Adım {env.steps}: HEDEFE ULAŞILDI! Reward: {env.total_reward:.1f}")
            elif reward == -30:
                print(f"❌ Adım {env.steps}: ÇARPIŞMA! Reward: {env.total_reward:.1f}")
            else:
                print(f"📊 Adım {env.steps}: Mesafe={distance:.2f}m | Reward={env.total_reward:.1f}")
        
        time.sleep(0.02)
    
    env.save_screenshot(action, distance, env.steps)
    
    print("-" * 50)
    print(f"\n📊 TEST SONUCU:")
    print(f"   • Toplam Adım: {env.steps}")
    print(f"   • Toplam Reward: {env.total_reward:.2f}")
    print(f"   • Son Mesafe: {distance:.2f}m")
    
    if env.total_reward > 50:
        print("   • ✅ BAŞARILI! Robot hareketli hedefe ulaştı!")
    elif env.total_reward > 0:
        print("   • ⚠️ KISMİ BAŞARI! Daha fazla eğitim gerekebilir")
    else:
        print("   • ❌ BAŞARISIZ! Robot hedefe ulaşamadı")
    
    print(f"\n📸 Görseller '{env.screenshot_dir}' klasörüne kaydedildi.")
    
    plt.ioff()
    plt.show()
    
    return env.total_reward


if __name__ == "__main__":
    MODEL_PATH = "dqn_moving_goal_ep1500.keras"
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model bulunamadı: {MODEL_PATH}")
        print("Mevcut model dosyaları:")
        for f in os.listdir('.'):
            if f.endswith('.keras'):
                print(f"   - {f}")
        MODEL_PATH = input("\nModel dosya adını girin: ").strip()
        if not MODEL_PATH:
            MODEL_PATH = "dqn_moving_goal_final.keras"
    
    run_dqn_test(MODEL_PATH)