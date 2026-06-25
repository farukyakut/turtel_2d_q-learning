# -*- coding: utf-8 -*-
"""
TurtleBot3 Q-Learning Test - BASİT VE ÇALIŞAN (50 Adımda Görsel)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math
import pickle
import os
import time
import warnings

warnings.filterwarnings('ignore')

# ==================== EĞİTİMDEKİ RobotEnv İLE AYNI ====================
class RobotEnv:
    def __init__(self, render=True):
        self.render_mode = render
        self.robot_radius = 0.25
        self.goal_radius = 0.25
        self.lidar_range = 3.0
        self.num_lidar = 7
        self.lidar_angles = np.linspace(-np.pi/2, np.pi/2, self.num_lidar)
        
        self.walls = [
            (-4, -4, 8, 0.4), (-4, 4, 8, 0.4),
            (-4, -4, 0.4, 8), (4, -4, 0.4, 8),
            (-2, -1, 1.2, 0.3), (1, 1, 1.2, 0.3),
            (0, -2, 0.3, 1.2), (-1, 2, 0.3, 1.0)
        ]
        
        # Klasör oluştur
        self.screenshot_dir = "test_screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
        
        if self.render_mode:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 7))
            self.ax = self.fig.add_subplot(111)
            self.fig.suptitle('TurtleBot3 Q-Learning Testi', fontsize=14, fontweight='bold')
        
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
    
    def _discretize_lidar(self, lidar_meters):
        discretized = []
        for dist in lidar_meters:
            if dist < 0.3:
                discretized.append(0)
            elif dist < 0.6:
                discretized.append(1)
            elif dist < 1.2:
                discretized.append(2)
            else:
                discretized.append(3)
        return tuple(discretized)
    
    def _discretize_goal(self, dist, angle):
        if dist < 0.4:
            dist_bucket = 0
        elif dist < 0.8:
            dist_bucket = 1
        elif dist < 1.5:
            dist_bucket = 2
        elif dist < 3.0:
            dist_bucket = 3
        else:
            dist_bucket = 4
        
        angle_normalized = angle / np.pi
        angle_bucket = min(7, max(0, int((angle_normalized + 1) * 4)))
        return dist_bucket, angle_bucket
    
    def reset(self):
        self.robot_x = -3.5
        self.robot_y = -3.5
        self.robot_theta = 0.0
        self.goal_x = 2.5
        self.goal_y = 2.5
        self.steps = 0
        self.max_steps = 600
        self.done = False
        self.trajectory_x = [self.robot_x]
        self.trajectory_y = [self.robot_y]
        self.total_reward = 0
        return self._get_discretized_state()
    
    def _get_discretized_state(self):
        lidar_meters = self._get_lidar()
        lidar_state = self._discretize_lidar(lidar_meters)
        
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx) - self.robot_theta
        angle = (angle + np.pi) % (2*np.pi) - np.pi
        
        dist_bucket, angle_bucket = self._discretize_goal(dist, angle)
        
        return lidar_state + (dist_bucket, angle_bucket)
    
    def step(self, action):
        angular_vel = [-1.0, -0.5, 0, 0.5, 1.0][action]
        linear_vel = 0.18
        self.robot_theta += angular_vel * 0.1
        self.robot_x += linear_vel * np.cos(self.robot_theta) * 0.1
        self.robot_y += linear_vel * np.sin(self.robot_theta) * 0.1
        self.steps += 1
        self.trajectory_x.append(self.robot_x)
        self.trajectory_y.append(self.robot_y)
        
        # Çarpışma kontrolü
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
            angle_to_goal = math.atan2(dy, dx) - self.robot_theta
            angle_to_goal = (angle_to_goal + np.pi) % (2*np.pi) - np.pi
            angle_reward = max(-0.5, 1 - (abs(angle_to_goal) / (np.pi/2))) * 0.8
            dist_reward = 3.0 / (dist + 0.5)
            
            lidar = self._get_lidar()
            min_dist = np.min(lidar)
            obstacle_penalty = 0
            if min_dist < 0.4:
                obstacle_penalty = -5 * (0.4 - min_dist) / 0.4
            elif min_dist < 0.8:
                obstacle_penalty = -1 * (0.8 - min_dist) / 0.4
            
            survival_reward = 0.05
            reward = angle_reward + dist_reward + obstacle_penalty + survival_reward
        
        self.total_reward += reward
        return self._get_discretized_state(), reward, self.done, dist
    
    def render(self, action, distance):
        """Tek bir render fonksiyonu - her adımda çağrılır"""
        if not self.render_mode:
            return
        
        try:
            self.ax.clear()
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title(f'Robot Simülasyonu | Adım: {self.steps} | Reward: {self.total_reward:.1f}', fontsize=12)
            
            # Duvarlar
            for wx, wy, ww, wh in self.walls:
                rect = Rectangle((wx - ww/2, wy - wh/2), ww, wh, color='gray', alpha=0.6)
                self.ax.add_patch(rect)
            
            # Yol izi
            if len(self.trajectory_x) > 1:
                self.ax.plot(self.trajectory_x, self.trajectory_y, 'b--', linewidth=1, alpha=0.5)
            
            # LIDAR
            lidar_meters = self._get_lidar()
            for i, angle in enumerate(self.lidar_angles):
                beam_angle = self.robot_theta + angle
                dist = lidar_meters[i]
                color = 'red' if dist < 0.4 else 'orange' if dist < 0.8 else 'green'
                self.ax.plot([self.robot_x, self.robot_x + dist*np.cos(beam_angle)],
                             [self.robot_y, self.robot_y + dist*np.sin(beam_angle)], 
                             color, linewidth=1, alpha=0.6)
            
            # Robot
            robot = Circle((self.robot_x, self.robot_y), self.robot_radius, color='blue', fill=False, linewidth=2)
            self.ax.add_patch(robot)
            self.ax.arrow(self.robot_x, self.robot_y, 0.5*np.cos(self.robot_theta), 
                          0.5*np.sin(self.robot_theta), head_width=0.2, head_length=0.2, 
                          fc='darkblue', ec='darkblue')
            
            # Hedef
            goal = Circle((self.goal_x, self.goal_y), self.goal_radius, color='red', alpha=0.8)
            self.ax.add_patch(goal)
            self.ax.text(self.goal_x, self.goal_y, '🎯', fontsize=16, ha='center', va='center')
            
            # Bilgi metni
            self.ax.text(self.robot_x, self.robot_y - 0.55, f'Hedefe: {distance:.2f}m', 
                        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Aksiyon bilgisi
            action_names = ['Sert Sol', 'Hafif Sol', 'Düz', 'Hafif Sağ', 'Sert Sağ']
            self.ax.text(-4.5, 4.5, f'Aksiyon: {action_names[action]}', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.pause(0.03)
            
        except Exception as e:
            pass
    
    def save_screenshot(self, action, distance, step):
        """50 adımda bir görsel kaydet"""
        try:
            # Yeni bir figure oluştur (mevcut olanı bozmadan)
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'TurtleBot3 Test - Adım {step}', fontsize=14, fontweight='bold')
            
            # Duvarlar
            for wx, wy, ww, wh in self.walls:
                rect = Rectangle((wx - ww/2, wy - wh/2), ww, wh, color='gray', alpha=0.6)
                ax.add_patch(rect)
            
            # Yol izi
            if len(self.trajectory_x) > 1:
                ax.plot(self.trajectory_x, self.trajectory_y, 'b--', linewidth=1, alpha=0.5, label='İz')
            
            # LIDAR
            lidar_meters = self._get_lidar()
            for i, angle in enumerate(self.lidar_angles):
                beam_angle = self.robot_theta + angle
                dist = lidar_meters[i]
                color = 'red' if dist < 0.4 else 'orange' if dist < 0.8 else 'green'
                ax.plot([self.robot_x, self.robot_x + dist*np.cos(beam_angle)],
                        [self.robot_y, self.robot_y + dist*np.sin(beam_angle)], 
                        color, linewidth=1.5, alpha=0.7)
            
            # Robot
            robot = Circle((self.robot_x, self.robot_y), self.robot_radius, color='blue', fill=False, linewidth=3)
            ax.add_patch(robot)
            ax.arrow(self.robot_x, self.robot_y, 0.5*np.cos(self.robot_theta), 
                     0.5*np.sin(self.robot_theta), head_width=0.2, head_length=0.2, 
                     fc='darkblue', ec='darkblue', linewidth=2)
            
            # Hedef
            goal = Circle((self.goal_x, self.goal_y), self.goal_radius, color='red', alpha=0.8)
            ax.add_patch(goal)
            ax.text(self.goal_x, self.goal_y, '🎯', fontsize=16, ha='center', va='center')
            
            # Bilgi metni
            ax.text(self.robot_x, self.robot_y - 0.6, f'Hedefe: {distance:.2f}m', 
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Detaylı bilgi paneli
            action_names = ['Sert Sol (-1.0)', 'Hafif Sol (-0.5)', 'Düz (0)', 'Hafif Sağ (+0.5)', 'Sert Sağ (+1.0)']
            
            info_text = f"""
╔══════════════════════════════════════╗
║         📊 TEST DURUM RAPORU         ║
╠══════════════════════════════════════╣
║  🤖 Robot: ({self.robot_x:.2f}, {self.robot_y:.2f})        ║
║  🎯 Hedef: ({self.goal_x:.1f}, {self.goal_y:.1f})          ║
║  📏 Mesafe: {distance:.2f} m                         ║
║  🧭 Açı: {math.degrees(self.robot_theta):.1f}°                    ║
║  🎮 Aksiyon: {action_names[action]}     ║
║  💰 Reward: {self.total_reward:.2f}                    ║
║  👣 Adım: {step} / {self.max_steps}                   ║
╚══════════════════════════════════════╝
"""
            ax.text(-4.8, 4.2, info_text, fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='#E8F4FD', edgecolor='blue', linewidth=1.5))
            
            ax.legend(loc='lower right')
            plt.tight_layout()
            
            # Kaydet
            filename = f"{self.screenshot_dir}/test_step_{step:04d}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📸 Görsel kaydedildi: {filename}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Görsel kaydetme hatası: {e}")


# ==================== TEST ====================
def run_test(model_path="qlearning_model_ep3000.pkl"):
    print("=" * 60)
    print("🤖 TURTLEBOT3 Q-LEARNING TESTİ")
    print("=" * 60)
    
    # Model yükle
    try:
        with open(model_path, 'rb') as f:
            q_table = pickle.load(f)
        print(f"✅ Q-Table yüklendi: {model_path}")
        print(f"📊 Q-Table boyutu: {len(q_table)} durum")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return
    
    env = RobotEnv(render=True)
    
    class TestAgent:
        def __init__(self, q_table):
            self.q_table = q_table
        
        def act(self, state):
            state_key = str(state)
            if state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            return np.random.randint(5)
    
    agent = TestAgent(q_table)
    
    state = env.reset()
    done = False
    
    print("\n🚀 Test başlıyor...")
    print("📌 Her 50 adımda görsel kaydedilecek")
    print("-" * 50)
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, distance = env.step(action)
        state = next_state
        
        # HER ADIMDA RENDER (canlı görüntü)
        env.render(action, distance)
        
        # HER 50 ADIMDA GÖRSEL KAYDET
        if env.steps % 50 == 0 and env.steps > 0:
            env.save_screenshot(action, distance, env.steps)
            
            # Durum mesajı
            if reward == 100:
                print(f"✅ Adım {env.steps}: HEDEFE ULAŞILDI! Reward: {env.total_reward:.1f}")
            elif reward == -30:
                print(f"❌ Adım {env.steps}: ÇARPIŞMA! Reward: {env.total_reward:.1f}")
            else:
                print(f"📊 Adım {env.steps}: Mesafe={distance:.2f}m | Reward={env.total_reward:.1f}")
        
        time.sleep(0.02)
    
    # Final görseli
    env.save_screenshot(action, distance, env.steps)
    
    print("-" * 50)
    print(f"\n📊 TEST SONUCU:")
    print(f"   • Toplam Adım: {env.steps}")
    print(f"   • Toplam Reward: {env.total_reward:.2f}")
    print(f"   • Son Mesafe: {distance:.2f}m")
    
    if env.total_reward > 50:
        print("   • ✅ BAŞARILI! Robot hedefe ulaştı!")
    elif env.total_reward > 0:
        print("   • ⚠️ KISMİ BAŞARI!")
    else:
        print("   • ❌ BAŞARISIZ!")
    
    print(f"\n📸 Görseller '{env.screenshot_dir}' klasörüne kaydedildi.")
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # En son modeli bul
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if model_files:
        latest_model = sorted(model_files)[-1]
        #print(f"📁 En son model: {latest_model}")
        run_test("qlearning_final_model.pkl")
    else:
        run_test("qlearning_final_model.pkl")