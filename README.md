#  TurtleBot3 Q-Learning ile Otonom Hedef Bulma

Bu proje, **Q-Learning** algoritması kullanarak TurtleBot3 robotunun **LIDAR sensörleri** ile çevresini algılayıp, engellerden kaçınarak hedef noktaya (kırmızı daire) ulaşmasını sağlayan bir **Pekiştirmeli Öğrenme** uygulamasıdır.

---

##  **Proje Sonuçları (Özet)**

| Metrik | Değer |
|--------|-------|
| **Toplam Episode** | 3000 |
| **Ortalama Reward (son 50)** | ~400 |
| **En İyi Reward** | ~1000 |
| **Hedefe Ulaşmak Gerekli İçin Minimum Puan** | ~250 |

###  **Eğitim Grafiği**

<img width="820" height="297" alt="Ekran görüntüsü 2026-05-06 161847" src="https://github.com/user-attachments/assets/a91918bf-daf4-400d-82b0-7ec4dbc2606b" />


> **Grafik Yorumu:** Başlangıçta reward'lar negatifken (robot sürekli çarpıyor veya hedefi bulamıyor), eğitim ilerledikçe reward'ların pozitif bölgeye çıktığı ve başarı için gerekli olan 250 reward değerini geçtiği görülüyor. Mavi çizgi her episode'un reward'ını, kırmızı çizgi 20 episode'lık hareketli ortalamayı göstermektedir.

---

##  **Test GIF'i (Robotun Hedefe Ulaşma Animasyonu)**

<img width="1460" height="1484" alt="robot_test_animation" src="https://github.com/user-attachments/assets/cf78c693-a0cd-4362-962a-647fac16bb35" />


> **GIF Yorumu:** Robot, başlangıç noktası olan **(-3.5, -3.5)** konumundan hareket ederek, LIDAR sensörleriyle duvarları algılayıp engellerden kaçınır ve hedef noktası olan **(2.5, 2.5)** konumuna ulaşmaya çalışır. GIF'te robotun izlediği yol (mavi kesikli çizgi), LIDAR ışınları (yeşil/turuncu/kırmızı çizgiler) ve hedefe olan mesafe gösterilmektedir.

---

##  **Q-Learning Algoritması**

### **Matematiksel Formül**

> Q-Learning, **Bellman Denklemi**'ni kullanarak Q-tablosunu günceller:
---
> Q(s,a) ← Q(s,a) + α × [r + γ × maxₐ' Q(s',a') − Q(s,a)]
---

| Parametre | Anlamı | Değeri |
|-----------|--------|--------|
| **Q(s,a)** | s durumunda a aksiyonunun değeri | - |
| **α (alpha)** | Öğrenme hızı | 0.15 |
| **r** | Alınan anlık ödül | -30 ila +100 |
| **γ (gamma)** | Gelecek ödül indirgeme faktörü | 0.98 |
| **maxₐ' Q(s',a')** | Yeni durumdaki en iyi aksiyon değeri | - |

---

##  **State Space (Durum Uzayı)**

Robotun algıladığı durum **3 ana bileşenden** oluşur:

### **1. LIDAR Verileri (7 sensör × 4 seviye)**

Robot, önüne bakan **7 LIDAR ışını** ile engellere olan mesafeyi ölçer. Her ışın **4 seviyeye** ayrılır:

| Seviye | Mesafe | Anlamı |
|--------|--------|--------|
| 0 | < 0.3 m | Çok yakın (acil tehlike) |
| 1 | 0.3 - 0.6 m | Yakın (tehlike) |
| 2 | 0.6 - 1.2 m | Orta mesafe |
| 3 | > 1.2 m | Uzak (güvenli) |

**LIDAR State Boyutu:** 7 × 4 = **28 kombinasyon**

### **2. Hedefe Uzaklık (5 seviye)**

| Seviye | Mesafe | Anlamı |
|--------|--------|--------|
| 0 | < 0.4 m | Çok yakın (hedefe neredeyse ulaştı) |
| 1 | 0.4 - 0.8 m | Yakın |
| 2 | 0.8 - 1.5 m | Orta |
| 3 | 1.5 - 3.0 m | Uzak |
| 4 | > 3.0 m | Çok uzak |

### **3. Hedefe Açı (8 seviye)**

Robotun hedefe doğru dönmesi gereken açı, **8 eşit parçaya** bölünür (her 45°'de bir).

| Açı Aralığı | Seviye |
|-------------|--------|
| -180° ile -135° | 0 |
| -135° ile -90° | 1 |
| -90° ile -45° | 2 |
| -45° ile 0° | 3 |
| 0° ile 45° | 4 |
| 45° ile 90° | 5 |
| 90° ile 135° | 6 |
| 135° ile 180° | 7 |

### **Toplam State Uzayı Boyutu:**
---
Toplam State = LIDAR (7 × 4) × Hedef Mesafe (5) × Hedef Açı (8) = 28 × 5 × 8 = 1.120
---

---

##  **Action Space (Aksiyon Uzayı)**

Robot **5 farklı aksiyon** alabilir:

| Aksiyon | Açısal Hız (rad/s) | Anlamı |
|---------|-------------------|--------|
| 0 | -1.0 | Sert Sola Dönüş |
| 1 | -0.5 | Hafif Sola Dönüş |
| 2 | 0 | Düz Git |
| 3 | +0.5 | Hafif Sağa Dönüş |
| 4 | +1.0 | Sert Sağa Dönüş |

**Doğrusal Hız:** Sabit **0.18 m/s** (hız sabit, sadece yön değiştirir)

---

##  **Ödül Fonksiyonu (Reward Function)**

Robotun aldığı her aksiyon sonrasında bir ödül/ceza alır:

| Durum | Ödül | Açıklama |
|-------|------|----------|
| Duvara çarpma | **-30** | Büyük ceza (ölümcül hata) |
| Hedefe ulaşma | **+100** | Büyük ödül (başarı) |
| Zaman aşımı (600 adım) | **-20** | Orta ceza |
| Yönelim ödülü | 0 - 0.8 | Hedefe doğru dönüş |
| Mesafe ödülü | 0.3 - 6.0 | Hedefe yaklaştıkça artar |
| Engel cezası | -5 ila 0 | Yakın engel varsa ceza |
| Yaşam ödülü | +0.05 | Her adımda (keşfi teşvik) |

---
