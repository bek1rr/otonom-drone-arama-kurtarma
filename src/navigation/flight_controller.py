#!/usr/bin/env python3
"""
Uçuş Kontrol Modülü
AirSim arayüzü
"""

import airsim
import numpy as np
import time
from typing import Tuple, Optional


class FlightController:
    """
    İHA Uçuş Kontrolü
    """
    
    def __init__(self, ip: str = "127.0.0.1"):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()
        
        self.home_position = None
        self.is_flying = False
        
        print("[FlightController] AirSim bağlantısı hazır")
    
    def takeoff(self, altitude: float = 20.0, speed: float = 5.0):
        """Kalkış yap"""
        print("[Flight] Kalkış hazırlığı...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        print("[Flight] Kalkış!")
        self.client.takeoffAsync().join()
        
        # İrtifaya çık
        print(f"[Flight] {altitude}m'ye çıkılıyor...")
        self.client.moveToZAsync(-altitude, speed).join()
        
        # Home pozisyonu kaydet
        self.home_position = self.get_position()
        self.is_flying = True
        
        print(f"[Flight] ✓ Kalkış tamamlandı. Home: {self.home_position}")
        return self.home_position
    
    def get_position(self) -> Tuple[float, float, float]:
        """Mevcut pozisyon (NED koordinatları)"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return (pos.x_val, pos.y_val, pos.z_val)
    
    def get_gps(self) -> dict:
        """GPS koordinatları"""
        gps = self.client.getGpsData()
        return {
            'latitude': gps.gnss.geo_point.latitude,
            'longitude': gps.gnss.geo_point.longitude,
            'altitude': gps.gnss.geo_point.altitude
        }
    
    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        speed: float = 5.0,
        timeout: float = 30.0
    ) -> bool:
        """
        Belirli noktaya git
        Returns: True = başarılı, False = zaman aşımı
        """
        self.client.moveToPositionAsync(x, y, z, speed)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos = self.get_position()
            dist = np.sqrt((pos[0]-x)**2 + (pos[1]-y)**2 + (pos[2]-z)**2)
            
            if dist < 1.5:  # 1.5m tolerans
                return True
            
            time.sleep(0.1)
        
        return False  # Zaman aşımı
    
    def rotate_to(self, yaw: float, duration: float = 3.0):
        """Belirli açıya dön"""
        self.client.rotateToYawAsync(yaw, duration).join()
    
    def loiter(
        self,
        center_x: float,
        center_y: float,
        altitude: float,
        radius: float = 3.0,
        duration: float = 10.0,
        callback=None
    ):
        """
        Daire çizerek bekle
        callback: her adımda çağrılacak fonksiyon (image -> None)
        """
        print(f"[Flight] {duration}s loiter...")
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            angle = (elapsed / duration) * 2 * np.pi
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            self.client.moveToPositionAsync(x, y, -altitude, 2.0)
            
            # Callback çağır (görüntü işleme için)
            if callback and step % 5 == 0:
                callback()
            
            step += 1
            time.sleep(0.1)
    
    def land(self):
        """İniş yap"""
        print("[Flight] İniş yapılıyor...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.is_flying = False
        print("[Flight] ✓ İniş tamamlandı")
    
    def get_image(self, camera: str = "front_center") -> Optional[np.ndarray]:
        """Kamera görüntüsü al"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)
            ])
            
            if responses:
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                return img_rgb[:, :, ::-1]  # BGR
        
        except Exception as e:
            print(f"[Flight] Kamera hatası: {e}")
        
        return None
    
    def rtl(self, speed: float = 8.0):
        """Return to Launch"""
        if self.home_position is None:
            print("[Flight] Home pozisyonu bilinmiyor!")
            return
        
        print("[Flight] Eve dönüş...")
        self.move_to(
            self.home_position[0],
            self.home_position[1],
            self.home_position[2],
            speed
        )
        self.land()