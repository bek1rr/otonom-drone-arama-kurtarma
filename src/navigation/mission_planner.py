#!/usr/bin/env python3
"""
Görev Planlama Modülü
Arama alanı ve rotalar
"""

import numpy as np
from typing import List, Tuple


class MissionPlanner:
    """
    Arama görevi planlayıcı
    """
    
    def __init__(
        self,
        center: Tuple[float, float],
        size: float = 100.0,
        altitude: float = 20.0,
        strip_width: float = 20.0
    ):
        self.center = center
        self.size = size
        self.altitude = altitude
        self.strip_width = strip_width
        
        self.waypoints = []
        self.current_index = 0
    
    def generate_zigzag_pattern(self) -> List[Tuple[float, float, float]]:
        """
        Zig-zag arama pattern'i oluştur
        Returns: [(x, y, z), ...]
        """
        waypoints = []
        num_strips = int(self.size / self.strip_width)
        
        start_x = self.center[0] - self.size / 2
        start_y = self.center[1] - self.size / 2
        
        for i in range(num_strips):
            y = start_y + (i * self.strip_width)
            
            if i % 2 == 0:
                # Soldan sağa
                waypoints.append((start_x, y, -self.altitude))
                waypoints.append((start_x + self.size, y, -self.altitude))
            else:
                # Sağdan sola
                waypoints.append((start_x + self.size, y, -self.altitude))
                waypoints.append((start_x, y, -self.altitude))
        
        self.waypoints = waypoints
        return waypoints
    
    def get_next_waypoint(self) -> Tuple[float, float, float]:
        """Sonraki waypoint'i al"""
        if self.current_index < len(self.waypoints):
            wp = self.waypoints[self.current_index]
            self.current_index += 1
            return wp
        return None
    
    def has_more_waypoints(self) -> bool:
        """Waypoint kaldı mı?"""
        return self.current_index < len(self.waypoints)
    
    def get_progress(self) -> float:
        """İlerleme yüzdesi"""
        if not self.waypoints:
            return 0.0
        return (self.current_index / len(self.waypoints)) * 100