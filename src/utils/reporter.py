#!/usr/bin/env python3
"""
Görev Raporlama Modülü
Eşzamanlı veri kaydı ve PDF/HTML harita üretimi
"""

import json
import time
from datetime import datetime
from pathlib import Path
import folium
from fpdf import FPDF

class MissionReporter:
    def __init__(self, output_base: str = "output"):
        self.start_time = datetime.now()
        self.mission_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.mission_dir = Path(output_base) / f"mission_{self.mission_id}"
        self.images_dir = self.mission_dir / "images"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.mission_dir / "targets.json"
        
        self.targets = []
        self.flight_path = []  
        
        print(f"[Reporter] Veri klasörü oluşturuldu: {self.mission_dir.name}")
        
        self._save_json()

    def log_waypoint(self, lat: float, lon: float):
        if not self.flight_path or (self.flight_path[-1] != [lat, lon]):
            self.flight_path.append([lat, lon])

    def log_target(self, target_info: dict, image_filename: str):
        target_info['image_file'] = image_filename
        target_info['timestamp'] = datetime.now().strftime("%H:%M:%S")
        self.targets.append(target_info)
        self._save_json()

    def _save_json(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.targets, f, indent=4, ensure_ascii=False)

    def generate_final_reports(self, home_gps: dict, stats: dict, total_frames: int):
        if not self.targets:
            print("[Reporter] Hiç hedef bulunamadı. Raporlar pas geçiliyor.")
            return

        print("[Reporter] Nihai raporlar derleniyor...")
        self._generate_map(home_gps)
        self._generate_pdf(stats, total_frames)
        print(f"[Reporter] KUSURSUZ! Raporlar şuraya kaydedildi: {self.mission_dir}")

    def _generate_map(self, home_gps: dict):
        m = folium.Map(location=[home_gps['latitude'], home_gps['longitude']], zoom_start=18)
        
        folium.Marker(
            [home_gps['latitude'], home_gps['longitude']],
            popup="KALKIŞ NOKTASI (HOME)",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(m)
        
        if len(self.flight_path) > 1:
            folium.PolyLine(self.flight_path, color="blue", weight=2.5, opacity=0.8).add_to(m)
            
        for idx, t in enumerate(self.targets, 1):
            lat, lon = t['gps']['latitude'], t['gps']['longitude']
            popup_html = f"<b>Hedef #{idx}</b><br>Sınıf: {t['class']}<br>Güven: %{t['confidence']*100:.1f}"
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_html, max_width=200),
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
        m.save(str(self.mission_dir / "map.html"))

    def _generate_pdf(self, stats: dict, total_frames: int):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        
        pdf.cell(0, 10, "PHOENIX RESCUE - GOREV RAPORU", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Gorev ID: {self.mission_id}", ln=True)
        pdf.cell(0, 10, f"Toplam Hedef: {len(self.targets)}", ln=True)
        pdf.cell(0, 10, f"Islenen Frame: {total_frames}", ln=True)
        pdf.cell(0, 10, f"Performans: {stats.get('avg_inference_ms', 0):.1f} ms/frame", ln=True)
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Bulunan Hedefler:", ln=True)
        pdf.set_font("Arial", '', 11)
        
        for idx, t in enumerate(self.targets, 1):
            lat, lon = t['gps']['latitude'], t['gps']['longitude']
            pdf.cell(0, 8, f"{idx}. Sinif: {t['class'].upper()} | Guven: {t['confidence']:.2f} | GPS: {lat:.6f}, {lon:.6f}", ln=True)
            
        pdf.output(str(self.mission_dir / "report.pdf"))
