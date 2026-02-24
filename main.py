#!/usr/bin/env python3
"""
Phoenix Rescue - Ana Sistem
"""

import cv2
import time
import sys
import numpy as np
from pathlib import Path

# Python path ayarla
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Modülleri import et
from perception.detector import ObjectDetector
from navigation.flight_controller import FlightController
from navigation.mission_planner import MissionPlanner
from perception.tracker import ObjectTracker


class PhoenixRescueMission:
    """
    Ana arama kurtarma görevi
    """

    def __init__(
        self,
        altitude: float = 15.0,
        search_size: float = 100.0,
        speed: float = 4.0
        # --- Target Centering ---
    
      ):
        self.altitude = altitude
        self.search_size = search_size
        self.speed = speed

        print("=" * 60)
        print("🚁 PHOENIX RESCUE - Başlatılıyor")
        print("=" * 60)

        self.flight = FlightController()
        self.detector = ObjectDetector(
            model_path=str(project_root / "models" / "yolo11n.onnx"),
            conf_threshold=0.25
        )
        self.tracker = ObjectTracker()
        self.seen_track_ids = set()
        self.track_frame_counter = {}
        self.planner = None
        self.targets_found = []
        self.frame_count = 0

        print("=" * 60)

    def run(self):
        try:
            home = self.flight.takeoff(self.altitude, self.speed)

            self.planner = MissionPlanner(
                center=(home[0], home[1]),
                size=self.search_size,
                altitude=self.altitude
            )
            waypoints = self.planner.generate_zigzag_pattern()
            print(f"[Mission] {len(waypoints)} waypoint oluşturuldu")

            for i, (x, y, z) in enumerate(waypoints):
                print(f"\n[Mission] Waypoint {i+1}/{len(waypoints)}")

                target_found = self.fly_and_scan(x, y, z)

                if target_found:
                    print(f"[Mission] 🎯 Hedef bulundu, sonraki waypoint'e geçiliyor...")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Mission] Kullanıcı durdurdu")
                    break

            self.flight.rtl()

        except KeyboardInterrupt:
            print("\n[Mission] Ctrl+C")
            self.flight.land()
        except Exception as e:
            print(f"\n[HATA] {e}")
            import traceback
            traceback.print_exc()
            self.flight.land()
        finally:
            self.report()
            cv2.destroyAllWindows()

    def fly_and_scan(self, x: float, y: float, z: float) -> bool:

        self.flight.client.moveToPositionAsync(x, y, z, self.speed)

        target_found = False

        while True:

            pos = self.flight.get_position()
            dist = ((pos[0]-x)**2 + (pos[1]-y)**2 + (pos[2]-z)**2) ** 0.5

            if dist < 2.0:
                break

            image = self.flight.get_image()

            if image is not None:

                detections = self.detector.detect(image)
                tracks = self.tracker.update(detections)
                # Track stabilite sayacı
                for t in tracks:
                    if t.id not in self.track_frame_counter:
                        self.track_frame_counter[t.id] = 0
                    self.track_frame_counter[t.id] += 1

                critical_tracks = [
                    t for t in tracks
                    if t.class_name in ['person', 'sports_ball']
                    and self.track_frame_counter.get(t.id, 0) >= 8
                ]

                viz = self.visualize(image, detections, tracks)
                cv2.imshow('Phoenix Rescue', viz)
                cv2.waitKey(1)

                # ✅ Hedef kontrolü DOĞRU yerde
                if critical_tracks:

                    best = max(critical_tracks, key=lambda t: t.confidence)

                    if best.confidence >= 0.30:

                        is_new = self.handle_target(best, image)

                        if is_new:
                            target_found = True

                            self.flight.client.moveToPositionAsync(
                                pos[0], pos[1], pos[2], 0.1
                            )

                            break

            self.frame_count += 1
            time.sleep(0.05)

        return target_found

    def handle_target(self, track, image) -> bool:

        # --- SMART APPROACH + CENTERING ---

        h, w = image.shape[:2]

        x1, y1, x2, y2 = track.bbox
        target_cx = (x1 + x2) / 2
        target_cy = (y1 + y2) / 2

        img_cx = w / 2
        img_cy = h / 2

        error_x = target_cx - img_cx
        error_y = target_cy - img_cy

        bbox_height = y2 - y1

        CENTER_THRESHOLD = 20
        TARGET_CLOSE_SIZE = 220   # yaklaşık 5 metre
        gain_xy = 0.002

        pos = self.flight.get_position()

        offset_x = -error_x * gain_xy
        offset_y = -error_y * gain_xy

        # Hedef uzaksa → yaklaş + ortala
        if bbox_height < TARGET_CLOSE_SIZE:

            new_x = pos[0] + offset_x + 1.5
            new_y = pos[1] + offset_y

            self.flight.client.moveToPositionAsync(
                new_x,
                new_y,
                -self.altitude,
                2.0
            )

            print("[Approach] Hedefe yaklaşılıyor ve ortalanıyor...")
            return False

        # Hedef yakın ama ortalı değilse → sadece hizala
        if abs(error_x) > CENTER_THRESHOLD or abs(error_y) > CENTER_THRESHOLD:

            new_x = pos[0] + offset_x
            new_y = pos[1] + offset_y

            self.flight.client.moveToPositionAsync(
                new_x,
                new_y,
                -self.altitude,
                1.5
            )

            print("[Centering] Yakın hedef ortalanıyor...")
            return False

        print("[Target] 5m içinde ve ortalı. Loiter başlıyor...")


        if track.id in self.seen_track_ids:
            print("[Filter] Aynı Track ID tekrar bulundu")
            return False
        
        gps = self.flight.get_gps()

        if not self.is_new_target(gps, min_distance=30.0):
            print(f"[Filter] Aynı hedef tekrar bulundu, loiter yapılmıyor")
            return False

        print(f"\n{'='*60}")
        print(f"🎯 YENİ HEDEF BULUNDU! #{len(self.targets_found) + 1}")
        print(f"{'='*60}")
        print(f"Track ID: {track.id}")
        print(f"Sınıf: {track.class_name}")
        print(f"Güven: {track.confidence:.3f}")
        print(f"GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}")

        target_info = {
            'track_id': track.id,
            'class': track.class_name,
            'confidence': track.confidence,
            'gps': gps,
            'frame': self.frame_count
        }

        self.seen_track_ids.add(track.id)
        self.targets_found.append(target_info)

        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        filename = f"target_{len(self.targets_found):03d}_{int(time.time())}.jpg"
        cv2.imwrite(str(output_dir / filename), image)
        print(f"📷 Kaydedildi: {filename}")

        print("[Target] Loiter yapılıyor...")
        time.sleep(3)

        print(f"{'='*60}\n")
        return True


    def visualize(self, image, detections, tracks):

        viz = image.copy()

        # Sarı: raw detections
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Kırmızı: trackler + ID
        for t in tracks:
            if t.class_name in ['person', 'sports_ball']:

                x1, y1, x2, y2 = map(int, t.bbox)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 3)

                cv2.putText(
                    viz,
                    f"ID {t.id} {t.class_name} {t.confidence:.2f}",
                    (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        cv2.putText(viz, "PHOENIX RESCUE", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(
            viz,
            f"Frame: {self.frame_count} | Targets: {len(self.targets_found)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return viz

    def is_new_target(self, gps: dict, min_distance: float = 15.0) -> bool:

        for t in self.targets_found:

            lat1, lon1 = gps['latitude'], gps['longitude']
            lat2, lon2 = t['gps']['latitude'], t['gps']['longitude']

            dx = (lon1 - lon2) * 111000 * np.cos(np.radians(lat1))
            dy = (lat1 - lat2) * 111000
            dist = np.sqrt(dx**2 + dy**2)

            if dist < min_distance:
                print(f"[Filter] Aynı hedef (mesafe: {dist:.1f}m) - Atlanıyor")
                return False

        return True

    def report(self):

        print("\n" + "=" * 60)
        print("🚁 GÖREV ÖZETİ")
        print("=" * 60)
        print(f"Toplam frame: {self.frame_count}")
        print(f"Bulunan hedef: {len(self.targets_found)}")

        for i, t in enumerate(self.targets_found, 1):
            print(f"  {i}. ID {t['track_id']} {t['class']} @ "
                  f"({t['gps']['latitude']:.6f}, {t['gps']['longitude']:.6f})")

        stats = self.detector.get_stats()
        print(f"\nPerformans: {stats.get('avg_inference_ms', 0):.1f}ms/frame")
        print("=" * 60)


def main():

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("AirSim hazır mı? 3 saniye...")
    time.sleep(3)

    mission = PhoenixRescueMission(
        altitude=15.0,
        search_size=100.0,
        speed=4.0
    )

    mission.run()


if __name__ == '__main__':
    main()