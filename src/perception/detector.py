#!/usr/bin/env python3
"""
Nesne Tespit Modülü
YOLOv11 entegrasyonu
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    """Tespit sonucu"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # 0-1
    class_id: int
    class_name: str
    center: Tuple[int, int]   # Merkez nokta
    
    def get_center(self) -> Tuple[int, int]:
        """Bounding box merkezi"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class ObjectDetector:
    """
    YOLOv11 Nesne Tespiti
    """
    
    # Arama kurtarma için önemli sınıflar
    TARGET_CLASSES = {
        0: 'person',           # İnsan
        32: 'sports_ball',     # Top (işaretleyici)
        24: 'backpack',        # Sırt çantası
        26: 'handbag',         # Çanta
        28: 'suitcase',        # Bavul
    }
    
    def __init__(
        self,
        model_path: str = "models/yolo11n.onnx",
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Model yükle
        self._load_model()
        
        # İstatistik
        self.inference_count = 0
        self.total_time = 0.0
    
    def _load_model(self):
        """ONNX model yükle"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {self.model_path}")
        
        # CPU kullan (daha stabil)
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        print(f"[Detector] Model yüklendi: {self.model_path.name}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü hazırla"""
        # Resize
        img = cv2.resize(image, self.input_size)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(
        self,
        outputs: np.ndarray,
        orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Çıktıyı işle"""
        predictions = np.squeeze(outputs).T
        
        # Confidence filter
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(predictions) == 0:
            return []
        
        # Boxes
        boxes = predictions[:, :4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # xywh -> xyxy
        boxes_xyxy = self._xywh2xyxy(boxes)
        
        # Scale
        scale_x = orig_shape[1] / self.input_size[0]
        scale_y = orig_shape[0] / self.input_size[1]
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )
        
        detections = []
        if len(indices) > 0:
            indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            
            for idx in indices:
                class_id = int(class_ids[idx])
                
                # Sadece hedef sınıfları
                if class_id in self.TARGET_CLASSES:
                    det = Detection(
                        bbox=boxes_xyxy[idx].astype(int),
                        confidence=float(scores[idx]),
                        class_id=class_id,
                        class_name=self.TARGET_CLASSES[class_id],
                        center=(0, 0)  # Sonradan hesaplanacak
                    )
                    # Merkez hesapla
                    x1, y1, x2, y2 = det.bbox
                    det.center = (int((x1+x2)/2), int((y1+y2)/2))
                    detections.append(det)
        
        return detections
    
    def _xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """Kutu formatı dönüşümü"""
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Ana tespit fonksiyonu"""
        import time
        start = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs[0], image.shape[:2])
        
        # İstatistik
        elapsed = time.time() - start
        self.total_time += elapsed
        self.inference_count += 1
        
        return detections
    
    def get_stats(self) -> dict:
        """Performans istatistikleri"""
        if self.inference_count == 0:
            return {}
        
        avg_time = self.total_time / self.inference_count
        return {
            'avg_inference_ms': avg_time * 1000,
            'fps': 1.0 / avg_time if avg_time > 0 else 0,
            'total_frames': self.inference_count
        }