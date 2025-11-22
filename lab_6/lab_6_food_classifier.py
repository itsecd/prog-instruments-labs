import json
import logging
import os
import traceback
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


class FoodClassifier:

    def __init__(
            self,
            model_path: str = "models/classifier/model.h5",
            class_mapping_path: str = "models/classifier/class_mapping.json"
    ):
        self.model = None
        self.class_mapping = {}
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.load_model()

    def load_model(self):
        """Загрузка модели и mapping'а классов."""
        try:
            if os.path.exists(self.model_path):
                with tf.device('/CPU:0'):
                    self.model = tf.keras.models.load_model(
                        self.model_path, compile=False
                    )

                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info(f"✅ TensorFlow model loaded from {self.model_path}")
            else:
                logger.warning(
                    f"❌ Model not found at {self.model_path}, using fallback"
                )
                self.model = None

            self._load_class_mapping()

        except Exception as e:
            logger.error(f"❌ Error loading TensorFlow model: {e}")
            self.model = None
            logger.info("🔄 Using fallback classifier")

    def _load_class_mapping(self):
        """Загрузка mapping'а числовых классов в названия еды."""
        try:
            if os.path.exists(self.class_mapping_path):
                with open(
                        self.class_mapping_path, 'r', encoding='utf-8'
                ) as f:
                    self.class_mapping = json.load(f)
                logger.info(
                    f"✅ Loaded class mapping with {len(self.class_mapping)} classes"
                )
            else:
                self.class_mapping = {
                    str(i): f"food_{i}" for i in range(101)
                }
                logger.info("🔄 Using default class mapping")

        except Exception as e:
            logger.error(f"❌ Error loading class mapping: {e}")
            self.class_mapping = {
                str(i): f"food_{i}" for i in range(101)
            }

    def predict(
            self, image: Image.Image, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Предсказание класса еды."""
        try:
            if self.model is not None:
                return self._predict_with_model(image, top_k)
            else:
                return self._predict_fallback(image, top_k)

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return self._predict_fallback(image, top_k)

    def _predict_with_model(
            self, image: Image.Image, top_k: int
    ) -> List[Dict[str, Any]]:
        """Предсказание с использованием TensorFlow модели."""
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            predictions = predictions[0]

            logger.info("🔍 === DEBUG PREDICTION ANALYSIS ===")
            logger.info(f"🔍 Predictions shape: {predictions.shape}")
            logger.info(f"🔍 Number of classes: {len(predictions)}")
            logger.info(f"🔍 Min confidence: {np.min(predictions):.6f}")
            logger.info(f"🔍 Max confidence: {np.max(predictions):.6f}")
            logger.info(f"🔍 Mean confidence: {np.mean(predictions):.6f}")

            top_10_indices = np.argsort(predictions)[-10:][::-1]
            logger.info("🔍 Top 10 predictions:")

            for i, idx in enumerate(top_10_indices):
                confidence = float(predictions[idx])
                class_id = str(idx)
                class_name = self.class_mapping.get(
                    class_id, f"class_{idx}"
                )
                logger.info(
                    f"🔍 #{i + 1}: idx={idx}, class='{class_name}', "
                    f"confidence={confidence:.6f}"
                )

            if 17 < len(predictions):
                confidence_17 = float(predictions[17])
                class_name_17 = self.class_mapping.get("17", "class_17")
                logger.info(
                    f"🔍 Specific check - idx=17: '{class_name_17}' "
                    f"with confidence={confidence_17:.6f}"
                )

            top_indices = np.argsort(predictions)[-top_k:][::-1]
            results = []

            for idx in top_indices:
                confidence = float(predictions[idx])
                class_id = str(idx)
                class_name = self.class_mapping.get(
                    class_id, f"class_{idx}"
                )

                results.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'class_id': int(idx)
                })

            logger.info(
                f"✅ TensorFlow prediction: {results[0]['class_name']} "
                f"({results[0]['confidence']:.3f})"
            )
            return results

        except Exception as e:
            logger.error(f"❌ TensorFlow prediction error: {e}")
            logger.error(traceback.format_exc())
            return self._predict_fallback(image, top_k)

    def _predict_fallback(
            self, image: Image.Image, top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback предсказание на основе эвристик."""
        main_classes = [
            "apple_pie", "pizza", "hamburger", "sushi", "steak",
            "salad", "chicken_curry", "pasta", "sandwich", "soup"
        ]

        dominant_color = self._get_dominant_color(image)

        if dominant_color == "green":
            predictions = ["salad", "guacamole", "seaweed_salad"]
        elif dominant_color == "brown":
            predictions = ["steak", "chicken_wings", "hamburger"]
        elif dominant_color == "orange":
            predictions = ["pizza", "carrot_cake", "cheese_plate"]
        elif dominant_color == "white":
            predictions = ["rice", "pasta", "bread_pudding"]
        else:
            predictions = main_classes[:top_k]

        results = []
        for i, food_class in enumerate(predictions[:top_k]):
            confidence = max(0.1, 0.9 - i * 0.2)
            results.append({
                'class_name': food_class,
                'confidence': round(confidence, 3),
                'class_id': i
            })

        logger.info(
            f"🔄 Fallback prediction: {results[0]['class_name']} "
            f"({results[0]['confidence']})"
        )
        return results

    def preprocess_image(
            self, image: Image.Image, target_size: tuple = (224, 224)
    ) -> np.ndarray:
        """Предобработка изображения для модели."""
        image = image.resize(target_size)
        img_array = np.array(image) / 255.0

        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _get_dominant_color(self, image: Image.Image) -> str:
        """Определение доминирующего цвета."""
        try:
            small_image = image.resize((50, 50))
            pixels = list(small_image.getdata())

            r_avg = sum(p[0] for p in pixels) / len(pixels)
            g_avg = sum(p[1] for p in pixels) / len(pixels)
            b_avg = sum(p[2] for p in pixels) / len(pixels)

            if g_avg > r_avg and g_avg > b_avg and g_avg > 100:
                return "green"
            elif r_avg > g_avg and r_avg > b_avg and r_avg > 150:
                return "orange"
            elif r_avg < 100 and g_avg < 100 and b_avg < 100:
                return "brown"
            elif r_avg > 200 and g_avg > 200 and b_avg > 200:
                return "white"
            else:
                return "mixed"
        except Exception:
            return "unknown"

    def get_available_classes(self) -> List[str]:
        """Получить список доступных классов."""
        return list(self.class_mapping.values())