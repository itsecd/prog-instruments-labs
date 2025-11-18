import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from app.services.food_database import FoodDatabase

logger = logging.getLogger(__name__)


@dataclass
class FoodInfo:
    """Data class for food information."""
    calories_per_100g: float
    typical_weight_g: int
    category: str


@dataclass
class EstimationResult:
    """Data class for calorie estimation result."""
    success: bool
    estimated_calories: Optional[int] = None
    confidence: float = 0.0
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class CalorieCalculator:
    def __init__(self, food_db_path: str = "data/food_ai.db"):
        self.food_db = FoodDatabase(food_db_path)
        self.food_db.initialize()
        self._init_extended_food_database()
        logger.info("✅ CalorieCalculator initialized with extended food database")

    def _init_extended_food_database(self):
        """Расширенная база данных продуктов Food-101"""
        self.extended_food_db = {
            "cheese_plate": FoodInfo(calories_per_100g=350, typical_weight_g=200, category="Dairy"),
            "sashimi": FoodInfo(calories_per_100g=120, typical_weight_g=150, category="Seafood"),
            "seaweed_salad": FoodInfo(calories_per_100g=45, typical_weight_g=100, category="Vegetable"),
            "grilled_salmon": FoodInfo(calories_per_100g=206, typical_weight_g=150, category="Seafood"),
            "deviled_eggs": FoodInfo(calories_per_100g=210, typical_weight_g=100, category="Eggs"),
            "club_sandwich": FoodInfo(calories_per_100g=250, typical_weight_g=200, category="Sandwich"),
            "apple_pie": FoodInfo(calories_per_100g=265, typical_weight_g=150, category="Dessert"),
            "pizza": FoodInfo(calories_per_100g=266, typical_weight_g=100, category="Fast Food"),
            "hamburger": FoodInfo(calories_per_100g=295, typical_weight_g=200, category="Fast Food"),
            "sushi": FoodInfo(calories_per_100g=150, typical_weight_g=50, category="Seafood"),
            "steak": FoodInfo(calories_per_100g=271, typical_weight_g=200, category="Meat"),
            "salad": FoodInfo(calories_per_100g=35, typical_weight_g=150, category="Vegetable"),
            "default_food": FoodInfo(calories_per_100g=200, typical_weight_g=150, category="Unknown")
        }

    def get_food_info(self, food_name: str) -> FoodInfo:
        """Получить информацию о продукте из расширенной базы"""
        food_info = self.extended_food_db.get(food_name.lower())

        if not food_info:
            food_info = self.food_db.get_food_info(food_name)
            if "calories_per_100g" in food_info:
                food_info = FoodInfo(
                    calories_per_100g=food_info["calories_per_100g"],
                    typical_weight_g=150,
                    category=food_info.get("category", "Unknown")
                )

        return food_info or self.extended_food_db["default_food"]

    def estimate_calories(self, food_class: str, coverage_ratio: float, confidence: float) -> EstimationResult:
        """Оценка калорийности"""
        try:
            food_info = self.get_food_info(food_class)

            if not food_info:
                logger.warning(f"⚠️ Food class '{food_class}' not found in database")
                return EstimationResult(
                    success=False,
                    error=f"Food class '{food_class}' not found in database",
                    confidence=0.0
                )

            typical_weight = food_info.typical_weight_g
            calories_per_100g = food_info.calories_per_100g

            estimated_weight = typical_weight * coverage_ratio
            estimated_calories = (calories_per_100g / 100) * estimated_weight

            confidence_factor = min(1.0, confidence * 10)
            final_calories = estimated_calories * confidence_factor

            self.food_db.add_prediction_to_history(
                food_class,
                confidence,
                final_calories
            )

            result = EstimationResult(
                success=True,
                estimated_calories=round(final_calories),
                confidence=round(confidence_factor, 2),
                details={
                    "food_class": food_class,
                    "calories_per_100g": calories_per_100g,
                    "estimated_weight_g": round(estimated_weight),
                    "coverage_ratio": round(coverage_ratio, 2),
                    "category": food_info.category
                }
            )

            logger.info(f"✅ Calorie estimation: {result.estimated_calories} kcal for {food_class}")
            return result

        except Exception as e:
            logger.error(f"❌ Calorie estimation error: {e}")
            return EstimationResult(
                success=False,
                error=str(e),
                confidence=0.0
            )

    def get_prediction_history(self, limit: int = 50):
        """Получить историю предсказаний"""
        return self.food_db.get_prediction_history(limit)

    def close(self):
        """Закрыть соединения"""
        self.food_db.close()