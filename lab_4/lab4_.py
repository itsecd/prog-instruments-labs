import logging
from typing import Dict, Any, Optional, List
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


class FoodDatabaseManager:
    """Manages food database operations."""

    def __init__(self, food_db_path: str = "data/food_ai.db"):
        self.food_db = FoodDatabase(food_db_path)
        self.food_db.initialize()
        self.extended_food_db = self._initialize_extended_database()
        logger.info("✅ FoodDatabaseManager initialized")

    def _initialize_extended_database(self) -> Dict[str, FoodInfo]:
        """Initialize extended Food-101 database."""
        return {
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
        """Get food information from extended database."""
        normalized_name = food_name.lower()

        # Try extended database first
        if normalized_name in self.extended_food_db:
            return self.extended_food_db[normalized_name]

        # Fallback to main database
        db_food_info = self.food_db.get_food_info(food_name)
        if db_food_info and "calories_per_100g" in db_food_info:
            return FoodInfo(
                calories_per_100g=db_food_info["calories_per_100g"],
                typical_weight_g=150,
                category=db_food_info.get("category", "Unknown")
            )

        # Return default if not found
        return self.extended_food_db["default_food"]

    def add_prediction_to_history(self, food_class: str, confidence: float, calories: float) -> None:
        """Add prediction to history."""
        self.food_db.add_prediction_to_history(food_class, confidence, calories)

    def get_prediction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get prediction history."""
        return self.food_db.get_prediction_history(limit)

    def close(self) -> None:
        """Close database connections."""
        self.food_db.close()


class CalorieEstimator:
    """Handles calorie estimation logic."""

    @staticmethod
    def calculate_estimated_weight(typical_weight: int, coverage_ratio: float) -> float:
        """Calculate estimated weight based on coverage ratio."""
        return typical_weight * coverage_ratio

    @staticmethod
    def calculate_base_calories(calories_per_100g: float, weight: float) -> float:
        """Calculate calories based on weight."""
        return (calories_per_100g / 100) * weight

    @staticmethod
    def calculate_confidence_factor(confidence: float) -> float:
        """Calculate confidence factor for estimation."""
        return min(1.0, confidence * 10)

    @staticmethod
    def create_success_result(calories: float, confidence: float, food_class: str,
                              food_info: FoodInfo, coverage_ratio: float,
                              estimated_weight: float) -> EstimationResult:
        """Create successful estimation result."""
        return EstimationResult(
            success=True,
            estimated_calories=round(calories),
            confidence=round(confidence, 2),
            details={
                "food_class": food_class,
                "calories_per_100g": food_info.calories_per_100g,
                "estimated_weight_g": round(estimated_weight),
                "coverage_ratio": round(coverage_ratio, 2),
                "category": food_info.category
            }
        )

    @staticmethod
    def create_error_result(error_message: str) -> EstimationResult:
        """Create error estimation result."""
        return EstimationResult(
            success=False,
            error=error_message,
            confidence=0.0
        )


class CalorieCalculator:
    """
    Main class for calorie calculation with separated concerns.
    Improved maintainability and testability.
    """

    def __init__(self, food_db_path: str = "data/food_ai.db"):
        self.db_manager = FoodDatabaseManager(food_db_path)
        self.estimator = CalorieEstimator()
        logger.info("✅ CalorieCalculator initialized with improved architecture")

    def get_food_info(self, food_name: str) -> FoodInfo:
        """Получить информацию о продукте из базы данных."""
        return self.db_manager.get_food_info(food_name)

    def estimate_calories(self, food_class: str, coverage_ratio: float, confidence: float) -> EstimationResult:
        """Оценка калорийности"""
        try:
            food_info = self.get_food_info(food_class)

            if not food_info:
                logger.warning(f"⚠️ Food class '{food_class}' not found in database")
                return self.estimator.create_error_result(f"Food class '{food_class}' not found in database")

            # Расчет компонентов
            estimated_weight = self.estimator.calculate_estimated_weight(
                food_info.typical_weight_g, coverage_ratio
            )
            base_calories = self.estimator.calculate_base_calories(
                food_info.calories_per_100g, estimated_weight
            )
            confidence_factor = self.estimator.calculate_confidence_factor(confidence)
            final_calories = base_calories * confidence_factor

            self.db_manager.add_prediction_to_history(
                food_class,
                confidence,
                final_calories
            )

            result = self.estimator.create_success_result(
                final_calories, confidence_factor, food_class,
                food_info, coverage_ratio, estimated_weight
            )

            logger.info(f"✅ Calorie estimation: {result.estimated_calories} kcal for {food_class}")
            return result

        except Exception as e:
            logger.error(f"❌ Calorie estimation error: {e}")
            return self.estimator.create_error_result(str(e))

    def get_prediction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получить историю предсказаний"""
        return self.db_manager.get_prediction_history(limit)

    def close(self):
        """Закрыть соединения"""
        self.db_manager.close()