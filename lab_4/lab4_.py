import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from app.services.food_database import FoodDatabase

logger = logging.getLogger(__name__)

# Constants for configuration
DEFAULT_DB_PATH = "data/food_ai.db"
DEFAULT_TYPICAL_WEIGHT = 150
DEFAULT_CALORIES_PER_100G = 200
CONFIDENCE_MULTIPLIER = 10
MAX_CONFIDENCE_FACTOR = 1.0
DEFAULT_PREDICTION_HISTORY_LIMIT = 50
CALORIES_PER_GRAM_DIVISOR = 100


@dataclass(frozen=True)
class FoodInfo:
    """
    Immutable data class for storing food nutritional information.

    Attributes:
        calories_per_100g: Calories per 100 grams of the food
        typical_weight_g: Typical serving weight in grams
        category: Food category (e.g., 'Dairy', 'Seafood', 'Vegetable')
    """
    calories_per_100g: float
    typical_weight_g: int
    category: str

    @classmethod
    def create_default(cls) -> 'FoodInfo':
        """Create a default FoodInfo instance."""
        return cls(
            calories_per_100g=DEFAULT_CALORIES_PER_100G,
            typical_weight_g=DEFAULT_TYPICAL_WEIGHT,
            category="Unknown"
        )


@dataclass(frozen=True)
class EstimationResult:
    """
    Immutable data class for calorie estimation results.

    Attributes:
        success: Whether the estimation was successful
        estimated_calories: Estimated calorie count (if successful)
        confidence: Confidence level of the estimation (0.0 to 1.0)
        error: Error message (if unsuccessful)
        details: Additional estimation details
    """
    success: bool
    estimated_calories: Optional[int] = None
    confidence: float = 0.0
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    @classmethod
    def create_success(
            cls,
            calories: float,
            confidence: float,
            food_class: str,
            food_info: FoodInfo,
            coverage_ratio: float,
            estimated_weight: float
    ) -> 'EstimationResult':
        """
        Create a successful estimation result.

        Args:
            calories: Estimated calorie count
            confidence: Confidence factor
            food_class: Name of the food item
            food_info: Food information used for calculation
            coverage_ratio: Coverage ratio used
            estimated_weight: Estimated weight in grams

        Returns:
            EstimationResult with success status and details
        """
        return cls(
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

    @classmethod
    def create_error(cls, error_message: str) -> 'EstimationResult':
        """
        Create an error estimation result.

        Args:
            error_message: Description of the error

        Returns:
            EstimationResult with error status
        """
        return cls(
            success=False,
            error=error_message,
            confidence=0.0
        )


class FoodDatabaseManager:
    """
    Manages food database operations including extended Food-101 database.

    This class handles all database interactions and provides a unified
    interface for accessing food information from multiple sources.
    """

    def __init__(self, food_db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the food database manager.

        Args:
            food_db_path: Path to the food database file
        """
        self.food_db = FoodDatabase(food_db_path)
        self.food_db.initialize()
        self.extended_food_db = self._initialize_extended_database()
        logger.info("✅ FoodDatabaseManager initialized")

    @staticmethod
    def _initialize_extended_database() -> Dict[str, FoodInfo]:
        """
        Initialize extended Food-101 database with common food items.

        Returns:
            Dictionary mapping food names to FoodInfo objects
        """
        food_data = {
            "cheese_plate": (350, 200, "Dairy"),
            "sashimi": (120, 150, "Seafood"),
            "seaweed_salad": (45, 100, "Vegetable"),
            "grilled_salmon": (206, 150, "Seafood"),
            "deviled_eggs": (210, 100, "Eggs"),
            "club_sandwich": (250, 200, "Sandwich"),
            "apple_pie": (265, 150, "Dessert"),
            "pizza": (266, 100, "Fast Food"),
            "hamburger": (295, 200, "Fast Food"),
            "sushi": (150, 50, "Seafood"),
            "steak": (271, 200, "Meat"),
            "salad": (35, 150, "Vegetable"),
        }

        return {
            name: FoodInfo(calories, weight, category)
            for name, (calories, weight, category) in food_data.items()
        }

    def get_food_info(self, food_name: str) -> Optional[FoodInfo]:
        """
        Get food information from extended database with fallback logic.

        Args:
            food_name: Name of the food item to look up

        Returns:
            FoodInfo object with nutritional information, or None if not found
        """
        normalized_name = food_name.lower()

        # Try extended database first
        if normalized_name in self.extended_food_db:
            return self.extended_food_db[normalized_name]

        # Fallback to main database
        db_food_info = self.food_db.get_food_info(food_name)
        if db_food_info and "calories_per_100g" in db_food_info:
            return FoodInfo(
                calories_per_100g=db_food_info["calories_per_100g"],
                typical_weight_g=DEFAULT_TYPICAL_WEIGHT,
                category=db_food_info.get("category", "Unknown")
            )

        return None

    def add_prediction_to_history(self, food_class: str, confidence: float, calories: float) -> None:
        """
        Add a prediction to the history database.

        Args:
            food_class: Name of the food item
            confidence: Confidence level of the prediction
            calories: Estimated calorie count
        """
        self.food_db.add_prediction_to_history(food_class, confidence, calories)

    def get_prediction_history(self, limit: int = DEFAULT_PREDICTION_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        Get prediction history from the database.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of prediction history entries
        """
        return self.food_db.get_prediction_history(limit)

    def close(self) -> None:
        """Close database connections and release resources."""
        self.food_db.close()


class CalorieEstimator:
    """
    Handles calorie estimation calculations and result formatting.

    This class contains pure calculation logic without side effects,
    making it easily testable and reusable.
    """

    @staticmethod
    def calculate_estimated_weight(typical_weight: int, coverage_ratio: float) -> float:
        """
        Calculate estimated weight based on coverage ratio.

        Args:
            typical_weight: Typical weight of the food item in grams
            coverage_ratio: Ratio of typical weight (0.0 to 1.0)

        Returns:
            Estimated weight in grams
        """
        if coverage_ratio < 0 or coverage_ratio > 1:
            raise ValueError("Coverage ratio must be between 0 and 1")
        return typical_weight * coverage_ratio

    @staticmethod
    def calculate_base_calories(calories_per_100g: float, weight: float) -> float:
        """
        Calculate calories based on weight and calories per 100g.

        Args:
            calories_per_100g: Calories per 100 grams
            weight: Weight in grams

        Returns:
            Total calories for the given weight
        """
        if weight < 0:
            raise ValueError("Weight cannot be negative")
        return (calories_per_100g / CALORIES_PER_GRAM_DIVISOR) * weight

    @staticmethod
    def calculate_confidence_factor(confidence: float) -> float:
        """
        Calculate confidence factor for calorie estimation.

        Args:
            confidence: Raw confidence level from AI model (0.0 to 1.0)

        Returns:
            Confidence factor adjusted for estimation (0.0 to 1.0)
        """
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return min(MAX_CONFIDENCE_FACTOR, confidence * CONFIDENCE_MULTIPLIER)

    def estimate_calories(
            self,
            food_info: FoodInfo,
            coverage_ratio: float,
            confidence: float
    ) -> float:
        """
        Perform complete calorie estimation.

        Args:
            food_info: Food information for calculation
            coverage_ratio: Ratio of typical weight
            confidence: Confidence level from AI model

        Returns:
            Final estimated calories
        """
        estimated_weight = self.calculate_estimated_weight(
            food_info.typical_weight_g, coverage_ratio
        )
        base_calories = self.calculate_base_calories(
            food_info.calories_per_100g, estimated_weight
        )
        confidence_factor = self.calculate_confidence_factor(confidence)
        return base_calories * confidence_factor


class CalorieCalculator:
    """
    Main class for calorie calculation with separated concerns.

    This class coordinates between database management and calorie estimation,
    providing a clean API for calorie calculation while maintaining
    improved maintainability and testability.
    """

    def __init__(self, food_db_path: str = DEFAULT_DB_PATH):
        """
        Initialize the calorie calculator.

        Args:
            food_db_path: Path to the food database file
        """
        self.db_manager = FoodDatabaseManager(food_db_path)
        self.estimator = CalorieEstimator()
        logger.info("✅ CalorieCalculator initialized with improved architecture")

    def get_food_info(self, food_name: str) -> FoodInfo:
        """
        Get food information from the database.

        Args:
            food_name: Name of the food item

        Returns:
            FoodInfo object with nutritional information
        """
        food_info = self.db_manager.get_food_info(food_name)
        return food_info or FoodInfo.create_default()

    def estimate_calories(self, food_class: str, coverage_ratio: float,
                          confidence: float) -> EstimationResult:
        """
        Estimate calories for given food class with coverage and confidence.

        Args:
            food_class: Name of the food item
            coverage_ratio: Ratio of typical weight (0.0 to 1.0)
            confidence: Confidence level from AI model (0.0 to 1.0)

        Returns:
            EstimationResult with calculation details
        """
        try:
            # Input validation
            if not food_class or not food_class.strip():
                return EstimationResult.create_error("Food class cannot be empty")

            if coverage_ratio < 0 or coverage_ratio > 1:
                return EstimationResult.create_error("Coverage ratio must be between 0 and 1")

            if confidence < 0 or confidence > 1:
                return EstimationResult.create_error("Confidence must be between 0 and 1")

            food_info = self.get_food_info(food_class)

            # Calculate calories
            final_calories = self.estimator.estimate_calories(
                food_info, coverage_ratio, confidence
            )

            estimated_weight = self.estimator.calculate_estimated_weight(
                food_info.typical_weight_g, coverage_ratio
            )

            # Save to history
            self.db_manager.add_prediction_to_history(
                food_class, confidence, final_calories
            )

            # Create result
            result = EstimationResult.create_success(
                final_calories,
                self.estimator.calculate_confidence_factor(confidence),
                food_class,
                food_info,
                coverage_ratio,
                estimated_weight
            )

            logger.info(f"✅ Calorie estimation: {result.estimated_calories} kcal for {food_class}")
            return result

        except ValueError as e:
            logger.warning(f"⚠️ Validation error in calorie estimation: {e}")
            return EstimationResult.create_error(str(e))
        except Exception as e:
            logger.error(f"❌ Calorie estimation error: {e}")
            return EstimationResult.create_error(f"Internal server error: {str(e)}")

    def get_prediction_history(self, limit: int = DEFAULT_PREDICTION_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        Get prediction history from the database.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of prediction history entries
        """
        return self.db_manager.get_prediction_history(limit)

    def close(self) -> None:
        """Close database connections and release resources."""
        self.db_manager.close()


# Factory function for easy instance creation
def create_calorie_calculator(db_path: str = DEFAULT_DB_PATH) -> CalorieCalculator:
    """
    Create and initialize a CalorieCalculator instance.

    Args:
        db_path: Path to the food database file

    Returns:
        Initialized CalorieCalculator instance
    """
    return CalorieCalculator(db_path)
