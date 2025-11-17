from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class BankConfig:
    """Bank configuration model"""
    url: str
    name: str
    product_type: str


@dataclass
class CardData:
    """Card data model"""
    url: str
    id: Optional[str] = None
    name: Optional[str] = None
    bank: Optional[str] = None
    rating: Optional[float] = None
    features: List[str] = field(default_factory=list)
    bonuses: Optional[Dict] = None
    tariffs: Optional[Dict] = None
    requirements: Optional[Dict] = None
    expertise: Optional[Dict] = None
    product_type: Optional[str] = None
    credit_limit: Optional[str] = None
    interest_rate: Optional[str] = None
    grace_period: Optional[str] = None
