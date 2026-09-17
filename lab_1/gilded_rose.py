"""Gilded Rose inventory update logic."""


class Item:
    """An item in the Gilded Rose inventory."""

    def __init__(self, name: str, sell_in: int, quality: int) -> None:
        """Initialize an item."""
        self.name = name
        self.sell_in = sell_in
        self.quality = quality

    def __repr__(self) -> str:
        """Return string representation of the item."""
        return f"{self.name}, {self.sell_in}, {self.quality}"


def update_quality(items: list[Item]) -> list[Item]:
    """Update quality and sell_in for all items."""
    for item in items:
        if item.name != "Aged Brie" and item.name != (
            "Backstage passes to a TAFKAL80ETC concert"
        ):
            if item.quality > 0 and item.name != "Sulfuras, Hand of Ragnaros":
                item.quality -= 1
        else:
            if item.quality < 50:
                item.quality += 1
                if item.name == "Backstage passes to a TAFKAL80ETC concert":
                    if item.sell_in < 11 and item.quality < 50:
                        item.quality += 1
                    if item.sell_in < 6 and item.quality < 50:
                        item.quality += 1
        if item.name != "Sulfuras, Hand of Ragnaros":
            item.sell_in -= 1
        if item.sell_in < 0:
            if item.name != "Aged Brie":
                if item.name != "Backstage passes to a TAFKAL80ETC concert":
                    if item.quality > 0 and item.name != "Sulfuras, Hand of Ragnaros":
                        item.quality -= 1
                else:
                    item.quality = 0
            else:
                if item.quality < 50:
                    item.quality += 1
    return items
