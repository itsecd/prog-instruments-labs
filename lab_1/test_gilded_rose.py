"""Tests for the Gilded Rose inventory update logic."""

import unittest

from gilded_rose import Item, update_quality


class GildedRoseTest(unittest.TestCase):
    """Test cases for update_quality."""

    def test_regular_item_degrades(self) -> None:
        """Regular item quality decreases by 1 each day."""
        items = [Item("+5 Dexterity Vest", 10, 20)]
        update_quality(items)
        self.assertEqual(items[0].quality, 19)
        self.assertEqual(items[0].sell_in, 9)

    def test_aged_brie_increases(self) -> None:
        """Aged Brie quality increases by 1 each day."""
        items = [Item("Aged Brie", 2, 0)]
        update_quality(items)
        self.assertEqual(items[0].quality, 1)

    def test_sulfuras_never_changes(self) -> None:
        """Sulfuras never changes quality or sell_in."""
        items = [Item("Sulfuras, Hand of Ragnaros", 0, 80)]
        update_quality(items)
        self.assertEqual(items[0].quality, 80)
        self.assertEqual(items[0].sell_in, 0)

    def test_backstage_passes_drop_to_zero_after_concert(self) -> None:
        """Backstage passes quality drops to 0 after the concert."""
        items = [Item("Backstage passes to a TAFKAL80ETC concert", 0, 20)]
        update_quality(items)
        self.assertEqual(items[0].quality, 0)


if __name__ == "__main__":
    unittest.main()
