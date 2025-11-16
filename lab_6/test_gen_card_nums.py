from card_finder import gen_card_nums
from consts import LAST_4_DIGITS


def test_gen_card_nums_basic():
    nums = gen_card_nums("519747")
    assert len(nums) == 1_000_000
    assert nums[0] == f"519747000000{LAST_4_DIGITS}"
    assert nums[-1] == f"519747999999{LAST_4_DIGITS}"
