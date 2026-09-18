import random
import re

import approvaltests

from daily_forecasts import print_sales_forecasts


def test_sales_forecasts(capsys):
    random.seed(42)
    print_sales_forecasts()
    output = capsys.readouterr().out
    scrubbed_output = re.sub(
        r"Forecast at time .*",
        "Forecast at time [TIME]",
        output,
    )
    approvaltests.verify(scrubbed_output)
