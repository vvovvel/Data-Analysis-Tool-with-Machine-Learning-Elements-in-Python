import pandas as pd
import pytest
from analysis.statistics import _summary_stats
from data.exceptions import InvalidDataError


def test_summary_stats_numeric_check():
    df = pd.DataFrame({
        'Sleep Duration': [6.0, 8.0, 10.0],
        'Age': [20, 30, 40],
        'Daily Steps': [1000, 2000, 3000]
    })

    stats = _summary_stats(df, ['Sleep Duration', 'Age'])

    assert stats.loc['Sleep Duration', 'mean'] == 8.0

    assert stats.loc['Age', 'max'] == 40.0

    assert stats.loc['Sleep Duration', 'min'] == 6.0

def test_summary_stats_raises_on_non_numeric():
    df = pd.DataFrame({
        'Sleep Duration': [6.0, 7.0],
        'Name': ['Alice', 'Bob']
    })
    with pytest.raises(InvalidDataError):
        _summary_stats(df, ['Name'])

# # How it works:
# Pytest "enters" the with block.
# It executes the code inside (summary_stats(df, ['Name'])).
# If the code raises exactly the InvalidDataError exception → the test passes.
# If the code does not raise an exception → the test fails, and pytest reports an error.
# If the code raises a different exception → the test also fails, and pytest shows a different error.