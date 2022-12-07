import datetime
import holidays
from pyspark.sql import DataFrame
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

def is_belgian_holiday(date: datetime.date) -> bool:
    be_holidays = holidays.country_holidays('BE')
    return date in be_holidays
    pass


def label_weekend(
    frame: DataFrame, colname: str = "date", new_colname: str = "is_weekend"
) -> DataFrame:
    return frame.withColumn(new_colname, dayofweek(colname).isin(1,7))
    """Adds a column indicating whether or not the attribute `colname`
    in the corresponding row is a weekend day."""
    pass


def label_holidays(
    frame: DataFrame,
    colname: str = "date",
    new_colname: str = "is_belgian_holiday",
) -> DataFrame:
    extract_be_holidays = udf(is_belgian_holiday, BooleanType())
    return frame.withColumn(new_colname, extract_be_holidays(colname))
    """Adds a column indicating whether or not the column `colname`
    is a holiday."""
    pass


def label_holidays2(
    frame: DataFrame,
    colname: str = "date",
    new_colname: str = "is_belgian_holiday",
) -> DataFrame:
    """Adds a column indicating whether or not the column `colname`
    is a holiday. An alternative implementation."""
    pass


def label_holidays3(
    frame: DataFrame,
    colname: str = "date",
    new_colname: str = "is_belgian_holiday",
) -> DataFrame:
    """Adds a column indicating whether or not the column `colname`
    is a holiday. An alternative implementation."""
    pass
