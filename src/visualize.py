from . import data as dat
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from typing import Literal
from upsetplot import plot as upset_plot


class SetDifference:
    def __init__(self, file_path):
        self.df, _ = dat.read_smart(file_path)

    def _df_to_upset_series(self, category_columns: list = None) -> pd.Series:
        """
        TODO: transform df to multi-idx series compatible with upsetplot pkg
        """
        df = self.df.copy()
        # Identify category columns (all columns in the DataFrame except 'cnt')
        if category_columns is None:
            category_columns = [col for col in df.columns if col != "cnt"]

        # data validation
        assert "cnt" in df.columns, "Count column 'cnt' not found in DataFrame."
        assert pd.api.types.is_numeric_dtype(
            df["cnt"]
        ), "Count column 'cnt' must be numeric."

        # transform 1/0 to True/False if source table is from SQL query
        df[category_columns] = df[category_columns].astype(bool)

        # Set the category columns as multi-index, keep only the cnt series
        df_indexed = df.set_index(category_columns)
        raw_output = df_indexed["cnt"]

        # Generate all possible combinations category columns:
        all_combinations = list(
            itertools.product([False, True], repeat=len(category_columns))
        )

        # Ouput buffer:
        all_combinations_df = pd.MultiIndex.from_tuples(
            all_combinations, names=category_columns
        )

        # Reindex input with all possible combinations.
        # missing groups will be filled with 0
        impute_output = raw_output.reindex(all_combinations_df, fill_value=0)

        # Reindexing sometimes change dtype to float if NaNs were involved before fill_value,
        # so this ensures consistency.
        impute_output = impute_output.astype(int)

        self.upset_data = impute_output

    def plot(
        self,
        method: Literal["upset", "venn"] = "upset",
        category_columns: list = None,
        **kwargs
    ):
        """
        Plot the data using upsetplot or bar plot.
        :param type: 'upset' for upset plot, 'bar' for bar plot
        :param category_columns: List of columns to use as categories in the upset plot
        :param kwargs: Additional keyword arguments for plotting functions
        """
        self._df_to_upset_series(category_columns)

        if type == "upset":
            upset_plot(self.upset_data, **kwargs)
            plt.show()
        elif type == "venn":
            pass
