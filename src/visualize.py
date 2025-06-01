from . import data as dat
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from typing import Literal
from upsetplot import plot as _plot_upset
from matplotlib_venn import venn2, venn3
import warnings

# block Upset plot FutureWarnings:
warnings.simplefilter(action="ignore", category=FutureWarning)


class SetDifference:
    def __init__(self, file_path):
        self.df, _ = dat.read_smart(file_path)

    def _df_to_upset_series(self) -> pd.Series:
        """
        TODO: transform df to multi-idx series compatible with upsetplot pkg
        """
        df = self.df.copy()
        # Identify category columns (all columns in the DataFrame except 'cnt')
        if self.category_columns is None:
            category_columns = [col for col in df.columns if col != "cnt"]
        else:
            category_columns = self.category_columns
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

        # save output as attribute:
        self.category_columns = category_columns
        self.set_diff_data = impute_output

    def _plot_venn2(self, **kwargs):
        val_10 = self.set_diff_data.loc[(True, False)]
        val_01 = self.set_diff_data.loc[(False, True)]
        val_11 = self.set_diff_data.loc[(True, True)]
        return venn2(
            subsets=(val_10, val_01, val_11), set_labels=self.category_columns, **kwargs
        )

    def _plot_venn3(self, **kwargs):
        val_100 = self.set_diff_data.loc[(True, False, False)]
        val_010 = self.set_diff_data.loc[(False, True, False)]
        val_001 = self.set_diff_data.loc[(False, False, True)]
        val_110 = self.set_diff_data.loc[(True, True, False)]
        val_101 = self.set_diff_data.loc[(True, False, True)]
        val_011 = self.set_diff_data.loc[(False, True, True)]
        val_111 = self.set_diff_data.loc[(True, True, True)]
        return venn3(
            subsets=(val_100, val_010, val_001, val_110, val_101, val_011, val_111),
            set_labels=self.category_columns,
            **kwargs
        )

    def plot(
        self,
        use_upset: bool = None,
        category_columns: list = None,
        title=None,
        **kwargs
    ):
        """
        Plot the data using upsetplot or bar plot.
        :param use_upset: If True, use upset plot; if False, use venn diagram
        :param category_columns: List of columns to use as categories in the upset plot
        :param title: Title of the plot
        :param kwargs: Additional keyword arguments for plotting functions
        """
        self.category_columns = category_columns
        self._df_to_upset_series()

        # Determine to use upset plot or venn diagram:
        if use_upset is None:
            use_upset = True if len(self.category_columns) > 3 else False

        if use_upset:
            upset_diagram = _plot_upset(self.set_diff_data, **kwargs)
            self.upset_diagram = upset_diagram
        else:
            if len(self.category_columns) == 2:
                self.venn_diagram = self._plot_venn2()
            elif len(self.category_columns) == 3:
                self.venn_diagram = self._plot_venn3()
            else:
                raise ValueError(
                    "Venn diagram supports only 2 or 3 categories, please set `use_upset=True`"
                )

        if title:
            plt.suptitle(title)
        plt.show()
