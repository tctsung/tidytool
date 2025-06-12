from . import data as dat
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from typing import Literal
from upsetplot import plot as _plot_upset
from matplotlib_venn import venn2, venn3, venn3_circles, venn2_circles
import warnings

# block Upset plot FutureWarnings:
warnings.simplefilter(action="ignore", category=FutureWarning)


## Bar Plot
def barplot_category_counts(
    df,
    col_category="category",
    col_count="count",
    title="Category Distribution",
    highlight_top_k=10,
    figsize=(12, 10),
    textbox_font_size=12,
):
    """
    Visualize categorical data as a bar chart with top categories highlighted.

    Parameters:
    df (pd.DataFrame): DataFrame with category and count columns.
    col_category (str): Name of the category column.
    col_count (str): Name of the count column.
    title (str): Custom title for the chart.
    highlight_top_k (int or None): The number of top categories to highlight.
                                   If None, all categories are plotted uniformly.
    figsize (tuple): Figure size for the plot.
    """
    df, _ = dat.read_smart(df)  # Ensure df is a DataFrame
    # input validation:
    assert (
        col_category in df.columns
    ), f"Column '{col_category}' not found in DataFrame."
    assert col_count in df.columns, f"Column '{col_count}' not found in DataFrame."
    total_count = df[col_count].sum()
    assert (
        total_count > 0
    ), "Total count must be greater than zero to calculate percentages."

    # --- 1. Modify highlight_top_k based on the number of categories ---
    num_categories = df.shape[0]
    if highlight_top_k is not None and highlight_top_k >= num_categories:
        highlight_top_k = None

    # Sort by count descending
    df_sorted = df.sort_values(col_count, ascending=False).reset_index(drop=True)

    # Keep only top 30 for plotting to maintain readability
    df_plot = df_sorted.head(30)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # --- 2 & 3. Conditional plotting logic ---
    if highlight_top_k is None:
        # Case 2: No highlighting, plot all bars in orange
        colors = ["#FF8C42"] * len(df_plot)
        bars_to_label = len(df_plot)
    else:
        # Case 3: Highlight top k categories
        num_orange = min(highlight_top_k, len(df_plot))
        colors = ["#FF8C42"] * num_orange + ["#D3D3D3"] * (len(df_plot) - num_orange)
        bars_to_label = num_orange

    # Create horizontal bar chart
    ax.barh(
        range(len(df_plot)),
        df_plot[col_count],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Invert y-axis so top categories appear at the top
    ax.invert_yaxis()
    x_min, x_max = ax.get_xlim()

    # Add category labels
    for i in range(bars_to_label):
        ax.text(
            df_plot[col_count].iloc[i] + (x_max * 0.01),
            i,
            df_plot[col_category].iloc[i],
            ha="left",
            va="center",
            fontsize=10,
        )

    # Add details for the highlighted case
    if highlight_top_k is not None and len(df_plot) > highlight_top_k:
        # dotted line to separate highlighted area
        line_pos = highlight_top_k - 0.5
        ax.axhline(y=line_pos, color="black", linestyle="--", linewidth=1.5)

        # Calculate statistics for the text box
        total_count = df_sorted[col_count].sum()
        top_k_count = df_sorted[col_count].head(highlight_top_k).sum()
        top_k_percentage = (top_k_count / total_count) * 100

        # --- 5. Move text box to the top right area ---
        text_x_position = x_max * 0.98
        text_y_position = line_pos + (len(df_plot) - line_pos) / 6

        ax.text(
            text_x_position,
            text_y_position,
            f"Top {highlight_top_k}: {top_k_percentage:.1f}%\n"
            f"Total categories: {len(df_sorted):,}\n"
            f"Total count: {total_count:,}",
            ha="right",
            va="center",
            fontsize=textbox_font_size,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Categories (ordered by count)", fontsize=12)

    # --- 6. Remove the black frame at right and top ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Remove y-axis ticks for a cleaner look
    ax.set_yticks([])

    # Add a light grid for better readability
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


## Venn Diagram and Upset Plot for Set Differences:
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
        fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes

        val_10 = self.set_diff_data.loc[(True, False)]
        val_01 = self.set_diff_data.loc[(False, True)]
        val_11 = self.set_diff_data.loc[(True, True)]

        # Convert counts to percentages
        total_count = self.set_diff_data.sum()
        subsets = (
            val_10 / total_count * 100,
            val_01 / total_count * 100,
            val_11 / total_count * 100,
        )

        # Draw Venn diagram with labels (no fill)
        venn_diagram = venn2(
            subsets=subsets,
            set_labels=self.category_columns,
            alpha=0.0,  # No fill (transparent)
            subset_label_formatter=lambda x: f"{x:.2f}%",  # Format as percentage
            ax=ax,  # Pass the axes to venn2
            **kwargs,
        )

        # Draw circles with custom colors and solid lines
        c = venn2_circles(
            subsets=subsets,
            linestyle="solid",
            linewidth=1,
            alpha=0.5,
            ax=ax,  # Pass the axes to venn2_circles
        )

        # Define custom colors for circles & labels
        set_colors = ["darkblue", "darkred"]  # Example colors, adjust as needed
        c[0].set_edgecolor(set_colors[0])  # Set color for first circle
        c[1].set_edgecolor(set_colors[1])  # Set color for second circle

        # Match label colors to circle outline colors
        for idx, label in enumerate(venn_diagram.set_labels or []):
            label.set_color(set_colors[idx])

        return fig

    def _plot_venn3(self, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 6))
        # get each subset value:
        val_100 = self.set_diff_data.loc[(True, False, False)]
        val_010 = self.set_diff_data.loc[(False, True, False)]
        val_110 = self.set_diff_data.loc[(True, True, False)]
        val_001 = self.set_diff_data.loc[(False, False, True)]
        val_101 = self.set_diff_data.loc[(True, False, True)]
        val_011 = self.set_diff_data.loc[(False, True, True)]
        val_111 = self.set_diff_data.loc[(True, True, True)]
        # Convert counts to percentages
        total_count = self.set_diff_data.sum()
        subsets = (
            val_100 / total_count * 100,
            val_010 / total_count * 100,
            val_110 / total_count * 100,
            val_001 / total_count * 100,
            val_101 / total_count * 100,
            val_011 / total_count * 100,
            val_111 / total_count * 100,
        )
        # Draw Venn diagram with labels:
        venn_diagram = venn3(
            subsets=subsets,
            set_labels=self.category_columns,
            alpha=0.0,  # No fill (transparent)
            subset_label_formatter=lambda x: f"{x:.2f}%",  # Format as percentage
            ax=ax,  # Pass the axes to venn3
            **kwargs,
        )
        c = venn3_circles(
            subsets=subsets,  # Ensure circles are drawn with the same colors
            linestyle="solid",  # Solid lines for the circles
            linewidth=1,
            alpha=0.5,  # slightly transparent
            ax=ax,  # Pass the axes to venn3
        )
        # define custom colors for circles & labels:
        set_colors = ["darkblue", "darkred", "darkgreen"]
        c[0].set_edgecolor(set_colors[0])  # Set color for first circle
        c[1].set_edgecolor(set_colors[1])  # Set color for second circle
        c[2].set_edgecolor(set_colors[2])  # Set color for third circle

        # Match label colors to circle outline colors
        for idx, label in enumerate(venn_diagram.set_labels or []):
            label.set_color(set_colors[idx])

        return fig

    def plot(
        self,
        use_upset: bool = None,
        category_columns: list = None,
        title=None,
        **kwargs,
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
