import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import src.utils as utils  # assuming script path is tidytool/
from collections import Counter
from pprint import pprint, pformat
from typing import Literal
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
import re


####### Helper functions #########
def read_smart(file_path, read_as_str=False):
    """
    TODO: read table from file_path
    Args:
        file_path (str): table file path
        read_as_str (bool): read all cols as string or not
    return (tuple):
        df, bad_lines
    """
    # supported methods for read()
    read_methods = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    # helper to save the bad lines:
    def bad_line_handler(bad_line):
        bad_lines.append(bad_line)
        return None

    # choose read method based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    read_method = read_methods.get(file_ext)
    read_args = {"dtype": str} if read_as_str else {}  # read all cols as str dtypes
    bad_lines = []  # buffer to save bad lines

    # read file:
    df = read_method(
        file_path,
        on_bad_lines=bad_line_handler,
        engine="python",
        **read_args,
    )
    if bad_lines:  # if bad lines exist
        logging.warning(f"ParserError: following bad lines are skipped:\n{bad_lines}")
    return df, bad_lines


class Data:
    def __init__(self, file_path, logging_level="info"):
        """
        TODO: check data quality, understand the data
        Args:
            file_path (str/pd.DF): The file path/DF to be analyzed. (only support csv for now)
            logging_level (str, optional): The logging level to be used. Defaults to "info".
        """
        # setup:
        utils.set_loggings(level=logging_level, func_name="tidytools.data.Data")
        self.file_path = file_path
        # load data:
        self.read()  # read file based on extension

    def read(self):
        """TODO: read input file | load input dataframe"""
        if isinstance(self.file_path, pd.DataFrame):  # load from DF
            self.raw = self.file_path
        else:
            # read file based on extension, all col as string
            self.raw, self.bad_lines = read_smart(self.file_path, read_as_str=True)
        self.df = self.raw.copy()  # buffer for processed data

    def info(
        self, status: Literal["before", "after"] = "after", head=False, max_unique=3
    ):
        """
        TODO: Some summary info of data, including data types, NA count, unique values, etc.
        Args:
            data (Literal['before', 'after']): State of data to be summarized
            head (bool, optional): If True, display first few rows of data
            max_unique (int, optional): Max no. of unique values to display for each feature
        Attrs:
            ov (pd.DataFrame): Each row represents a feature info
                - dtype: Data type of each feature.
                - NA_count: Proportion of missing values in each feature.
                - n_unique: Number of unique values in each feature.
                - examples: Examples of unique values in each feature.
        """
        df = self.before if status == "before" else self.after  # select data
        pd.set_option(
            "display.max_rows", max(df.shape[1], 10)
        )  # to display all features

        top_unique = lambda x, n=max_unique: x.unique()[:n]  # get top n unique values

        info = df.apply(
            lambda x: (x.dtype, x.isna().mean(), x.nunique(), top_unique(x)), axis=0
        ).T
        info.columns = ["dtype", "NA_count", "n_unique", "examples"]
        info_str = pformat(info)

        # collect df.head()
        pd.set_option("display.max_columns", df.shape[1])  # to display all features
        head_info = df.head() if head else "Skipped"

        # display basic info of data
        logging.critical(
            f"""Status: {status}\n
Table Dimension: {df.shape}\n
Data types summary:\n{df.dtypes.value_counts()}\n
Head of data:\n{head_info}\n
Data info (Data.ov):\n{info_str}
"""
        )
        self.ov = info

    def str_process(self, case: Literal["raw", "upper", "lower"] = "raw"):
        """
        TODO: string processing, including space stripping, case-changing
        """

        def clean_str(x):
            # x: pandas series
            return (
                x.replace(r"['\"]", "", regex=True)  # rm ' and "
                .str.strip()  # strip space
                .replace(r"\s+", " ", regex=True)  # multiple space to one space
            )

        # string cleaning:
        self.after = self.after.apply(
            lambda x: (clean_str(x) if x.dtype == "object" else x)
        )
        # change case:
        if case == "upper":
            self.after = self.after.apply(
                lambda x: x.str.upper() if x.dtype == "object" else x
            )
        elif case == "lower":
            self.after = self.after.apply(
                lambda x: x.str.lower() if x.dtype == "object" else x
            )

    def clean_header(self, keep_space=False):
        """Strip space & single/double quotes for column names."""
        ori_colnames = self.before.columns
        colnames = (
            self.before.columns.str.replace(r"['\"]", "", regex=True)  # rm quotes
            .str.replace(r"\s+", " ", regex=True)  # long space to single space
            .str.strip()  # strip space
        )
        if not keep_space:
            colnames = colnames.str.replace(" ", "_")  # turn space to underscore
        self.after.columns = colnames
        name_log = ""  # buffer to changed names
        cnt = 0  # count changed names
        for ori, new in zip(ori_colnames, colnames):  # display changed names
            if ori != new:
                cnt += 1
                name_log += f"{ori} -> {new}\n"
        logging.info(
            f"`clean_header` completed. {cnt} column names were updated:\n{name_log}"
        )
        self.ori_colnames = ori_colnames

    def replace_with_na(self, na_vals=[" ", "", "?"]):
        """TODO: Replace na_candidates with pd.NA"""
        # merge list into regex pattern:
        na_vals.extend([np.nan, None])  # standardize na values

        # get colnames of each gp:
        float_cols = self.after.select_dtypes(include=["float"]).columns
        date_cols = self.after.select_dtypes(include=["datetime"]).columns
        other_cols = self.after.select_dtypes(exclude=["float", "datetime"]).columns

        # use np.nan for float:
        self.after[float_cols] = self.after[float_cols].replace(na_vals, np.nan)

        # use pd.NaT for datetime:
        self.after[date_cols] = self.after[date_cols].replace(na_vals, pd.NaT)

        # use pd.NA for other types:
        self.after[other_cols] = self.after[other_cols].replace(na_vals, pd.NA)

    def clean(
        self, na_vals=[" ", "", "?", np.nan, None], case="raw", header_keep_space=False
    ):
        self._clean_header(keep_space=header_keep_space)
        self._str_process(case=case)
        self._replace_with_na(na_vals=na_vals)


class SmartDtype:
    def __init__(
        self, input, tolerance=0.95, timezone=None, max_size=1000, random_state=10
    ):
        """
        Helper for class Data
        TODO: transform dtype based on missingness
        """
        # setup args:
        if isinstance(input, pd.DataFrame):
            self.df = input.astype(str)  # force all dtypes to str
        else:
            self.df, _ = read_smart(input, read_as_str=True)
        self.tolerance = tolerance
        self.timezone = (
            ZoneInfo(timezone) if timezone else datetime.now().astimezone().tzinfo
        )

        # limit subset size:
        if self.df.shape[0] > max_size:
            self.subset = self.df.sample(n=max_size, random_state=random_state)
        else:
            self.subset = self.df.copy()
        # buffers:
        self.dtypes = dict()

    def diagnosis(self):
        colnames = self.df.columns
        # functions to check dtype in order
        check_funcs = [self._bool, self._time, self._date, self._numeric]
        for col in colnames:
            for check in check_funcs:
                if check(col):
                    print(col, str(check))
                    break
            if col not in self.dtypes:
                self.dtypes[col] = {"dtype": "string"}

    def _time(self, col):
        """
        TODO: try transform to timedelta dtype; matches HH:MM, HH:MM:SS, day, hour, min, sec
        Args:
            col: column name in self.df
        Eg. "02:03" -> 2hr 3 min
        return:
            TRUE if dtypes dictionary is updated
            FALSE if ideal dtype not identified
        """
        feature = (
            self.subset[col].dropna().astype(str)
        )  # Ensure string type for regex matching
        time_pattern = r"^\d{1,2}:\d{2}(:\d{2})?$"  # Matches HH:MM or HH:MM:SS
        words = "|".join(["day", "hour", "minute", "second", "min", "sec", "hr"])
        match_rate = (
            feature.str.match(time_pattern).mean()
            + feature.str.contains(words, case=False).mean()
        )
        if match_rate > self.tolerance:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                transformed = pd.to_timedelta(feature, errors="coerce")
            success_rate = transformed.notna().mean()  # Calculate success rate
            if success_rate > self.tolerance:
                self.dtypes[col] = {
                    "dtype": "timedelta",
                    "success_rate": float(success_rate),
                }
                return True

    def _date(self, col):
        """
        TODO: try transform to datetime dtype
        Args:
            col: column name in self.df
        Eg. 20240703 -> 2024/07/03 in datetime format
        return:
            TRUE if dtypes dictionary is updated
            FALSE if ideal dtype not identified
        """
        feature = self.subset[col].dropna()
        with warnings.catch_warnings():  # disable warning for incorrect types:
            warnings.simplefilter("ignore", category=UserWarning)
            transformed = pd.to_datetime(feature, errors="coerce")
        # check success rate:
        success_rate = transformed.notna().mean()  # success rate exclude NA
        # record the dtype:
        if success_rate > self.tolerance:
            self.dtypes[col] = {"dtype": "date", "success_rate": float(success_rate)}
            return True

    def _bool(self, col, pairs=[["no", "yes"], [False, True], [0, 1]]):
        """
        TODO: try transform to boolean dtype
        Only col with unique value as one of the provided pairs will be labeled as boolean
        Args:
            col: column name in self.df
            pairs: nested list, pairs that should be treated as boolean: [false_val, true_val]
        """
        feature = (
            self.subset[col].dropna().str.lower()
        )  # turn to lower case for comparison
        unique_values = set(feature.unique())
        if len(unique_values) == 2:
            for pair in pairs:
                pair = set(map(lambda x: str(x).lower(), pair))  # turn to lower case
                if pair == unique_values:
                    false_val, true_val = pair
                    # mapping param is for dtype transformation
                    self.dtypes[col] = {
                        "dtype": "boolean",
                        "mapping": {true_val: True, false_val: False},
                    }
                    return True

    def _numeric(self, col):
        feature = self.subset[col].dropna()
        # turn to numeric:
        transformed = pd.to_numeric(feature, errors="coerce")
        success_rate = transformed.notna().mean()
        if success_rate > self.tolerance:  # if can be treated as numeric
            if (transformed % 1 == 0).all():
                dtype = "integer"
            else:
                dtype = "float"
            self.dtypes[col] = {"dtype": dtype, "success_rate": float(success_rate)}
            return True

    def _bool_transform(self, col):
        mapping = self.dtypes[col]["mapping"]  # mapping for boolean transformation
        self.df[col] = self.df[col].str.lower().map(mapping)  # transform


####### Helper functions ########
