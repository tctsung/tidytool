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


class Data:
    read_methods = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    def __init__(self, file, logging_level="info", display=True):
        """
        TODO: check data quality, understand the data
        Args:
            file (str/pd.DF): The file path/DF to be analyzed. (only support csv for now)
            logging_level (str, optional): The logging level to be used. Defaults to "info".
        """
        # settings:
        utils.set_loggings(level=logging_level, func_name="EDA.Data")
        self.file = file
        # load data:
        self.load()
        # display basic info of data
        if display:
            self.info()

    def load(self):
        """Wrapper for different load_fileext methods"""
        if isinstance(self.file, pd.DataFrame):  # load from DF
            self.before = self.file
        else:
            # define a pd func to read file:
            file_ext = os.path.splitext(self.file)[1].lower()
            self.read_method = Data.read_methods.get(file_ext)
            self.load_force()
        self.after = self.before.copy()  # buffer for processed data

    def load_force(self):
        """TODO: load data from csv file, will skip bad lines if needed"""
        try:
            self.before = self.read_method(self.file)
        except:
            logging.warning(
                "Input file ParserError. Bad lines are skipped & saved in .bad_lines"
            )
            bad_lines = []  # buffer to save bad lines

            def bad_line_handler(bad_line):
                bad_lines.append(bad_line)  # save the bad line content
                return None

            # save data as attr
            self.before = self.read_method(
                self.file, on_bad_lines=bad_line_handler, engine="python"
            )
            self.bad_lines = bad_lines

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
                x.replace(r"['\"]", "", regex=True)
                .str.strip()
                .replace(r"\s+", " ", regex=True)
            )

        # strip space:
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
        self, df, tolerance=0.95, timezone=None, max_size=1000, random_state=10
    ):
        """
        TODO: transform dtype based on missingness
        """
        # setup args:
        self.df = df
        self.tolerance = tolerance
        self.timezone = (
            ZoneInfo(timezone) if timezone else datetime.now().astimezone().tzinfo
        )
        if df.shape[0] > max_size:
            self.subset = df.sample(n=max_size, random_state=random_state)
        else:
            self.subset = df.copy()
        # buffers:
        self.dtypes = dict()

    def diagnosis(self):
        colnames = self.df.columns
        for col in colnames:
            if col not in self.dtypes:  # if proper dtype not identified
                continue

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
                self.dtypes[col] = "timedelta"
                return True
        return False

    def _date(self, col):
        """
        TODO: try transform to date dtype
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
            formats = {
                "ymd": pd.to_datetime(
                    feature, errors="coerce", exact=False, yearfirst=True
                ),
                "dmy": pd.to_datetime(
                    feature, errors="coerce", exact=False, dayfirst=True
                ),
                "mdy": pd.to_datetime(feature, errors="coerce", exact=False),
            }
        # check success rate:
        best_format = max(formats, key=lambda k: formats[k].notna().mean())
        success_rate = formats[best_format].notna().mean()  # success rate exclude NA
        # record the dtype:
        if success_rate > self.tolerance:
            self.dtypes[col] = "date-" + best_format


####### Helper functions ########


####### Helper functions ########
