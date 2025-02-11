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


class DataClean:
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
            self.raw, self.bad_lines = read_smart(self.file_path, read_as_str=False)
            # buffer for processed data:
            self.df, _ = read_smart(self.file_path, read_as_str=True)

    def head(self):
        pd.set_option(
            "display.max_columns", self.raw.shape[1]
        )  # to display all features
        raw_head = pformat(self.raw.head())
        df_head = pformat(self.df.head())
        logging.critical(
            f"""
Raw data:\n{raw_head}
Processed data:\n{df_head}
"""
        )

    def _update_dtype(self):
        smartd = SmartDtype(input=self.df)
        smartd.transform()  # turn self.df to recommended dtypes
        self.df = smartd.df
        # self.recommend_dtypes = pd.DataFrame.from_dict(smartd.dtypes, orient="index")

    def summary(self, max_unique=3):
        """
        TODO: Some summary info of data, including data types, NA count, unique values, etc.
        Args:
            head (bool, optional): If True, display first few rows of data
            max_unique (int, optional): Max no. of unique values to display for each feature
        Attrs:
            info (pd.DataFrame): Each row represents a feature info
                - dtype: Data type of each feature.
                - NA_count: Proportion of missing values in each feature.
                - n_unique: Number of unique values in each feature.
                - examples: Examples of unique values in each feature.
        """

        def get_summary(df, label):
            # get top n unique values without NA
            top_unique = lambda x, n=max_unique: x.dropna().unique()[:n]
            res = df.apply(
                lambda x: (x.dtype, x.isna().mean(), x.nunique(), top_unique(x)), axis=0
            ).T
            res.columns = pd.MultiIndex.from_product(
                [[label], ["dtype", "NA_count", "n_unique", "examples"]]
            )
            return res

        # Get summary for raw and transformed data
        raw_info = get_summary(self.raw, "Raw")
        transformed_info = get_summary(self.df, "Transformed")
        raw_info.index = transformed_info.index = (
            raw_info.index + " -> " + transformed_info.index
        )
        # Combine both into a MultiIndex DataFrame
        info = pd.concat([raw_info, transformed_info], axis=1)

        # display all features
        pd.set_option("display.max_rows", max(self.raw.shape[1], 10))
        info_str = pformat(info)

        # display basic info of data
        logging.critical(f"Table Dimension: {self.raw.shape}")
        self.info = info
        display(info)

    def _str_process(self, case: Literal["raw", "upper", "lower"] = "raw"):
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
        self.df = self.df.apply(lambda x: (clean_str(x) if x.dtype == "object" else x))
        # change case:
        if case == "upper":
            self.df = self.df.apply(
                lambda x: x.str.upper() if x.dtype == "object" else x
            )
        elif case == "lower":
            self.df = self.df.apply(
                lambda x: x.str.lower() if x.dtype == "object" else x
            )

    def _clean_header(self, keep_space=False):
        """Strip space & single/double quotes for column names."""
        ori_colnames = self.raw.columns
        colnames = (
            self.raw.columns.str.replace(r"['\"]", "", regex=True)  # rm quotes
            .str.replace(r"\s+", " ", regex=True)  # long space to single space
            .str.strip()  # strip space
        )
        if not keep_space:
            colnames = colnames.str.replace(" ", "_")  # turn space to underscore
        self.df.columns = colnames
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

    def _replace_with_na(self, na_vals=[" ", "", "?", "nan", "NA"]):
        """TODO: Replace na_candidates with pd.NA"""
        # merge list into regex pattern:
        na_vals.extend([np.nan, None])  # standardize na values

        # get colnames of each gp:
        float_cols = self.df.select_dtypes(include=["float"]).columns
        date_cols = self.df.select_dtypes(include=["datetime"]).columns
        other_cols = self.df.select_dtypes(exclude=["float", "datetime"]).columns

        # use np.nan for float:
        self.df[float_cols] = self.df[float_cols].replace(na_vals, np.nan)

        # use pd.NaT for datetime:
        self.df[date_cols] = self.df[date_cols].replace(na_vals, pd.NaT)

        # use pd.NA for other types:
        self.df[other_cols] = self.df[other_cols].replace(na_vals, pd.NA)

    def clean(
        self,
        na_vals=[" ", "", "?", "nan", "NA"],
        case="raw",
        header_keep_space=False,
    ):
        self._clean_header(keep_space=header_keep_space)
        self._str_process(case=case)
        self._update_dtype()
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
            # missing value could be string "nan":
            self.df.replace(["nan"], pd.NA, inplace=True)
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

    def transform(self):
        """
        TODO: transform data type based on diagnosis
        """
        self.diagnosis(transform=True)

    def diagnosis(self, transform=False):
        """
        TODO: identify proper dtype for each col
        Args:
            transform (bool): if True, will transform the dtype directly
        Return:
            self.dtypes (dict): contain the recommended dtypes
        """
        colnames = self.df.columns
        # functions to check dtype in order
        check_funcs = [self._bool, self._time, self._date, self._numeric]
        for col in colnames:
            if col not in self.dtypes:
                for check in check_funcs:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        if check(col):  # check dtype
                            break
                # set dtype to string if none of the check is True:
                self.dtypes[col] = self.dtypes.get(col, {"dtype": "string"})
            if transform:
                self._transform(col)

    def _transform(self, col):
        """
        TODO: transform dtypes based on self.dtypes
        """
        dtype = self.dtypes[col]["dtype"]
        feature = self.df[col]
        if dtype == "boolean":  # transform to T/F
            mapping = self.dtypes[col]["mapping"]
            self.df[col] = feature.str.lower().map(mapping).astype(bool)
        elif dtype == "timedelta":
            self.df[col] = pd.to_timedelta(feature, errors="coerce")
        elif dtype == "date":
            self.df[col] = pd.to_datetime(feature, errors="coerce")
            self.df[col] = self.df[col].dt.tz_localize(self.timezone)  # set timezone
        elif dtype == "float":
            self.df[col] = pd.to_numeric(feature, errors="coerce")
        elif dtype == "integer":
            self.df[col] = pd.to_numeric(feature, errors="coerce").astype("Int64")

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
        # count no. of digits
        digit_len = feature.str.replace(r"\D", "", regex=True).str.len()
        # must contain 6 digits to be considered a date
        if (digit_len >= 6).mean() > self.tolerance:
            transformed = pd.to_datetime(feature, errors="coerce")
            # check success rate:
            success_rate = transformed.notna().mean()  # success rate exclude NA
            # record the dtype:
            if success_rate > self.tolerance:
                self.dtypes[col] = {
                    "dtype": "date",
                    "success_rate": float(success_rate),
                }
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


####### Helper functions ########
