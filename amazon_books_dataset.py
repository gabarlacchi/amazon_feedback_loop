import pandas as pd
import os
import json 
from datetime import datetime
import pickle 
import re
import matplotlib.pyplot as plt 
import numpy as np
from dateutil.relativedelta import relativedelta
import traceback
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import itertools

from utils import DotDict

class AmazonBooksDataset:
    def __init__(self, config: DotDict, **kwargs):
        self.config = config

        self.dataset_root_path = f"./data"
        self.dataset_filename = "amazon_books.jsonl"
        self.dataset_features_filename = "meta_amazon_books.jsonl"

        self.item_level = self.config.item_level

        if not os.path.exists(os.path.join(self.dataset_root_path, self.dataset_filename)):
            raise Exception(f"\n Dataset not found -> {os.path.join(self.dataset_root_path, self.dataset_filename)} does not exists \n")
        
        try:
            if (re.match(r"^\d{4}-\d{4}$", self.config.time_window) is not None):
                splitted = (self.config.time_window).split("-")
                self.y1, self.y2 = int(splitted[0]), int(splitted[1])
                self.n_years = (self.y2 - self.y1) + 1
            else:
                raise Exception(f"\n Time window not recognized -> {self.config.time_window}. It must be something like '2011-2012' \n")
        except Exception as e:
            print(traceback.format_exc())

        self.distribution_timeline_path = f"./cache/distribution_timeline_amazon_books.csv"
        self.unrolled_dataset_total_path = f"./cache/unrolled_total_{self.y1}-{self.y2}_amazon_books.csv"
        self.users_rating_distribution_path = f"./cache/users_rating_distribution_{self.y1}-{self.y2}_amazon_books.csv"
        self.global_fallback_rating_distribution_path = f"./cache/global_fallback_rating_distribution_{self.y1}-{self.y2}_amazon_books.csv"
        self.cache_mapping_strings_int_path_users = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_users_amazon_books.csv"
        self.cache_mapping_strings_int_path_items = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_items_amazon_books.csv"
        self.dump_dataset = f"./cache/real_dataset_sim_amazon_books"
        os.makedirs(self.dump_dataset, exist_ok=True)

    def setup(self)-> None:
        if not self.config.use_cache or not os.path.exists(self.unrolled_dataset_total_path):
            # N = 100_000
            # rows = []
            # with open(os.path.join(self.dataset_root_path, self.dataset_filename), "r", encoding="utf-8") as f:
            #     for line in itertools.islice(f, N):
            #         rows.append(json.loads(line))
            rows = []
            with open(
                os.path.join(self.dataset_root_path, self.dataset_filename),
                "r",
                encoding="utf-8"
            ) as f:
                for line in f:
                    rows.append(json.loads(line))

            df_inter = pd.DataFrame(rows)

            cols = [c for c in ["user_id", "asin",  "parent_asin", "rating", "timestamp", "verified_purchase"] if c in df_inter.columns]
            df_inter = df_inter[cols].copy()

            if "rating" in df_inter.columns:
                df_inter["rating"] = pd.to_numeric(df_inter["rating"], errors="coerce")

            if "timestamp" in df_inter.columns:
                df_inter["timestamp"] = pd.to_numeric(df_inter["timestamp"], errors="coerce")
                df_inter["date"] = (
                    pd.to_datetime(df_inter["timestamp"], unit="ms", utc=True, errors="coerce").dt.floor("D")
                )
            
            df_inter = df_inter[df_inter["verified_purchase"] == True]

            mask = (
                (df_inter["date"].dt.year >= self.y1) &
                (df_inter["date"].dt.year <= self.y2)
            )
            df_inter = df_inter.loc[mask]

            df_inter = df_inter.sort_values("date", kind="mergesort")
            
            # FEATURES
            rows = []
            with open(
                os.path.join(self.dataset_root_path, self.dataset_features_filename),
                "r",
                encoding="utf-8"
            ) as f:
                for line in f:
                    rows.append(json.loads(line))

            df_features = pd.DataFrame(rows)

            to_keep = [
                "parent_asin",
                "categories",
                "price",
                "description",
                "details"
            ]
            keep = [c for c in to_keep if c in df_features.columns]
            meta = df_features[keep].copy()

            # INTERACTION FEATURE(S)

            # PRICE
            if "price" in meta.columns:
                meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
                meta["log_price"] = np.log1p(meta["price"])

            # ITEM FEATURE(S)

            # CATEGORIES
            # Store as tokens because most recbole models have tokens logic
            if "categories" in meta.columns:
                meta["category_seq"] = meta["categories"].apply(
                    lambda x: self.categories_to_token_seq(
                        x,
                        drop_roots=True,
                        max_len=30,
                        sep="|"
                    )
                )
            else:
                meta["category_seq"] = None
            
            meta = meta.drop(columns=["categories"])

            # DESCRIPTION
            if "description" in meta.columns:
                meta["desc_text"] = meta["description"].apply(lambda x: self.build_descr_features(x, max_chars=1000))

            meta = meta.drop(columns=["description"])

            # DETAILS
            # It contains publisher and language of the book
            if "details" in meta.columns:
                extracted = meta["details"].apply(self.extract_details_info)

                meta["publisher"] = extracted.apply(lambda x: x[0])
                meta["language"] = extracted.apply(lambda x: x[1])

                meta = meta.drop(columns=["details"])

            meta = meta.drop_duplicates(subset=["parent_asin"])

            # Merge dataframe with items features
            df_final = df_inter.merge(meta, on="parent_asin", how="left", validate="m:1")

            n_rows_missing = df_final["publisher"].isna().sum()
            total_rows = len(df_final)

            print(f"Rows without meta: {n_rows_missing} / {total_rows} "
                f"({n_rows_missing / total_rows:.2%})")
            
            # Delete items interactions tat have no conncetion with features data
            df_final = df_final[df_final["publisher"].notna()]
            self.unrolled_dataset_total = df_final.copy()

            rename_dict = {
                'asin': 'item_id',
            }

            self.unrolled_dataset_total = self.unrolled_dataset_total.rename(columns=rename_dict)

            self.unrolled_dataset_total.drop(columns=["parent_asin"])

            self.unrolled_dataset_total.to_csv(self.unrolled_dataset_total_path)

            # self.user_rating_distribution()
            # self.user_rating_distributions.to_csv(self.users_rating_distribution_path)
            # self.global_rating_distribution.to_csv(self.global_fallback_rating_distribution_path)
        else:
            self.unrolled_dataset_total = pd.read_csv(self.unrolled_dataset_total_path, index_col=0)
            # self.user_rating_distributions = pd.read_csv(self.users_rating_distribution_path, index_col=0)
            # self.global_rating_distribution = pd.read_csv(self.global_fallback_rating_distribution_path, index_col=0)

    def user_rating_distribution(self):
        if "rating" in self.unrolled_dataset_total.columns:
            user_rating_dist = (
                self.unrolled_dataset_total
                .groupby(['user_id', 'rating'])
                .size()
                .groupby(level=0)
                .apply(lambda x: (x / x.sum()).to_dict())
                .to_dict()
            )
            
            self.user_rating_distributions = user_rating_dist
            
            # Fallback distribution in case of errors (just get the global distribution)
            global_rating_dist = (
                self.unrolled_dataset_total['rating']
                .value_counts(normalize=True)
                .to_dict()
            )
            self.global_rating_distribution = global_rating_dist
        else:
            raise f"\n Rating column not found in dataset \n"

    def real_dataset_save_cache(self, start_date: datetime, end_date: datetime, users: list, items: list):
        if not self.config.use_cache:
            df = self.unrolled_dataset_total
            # Filter by users and items partecipating to the simulation, in case
            if users is not None and items is not None:
                df = df[(df.user_id.isin(users)) & (df.item_id.isin(items))]
            df = df[["user_id", "item_id", "date", "timestamp"]]

            df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
            df['timestamp'] = df.date.values.astype(np.int64) // 10 ** 9

            start_dataset_initialization = datetime(self.y1, 1, 1)
            end_dataset_initialization = start_date - relativedelta(days=1)
            d_init = df[(df['date'].dt.date >= start_dataset_initialization.date()) & (df['date'].dt.date <= end_dataset_initialization)]
            d_init.to_csv(os.path.join(self.dump_dataset, "epoch_0.csv"))
            
            epoch = 1
            start_simulation_date = end_dataset_initialization + relativedelta(days=1)
            end_date = start_simulation_date + relativedelta(months=1) - relativedelta(days=1)
            while epoch < self.config.epochs+1:
                d = df[(df['date'].dt.date >= start_simulation_date) & (df['date'].dt.date <= end_date)]
                d.to_csv(os.path.join(self.dump_dataset, f"epoch_{epoch}.csv"))
                start_simulation_date = (start_simulation_date.replace(day=1) + relativedelta(months=1)).replace(year=start_simulation_date.year + (start_simulation_date.month // 12))
                end_date = start_simulation_date + relativedelta(months=1) - relativedelta(days=1)
                print(f"\n Epoch start {start_simulation_date.strftime('%Y-%m-%d')}, end in {end_date.strftime('%Y-%m-%d')}")
                epoch += 1
                if end_date.year == self.y2 and end_date.month == 12:
                    break
        
        else:
            if os.path.exists(self.dump_dataset) and os.path.isdir(self.dump_dataset):
                files = [f for f in os.listdir(self.dump_dataset) if os.path.isfile(os.path.join(self.dump_dataset, f))]
                if len(files) > 1:
                    pass
                else:
                    raise Exception(f"\n Cache not found or not well formatted -> {self.dump_dataset} \n")

    def strategy_simulation_info(self, start_date: datetime, end_date: datetime, users: list, items: list, use_cache: bool = True) -> dict:
        """
            This function define a dict than can be easily accessed later on in order to know what user entered and what has bought, for each date

            It's called from feedback loop class in order to give start and end date
        """
        
        if not self.config.use_cache or not os.path.exists(self.distribution_timeline_path):
            df = self.unrolled_dataset_total
            df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
            df['timestamp'] = df.date.values.astype(np.int64) // 10 ** 9

            unique_dates = df['date'].dt.date.unique()
            result = {}
            for date in unique_dates:
                daily_data = df[df['date'].dt.date == date]
                daily_user_items = daily_data.groupby('user_id')['item_id'].apply(list).reset_index()
                date_str = date.strftime('%Y-%m-%d')
                result[date_str] = {}
                for index, row in daily_user_items.iterrows():
                    result[date_str][row["user_id"]] = row["item_id"]
            self.distribution_dict = result
            with open(self.distribution_timeline_path, "wb") as f:
                pickle.dump(self.distribution_dict, f)
        else:
            with open(self.distribution_timeline_path, "rb") as f:
                self.distribution_dict = pickle.load(f)
        return self.distribution_dict

    def normalize_tokens(self, s: str):
        s = str(s).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" ", "_")
        return s
    
    def categories_to_token_seq(self, cats, drop_roots=True, max_len=30, sep="|"):
        # Input: 
        ROOT_STOP = {"books"}

        if not isinstance(cats, list):
            return None

        out = []
        seen = set()

        for c in cats:
            if c is None:
                continue
            c_str = str(c).strip()
            if not c_str:
                continue

            c_norm = self.normalize_tokens(s = c_str)
            key = c_norm.lower()

            if drop_roots and key in ROOT_STOP:
                continue

            if key in seen:
                continue
            seen.add(key)
            out.append(c_norm)

            if max_len is not None and len(out) >= max_len:
                break

        return sep.join(out) if out else None

    def build_descr_features(self, description, max_chars=2000):
        if isinstance(description, list):
            parts = [self.normalize_tokens(x) for x in description if self.normalize_tokens(x)]
            text = " ".join(parts)
        elif isinstance(description, str):
            text = self.normalize_tokens(description)
        else:
            text = ""
        
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]

        return text

    def extract_details_info(self, details):
        def clean_publisher(s):
            if s is None:
                return None
            s = str(s).strip()
            if not s:
                return None

            # Remove anything after ';' (edition info)
            s = s.split(";")[0]

            # Remove parenthetical content
            s = re.sub(r"\(.*?\)", "", s)

            # Normalize whitespace
            s = re.sub(r"\s+", " ", s).strip()

            return s if s else None

        def clean_language(s):
            if s is None:
                return None
            s = str(s).strip()
            if not s:
                return None

            # Normalize capitalization (English, French, German, etc.)
            s = s.title()

            return s 
        
        if not isinstance(details, dict):
            return None, None

        publisher = clean_publisher(details.get("Publisher"))
        language = clean_language(details.get("Language"))

        return publisher, language


     
                