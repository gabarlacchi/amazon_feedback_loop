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

from utils import DotDict

class LastFMDataset:
    def __init__(self, config: DotDict, **kwargs):
        self.config = config
        self.dataset_root_path = f"./data/lastfm-dataset-1K"
        self.dataset_filename = "userid-timestamp-artid-artname-traid-traname.tsv"
        self.dataset_features_filename = "userid-profile.tsv"
        self.item_level = self.config.item_level # must be a column within the dataframe
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
        self.distribution_timeline_path = f"./cache/distribution_timeline_lastfm.csv"
        self.unrolled_dataset_total_path = f"./cache/unrolled_total_{self.y1}-{self.y2}_lastfm.csv"
        self.users_rating_distribution_path = f"./cache/users_rating_distribution_{self.y1}-{self.y2}_lastfm.csv"
        self.global_fallback_rating_distribution_path = f"./cache/global_fallback_rating_distribution_{self.y1}-{self.y2}_lastfm.csv"
        self.cache_mapping_strings_int_path_users = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_users_lastfm.csv"
        self.cache_mapping_strings_int_path_items = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_items_lastfm.csv"
        self.dump_dataset = f"./cache/real_dataset_sim_lastfm"
        os.makedirs(self.dump_dataset, exist_ok=True)

        self.item_id_col = self.config.item_level
    
    def setup(self) -> None:
        if not self.config.use_cache or not os.path.exists(self.unrolled_dataset_total_path):
            # Load main listening data
            df = pd.read_csv(
                os.path.join(self.dataset_root_path, self.dataset_filename),
                sep='\t',
                names=['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name'],
                encoding='utf-8',
                on_bad_lines='skip'
            )

            df['date'] = pd.to_datetime(df['timestamp'])
            df = df.dropna()
            df = df[df['date'].dt.year.between(self.y1, self.y2)]
            
            # Load user profile features
            user_features_path = os.path.join(self.dataset_root_path, self.dataset_features_filename)
            if os.path.exists(user_features_path):
                df_features = pd.read_csv(
                    user_features_path,
                    sep='\t',
                    header=None,
                    index_col=False,
                    names=['user_id', 'gender', 'age', 'country'],
                    encoding='utf-8',
                    on_bad_lines='skip'
                )

                # Merge features with main dataset
                self.unrolled_dataset_total = df.merge(df_features, on='user_id', how='left')

            to_remove = ["artist_name", "track_name"]
            if self.item_id_col == "artist_id":
                to_remove += ["track_id"]
                self.unrolled_dataset_total = self.unrolled_dataset_total.rename({"artist_id": "item_id"}, axis=1)
            elif self.item_id_col == "track_id":
                to_remove += ["artist_id"]
                self.unrolled_dataset_total = self.unrolled_dataset_total.rename({"track_id": "item_id"}, axis=1)
            else:
                raise Exception(f"{self.item_id_col} does not correspond to a columns or something's wrong")
            
            self.unrolled_dataset_total = self.unrolled_dataset_total.drop(to_remove, axis=1)

            # There are lots of interactions in the same timestamps. Reduce to a reasonable number max
            # Verbose analys - it print usefull statistics

            original_count = len(self.unrolled_dataset_total)
            print(f"Original number of interactions: {original_count:,}")

            feature_cols = ['gender', 'age', 'country']
            existing_feature_cols = [col for col in feature_cols if col in self.unrolled_dataset_total.columns]

            if existing_feature_cols:
                print("\nNaN counts per feature column:")
                for col in existing_feature_cols:
                    nan_count = self.unrolled_dataset_total[col].isna().sum()
                    print(f"  {col}: {nan_count:,} ({nan_count/len(self.unrolled_dataset_total)*100:.2f}%)")
                
                # Count NaN values per row across feature columns
                nan_count_per_row = self.unrolled_dataset_total[existing_feature_cols].isna().sum(axis=1)
                
                print("\nDistribution of NaN values per row:")
                for i in range(len(existing_feature_cols) + 1):
                    count = (nan_count_per_row == i).sum()
                    print(f"  {i} NaN(s): {count:,} rows ({count/len(self.unrolled_dataset_total)*100:.2f}%)")
                
                # Remove rows with 2 or more NaN values in feature columns
                rows_to_remove = nan_count_per_row >= 2
                
                print(f"\nRows with 2+ NaN values (to be removed): {rows_to_remove.sum():,}")
                
                self.unrolled_dataset_total = self.unrolled_dataset_total[~rows_to_remove]
                
                removed_count = original_count - len(self.unrolled_dataset_total)
                print(f"\nInteractions removed due to NaN (2+ columns): {removed_count:,} ({removed_count/original_count*100:.2f}%)")
                print(f"Remaining interactions: {len(self.unrolled_dataset_total):,}")
                
                # Show what remains
                remaining_nan_per_row = self.unrolled_dataset_total[existing_feature_cols].isna().sum(axis=1)
                print(f"\nAfter removal - rows with 1 NaN: {(remaining_nan_per_row == 1).sum():,}")
                print(f"After removal - rows with 0 NaN: {(remaining_nan_per_row == 0).sum():,}")
            else:
                print("No feature columns found - skipping NaN removal")
            
            grouped = self.unrolled_dataset_total.groupby(['user_id', 'item_id', 'timestamp', 'date']).size().reset_index(name='count')

            print(f"\nTotal unique (user, item, timestamp) combinations: {len(grouped):,}")
            duplicates_count = (grouped['count'] > 1).sum()
            print(f"Combinations with duplicates (count > 1): {duplicates_count:,}")

            if duplicates_count > 0:
                
                # Apply limit
                limit = 10
                exceeds_limit = grouped[grouped['count'] > limit]
                
                if len(exceeds_limit) > 0:
                    print(f"Combinations exceeding limit: {len(exceeds_limit):,}")
                    print(f"Total interactions in these combinations: {exceeds_limit['count'].sum():,}")
                    print(f"Interactions that would be removed: {(exceeds_limit['count'] - limit).sum():,}")
                    
                    before_limit = len(self.unrolled_dataset_total)
                    
                    # Keep only first N occurrences for each (user, item, timestamp) group
                    self.unrolled_dataset_total = self.unrolled_dataset_total.groupby(
                        ['user_id', 'item_id', 'timestamp']
                    ).head(limit).reset_index(drop=True)
                    
                    after_limit = len(self.unrolled_dataset_total)
                    removed_by_limit = before_limit - after_limit
                    
                    print(f"\nInteractions before limit: {before_limit:,}")
                    print(f"Interactions after limit: {after_limit:,}")
                    print(f"Interactions removed by limit: {removed_by_limit:,} ({removed_by_limit/before_limit*100:.2f}%)")
                else:
                    print(f"\nNo combinations exceed limit of {limit} - no filtering needed")
            else:
                print("\n No duplicate (user, item, timestamp) combinations found")

            self.unrolled_dataset_total.to_csv(self.unrolled_dataset_total_path)
        else:
            self.unrolled_dataset_total = pd.read_csv(self.unrolled_dataset_total_path, index_col=0)

    def real_dataset_save_cache(self, start_date: datetime, end_date: datetime, users: list, items: list):
        """
            Run the loop over the selected epochs and save the results into the cache folder, as dump of the real dataset
            It takes the format will have the results of the simulation

            ! THE FIRST DATAFRAME OF THE LIST CONTAINS THE X-MONTHS INITIALIZATION DATA !
            ! THE REST OF THE DATAFRAMES WILL REFER TO EACH EPOCH (USUALLY 1 MONTH) EACH ONE !
        """
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
            
            # Sanity check of wrote at the head
            # d_init = pd.read_csv(os.path.join(self.dump_dataset, "epoch_0.csv"), index_col=0)

            # months = [datetime.strptime(date, '%Y-%m-%d').month for date in d_init['date'].unique()]
            # if len(months) != self.config.epochs:
            #     raise Exception(f"\n The real dump of the dataset is wrong. The initialization (first dataframe) contins a different number of epochs")
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