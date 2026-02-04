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

class MovielensDataset:
    def __init__(self, config: DotDict, **kwargs):
        self.config = config

        self.ratings_path = "./data/ml-32m/ratings.csv"
        self.movies_path = "./data/ml-32m/movies.csv"
        self.tags_path = "./data/ml-32m/tags.csv"

        self.item_level = self.config.item_level # must be a column within the dataframe
        # if not os.path.exists(os.path.join(self.dataset_root_path, self.dataset_filename)):
        #     raise Exception(f"\n Dataset not found -> {os.path.join(self.dataset_root_path, self.dataset_filename)} does not exists \n")
        try:
            if (re.match(r"^\d{4}-\d{4}$", self.config.time_window) is not None):
                splitted = (self.config.time_window).split("-")
                self.y1, self.y2 = int(splitted[0]), int(splitted[1])
                self.n_years = (self.y2 - self.y1) + 1
            else:
                raise Exception(f"\n Time window not recognized -> {self.config.time_window}. It must be something like '2011-2012' \n")
        except Exception as e:
            print(traceback.format_exc())
        self.distribution_timeline_path = f"./cache/distribution_timeline_movielens.csv"
        self.unrolled_dataset_total_path = f"./cache/unrolled_total_{self.y1}-{self.y2}_movielens.csv"
        self.features_dataset_total_path = f"./cache/unrolled_total_{self.y1}-{self.y2}_features_movielens.csv"
        self.users_rating_distribution_path = f"./cache/users_rating_distribution_{self.y1}-{self.y2}_movielens.csv"
        self.global_fallback_rating_distribution_path = f"./cache/global_fallback_rating_distribution_{self.y1}-{self.y2}_movielens.csv"
        self.cache_mapping_strings_int_path_users = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_users_movielens.csv"
        self.cache_mapping_strings_int_path_items = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_items_movielens.csv"
        self.dump_dataset = f"./cache/real_dataset_sim_lastfm"
        os.makedirs(self.dump_dataset, exist_ok=True)

        self.item_id_col = self.config.item_level

    def setup(self) -> None:
        if not self.config.use_cache or not os.path.exists(self.unrolled_dataset_total_path):
            df_ratings = pd.read_csv(
                self.ratings_path,
                dtype={
                    'userId': 'int32',
                    'movieId': 'int32', 
                    'rating': 'float32',
                    'timestamp': 'int64'
                }
            )
            
            df_movies = pd.read_csv(
                self.movies_path,
                dtype={'movieId': 'int32'}
            )
            
            df_tags = pd.read_csv(
                self.tags_path,
                dtype={
                    'userId': 'int32',
                    'movieId': 'int32',
                    'timestamp': 'int64'
                }
            )
            
            df_ratings['date'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

            df_ratings['year'] = df_ratings['date'].dt.year
            df_ratings = df_ratings[
                (df_ratings['year'] >= self.y1) & 
                (df_ratings['year'] <= self.y2)
            ]
            df_ratings.drop(columns=["year"])

            df_ratings = df_ratings.merge(
                df_movies,
                on='movieId',
                how='left'
            )

            df_tags['date'] = pd.to_datetime(df_tags['timestamp'], unit='s')
            df_tags['year'] = df_tags['date'].dt.year
            df_tags = df_tags[
                (df_tags['year'] >= self.y1) & 
                (df_tags['year'] <= self.y2)
            ].copy()
            df_tags.drop(columns=['year'], inplace=True)
            
            # Tags are noisy, keep the first 10
            df_tags_agg = df_tags.groupby('movieId')['tag'].apply(
                lambda x: '|'.join(x.astype(str).str.lower().unique()[:10])
            ).reset_index()
            df_tags_agg.columns = ['movieId', 'tags']
            
            # Merge tags into movies dataframe
            df_movies = df_movies.merge(
                df_tags_agg,
                on='movieId',
                how='left'
            )
            
            # Fill movies without tags
            df_movies['tags'] = df_movies['tags'].fillna('')

            self.unrolled_dataset_total = df_ratings
            self.df_movies = df_movies

            self.unrolled_dataset_total.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}, inplace=True)
            self.df_movies.rename(columns={'movieId': 'item_id'}, inplace=True)

            self.unrolled_dataset_total.drop(columns=["year", "title", "genres"], inplace=True)

            print(list(self.unrolled_dataset_total.columns))
            print(list(self.df_movies.columns))
            print(df_movies.head(10))
            print(self.unrolled_dataset_total.head(10))

            self.unrolled_dataset_total.to_csv(self.unrolled_dataset_total_path)
            self.df_movies.to_csv(self.features_dataset_total_path)
            
        else:
            self.unrolled_dataset_total = pd.read_csv(self.unrolled_dataset_total_path)
            self.df_movies = pd.read_csv(self.features_dataset_total_path)

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
        

