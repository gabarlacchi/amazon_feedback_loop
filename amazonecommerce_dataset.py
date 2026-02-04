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


class AmazonECommerceDataset:
    def __init__(self, config: DotDict, **kwargs):
        self.config = config

        self.dataset_root_path = f"./data"
        self.dataset_filename = "amazon-purchases.csv"
        self.dataset_features_filename = "survey.csv"

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

        self.distribution_timeline_path = f"./cache/distribution_timeline_amazonecommerce.csv"
        self.unrolled_dataset_total_path = f"./cache/unrolled_total_{self.y1}-{self.y2}.csv"
        self.cache_mapping_strings_int_path_users = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_users.csv"
        self.cache_mapping_strings_int_path_items = f"./cache/mapping_strings_int_{self.y1}-{self.y2}_users.csv"
        self.user_id = "user_id:token"
        self.item_id = "tracks_id:token"
        self.timestamp = "timestamp:float"
        self.dump_dataset = f"./cache/real_dataset_sim_amazon"
        os.makedirs(self.dump_dataset, exist_ok=True)
    
    def setup(self)-> None:
        if not self.config.use_cache or not os.path.exists(self.unrolled_dataset_total_path):
            df_dataset = pd.read_csv(os.path.join(self.dataset_root_path, self.dataset_filename), sep=",", low_memory=False)
            
            df_features = pd.read_csv(os.path.join(self.dataset_root_path, self.dataset_features_filename), sep=",", low_memory=False)
            df_features = df_features[
                [
                    "Survey ResponseID",
                    "Q-demos-state",
                    "Q-demos-age", 
                    "Q-demos-race", 
                    "Q-demos-education",
                    "Q-demos-income",
                    "Q-demos-gender",
                    "Q-amazon-use-how-oft",
                    "Q-substance-use-cigarettes",
                    "Q-personal-diabetes",
                ]
            ]
            rename_dict = {
                'Q-demos-state': 'us_origin_state',
                'Q-demos-age': 'age',
                'Q-demos-race': 'race',
                'Q-demos-education': 'education',
                'Q-demos-income': 'income',
                'Q-demos-gender': 'gender',
                'Q-amazon-use-how-oft': 'how_often_use_amazon',
                'Q-substance-use-cigarettes': 'smoke_cigarettes',
                'Q-personal-diabetes': 'has_diabet',
                'Survey ResponseID': 'user_id'
            }

            df_features = df_features.rename(columns=rename_dict)

            nan_columns = ['Order Date', 'Quantity', 'Shipping Address State', 'ASIN/ISBN (Product Code)', 'Category', 'Survey ResponseID']
            df_dataset = df_dataset.dropna(subset=nan_columns)

            # Assure date format
            if not is_datetime(df_dataset['Order Date']):
                parsed = pd.to_datetime(df_dataset['Order Date'], errors="coerce")  # set dayfirst=True if needed
                bad = df_dataset.loc[parsed.isna() & df_dataset['Order Date'].notna(), 'Order Date']
                if not bad.empty:
                    # You can inspect or fix these rows; abort instead of silently sorting wrong
                    raise ValueError(f"{len(bad)} values in '{'Order Date'}' failed to parse, e.g.: {bad.head().tolist()}")
                df_dataset['Order Date'] = parsed
            
            df_dataset = df_dataset.sort_values('Order Date', na_position="last")
            
            rename_dict = {
                    'Order Date': 'date',
                    'Purchase Price Per Unit': 'price_per_unit',
                    'Quantity': 'quantity',
                    'Shipping Address State': 'state',
                    'Survey ResponseID': 'user_id',
                    'Category': 'category',
                    "ASIN/ISBN (Product Code)": 'item_id'
                }

            df_dataset = df_dataset.rename(columns=rename_dict)

            unroll_df = []
            for _, row in df_dataset.iterrows():
                for _ in range(int(row['quantity'])):
                    new_row = row
                    new_row["quantity"] = 1
                    unroll_df.append(new_row)
            
            df_unrolled_dataset_total = pd.DataFrame(unroll_df, columns=df_dataset.columns)

            self.unrolled_dataset_total = pd.merge(df_unrolled_dataset_total, df_features, how="left", on="user_id")

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
    
    def get_trasformed_dataset(self):
        """
            App function to map ids
        """
        df = self.unrolled_dataset_total.copy()

        # Map user_id to int
        unique_user_ids = df['user_id'].unique()
        user_id_mapping = {id_str: idx for idx, id_str in enumerate(unique_user_ids)}
        df['user_id_int'] = df['user_id'].map(user_id_mapping)
        user_mapping_df = pd.DataFrame(list(user_id_mapping.items()), columns=['original_user_id', 'int_user_id'])
        user_mapping_df.to_csv(self.cache_mapping_strings_int_path_users, index=False)
        # Map item_id to int (separate ID space)
        unique_item_ids = df['item_id'].unique()
        item_id_mapping = {id_str: idx for idx, id_str in enumerate(unique_item_ids)}
        df['item_id_int'] = df['item_id'].map(item_id_mapping)

        item_mapping_df = pd.DataFrame(list(item_id_mapping.items()), columns=['original_item_id', 'int_item_id'])
        item_mapping_df.to_csv(self.cache_mapping_strings_int_path_items, index=False)

        self.unrolled_dataset_total = df
    
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