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

from utils import DotDict


class AmazonECommerceDataset:
    def __init__(self, config: DotDict, **kwargs):
        self.config = config

        self.dataset_root_path = f"./data"
        self.dataset_filename = "amazon-purchases.csv"

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
    
    def setup(self) -> None:
        # if not os.path.exists(self.unrolled_dataset_total_path):
        if not self.config.use_cache or not os.path.exists(self.unrolled_dataset_total_path):
            df_original = pd.read_csv(os.path.join(self.dataset_root_path, self.dataset_filename), sep=",", low_memory=False)
            # Clean up
            nan_columns = ['Order Date', 'Quantity', 'Shipping Address State', 'ASIN/ISBN (Product Code)', 'Category', 'Survey ResponseID']
            df_filtered = df_original.dropna(subset=nan_columns)
            df_filtered = df_filtered.sort_values('Order Date')

            # By default use category of the items as item_it
            rename_dict = {
                'Order Date': 'date',
                'Purchase Price Per Unit': 'price_per_unit',
                'Quantity': 'quantity',
                'Shipping Address State': 'state',
                'ASIN/ISBN (Product Code)': 'aisin',
                'Category': 'item_id', # GROUP BY CATEGORY
                'Survey ResponseID': 'user_id'
            }

            df_filtered = df_filtered.rename(columns=rename_dict)

            # Unroll to be sure about implicit feedback format
            new_working_df = []
            for _, row in df_filtered.iterrows():
                for _ in range(int(row['quantity'])):
                    new_row = row
                    new_row["quantity"] = 1
                    new_working_df.append(new_row)
            
            self.unrolled_dataset_total = pd.DataFrame(new_working_df, columns=df_filtered.columns)
            self.unrolled_dataset_total.to_csv(self.unrolled_dataset_total_path)

        elif self.config.use_cache and os.path.exists(self.unrolled_dataset_total_path):
            self.unrolled_dataset_total = pd.read_csv(self.unrolled_dataset_total_path)
        else:
            raise Exception(f"\n Cache not found -> {self.unrolled_dataset_total_path} \n")
        
        # Set up dataframe to be used for the simulation from now on
        self.unrolled_dataset_total = self.unrolled_dataset_total[['user_id', 'item_id', 'date']]
        self.unrolled_dataset_total['date'] = pd.to_datetime(self.unrolled_dataset_total['date'], format="%Y-%m-%d")

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