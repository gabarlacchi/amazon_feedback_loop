from datetime import date, datetime, timedelta
from datetime import time as time_datetime
from dateutil.relativedelta import relativedelta
import dask.dataframe as dd
import pandas as pd
import os
import random
import pickle
import numpy as np
from logging import getLogger
import logging
from tqdm import tqdm 
import time
import torch
import matplotlib.pyplot as plt 
import math
import json
import networkx as nx
from scipy.sparse import csr_matrix
import traceback 
from multiprocessing import Pool
import torch.multiprocessing
import re
import shutil

from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.data import create_dataset, data_preparation
# from recbole.model.general_recommender import Pop, Random, MultiVAE
from recbole.trainer import Trainer
from recbole.data.interaction import Interaction
from recbole.trainer import HyperTuning
from recbole.utils import get_model, get_trainer

from utils import DotDict, get_consistent_users, _setup_repro
# from custom_models import UserKNN, IndividualRandom, IndividualPopularity, LightGCN, BPR, SpectralCF, NeuMF, NNCF
from custom_models import NeuMF, BPR, UserKNN, SpectralCF, FM, DeepFM, ItemKNN, EASE, NFM, DCNV2
from amazonecommerce_dataset import AmazonECommerceDataset
from choice_model import ChoiceModel

class AmazonECommerceFeedbackLoop():
    def __init__(self, config: DotDict, initialization_dataset: AmazonECommerceDataset, **kwargs):

        self.master_seed = int(getattr(config, "seed", 42))
        self.rng = _setup_repro(self.master_seed)

        self.config = config
        self.initialization_dataset = initialization_dataset

        self.first_avialable_date = date(self.initialization_dataset.y1, 1, 1)
        self.last_avialable_date = date(self.initialization_dataset.y2, 12, 31)

        self.model_name_config = self.config.recommender_model.model_name
        self.model_name_recbole = None
        
        self.metric_dict = {
            "epoch_index": 0,
        }

        self.train_window_months = int(getattr(self.config, "fixed_window_size", 6))
        # Every epoch: move the window size or consider from the start moving last month?
        self.use_all_data = bool(getattr(self.config, "use_all_from_start", True))

        self.unrolled_dataset_path = initialization_dataset.unrolled_dataset_total_path
        self.distribution_timeline_path = initialization_dataset.distribution_timeline_path

        # Define work folder, to remove at the end of the simulation
        self.tmp_folder =f"./{self.config.recbole_folder}"
        self.tmp_dataset_folder = "experiment_dataset"
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
        if not os.path.exists(os.path.join(self.tmp_folder, self.tmp_dataset_folder)):
            os.makedirs(os.path.join(self.tmp_folder, self.tmp_dataset_folder))

        self.test_client_list = "./test_client_ids.pkl"

        self.top_k_users_scores = {}
        self.all_users_scores = {}

        self.usr_strategy_ranking = {}
    
    def set_repetition_seed(self, p: float, rep_idx: int):
        """
        Derive a deterministic child seed from (master_seed, p, rep_idx),
        then re-seed NumPy, Python, and PyTorch + RecBole.
        """
        p_key = int(round(float(p) * 1_000_000))  # stable integer key for p
        ss = np.random.SeedSequence(self.master_seed, spawn_key=[p_key, rep_idx])
        # main RNG for all NumPy draws in this repetition
        self.rng = np.random.default_rng(ss)

        # Also seed Python's random from the same seed space (if you still use it anywhere)
        import random as pyrandom
        py_child = ss.spawn(1)[0]
        pyrandom.seed(int(py_child.generate_state(1, dtype=np.uint32)[0]))

       
        try:
            torch.manual_seed(self.master_seed + p_key + rep_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.master_seed + p_key + rep_idx)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        if hasattr(self, "parameter_dict"):
            self.parameter_dict["seed"] = int(ss.generate_state(1, dtype=np.uint32)[0])
            self.parameter_dict["reproducibility"] = True
    
    def init_experiment(self) -> None:
        # Set up unrolled dataset original with columns based on the item id chosen
        df_dataset_total = self.initialization_dataset.unrolled_dataset_total
        self.item_id_col = self.config.item_level
        if not self.item_id_col in list(df_dataset_total.columns):
            raise Exception(f"item level must be a column of the dataeset -> {self.item_id_col} not found in dataset")

        if self.item_id_col == "category":
            # Remove item_id and title cols
            df_dataset_total.drop(["item_id", "Title"], axis=1, inplace=True)
            df_dataset_total = df_dataset_total.rename({"category": "item_id"}, axis=1)

            self.initialization_dataset.unrolled_dataset_total = df_dataset_total

        df_dataset_total['date'] = pd.to_datetime(df_dataset_total['date'], format="%Y-%m-%d")
        df_dataset_total['timestamp'] = df_dataset_total.date.values.astype(np.int64) // 10 ** 9

        # Get the first available date 
        self.cold_start_start_date = df_dataset_total['date'].min().date()
        self.cold_start_end_date = self.cold_start_start_date + relativedelta(months=self.config.cold_start_months) - relativedelta(days=1)

        self.cold_start_start_datetime = datetime(self.cold_start_start_date.year, self.cold_start_start_date.month, self.cold_start_start_date.day)
        self.cold_start_end_datetime = datetime(self.cold_start_end_date.year, self.cold_start_end_date.month, self.cold_start_end_date.day)

        start_date = datetime(self.initialization_dataset.y1, 1, 1)
        end_date = start_date + relativedelta(months=self.config.cold_start_months) - relativedelta(days=1)

        self.dataset_unrolled_cold_start = df_dataset_total[(df_dataset_total['date'].dt.date >= start_date.date()) & (df_dataset_total['date'].dt.date <= end_date.date())]
        
        # Select users that interacted at least once per month for every year
        self.users_ids = get_consistent_users(df=df_dataset_total, user_col="user_id", min_months_per_year=1)
        sampled_users = pd.Series(self.users_ids)
        self.dataset_unrolled_cold_start = self.dataset_unrolled_cold_start[self.dataset_unrolled_cold_start['user_id'].isin(sampled_users)]

        self.start_experiment_date = self.cold_start_end_date + timedelta(days=1)

        # Reomve users with less than 10 interactions in the initialization phase
        user_counts = self.dataset_unrolled_cold_start["user_id"].value_counts()
        active_users = user_counts[user_counts >= 10].index
        self.dataset_unrolled_cold_start = self.dataset_unrolled_cold_start[self.dataset_unrolled_cold_start["user_id"].isin(active_users)]
        
        # If max_users is not null and an integer, randomly select max_users, delete the others
        if self.config.max_users is not None and isinstance(self.config.max_users, int):
            print(f"\n --- RANDOMLY SELECT {self.config.max_users} FOR THE SIMULATION --- \n")
            all_users = self.dataset_unrolled_cold_start["user_id"].unique()
            selected_users = self.rng.choice(all_users, size=self.config.max_users, replace=False)
            self.dataset_unrolled_cold_start = self.dataset_unrolled_cold_start[
                self.dataset_unrolled_cold_start["user_id"].isin(selected_users)
            ]

        self.users_ids = self.dataset_unrolled_cold_start.user_id.unique().tolist()
        self.items_ids = self.dataset_unrolled_cold_start.item_id.unique().tolist()

        print(f"\n Initialization of the experiment. \n The number of users partecipating at the simulation is: {len(self.users_ids)}. \n The number of items partecipaing at the simulation is: {len(self.items_ids)}. \n")
        
        # If use_cahe=False or no cache files exist, these functions will take time
        self.initialization_dataset.real_dataset_save_cache(start_date=self.start_experiment_date, end_date=self.last_avialable_date, users=self.users_ids, items=self.items_ids)
        self.experiment_distribution_dict = self.initialization_dataset.strategy_simulation_info(start_date=self.start_experiment_date, end_date=self.last_avialable_date, users=self.users_ids, items=self.items_ids)

    # HELPER FUNCTIONS

    @staticmethod
    def _to_bool01(series, true_vals=["yes", "Yes", "y", "Y", "1"], false_vals=["no", "No", "n", "N", "0"]):
        s = series.astype(str).str.strip().str.lower()
        arr = np.where(
            s.isin(true_vals), 1.0,
            np.where(s.isin(false_vals), 0.0, np.nan)
        )
        return pd.Series(arr, index=series.index, dtype="float")
    
    @staticmethod
    def _ensure_numeric(series, fill=0.0):

        return pd.to_numeric(series, errors="coerce").fillna(fill).round(2)
    
    @staticmethod
    def _simple_tokenize(text, max_len=16):
        if pd.isna(text):
            return ""
        return " ".join(str(text).strip().lower().split()[:max_len])

    @staticmethod
    def _rome_midnight_and_epoch(dt_series):
        dt = pd.to_datetime(dt_series, utc=True, errors="coerce").dt.tz_convert("Europe/Rome").dt.normalize()
        ts = (dt.astype("int64") // 10**9).astype(float)

        return dt, ts

    @staticmethod
    def _pre_process_dataset_features(df):
        df = df.copy()

        # USERS FEATURES
        # Age
        age_map = {
            '18 - 24 years': 21,
            '25 - 34 years': 30,
            '35 - 44 years': 40,
            '45 - 54 years': 50,
            '55 - 64 years': 60,
            '65 and older': 70
        }
        df['age'] = df['age'].map(age_map).astype('float32')

        # Income
        income_map = {
            'less than $25,000': 12.5,
            '$25,000 - $49,999': 37.5,
            '$50,000 - $74,999': 62.5,
            '$75,000 - $99,999': 87.5,
            '$100,000 - $149,999': 125.0,
            '$150,000 or more': 175.0
        }
        df['income'] = df['income'].map(income_map).astype('float32')

        # Education
        education_map = {
            'some high school or less': 'some_high_school',
            'high school diploma or ged': 'high_school',
            "bachelor's degree": 'bachelor',
            'graduate or professional degree (ma, ms, mba, etc.)': 'graduate',
            'prefer not to say': 'unknown'
        }
        df['education'] = (
            df['education']
            .map(education_map)
            .fillna('unknown')
            .astype('string')
        )

        # Gender
        gender_map = {
            'Male': 'M',
            'Female': 'F',
            'Other': 'O',
            'prefer not to say': 'unknown'
        }

        df['gender'] = (
            df['gender']
            .astype('string')
            .map(gender_map)
            .fillna('unknown')
        )

        race_categories = [
            'white or caucasian',
            'black or african american',
            'asian',
            'american indian/native american or alaska native',
            'native hawaiian or other pacific islander',
            'other'
        ]

        # normalize race text
        df['race'] = (
            df['race']
            .astype('string')
            .str.lower()
            .fillna('')
        )
        #Group the race
        race_categories = [
            'white or caucasian',
            'black or african american',
            'asian',
            'american indian/native american or alaska native',
            'native hawaiian or other pacific islander',
            'other'
        ]

        for race in race_categories:
            col_name = 'race_' + re.sub(r'[^a-z]+', '_', race).strip('_')
            df[col_name] = df['race'].str.contains(race, regex=False).astype('float32')

        df = df.drop(columns=['race'])

        # Amazon frequency
        freq_map = {
            'less than 5 times per month': 2.5,
            '5 - 10 times per month': 7.5,
            'more than 10 times per month': 15.0
        }

        df['amazon_freq'] = (
            df['how_often_use_amazon']
            .map(freq_map)
            .astype('float32')
        )
        df = df.drop(columns=['how_often_use_amazon'])
        
        df['user_id'] = df['user_id'].astype('string')

        return df
    
    @staticmethod
    def _unroll_by_interaction_count(df: pd.DataFrame):
        if "interaction_count:float" not in df.columns:
            return df.copy()

        cnt = (
            pd.to_numeric(df["interaction_count:float"], errors="coerce")
            .fillna(1)
            .astype(int)
            .clip(lower=1)
        )
        idx = np.repeat(np.arange(len(df)), cnt.to_numpy())
        unrolled = df.iloc[idx].copy()
        unrolled["interaction_count:float"] = 1.0

        return unrolled

    def init_recbole_model(self, is_first_init=False, warm_start=True, results_path=None):

        # Some models need custom implemention, add here
        CUSTOM_MODELS = {
            "NeuMF",
            "BPR",
            "UserKNN",
            "ItemKNN",
            "SpectralCF",
            "FM",
            "DeepFM",
            "EASE",
            "NFM",
            "FM",
            "DCNV2"
        }
        custom_model_map = {
            "NeuMF": NeuMF,
            "BPR": BPR,
            "UserKNN": UserKNN,
            "ItemKNN": ItemKNN,
            "SpectralCF": SpectralCF,
            "FM": FM,
            "DeepFM": DeepFM,
            "EASE": EASE,
            "NFM": NFM,
            "FM": FM,
            "DCNV2": DCNV2
        }

        if is_first_init or not hasattr(self, 'model_config'):
            name_map = {
                "Collective Popularity": "Pop",
                "Collective Random": "Random",
                "CF_KNN_item": "ItemKNN",
                "CF_KNN_user": "UserKNN",
                "MultiVAE": "MultiVAE",
                "BPR": "BPR",
                "LightGCN": "LightGCN",
                "SpectralCF": "SpectralCF",
                "NGCF":"NGCF",
                "SGL": "SGL",
                "EASE": "EASE",
                "NeuMF": "NeuMF",
                "DeepFM": "DeepFM", 
                "xDeepFM": "xDeepFM",
                "DCNV2": "DCNV2",
                "FM": "FM",
                "NFM": "NFM"
            }

            try:
                self.raw_model_name = self.config.recommender_model.model_name
                self.model_name_recbole = name_map[self.raw_model_name]
                self.use_custom_model = self.model_name_recbole in CUSTOM_MODELS
            except Exception as e:
                traceback.print_exc()
                raise Exception(f"{e if isinstance(e, KeyError) else str(e)}")

            CONTEXT_AWARE_MODELS = {"DeepFM", "xDeepFM", "DCNV2", "FM", "NFM"}
            needs_features = self.model_name_recbole in CONTEXT_AWARE_MODELS
            # !!
            needs_features = True

            base_inter_cols = ["user_id", "item_id", "timestamp", "label"]

            # if needs_features:
            #     base_inter_cols.append("interaction_count")
            
            if needs_features:
                user_cols = ["user_id", "age", "country", "gender"]
                item_cols = ["item_id"]

            else:
                user_cols = None
                item_cols = None

            self.parameter_dict = {
                'data_path': self.tmp_folder,
                'checkpoint_dir': os.path.join(self.tmp_folder, "checkpoints"),

                "USER_ID_FIELD": "user_id",
                "ITEM_ID_FIELD": "item_id",
                "TIME_FIELD": "timestamp",
                "LABEL_FIELD": "label",

                "load_col": {
                    "inter": base_inter_cols,
                },

                # Training settings
                "epochs": getattr(self.config.recommender_model, "epochs", 10),
                "eval_args": {
                    "group_by": "user", 
                    "order": "TO",
                    "mode": "full"
                },
                "benchmark_filename": ["train", "val", "test"],
                
                # Reproducibility
                "reproducibility": True,
                "seed": self.master_seed,

                # Evaluation metrics
                # In KDD catena for some reasone does not works
                # "metrics": ["NDCG", "Recall", "Precision", "Hit", "ItemCoverage", "MRR", "MAP"],
                "metrics": ["NDCG", "Recall", "Precision", "Hit", "MRR", "MAP"],
                "topk": 10,
                "valid_metric": "NDCG@10",

                # Negative sampling for implicit feedback
                "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},

                # GPU settings
                "use_gpu": False,
                "gpu_id": 0,
            }

            if user_cols:
                self.parameter_dict["load_col"]["user"] = user_cols
            if item_cols:
                self.parameter_dict["load_col"]["item"] = item_cols

            # Helper function to add parameters if they exist
            def add_param(attr, key=None):
                if model_cfg and hasattr(model_cfg, attr):
                    val = getattr(model_cfg, attr)
                    if val is not None:
                        self.parameter_dict[key or attr] = val

            model_cfg = getattr(self.config, "recommender_model", None)

            add_param("learning_rate")
            add_param("epochs")

            # Model-specific parameters
            MODEL_PARAMS = {
                "Random": [],
                "ItemKNN": [("k", "k"), ("shrink", "shrink")],
                "UserKNN": [("k", "k"), ("shrink", "shrink")],
                "BPR": [("reg_weight", "reg_weight")],
                "LightGCN": [("reg_weight", "reg_weight"), ("n_layers", "n_layers")],
                "NeuMF": [
                    ("mlp_hidden_size", "mlp_hidden_size"),
                    ("user_embedding_size", "mlp_embedding_size"),
                    ("item_embedding_size", "mf_embedding_size"),
                    ("dropout_prob", "dropout_prob"),
                ],
                "MultiVAE": [
                    ("mlp_hidden_size", "mlp_hidden_size"),
                    ("dropout_prob", "dropout_prob"),
                    ("latent_dimension", "latent_dimension")
                ],
                "SpectralCF": [("n_layers", "n_layers"), ("reg_weight", "reg_weight")],
                "NGCF": [
                    ("embedding_size", "embedding_size"),
                    ("hidden_size_list", "hidden_size_list"),
                    ("node_dropout", "node_dropout"),
                    ("message_dropout", "message_dropout"),
                    ("reg_weight", "reg_weight")
                ],
                "SGL": [
                    ("embedding_size", "embedding_size"),
                    ("n_layers", "n_layers"),
                    ("reg_weight", "reg_weight"),
                    ("ssl_tau", "ssl_tau"),
                    ("ssl_weight", "ssl_weight"),
                    ("drop_ratio", "drop_ratio"),
                    ("type", "type")
                ],
                "EASE": [("reg_weight", "reg_weight")],
                "DeepFM": [
                    ("embedding_size", "embedding_size"),
                    ("mlp_hidden_size", "mlp_hidden_size"),
                    ("dropout_prob", "dropout_prob")
                ],
                "xDeepFM": [
                    ("embedding_size", "embedding_size"),
                    ("mlp_hidden_size", "mlp_hidden_size"),
                    ("dropout_prob", "dropout_prob"),
                    ("reg_weight", "reg_weight"),
                    ("cin_layer_size", "cin_layer_size"),
                    ("direct", "direct")
                ],
                "DCNV2": [
                    ("embedding_size", "embedding_size"),
                    ("cross_layer_num", "cross_layer_num"),
                    ("mlp_hidden_size", "mlp_hidden_size"),
                    ("dropout_prob", "dropout_prob"),
                    ("reg_weight", "reg_weight"),
                    ("structure", "structure"),
                    ("mixed", "mixed"),
                    ("expert_num", "expert_num"),
                    ("low_rank", "low_rank")
                ],
                "FM": [("embedding_size", "embedding_size")],
                "NFM": [
                    ("dropout_prob", "dropout_prob"),
                    ("mlp_hidden_size", "mlp_hidden_size")
                ]
            }

            for attr, key in MODEL_PARAMS[self.model_name_recbole]:
                add_param(attr, key)
            
            # NOT IT WORKS FOR STANDARD MODELS
            # FOR USER KNN THE LIBRARY DOES NOT RECOGNIZE THE MODEL
            # SO FAR, I DID NOT UNDERSTAND HOW TO HANDLE IT 
            # IF I UNCOMMENT THE FOLLOWING, THE OTHER CUSTOM MODELS WORKS DIFFERENTLY BECAUSE THE DEFAULT PARAMETERS ARE DIFFERENT
            # try:
            #     # For custom models, use a dummy valid model name for Config validation
            #     config_model_name = "BPR" if self.use_custom_model else self.model_name_recbole
                
            #     self.model_config = Config(
            #         model=config_model_name,  # Use dummy name for custom models
            #         dataset="experiment_dataset",
            #         config_dict=self.parameter_dict
            #     )
                
            #     # Override with actual model name after Config is created
            #     if self.use_custom_model:
            #         self.model_config["model"] = self.model_name_recbole
                
            #     init_seed(self.model_config["seed"], self.model_config["reproducibility"])
            #     init_logger(self.model_config)
            #     self.logger = getLogger()
            #     self.logger.info(f"[Epoch 0] Full initialization complete")
            # except Exception as e:
            #     raise Exception(f"Error during the configuration of the model -> {e}")

            # print(self.parameter_dict)
            # exit()

            try:
                self.model_config = Config(
                    model=self.model_name_recbole,
                    dataset="experiment_dataset",
                    config_dict=self.parameter_dict
                )
                init_seed(self.model_config["seed"], self.model_config["reproducibility"])
                init_logger(self.model_config)
                self.logger = getLogger()
                self.logger.info(f"[Epoch 0] Full initialization complete")
            except Exception as e:
                raise Exception(f"Error during the configuration of the model -> {e}")

            try:
                self.recbole_dataset = create_dataset(self.model_config)
                self.train_data, self.valid_data, self.test_data = data_preparation(
                    config=self.model_config,
                    dataset=self.recbole_dataset
                )
                self.logger.info(self.train_data)
            except Exception as e:
                print(traceback.format_exc())
                raise Exception(f"Error during the initialization of the dataset -> {e}")

            if self.use_custom_model:
                model_cls = custom_model_map[self.model_name_recbole]
            else:
                model_cls = get_model(self.model_config["model"])

            self.recbole_model = model_cls(
                self.model_config, 
                self.train_data.dataset
            ).to(self.model_config["device"])
            
            self._first_init_complete = True

            # !! Roughly initiated
            evaluate_initial = True

            if evaluate_initial:
                self.logger.info("[Epoch 0] Evaluating initial model before simulation")
                initial_metrics = self.evaluate_initial_model(show_progress=False, save_dir=results_path)
                self.logger.info(f"[Epoch 0] Initial metrics: {initial_metrics}")

        else:
            try:
                self.recbole_dataset = create_dataset(self.model_config)
                self.train_data, self.valid_data, self.test_data = data_preparation(
                    config=self.model_config,
                    dataset=self.recbole_dataset
                )

            except Exception as e:
                print(traceback.format_exc())
                raise Exception(f"Error during incremental dataset update -> {e}")
        
            if warm_start and hasattr(self, 'recbole_model'):
                old_n_users = self.recbole_model.n_users
                old_n_items = self.recbole_model.n_items
                new_n_users = self.train_data.dataset.user_num
                new_n_items = self.train_data.dataset.item_num
                
                if new_n_users > old_n_users or new_n_items > old_n_items:
                    self.logger.warning(
                        f"[Warm Start] Vocabulary expanded: "
                        f"users {old_n_users}→{new_n_users}, "
                        f"items {old_n_items}→{new_n_items}"
                    )
                    if self.use_custom_model:
                        model_cls = custom_model_map[self.model_name_recbole]
                    else:
                        model_cls = get_model(self.model_config["model"])
                    self.recbole_model = model_cls(
                        self.model_config, 
                        self.train_data.dataset
                    ).to(self.model_config["device"])
                else:
                    pass
            else:
                if self.use_custom_model:
                    model_cls = custom_model_map[self.model_name_recbole]
                else:
                    model_cls = get_model(self.model_config["model"])
                self.recbole_model = model_cls(
                    self.model_config, 
                    self.train_data.dataset
                ).to(self.model_config["device"])

    def tuning_hyperparameters(self) -> None:
        hyper_file = f"tuning_parameters/{self.model_name_config}.hyper"
        export_result_file = f"tuning_parameters/{self.model_name_config}.result"
        if self.model_name_config in ["Collective Random", "Collective Popularity"]:
            return None
        if os.path.exists(export_result_file):
            print(f"\n Tuning of the model {self.model_name_config} already in the folder. Skip \n")
            return None
        
        def _objective_function(params_dict=None, config_file_list=None):
            # Custom models that need special handling
            # UserKNN does not exists within the library
            CUSTOM_MODELS = {"UserKNN"}
            custom_model_map = {
                "UserKNN": UserKNN
            }
            
            is_custom = self.model_name_recbole in CUSTOM_MODELS
            
            # Use dummy model name for custom models to pass Config validation
            config_model_name = "Pop" if is_custom else self.model_name_recbole
            
            config = Config(
                model=config_model_name,
                dataset='experiment_dataset', 
                config_file_list=config_file_list,
                config_dict={**self.parameter_dict, **(params_dict or {})}
            )
            
            # Override with actual model name for custom models
            if is_custom:
                config["model"] = self.model_name_recbole
            
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config=config, dataset=dataset)
            
            # Use custom model class if needed
            if is_custom:
                model_cls = custom_model_map[self.model_name_recbole]
            else:
                model_cls = get_model(config["model"])
            
            model = model_cls(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
            test_result = trainer.evaluate(test_data)
            
            return {
                'model': self.model_name_config,
                'best_valid_score': best_valid_score,
                'valid_score_bigger': config['valid_metric_bigger'],
                'best_valid_result': best_valid_result,
                'test_result': test_result
            }
                
        if not os.path.exists(hyper_file):
            raise Exception(f"Cannot find the hyper file for model {self.model_name_config} -> {hyper_file}")
        
        hp = HyperTuning(objective_function=_objective_function, algo='exhaustive', max_evals=100, 
                        params_file=hyper_file, params_dict=self.parameter_dict)
        hp.run()
        hp.export_result(output_file=export_result_file)

        self.parameter_dict.update(hp.best_params)
        return None
    
    @staticmethod
    def _handle_cold_start_users(train_df, val_df, test_df):
        '''
        When split by time, a user can be in the val and/or test set but not in the training period.
        Add the first val interaction within the training period
        '''
        train_min_ts = train_df["timestamp:float"].min()
        train_max_ts = train_df["timestamp:float"].max()
        train_users = set(train_df["user_id:token"].unique())
        
        # Check validation set
        val_users = val_df["user_id:token"].unique()
        val_cold_users = set(val_users) - train_users
        if val_cold_users:
            # Suppose the interactions are sorted
            val_cold_mask = val_df["user_id:token"].isin(val_cold_users)
            first_interactions = val_df[val_cold_mask].groupby("user_id:token", sort=False).head(1).copy()
            np.random.seed(42)
            new_timestamps = np.random.uniform(
                train_min_ts, train_max_ts, size=len(first_interactions)
            )
            first_interactions["timestamp:float"] = new_timestamps
            # UPDATE THE DATE COLUMN TOO
            first_interactions["date"] = pd.to_datetime(new_timestamps, unit='s', utc=True).tz_convert("Europe/Rome").normalize()
            
            # Concatenate to training
            train_df = pd.concat([train_df, first_interactions], ignore_index=True)
            # Remove from validation using index
            val_df = val_df.drop(first_interactions.index).reset_index(drop=True)
        
        # Check test set
        train_users = set(train_df["user_id:token"].unique())
        test_users = test_df["user_id:token"].unique()
        test_cold_users = set(test_users) - train_users
        if test_cold_users:
            test_cold_mask = test_df["user_id:token"].isin(test_cold_users)
            first_interactions = test_df[test_cold_mask].groupby("user_id:token", sort=False).head(1).copy()
            new_timestamps = np.random.uniform(
                train_min_ts, train_max_ts, size=len(first_interactions)
            )
            first_interactions["timestamp:float"] = new_timestamps
            # UPDATE THE DATE COLUMN TOO
            first_interactions["date"] = pd.to_datetime(new_timestamps, unit='s', utc=True).tz_convert("Europe/Rome").normalize()
            
            train_df = pd.concat([train_df, first_interactions], ignore_index=True)
            test_df = test_df.drop(first_interactions.index).reset_index(drop=True)
        
        if val_cold_users or test_cold_users:
            train_df = train_df.sort_values(["user_id:token", "timestamp:float"], ignore_index=True)
        
        return train_df, val_df, test_df

    def evaluate_initial_model(self, save_dir: str = None, show_progress: bool = False):

        if not hasattr(self, 'recbole_model') or not hasattr(self, 'train_data'):
            raise RuntimeError("Model must be initialized first. Call init_recbole_model(is_first_init=True)")
        
        # Set up save directory
        save_stem = os.path.join(save_dir, "epoch_0")
        
        # Create trainer
        trainer = Trainer(self.model_config, self.recbole_model)
        
        self.logger.info("[Epoch 0] Starting initial model training and evaluation")
        self._fit_with_tracking(
            trainer=trainer,
            save_stem=save_stem,
            show_progress=show_progress,
            save_plots=False
        )

        # Load the final test metrics that were saved
        metrics_file = f"{save_stem}_final_test_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                final_metrics = json.load(f)
            self.logger.info(f"[Epoch 0] Initial evaluation complete. Metrics saved to {metrics_file}")
            return final_metrics
        else:
            self.logger.warning(f"[Epoch 0] Metrics file not found at {metrics_file}")
            return None
    
    def _build_window_for_epoch(self, k: int = 0, new_interactions = None):
        # FIRST INITIALIZATION OF THE DATASET
        if k == 0:
            # --- Features update only at the start ---

            # Delete tmp folder content every tim a new simulation goes 
            try:
                shutil.rmtree(self.tmp_folder)
                print("Folder deleted successfully")
            except FileNotFoundError:
                print("Folder does not exist")
            except PermissionError:
                print("Permission denied")
            except OSError as e:
                print(f"Error deleting folder: {e}")
            # R create empty
            if not os.path.exists(self.tmp_folder):
                os.makedirs(self.tmp_folder)
            if not os.path.exists(os.path.join(self.tmp_folder, self.tmp_dataset_folder)):
                os.makedirs(os.path.join(self.tmp_folder, self.tmp_dataset_folder))

            df = self.dataset_unrolled_cold_start.copy()

            # Features pre-processing
            df["age"] = df.get("age").map(self._simple_tokenize).fillna("")
            df["race"] = df.get("race").map(self._simple_tokenize).fillna("")
            df["education"] = df.get("education").map(self._simple_tokenize).fillna("")
            df["income"] = df.get("income").map(self._simple_tokenize).fillna("")
            df["gender"] = df.get("gender").astype(str).str.strip().str.title()
            df.loc[~df["gender"].isin(["Male", "Female", "Other"]), "gender"] = "Other"
            df["how_often_use_amazon"] = df.get("how_often_use_amazon").map(self._simple_tokenize).fillna("")
            df["smoke_cigarettes"] = self._to_bool01(df.get("smoke_cigarettes")).fillna(0)
            df["has_diabet"] = self._to_bool01(df.get("has_diabet")).fillna(0)
            
            if self.item_id_col == "item_id":
                df["category"] = df.get("category").map(self._simple_tokenize).fillna("")
                df["title"] = df.get("Title").map(self._simple_tokenize).fillna("")

            df = df.loc[df["user_id"].isin(self.users_ids)
                        & df["item_id"].isin(self.items_ids)].copy()

            base_time_col = "timestamp" if "timestamp" in df.columns else "date"
            df["__rome_date"], _ = self._rome_midnight_and_epoch(df[base_time_col])

            categorical_cols = ["user_id", "item_id", "age", "race", "education", 
                                "income", "gender", "how_often_use_amazon"]
            if self.item_id_col == "item_id":
                categorical_cols.extend(["category"])  # Don't convert 'title' as it's a sequence

            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')

            # User features
            user_feats = (
                df.sort_values("__rome_date")
                .drop_duplicates("user_id", keep="last")
                .loc[:, ["user_id", "age", "race", "education", "income", "gender",
                        "how_often_use_amazon", "smoke_cigarettes", "has_diabet"]]
            )
            user_feats = user_feats[user_feats["user_id"].isin(self.users_ids)]
            
            # Pre-processing user features - handle text and semantic concerns
            user_feats = self._pre_process_dataset_features(df=user_feats)

            user_df = user_feats.rename(columns={
                "user_id": "user_id:token",
                "age": "age:float",
                "income": "income:float",
                "amazon_freq": "amazon_freq:float", 
                "education": "education:token",
                "gender": "gender:token",
                "race_white_or_caucasian": "race_white_or_caucasian:float",
                "race_black_or_african_american": "race_black_or_african_american:float",
                "race_asian": "race_asian:float",
                "race_american_indian_native_american_or_alaska_native": "race_american_indian_native_american_or_alaska_native:float",
                "race_native_hawaiian_or_other_pacific_islander": "race_native_hawaiian_or_other_pacific_islander:float",
                "race_other": "race_other:float",
                "smoke_cigarettes": "smoke_cigarettes:float",
                "has_diabet": "has_diabet:float",
            })

            # !!!
            # user_df = user_df[["user_id:token", "age:float"]]
            user_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.user"), index=False, sep='\t') 

            # Item features
            if self.item_id_col == "item_id":
                item_feats = (
                    df.sort_values("__rome_date")
                    .drop_duplicates("item_id", keep="last")
                    .loc[:, ["item_id", f"category", "title"]]
                )
                item_feats = item_feats[item_feats["item_id"].isin(self.items_ids)]

                item_df = item_feats.rename(columns={
                    f"item_id": "item_id:token",
                    f"category": "category:token",
                    "title": "title:token_seq", 
                })
            elif self.item_id_col == "category":
                item_feats = (
                    df.sort_values("__rome_date")
                    .drop_duplicates("item_id", keep="last")
                    .loc[:, ["item_id"]]
                )
                item_feats = item_feats[item_feats["item_id"].isin(self.items_ids)]

                item_df = item_feats.rename(columns={
                    f"item_id": "item_id:token",
                })

            item_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.item"), index=False, sep='\t') 

            self.user_cols_recbole = [c.split(":")[0] for c in user_df.columns] 
            self.item_cols_recbole = [c.split(":")[0] for c in item_df.columns]

            working_df = self.dataset_unrolled_cold_start.copy()

            working_df["user_id"] = working_df["user_id"].astype('category')
            working_df["item_id"] = working_df["item_id"].astype('category')

            train_len = self.train_window_months

            working_df["date"] = pd.to_datetime(working_df["date"], utc=True)
            working_df["date"] = working_df["date"].dt.tz_convert("Europe/Rome").dt.normalize()

            working_df["timestamp"] = (working_df["date"].astype("int64") // 10**9).astype(float)
            working_df["month"] = working_df["date"].dt.to_period("M")

            months = np.sort(working_df["month"].unique())
            
            train_start_idx = 0
            train_end_idx = train_len - 3
            val_idx  = train_end_idx + 1
            test_idx = train_end_idx + 2

            train_start = months[train_start_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            train_end = months[train_end_idx].to_timestamp(how="end").tz_localize("Europe/Rome")
            val_start = months[val_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            val_end = months[val_idx].to_timestamp(how="end").tz_localize("Europe/Rome")
            test_start = months[test_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            test_end = months[test_idx].to_timestamp(how="end").tz_localize("Europe/Rome")

            grp = working_df.groupby(["user_id", "item_id", "timestamp"], as_index=False, observed=True)
            grouped_counts = grp.size().rename(columns={"size": "interaction_count"})
            grouped_dates = grp["date"].first().reset_index()
            grouped_dates = grouped_dates.rename(columns={"date": "date"})
            grouped = grouped_counts.merge(
                grouped_dates,
                on=["user_id", "item_id", "timestamp"],
                how="inner",
            )
            grouped["label"] = 1.0

            cols = [
                "user_id:token", "item_id:token", "timestamp:float",
                "label:float", "interaction_count:float", "date"
            ]

            grouped = grouped.rename(columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token",
                "timestamp": "timestamp:float",
                "label": "label:float",
                "interaction_count": "interaction_count:float",
            })
            

            # Masks
            train_mask = grouped["date"].between(train_start, train_end, inclusive="both")
            val_mask = grouped["date"].between(val_start, val_end, inclusive="both")
            test_mask = grouped["date"].between(test_start, test_end, inclusive="both")

            # Headers
            cols = [
                "user_id:token", "item_id:token", "timestamp:float",
                "label:float", "interaction_count:float", "date"
            ]
            grouped = grouped.rename(columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token",
                "timestamp": "timestamp:float",
                "label": "label:float",
                "interaction_count": "interaction_count:float",
            })

            # Split frames
            self.working_train_df = grouped.loc[train_mask, cols].sort_values(["user_id:token", "timestamp:float"])
            self.working_val_df = grouped.loc[val_mask, cols].sort_values(["user_id:token", "timestamp:float"])
            self.working_test_df = grouped.loc[test_mask, cols].sort_values(["user_id:token", "timestamp:float"])

            self.working_train_df, self.working_val_df, self.working_test_df = self._handle_cold_start_users(
                self.working_train_df, 
                self.working_val_df, 
                self.working_test_df
            )

            print(f"\n Train dates: {self.working_train_df.date.min()} - {self.working_train_df.date.max()} \n")
            print(f"\n Val dates: {self.working_val_df.date.min()} - {self.working_val_df.date.max()} \n")
            print(f"\n Test dates: {self.working_test_df.date.min()} - {self.working_test_df.date.max()} \n")

            expanded_train = self._unroll_by_interaction_count(
                df=self.working_train_df.drop(columns=["date"])
            )
            expanded_val = self._unroll_by_interaction_count(
                df=self.working_val_df.drop(columns=["date"])
            )
            expanded_test = self._unroll_by_interaction_count(
                df=self.working_test_df.drop(columns=["date"])
            )

            all_items_from_file = set(item_df["item_id:token"].astype(str))
            all_items_in_interactions = set(
                pd.concat([
                    expanded_train["item_id:token"], 
                    expanded_val["item_id:token"], 
                    expanded_test["item_id:token"]
                ]).astype(str).unique()
            )
        
            cold_start_items = all_items_from_file - all_items_in_interactions
            if cold_start_items:
                print(f"⚠️  Found {len(cold_start_items)} cold-start items without interactions")

                dummy_user = expanded_train["user_id:token"].iloc[0]
                dummy_timestamp = expanded_train["timestamp:float"].min()

                dummy_rows = []
                for item_id in cold_start_items:
                    dummy_rows.append({
                        "user_id:token": dummy_user,
                        "item_id:token": item_id,
                        "timestamp:float": dummy_timestamp,
                        "label:float": 0.0,  # Negative label (no actual interaction)
                        "interaction_count:float": 1.0
                    })
                
                dummy_df = pd.DataFrame(dummy_rows)

                expanded_train = pd.concat([dummy_df, expanded_train], ignore_index=True)
                expanded_train = expanded_train.sort_values(["user_id:token", "timestamp:float"])

            expanded_train.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.train.inter"),
                                index=False, sep="\t")
            expanded_val.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.val.inter"),
                                index=False, sep="\t")
            expanded_test.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.test.inter"),
                                index=False, sep="\t")
            
            self._cached_historical_df = pd.concat([expanded_train, expanded_val, expanded_test], ignore_index=True)

        else:
            DATASET_TYPE = "implicit"  # "explicit"
            EXPLICIT_THRESHOLD = 3.0

            filtered_interactions = [
                inter for inter in new_interactions 
                if inter["user_id"] in self.users_ids and inter["item_id"] in self.items_ids
            ]
            if not filtered_interactions:
                print(f"Warning: No valid interactions after filtering at epoch {k}")
                return

            # Create new interactions DataFrame efficiently
            df_new_inter = pd.DataFrame(filtered_interactions)
            df_new_inter = df_new_inter[["user_id", "item_id", "timestamp"]].rename(columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token", 
                "timestamp": "timestamp:float"
            })
            df_new_inter["label:float"] = 1.0
            df_new_inter["interaction_count:float"] = 1.0

            if not hasattr(self, '_cached_historical_df'):
                raise Exception("Historical data not initialized. Run with k=0 first.")
            
            # print(self._cached_historical_df)
            # print(self._cached_historical_df[self._cached_historical_df["item_id:token"] == "SPLASH_POOL"])
            # exit()
            # Concatenate once
            working_df = pd.concat([self._cached_historical_df, df_new_inter], ignore_index=True)

            # if "date" not in working_df.columns:
            #     working_df["date"] = pd.to_datetime(working_df["timestamp:float"], unit="s").dt.date

            working_df["date"] = (
                pd.to_datetime(working_df["timestamp:float"], unit="s", utc=True)
                .dt.tz_convert("Europe/Rome")
                .dt.normalize()
            )
            working_df["month"] = working_df["date"].dt.to_period("M")

            # Get sorted unique months
            months = np.sort(working_df["month"].unique())

            # !!
            # Retrainig with all the available data but at most last max_months_to_keep months, then slice window
            # Alway stay with the the initialization perdio data
            max_months_to_keep = 12
            init_months = months[:self.config.cold_start_months]

            if len(months) > max_months_to_keep:
                recent_months = months[-max_months_to_keep:]
                months_to_keep = pd.Index(init_months).union(pd.Index(recent_months))
                working_df = working_df[working_df["month"].isin(months_to_keep)].copy()
                months = np.sort(working_df["month"].unique())

            train_len = self.train_window_months
            cold_start_len = self.config.cold_start_months
            n_months_total = len(months)

            if cold_start_len < 4:
                raise Exception("\nNeed at least 4 months of initialization (>=2 train, 1 val, 1 test).")
            months_available = min(cold_start_len + k, n_months_total)

            if months_available < 4:
                raise Exception("Not enough months available to build train/val/test windows at this epoch.")
            
            train_end_idx = months_available - 3
            val_idx = train_end_idx + 1
            test_idx = train_end_idx + 2

            if self.use_all_data:
                train_start_idx = 0
            else:
                if train_len is None or train_len <= 0:
                    raise ValueError("train_window_months must be a positive integer when use_all_data=False.")
                window = min(train_len, train_end_idx + 1)
                train_start_idx = train_end_idx - window + 1

            train_months = months[train_start_idx:train_end_idx + 1]
            val_month = months[val_idx]
            test_month = months[test_idx]

            working_df["interaction_count:float"] = working_df.groupby(
                ["user_id:token", "item_id:token", "timestamp:float"]
            )["month"].transform("count").astype(float)

            working_df = working_df.drop_duplicates(
                subset=["user_id:token", "item_id:token", "timestamp:float"]
            )
            
            print(f"\n Epoch {k} - Train months: {train_months}, Val month: {val_month}, Test month: {test_month} \n")

            train_mask = working_df["month"].isin(train_months)
            val_mask = working_df["month"] == val_month
            test_mask = working_df["month"] == test_month

            # Select columns needed for RecBole
            cols = ["user_id:token", "item_id:token", "timestamp:float", "label:float", "interaction_count:float"]

            # Create splits
            train_data = working_df[train_mask][cols].copy()
            val_data = working_df[val_mask][cols].copy()
            test_data = working_df[test_mask][cols].copy()

            self.working_train_df = train_data.sort_values(
                ["user_id:token", "timestamp:float"], 
                ignore_index=True
            )
            self.working_val_df = val_data.sort_values(
                ["user_id:token", "timestamp:float"], 
                ignore_index=True
            )
            self.working_test_df = test_data.sort_values(
                ["user_id:token", "timestamp:float"], 
                ignore_index=True
            )

            self.working_train_df, self.working_val_df, self.working_test_df = self._handle_cold_start_users(
                self.working_train_df, 
                self.working_val_df, 
                self.working_test_df
            )

            expanded_train = self._unroll_by_interaction_count(
                df=self.working_train_df
            )
            expanded_val = self._unroll_by_interaction_count(
                df=self.working_val_df
            )
            expanded_test = self._unroll_by_interaction_count(
                df=self.working_test_df
            )
            
            if "date" in expanded_train.columns:
                expanded_train = expanded_train.drop(columns=["date"])
            if "date" in expanded_val.columns:
                expanded_val = expanded_val.drop(columns=["date"])
            if "date" in expanded_test.columns:
                expanded_test = expanded_test.drop(columns=["date"])

            expanded_train.to_csv(
                os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.train.inter"),
                index=False, 
                sep="\t"
            )
            expanded_val.to_csv(
                os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.val.inter"),
                index=False, 
                sep="\t"
            )
            expanded_test.to_csv(
                os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.test.inter"),
                index=False, 
                sep="\t"
            )

            all_expanded = pd.concat([expanded_train, expanded_val, expanded_test], ignore_index=True)
            
            self._cached_historical_df = all_expanded[cols].copy()

    def _build_window_for_epoch_old(self, k: int = 0, new_interactions = None):
        # FIRST INITIALIZATION OF THE DATASET
        if k == 0:
            # --- Features update only at the start ---

            # Delete tmp folder content every tim a new simulation goes 
            try:
                shutil.rmtree(self.tmp_folder)
                print("Folder deleted successfully")
            except FileNotFoundError:
                print("Folder does not exist")
            except PermissionError:
                print("Permission denied")
            except OSError as e:
                print(f"Error deleting folder: {e}")
            # R create empty
            if not os.path.exists(self.tmp_folder):
                os.makedirs(self.tmp_folder)
            if not os.path.exists(os.path.join(self.tmp_folder, self.tmp_dataset_folder)):
                os.makedirs(os.path.join(self.tmp_folder, self.tmp_dataset_folder))

            df = self.dataset_unrolled_cold_start.copy()

            # Features pre-processing
            df["age"] = df.get("age").map(self._simple_tokenize).fillna("")
            df["race"] = df.get("race").map(self._simple_tokenize).fillna("")
            df["education"] = df.get("education").map(self._simple_tokenize).fillna("")
            df["income"] = df.get("income").map(self._simple_tokenize).fillna("")
            df["gender"] = df.get("gender").astype(str).str.strip().str.title()
            df.loc[~df["gender"].isin(["Male", "Female", "Other"]), "gender"] = "Other"
            df["how_often_use_amazon"] = df.get("how_often_use_amazon").map(self._simple_tokenize).fillna("")
            df["smoke_cigarettes"] = self._to_bool01(df.get("smoke_cigarettes")).fillna(0)
            df["has_diabet"] = self._to_bool01(df.get("has_diabet")).fillna(0)
            
            if self.item_id_col == "item_id":
                df["category"] = df.get("category").map(self._simple_tokenize).fillna("")
                df["title"] = df.get("Title").map(self._simple_tokenize).fillna("")

            df = df.loc[df["user_id"].isin(self.users_ids)
                        & df["item_id"].isin(self.items_ids)].copy()

            base_time_col = "timestamp" if "timestamp" in df.columns else "date"
            df["__rome_date"], _ = self._rome_midnight_and_epoch(df[base_time_col])

            categorical_cols = ["user_id", "item_id", "age", "race", "education", 
                                "income", "gender", "how_often_use_amazon"]
            if self.item_id_col == "item_id":
                categorical_cols.extend(["category"])  # Don't convert 'title' as it's a sequence

            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')

            # User features
            user_feats = (
                df.sort_values("__rome_date")
                .drop_duplicates("user_id", keep="last")
                .loc[:, ["user_id", "age", "race", "education", "income", "gender",
                        "how_often_use_amazon", "smoke_cigarettes", "has_diabet"]]
            )
            user_feats = user_feats[user_feats["user_id"].isin(self.users_ids)]
            
            # Pre-processing user features - handle text and semantic concerns
            user_feats = self._pre_process_dataset_features(df=user_feats)

            user_df = user_feats.rename(columns={
                "user_id": "user_id:token",
                "age": "age:float",
                "income": "income:float",
                "amazon_freq": "amazon_freq:float", 
                "education": "education:token",
                "gender": "gender:token",
                "race_white_or_caucasian": "race_white_or_caucasian:float",
                "race_black_or_african_american": "race_black_or_african_american:float",
                "race_asian": "race_asian:float",
                "race_american_indian_native_american_or_alaska_native": "race_american_indian_native_american_or_alaska_native:float",
                "race_native_hawaiian_or_other_pacific_islander": "race_native_hawaiian_or_other_pacific_islander:float",
                "race_other": "race_other:float",
                "smoke_cigarettes": "smoke_cigarettes:float",
                "has_diabet": "has_diabet:float",
            })

            # !!!
            # user_df = user_df[["user_id:token", "age:float"]]
            user_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.user"), index=False, sep='\t') 

            # Item features
            if self.item_id_col == "item_id":
                item_feats = (
                    df.sort_values("__rome_date")
                    .drop_duplicates("item_id", keep="last")
                    .loc[:, ["item_id", f"category", "title"]]
                )
                item_feats = item_feats[item_feats["item_id"].isin(self.items_ids)]

                item_df = item_feats.rename(columns={
                    f"item_id": "item_id:token",
                    f"category": "category:token",
                    "title": "title:token_seq", 
                })
            elif self.item_id_col == "category":
                item_feats = (
                    df.sort_values("__rome_date")
                    .drop_duplicates("item_id", keep="last")
                    .loc[:, ["item_id"]]
                )
                item_feats = item_feats[item_feats["item_id"].isin(self.items_ids)]

                item_df = item_feats.rename(columns={
                    f"item_id": "item_id:token",
                })

            item_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.item"), index=False, sep='\t') 

            self.user_cols_recbole = [c.split(":")[0] for c in user_df.columns] 
            self.item_cols_recbole = [c.split(":")[0] for c in item_df.columns]

            self._user_features_df = user_df.copy()
            self._item_features_df = item_df.copy()

            working_df = self.dataset_unrolled_cold_start.copy()

            working_df["user_id"] = working_df["user_id"].astype('category')
            working_df["item_id"] = working_df["item_id"].astype('category')

            train_len = self.train_window_months

            working_df["date"] = pd.to_datetime(working_df["date"], utc=True)
            working_df["date"] = working_df["date"].dt.tz_convert("Europe/Rome").dt.normalize()

            working_df["timestamp"] = (working_df["date"].astype("int64") // 10**9).astype(float)
            working_df["month"] = working_df["date"].dt.to_period("M")

            months = np.sort(working_df["month"].unique())
            
            train_start_idx = 0
            train_end_idx = train_len - 3
            val_idx  = train_end_idx + 1
            test_idx = train_end_idx + 2

            train_start = months[train_start_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            train_end = months[train_end_idx].to_timestamp(how="end").tz_localize("Europe/Rome")
            val_start = months[val_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            val_end = months[val_idx].to_timestamp(how="end").tz_localize("Europe/Rome")
            test_start = months[test_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            test_end = months[test_idx].to_timestamp(how="end").tz_localize("Europe/Rome")

            grp = working_df.groupby(["user_id", "item_id", "timestamp"], as_index=False, observed=True)
            grouped_counts = grp.size().rename(columns={"size": "interaction_count"})
            grouped_dates = grp["date"].first().reset_index()
            grouped_dates = grouped_dates.rename(columns={"date": "date"})
            grouped = grouped_counts.merge(
                grouped_dates,
                on=["user_id", "item_id", "timestamp"],
                how="inner",
            )
            grouped["label"] = 1.0

            cols = [
                "user_id:token", "item_id:token", "timestamp:float",
                "label:float", "interaction_count:float", "date"
            ]

            grouped = grouped.rename(columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token",
                "timestamp": "timestamp:float",
                "label": "label:float",
                "interaction_count": "interaction_count:float",
            })
            

            # Masks
            train_mask = grouped["date"].between(train_start, train_end, inclusive="both")
            val_mask = grouped["date"].between(val_start, val_end, inclusive="both")
            test_mask = grouped["date"].between(test_start, test_end, inclusive="both")

            # Headers
            cols = [
                "user_id:token", "item_id:token", "timestamp:float",
                "label:float", "interaction_count:float", "date"
            ]
            grouped = grouped.rename(columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token",
                "timestamp": "timestamp:float",
                "label": "label:float",
                "interaction_count": "interaction_count:float",
            })

            # Split frames
            self.working_train_df = grouped.loc[train_mask, cols].sort_values(["user_id:token", "timestamp:float"])
            self.working_val_df = grouped.loc[val_mask, cols].sort_values(["user_id:token", "timestamp:float"])
            self.working_test_df = grouped.loc[test_mask, cols].sort_values(["user_id:token", "timestamp:float"])

            self.working_train_df, self.working_val_df, self.working_test_df = self._handle_cold_start_users(
                self.working_train_df, 
                self.working_val_df, 
                self.working_test_df
            )

            print(f"\n Train dates: {self.working_train_df.date.min()} - {self.working_train_df.date.max()} \n")
            print(f"\n Val dates: {self.working_val_df.date.min()} - {self.working_val_df.date.max()} \n")
            print(f"\n Test dates: {self.working_test_df.date.min()} - {self.working_test_df.date.max()} \n")

            expanded_train = self._unroll_by_interaction_count(
                df=self.working_train_df.drop(columns=["date"])
            )
            expanded_val = self._unroll_by_interaction_count(
                df=self.working_val_df.drop(columns=["date"])
            )
            expanded_test = self._unroll_by_interaction_count(
                df=self.working_test_df.drop(columns=["date"])
            )

            expanded_train.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.train.inter"),
                                index=False, sep="\t")
            expanded_val.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.val.inter"),
                                index=False, sep="\t")
            expanded_test.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.test.inter"),
                                index=False, sep="\t")
            
            self._cached_historical_df = pd.concat([expanded_train, expanded_val, expanded_test], ignore_index=True)

        else:
            # When tested, apply to the initialization block too
            DATASET_TYPE = "implicit"  # "explicit"
            EXPLICIT_THRESHOLD = 3.0

            filtered_interactions = [
                inter for inter in new_interactions 
                if inter["user_id"] in self.users_ids and inter["item_id"] in self.items_ids
            ]
            if not filtered_interactions:
                print(f"Warning: No valid interactions after filtering at epoch {k}")
                return

            # Create new interactions DataFrame efficiently
            df_new_inter = pd.DataFrame(filtered_interactions)
            df_new_inter = df_new_inter[["user_id", "item_id", "timestamp"]].rename(columns={
                "user_id": "user_id:token",
                "item_id": "item_id:token", 
                "timestamp": "timestamp:float"
            })

            if not hasattr(self, '_cached_historical_df'):
                raise Exception("Historical data not initialized. Run with k=0 first.")

            # Concatenate once
            # working_df = pd.concat([self._cached_historical_df, df_new_inter], ignore_index=True)
            working_df = pd.concat([self._cached_historical_df, df_new_inter], ignore_index=True)
            working_df = working_df.drop_duplicates(
                subset=["user_id:token", "item_id:token", "timestamp:float"],
                keep="last"
            )

            # Create date and month columns (simplified, no timezone)
            if "date" not in working_df.columns:
                working_df["date"] = pd.to_datetime(working_df["timestamp:float"], unit="s").dt.date

            working_df["month"] = pd.to_datetime(working_df["date"]).dt.to_period("M")

            # Get sorted unique months
            months = np.sort(working_df["month"].unique())

            # !!
            # Retrainig with all the available data but at most last max_months_to_keep months, then slice window
            # Alway stay with the the initialization period data
            max_months_to_keep = 12
            init_months = months[:self.config.cold_start_months]

            if len(months) > max_months_to_keep:
                recent_months = months[-max_months_to_keep:]
                months_to_keep = pd.Index(init_months).union(pd.Index(recent_months))
                working_df = working_df[working_df["month"].isin(months_to_keep)].copy()
                months = np.sort(working_df["month"].unique())

            train_len = self.train_window_months
            cold_start_len = self.config.cold_start_months
            n_months_total = len(months)

            if cold_start_len < 4:
                raise Exception("\nNeed at least 4 months of initialization (>=2 train, 1 val, 1 test).")
            months_available = min(cold_start_len + k, n_months_total)

            if months_available < 4:
                raise Exception("Not enough months available to build train/val/test windows at this epoch.")
            
            train_end_idx = months_available - 3
            val_idx = train_end_idx + 1
            test_idx = train_end_idx + 2

            if self.use_all_data:
                train_start_idx = 0
            else:
                if train_len is None or train_len <= 0:
                    raise ValueError("train_window_months must be a positive integer when use_all_data=False.")
                window = min(train_len, train_end_idx + 1)
                train_start_idx = train_end_idx - window + 1

            train_months = months[train_start_idx:train_end_idx + 1]
            val_month = months[val_idx]
            test_month = months[test_idx]

            if DATASET_TYPE == "implicit":
                working_df["interaction_count:float"] = working_df.groupby(
                    ["user_id:token", "item_id:token", "timestamp:float"]
                )["month"].transform("count").astype(float)

                working_df = working_df.drop_duplicates(
                    subset=["user_id:token", "item_id:token", "timestamp:float"]
                )

                working_df["label:float"] = 1.0
            
            elif DATASET_TYPE == "explicit":
                if "rating" not in working_df.columns:
                    raise ValueError("Explicit feedback requires 'rating' column")

                working_df["label:float"] = (working_df["rating"] >= EXPLICIT_THRESHOLD).astype(float)
                working_df["interaction_count:float"] = working_df["rating"].astype(float)

                working_df = working_df.sort_values("timestamp:float").drop_duplicates(
                    subset=["user_id:token", "item_id:token"], 
                    keep="last"
                )
            
            else:
                raise ValueError(f"Invalid DATASET_TYPE: {DATASET_TYPE}. Must be 'implicit' or 'explicit'")
            
            train_mask = working_df["month"].isin(train_months)
            val_mask = working_df["month"] == val_month
            test_mask = working_df["month"] == test_month

            # Select columns needed for RecBole
            cols = ["user_id:token", "item_id:token", "timestamp:float", "label:float", "interaction_count:float"]

            # Create splits
            train_data = working_df[train_mask][cols].copy()
            val_data = working_df[val_mask][cols].copy()
            test_data = working_df[test_mask][cols].copy()

            self.working_train_df = train_data.sort_values(
                ["user_id:token", "timestamp:float"], 
                ignore_index=True
            )
            self.working_val_df = val_data.sort_values(
                ["user_id:token", "timestamp:float"], 
                ignore_index=True
            )
            self.working_test_df = test_data.sort_values(
                ["user_id:token", "timestamp:float"], 
                ignore_index=True
            )

            self.working_train_df, self.working_val_df, self.working_test_df = self._handle_cold_start_users(
                self.working_train_df, 
                self.working_val_df, 
                self.working_test_df
            )

            

            expanded_train.to_csv(
                os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.train.inter"),
                index=False, 
                sep="\t"
            )
            expanded_val.to_csv(
                os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.val.inter"),
                index=False, 
                sep="\t"
            )
            expanded_test.to_csv(
                os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.test.inter"),
                index=False, 
                sep="\t"
            )

            # Update cached historical data with expanded versions
            self._cached_historical_df = pd.concat([expanded_train, expanded_val, expanded_test], ignore_index=True)

            # CRITICAL: Always write complete user/item files to maintain vocabulary
            if hasattr(self, '_user_features_df') and hasattr(self, '_item_features_df'):
                self._user_features_df.to_csv(
                    os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.user"), 
                    index=False, 
                    sep='\t'
                )
                self._item_features_df.to_csv(
                    os.path.join(self.tmp_folder, self.tmp_dataset_folder, "experiment_dataset.item"), 
                    index=False, 
                    sep='\t'
                )
                print(f"Epoch {k}: Wrote {len(self._user_features_df)} users and {len(self._item_features_df)} items")
            else:
                raise Exception("User and item feature dataframes not initialized!")

    def init_choice_model(self) -> None:
        df_init = self.dataset_unrolled_cold_start.copy()

        self.user_choice_model = ChoiceModel(interaction_df=df_init, config=self.config, refresh=True)
        self.user_choice_model.setup()

    def recom_choice_model(self, curr_epoch: int, user_id_recbole: int) -> list:
        tau = self.config.user_strategy["tau"]
        recbole_dataset = self.recbole_dataset
        user_id = recbole_dataset.id2token(recbole_dataset.uid_field, user_id_recbole)
        items, scores = self.user_choice_model.predict_for_a_user(user_id=user_id, tau=tau)
        probs = np.array(scores, dtype=float)
        probs = probs / probs.sum()
        sampled_index = int(self.rng.choice(len(items), p=probs))
        selected_item_id = items[sampled_index]
        try:
            item_id_recbole = recbole_dataset.token2id(recbole_dataset.iid_field, str(selected_item_id))
        except ValueError:
            raise f"\n Item ID: {selected_item_id} not found in RecBole dataset vocabulary. \n"
        # while item_id_recbole not in self.items_ids_recbole:
        #     pass
        return item_id_recbole

    def _fit_with_tracking(self, trainer: Trainer, save_stem: str, show_progress: bool = False, save_plots: bool = False):
        all_validation_results = []
        all_test_results = []
        all_train_losses = []

        # ---- store originals ----
        original_valid_epoch = trainer._valid_epoch
        original_train_epoch = trainer._train_epoch

        # ---- wrappers ----
        def custom_train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False):
            train_loss = original_train_epoch(train_data, epoch_idx, loss_func, show_progress)
            all_train_losses.append(train_loss)
            return train_loss

        def custom_valid_epoch(valid_data, show_progress=False):
            valid_result = original_valid_epoch(valid_data, show_progress)
            all_validation_results.append(valid_result)
            # evaluate on test data every validation
            test_result = trainer.evaluate(self.test_data, load_best_model=False, show_progress=False)
            all_test_results.append(test_result)
            return valid_result

        trainer._train_epoch = custom_train_epoch
        trainer._valid_epoch = custom_valid_epoch

        best_valid_result = trainer.fit(
            train_data=self.train_data,
            valid_data=self.valid_data,
            show_progress=show_progress,
            saved=False
        )

        data_for_df = []
        max_epochs = max(len(all_train_losses), len(all_validation_results), len(all_test_results))
        for epoch in range(max_epochs):
            row = {"epoch": epoch}
            if epoch < len(all_train_losses):
                row["train_loss"] = all_train_losses[epoch]
            if epoch < len(all_validation_results):
                valid_data = all_validation_results[epoch]
                if isinstance(valid_data, dict):
                    for k, v in valid_data.items():
                        row[f"valid_{k}"] = v
                elif isinstance(valid_data, tuple):
                    if len(valid_data) >= 2 and isinstance(valid_data[1], dict):
                        for k, v in valid_data[1].items():
                            row[f"valid_{k}"] = v
                    elif len(valid_data) >= 1:
                        row["valid_score"] = valid_data[0]
            if epoch < len(all_test_results):
                for k, v in all_test_results[epoch].items():
                    row[f"test_{k}"] = v
            data_for_df.append(row)

        results_df = pd.DataFrame(data_for_df).set_index("epoch")
        val_cols = [c for c in results_df.columns if c.startswith("valid_")]
        test_cols = [c for c in results_df.columns if c.startswith("test_")]
        validation_df = results_df[val_cols].rename(columns=lambda c: c.replace("valid_", ""))
        test_df = results_df[test_cols].rename(columns=lambda c: c.replace("test_", ""))
        loss_df = results_df[["train_loss"]].copy() if "train_loss" in results_df.columns else pd.DataFrame()

        # ---- save CSVs ----
        base = save_stem  # e.g., ".../train_logs/sim_epoch_3"
        os.makedirs(os.path.dirname(base), exist_ok=True)
        if not loss_df.empty:
            loss_df.to_csv(f"{base}_loss.csv")
        if not validation_df.empty:
            validation_df.to_csv(f"{base}_validation.csv")
        if not test_df.empty:
            test_df.to_csv(f"{base}_test.csv")

        # ---- save final test metrics JSON (last epoch of this training run) ----
        final_test_metrics = all_test_results[-1] if all_test_results else {}
        metrics_to_save = ["Precision", "Recall", "Hit", "NDCG", "ItemCoverage", "MRR", "MAP", "AveragePopularity"]
        out_json = {}
        for metric_name in metrics_to_save:
            found = {}
            for k, v in final_test_metrics.items():
                if metric_name.lower() in k.lower():
                    found[k] = v
            out_json[metric_name] = found or None
        out_json["model_name"] = self.model_name_config
        out_json["dataset"] = "amazon"
        out_json["total_epochs"] = len(all_test_results)
        out_json["final_epoch"] = len(all_test_results) - 1
        with open(f"{base}_final_test_metrics.json", "w") as f:
            json.dump(out_json, f, indent=2, default=str)
    
    def recom_recbole_model(self, curr_epoch: str, user_id_recbole: int, K_horizon: int = None):
        '''
            Now it considers the epoch in order to optimize the ranking request since the models are updated only once per epoch
        '''
        if K_horizon is None:
            K_horizon = len(self.items_ids)-5 # To avoid overlaps
        if self.recbole_model is None:
            raise Exception(f"No model in the session")

        if self.model_name_config in ["Individual Random", "Individual Popularity"]:
            user_interacted_items = self.recbole_dataset.inter_feat[self.recbole_dataset.inter_feat["user_id"] == user_id_recbole]["item_id"].numpy()
            user_interacted_items = list(set(list(user_interacted_items)))
            user_interacted_items = [int(l) for l in user_interacted_items]
        
        interaction_batch = Interaction(
            {
                'user_id': torch.tensor([user_id_recbole])
            }
        )
        with torch.no_grad():
            try:
                SCORES_PATH = "./"
                item_scores = self.recbole_model.full_sort_predict(interaction_batch).cpu()

                # print(f"=== Score Statistics for User {user_id_recbole} ===")
                # print(f"Min score: {item_scores.min().item():.4f}")
                # print(f"Max score: {item_scores.max().item():.4f}")
                # print(f"Mean score: {item_scores.mean().item():.4f}")
                # print(f"Std score: {item_scores.std().item():.4f}")
                # print(f"Median score: {item_scores.median().item():.4f}")

                # # Check for NaN/Inf
                # print(f"NaN values: {torch.isnan(item_scores).sum().item()}")
                # print(f"Inf values: {torch.isinf(item_scores).sum().item()}")

                # # Score distribution
                # print(f"Scores > 0.5: {(item_scores > 0.5).sum().item()} items")
                # print(f"Scores > 0.7: {(item_scores > 0.7).sum().item()} items")
                # print(f"Scores < 0.3: {(item_scores < 0.3).sum().item()} items")

                # exit()

            except Exception as e:
                traceback.print_exc()
                raise Exception(f"{e}")
        try:
            if curr_epoch not in self.top_k_users_scores:
                self.top_k_users_scores[curr_epoch] = {}
                self.all_users_scores[curr_epoch] = {}
            if user_id_recbole not in self.top_k_users_scores[curr_epoch]:
                # Batch not considered, so process 1 user per time
                if item_scores.dim() == 2 and item_scores.size(0) == 1:
                    item_scores = item_scores.squeeze(0)

                shifted_scores = item_scores - torch.min(item_scores) 
                probabilities = shifted_scores / torch.sum(shifted_scores)
                top_k_indices = torch.topk(probabilities, K_horizon, dim=0)[1]
                
                top_k_probs = probabilities[top_k_indices]
                # print({top_k_probs.min(), top_k_probs.max()})
                top_k_probs = top_k_probs / torch.sum(top_k_probs)
                
                has_nan = np.isnan(top_k_probs).any()
                if has_nan:
                    our_user_id = self.recbole_dataset.id2token(self.recbole_dataset.uid_field, user_id_recbole)
                    raise ValueError(f"\n NaN values got from recom model computation - cached: {top_k_probs}  for uer: {our_user_id}\n")
                    
                self.top_k_users_scores[curr_epoch][user_id_recbole] = {}
                self.all_users_scores[curr_epoch][user_id_recbole] = {}
                self.all_users_scores[curr_epoch][user_id_recbole]["tot_scores"] = shifted_scores.cpu().numpy()
                self.top_k_users_scores[curr_epoch][user_id_recbole]["top_k_items"] = top_k_indices.cpu().numpy()
                self.top_k_users_scores[curr_epoch][user_id_recbole]["top_k_scores"] = top_k_probs.cpu().numpy()
                items_id_list = top_k_indices.cpu().numpy()
                items_token_list = [self.recbole_dataset.id2token("item_id", i) for i in items_id_list]
                self.top_k_users_scores[curr_epoch][user_id_recbole]["top_k_items_tokens"] = items_token_list
            else:
                top_k_probs = torch.from_numpy(np.array(self.top_k_users_scores[curr_epoch][user_id_recbole]["top_k_scores"]))
                top_k_indices = torch.from_numpy(np.array(self.top_k_users_scores[curr_epoch][user_id_recbole]["top_k_items"]))
                arr = self.top_k_users_scores[curr_epoch][user_id_recbole]["top_k_scores"]
                has_nan = np.isnan(arr).any()
                if has_nan:
                    raise ValueError(f"\n NaN values got from saved dictionary - cahed: {arr} \n")
            
            arr = top_k_probs.cpu().numpy()
            has_nan = np.isnan(arr).any()
            if has_nan:
                our_user_id = self.recbole_dataset.id2token(self.recbole_dataset.uid_field, user_id_recbole)
                print(f"\n User: {our_user_id} suggestions contain NaN: {arr}")
                raise ValueError(f"\n Probabilities array given by recommender model contains Nan values")
                
            local_selected_index = int(self.rng.choice(len(top_k_probs), p=top_k_probs.cpu().numpy()))
            selected = int(top_k_indices[local_selected_index])
            while selected == 0:
                local_selected_index = int(self.rng.choice(len(top_k_probs), p=top_k_probs.cpu().numpy()))
                selected = int(top_k_indices[local_selected_index])
            
        except Exception as e:
            traceback.print_exc()
            print(e)
            raise
        try:      
            return selected
        except Exception as e:
            traceback.print_exc()
    
    def run_feedback_loop(self, p: float|str, results_path: str, results_scores_path: str, k_horizon: int):
        if self.config.delta_training_epoch > self.config.epochs:
            raise Exception(f"Delta training epoch must be lower than epochs")
    
        epoch = 1
        end_dataset_initialization = self.start_experiment_date - relativedelta(days=1)
        start_simulation_date = end_dataset_initialization + relativedelta(days=1)
        end_date = start_simulation_date + relativedelta(months=1) - relativedelta(days=1)

        df_timeline_shops = self.experiment_distribution_dict.copy()
        while epoch < self.config.epochs+1:
            start_e = time.time()

            epochs_interactions_df = []
            print(f"\n --- Epoch n° {epoch} with model: {self.model_name_config} --- \n")

            # Get dates from the pre-computed distribution
            epoch_dates = [start_simulation_date + relativedelta(days=i) for i in range((end_date-start_simulation_date).days+1)]

            print(f"\n Epoch {epoch} dates: {[d.strftime('%Y-%m-%d') for d in epoch_dates]} \n")
            start_dates = time.time()
            for date in epoch_dates:
                date_str = date.strftime("%Y-%m-%d")
                try:
                    users_data = df_timeline_shops[date_str] # List of dict
                except KeyError:
                    continue
                if len(users_data) == 0:
                    print(f"\n No users entered in simulation in date {date_str} with p={p} \n")
                    continue
                else:
                    valid_users = {k: v for k, v in users_data.items() if k in self.users_ids}
                    if not len(valid_users):
                        # print(f"\n After filtering, no valid users found in date {date_str} with p={p} \n")
                        continue 
                                
                timestamp_ = datetime.combine(date, time_datetime(hour=15))
                recbole_timestamp = int(timestamp_.timestamp())

                for user, items in valid_users.items():
                    valid_items = items
                    
                    if len(valid_items) == 0:
                        continue
                    basket_size = len(valid_items)
                    user_id_recbole = self.recbole_dataset.token2id(self.recbole_dataset.uid_field, user)

                    use_recommender_mask = (self.rng.random(basket_size) < p)

                    n_recommender = np.sum(use_recommender_mask)
                    n_choice = basket_size - n_recommender
                    
                    recommender_items = []
                    if n_recommender > 0:
                        for _ in range(n_recommender):
                            recbole_item_id = self.recom_recbole_model(curr_epoch=epoch, user_id_recbole=user_id_recbole, K_horizon=k_horizon)
                            recommender_items.append(recbole_item_id)
                    
                    choice_items = []
                    if n_choice > 0:
                        if self.config.user_strategy["model_name"] == "Custom choice model":
                            for _ in range(n_choice):
                                recbole_item_id = self.recom_choice_model(curr_epoch=epoch, user_id_recbole=user_id_recbole)
                                choice_items.append(recbole_item_id)
                        else:
                            raise Exception(f"\n Only choice model is supported")
                        
                    rec_iter = iter(recommender_items)
                    choice_iter = iter(choice_items)

                    for use_rec in use_recommender_mask:
                        if use_rec:
                            recbole_item_id = next(rec_iter)
                        else:
                            recbole_item_id = next(choice_iter)

                        our_item_id = self.recbole_dataset.id2token(self.recbole_dataset.iid_field, recbole_item_id)

                        def is_bad_item_id(x) -> bool:
                            # Expect string-like token
                            return isinstance(x, (np.ndarray, list, dict, set, tuple)) or x is None
                        
                        if is_bad_item_id(our_item_id):
                            print(f"\n Bad token item ID. Found: {int(recbole_item_id) if hasattr(recbole_item_id, '__int__') else recbole_item_id} of type: {type(our_item_id).__name__}")
                            continue

                        single_interaction_df = {
                            "user_id": user,
                            "item_id": our_item_id,
                            "date": date,
                            "timestamp": recbole_timestamp,
                        }
                        epochs_interactions_df.append(single_interaction_df)
            end_dates = time.time()

            if (end_dates - start_dates) <= 60:
                print(f"\n Update - build window, init model finished in {end_dates - start_dates} seconds --- \n")
            else:
                print(f"\n Update - build window, init model finished in {(end_dates - start_dates)/60} minutes --- \n")

            if (epoch % self.config.delta_training_epoch) == 0:
                start_try = time.time()
                st1 = time.time()
                self.user_choice_model.update(new_interactions=epochs_interactions_df, epoch=epoch)
                end1 = time.time()
                st2 = time.time()
                self._build_window_for_epoch(k=epoch, new_interactions=epochs_interactions_df)
                end2 = time.time()
                st3 = time.time()
                self.init_recbole_model(is_first_init=False, warm_start=False) # warm start right True now is not working
                end3 = time.time()
                end_try = time.time()

                print(f"\n UPDATE finished in {end1 - st1} seconds --- \n")

                print(f"\n BUILD WINDOW finished in {end2 - st2} seconds --- \n")

                print(f"\n INIT MODEL finished in {end3 - st3} seconds --- \n")

                if (end_try - start_try) <= 60:
                    print(f"\n Update - build window, init model finished in {end_try - start_try} seconds --- \n")
                else:
                    print(f"\n Update - build window, init model finished in {(end_try - start_try)/60} minutes --- \n")

                trainer = Trainer(self.model_config, self.recbole_model)

                # where to save per-simulation-epoch training logs
                trainlogs_root = os.path.join(results_path, "train_logs")
                if not os.path.exists(trainlogs_root):
                    os.makedirs(trainlogs_root)
                save_stem = os.path.join(trainlogs_root, f"sim_epoch_{epoch}")
                start_fit = time.time()
                # re-training strategy
                self._fit_with_tracking(
                    trainer=trainer,
                    save_stem=save_stem,
                    show_progress=False,
                    save_plots=False 
                )
                end_fit = time.time()
                if (end_fit - start_fit) <= 60:
                    print(f"\n Fit with track finished in {end_fit - start_fit} seconds --- \n")
                else:
                    print(f"\n Fit with track finished in {(end_fit - start_fit)/60} minutes --- \n")

            if (len(epochs_interactions_df) == 0):
                raise Exception(f"ZERO Interactions in epoch {epoch}")
            
            end_e = time.time()
            
            if self.config.experiment_mode == "p-validation":
                if (end_e - start_e) <= 60:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- p = {p} --- K AV items = {k_horizon} \n"
                    )
                else:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- p = {p} --- K AV items = {k_horizon} \n"
                    )

            elif self.config.experiment_mode == "compare-models":
                model_name = self.config.recommender_model['model_name']
                if (end_e - start_e) <= 60:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- recom model = {model_name} --- p = {p} --- K AV items = {k_horizon} \n"
                    )
                else:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- recom model = {model_name} --- p = {p} --- K AV items = {k_horizon} \n"
                    )

            elif self.config.experiment_mode == "recom_model_test":
                if (end_e - start_e) <= 60:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- p = {p} --- K AV items = {k_horizon} \n"
                    )
                else:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- p = {p} --- K AV items = {k_horizon} \n"
                    )

            elif self.config.experiment_mode == "k_items_evaluation":
                model_name = self.config.recommender_model['model_name']
                if (end_e - start_e) <= 60:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- recom model = {model_name} --- p = {p} --- K AV items = {k_horizon} \n"
                    )
                else:
                    print(
                        f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes "
                        f"with total new interactions: {len(epochs_interactions_df)} "
                        f"-- recom model = {model_name} --- p = {p} --- K AV items = {k_horizon} \n"
                    )
            
            print(
                f"\n The epoch worked from "
                f"{start_simulation_date.strftime('%Y-%m-%d')} "
                f"to {end_date.strftime('%Y-%m-%d')} \n"
            )

            start_simulation_date = (start_simulation_date.replace(day=1) + relativedelta(months=1)).replace(year=start_simulation_date.year + (start_simulation_date.month // 12))
            end_date = start_simulation_date + relativedelta(months=1) - relativedelta(days=1)

            epoch_interactions_df = pd.DataFrame(epochs_interactions_df)
            epoch_interactions_df.to_csv(os.path.join(results_path, f"epoch_{epoch}.csv"))

            # !!

            with open(os.path.join(results_scores_path, f"top_k_scores_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(self.top_k_users_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(results_scores_path, f"all_scores_epoch_{epoch}.pkl"), "wb") as f:
                pickle.dump(self.all_users_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
            if p == 0:
                with open(os.path.join(results_scores_path, f"user_strategy_scores_epoch_{epoch}.pkl"), "wb") as f:
                    pickle.dump(self.usr_strategy_ranking, f, protocol=pickle.HIGHEST_PROTOCOL)   

            self.top_k_users_scores = {}
            self.all_users_scores = {}
            self.usr_strategy_ranking = {}

            epoch += 1

        return None