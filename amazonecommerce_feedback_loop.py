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

from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import Pop, Random, MultiVAE, ItemKNN
from recbole.trainer import Trainer
from recbole.data.interaction import Interaction
from recbole.trainer import HyperTuning
from recbole.utils import get_model, get_trainer

from utils import DotDict, get_consistent_users, _setup_repro
from custom_models import UserKNN, IndividualRandom, IndividualPopularity, LightGCN, BPR, SpectralCF, NeuMF, NNCF
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

        # Reomve users with less than 10 interactions in the initializatio phase
        user_counts = self.dataset_unrolled_cold_start["user_id"].value_counts()
        active_users = user_counts[user_counts >= 10].index
        self.dataset_unrolled_cold_start = self.dataset_unrolled_cold_start[self.dataset_unrolled_cold_start["user_id"].isin(active_users)]

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

    def init_recbole_model(self):
        name_map = {
            "Collective Popularity": "Pop",
            "Collective Random": "Random",
            "CF_KNN_item": "ItemKNN",
            "MultiVAE": "MultiVAE",
            "BPR": "BPR",
            "LightGCN": "LightGCN",
            "SpectralCF": "SpectralCF",
            "NeuMF": "NeuMF",
            "DeepFM": "DeepFM", 
            "xDeepFM": "xDeepFM",
            "DCN V2": "DCNv2"
        }
        try:
            self.raw_model_name = self.config.recommender_model.model_name
            self.model_name_recbole = name_map[self.raw_model_name]
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"{e if isinstance(e, KeyError) else str(e)}")

        self.parameter_dict = {
            'data_path': self.tmp_folder,
            'checkpoint_dir': os.path.join(self.tmp_folder, "checkpoints"),

            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "TIME_FIELD": "timestamp",
            "LABEL_FIELD": "label",

            "load_col": {
                "inter": ["user_id", "item_id", "timestamp", "label", "price_per_unit", "interaction_count"],
                "user":  ["user_id", "age", "race", "education", "income", "gender",
                        "how_often_use_amazon", "smoke_cigarettes", "has_diabet"],
                "item":  ["item_id", "category", "title"],
            },

            # training / eval
            "epochs": getattr(self.config.recommender_model, "train_epochs", 10),
            "eval_args": {"group_by": "user", "order": "TO", "mode": "full"},
            "benchmark_filename": ["train", "val", "test"],
            "reproducibility": True,
            "seed": self.master_seed,

            "metrics": ["NDCG", "Recall", "Precision", "Hit", "ItemCoverage", "MRR", "MAP"],
            "topk": 10,
            "valid_metric": "NDCG@10",

            # implicit negatives
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 3},

            "use_gpu": torch.cuda.is_available(),
            "gpu_id": 0,
        }

        # Optional per-model parameters pulled from config
        model_cfg = getattr(self.config, "recommender_model", None)

        def add_if(attr, key=None):
            if hasattr(model_cfg, attr):
                self.parameter_dict[key or attr] = getattr(model_cfg, attr)
        
        add_if("learning_rate", "learning_rate")

        if self.model_name_recbole == "ItemKNN":
            add_if("k", "k")
            add_if("shrink", "shrink")
            self.parameter_dict.setdefault("similarity", "cosine")
            self.parameter_dict.setdefault("normalize", True)
        elif self.model_name_recbole == "BPR":
            add_if("reg_weight", "reg_weight")
        elif self.model_name_recbole == "LightGCN":
            add_if("reg_weight", "reg_weight")
            add_if("n_layers", "n_layers")
        elif self.model_name_recbole == "NeuMF":
            add_if("mlp_hidden_size", "mlp_hidden_size")
            add_if("user_embedding_size", "mlp_embedding_size")
            add_if("item_embedding_size", "mf_embedding_size")
        elif self.model_name_recbole == "MultiVAE":
            add_if("user_hidden_size_list", "user_hidden_size_list")
            add_if("dropout_prob", "dropout_prob")
            self.parameter_dict.setdefault("latent_dim", 50)
        elif self.model_name_recbole == "SpectralCF":
            add_if("n_layers", "n_layers")
            add_if("reg_weight", "reg_weight")
        elif self.model_name_recbole == "DeepFM":
            self.parameter_dict.setdefault("embedding_size", 16)
            self.parameter_dict.setdefault("mlp_hidden_size", [256, 128])
            self.parameter_dict.setdefault("dropout_prob", 0.2)
            self.parameter_dict.setdefault("reg_weight", 1e-6)
            add_if("embedding_size")
            add_if("mlp_hidden_size")
            add_if("dropout_prob")
            add_if("reg_weight")

        elif self.model_name_recbole == "DeepFM":
            # FM (second-order) + MLP over concatenated embeddings
            self.parameter_dict.setdefault("embedding_size", 16)
            self.parameter_dict.setdefault("mlp_hidden_size", [256, 128])
            self.parameter_dict.setdefault("dropout_prob", 0.2)
            self.parameter_dict.setdefault("reg_weight", 1e-6)
            add_if("embedding_size")
            add_if("mlp_hidden_size")
            add_if("dropout_prob")
            add_if("reg_weight")

        elif self.model_name_recbole == "xDeepFM":
            # CIN (explicit higher-order) + MLP (implicit) + linear/FMs
            self.parameter_dict.setdefault("embedding_size", 16)
            self.parameter_dict.setdefault("mlp_hidden_size", [256, 128])
            self.parameter_dict.setdefault("cin_layer_size", [16, 16])
            self.parameter_dict.setdefault("dropout_prob", 0.2)
            self.parameter_dict.setdefault("reg_weight", 1e-6)
            add_if("embedding_size")
            add_if("mlp_hidden_size")
            add_if("cin_layer_size")
            add_if("dropout_prob")
            add_if("reg_weight")

        elif self.model_name_recbole == "DCNv2":
            # Cross network v2 + deep MLP
            self.parameter_dict.setdefault("embedding_size", 16)
            self.parameter_dict.setdefault("mlp_hidden_size", [256, 128])
            self.parameter_dict.setdefault("cross_layer_num", 3)   # number of cross layers
            # low_rank and num_experts are optional in some RecBole versions of DCNv2
            add_if("embedding_size")
            add_if("mlp_hidden_size")
            add_if("cross_layer_num")
            add_if("low_rank")
            add_if("num_experts")

        add_if("epochs", "epochs")

        # Remove None
        self.parameter_dict = {k: v for k, v in self.parameter_dict.items() if v is not None}

        try:
            self.model_config = Config(
                model=self.model_name_recbole,
                dataset="experiment_dataset",
                config_dict=self.parameter_dict
            )
            init_seed(self.model_config["seed"], self.model_config["reproducibility"])
            init_logger(self.model_config)
            self.logger = getLogger()
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

        model_cls = get_model(self.model_config["model"])
        self.recbole_model = model_cls(self.model_config, self.train_data.dataset).to(self.model_config["device"])

    def tuning_hyperparameters(self) -> None:
        hyper_file = f"tuning_parameters/{self.model_name_config}.hyper"
        export_result_file = f"tuning_parameters/{self.model_name_config}.result"
        if self.model_name_config in ["Collective Random", "Collective Popularity"]:
            return None
        if os.path.exists(export_result_file):
            print(f"\n Tuning of the model {self.model_name_config} already in the folder. Skip \n")
            return None
        def _objective_function(params_dict=None, config_file_list=None):
            config = Config(
                model=self.model_name_recbole,
                dataset='experiment_dataset', 
                config_file_list=config_file_list,
                config_dict={**self.parameter_dict, **(params_dict or {})}
            )

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config=config, dataset=dataset)

            # model_class = self.recbole_model_class
            # model = model_class(config, train_data.dataset).to(config['device'])
            model_cls = get_model(self.model_config["model"])
            self.recbole_model = model_cls(self.model_config, self.train_data.dataset).to(self.model_config["device"])

            trainer = Trainer(config, self.recbole_model)
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
    
    def _build_window_for_epoch(self, k: int = 0, new_interactions = None):
        # FIRST INITIALIZATION OF THE DATASET
        if k == 0:
            # --- Features update only at the start ---

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

            # User features
            user_feats = (
                df.sort_values("__rome_date")
                .drop_duplicates("user_id", keep="last")
                .loc[:, ["user_id", "age", "race", "education", "income", "gender",
                        "how_often_use_amazon", "smoke_cigarettes", "has_diabet"]]
            )
            user_feats = user_feats[user_feats["user_id"].isin(self.users_ids)]

            user_df = user_feats.rename(columns={
                "user_id": "user_id:token",
                "age": "age:token",
                "race": "race:token",
                "education": "education:token",
                "income": "income:token",
                "gender": "gender:token",
                "how_often_use_amazon": "how_often_use_amazon:token",
                "smoke_cigarettes": "smoke_cigarettes:float",
                "has_diabet": "has_diabet:float",
            })

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

            user_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.user"), index=False, sep='\t') 
            item_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.item"), index=False, sep='\t') 

            working_df = self.dataset_unrolled_cold_start.copy()

            train_len = self.train_window_months

            working_df["date"] = pd.to_datetime(working_df["date"], utc=True)
            # To avoid some outlier errors
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

            grouped = (
                working_df
                .groupby(["user_id", "item_id", "timestamp"], as_index=False)
                .agg(
                    interaction_count=("item_id", "size"),
                    date=("date", "first"),
                )
            )
            grouped["label"] = 1.0  # ! Use binary implicit positives !

            # Masks
            train_mask = grouped["date"].between(train_start, train_end, inclusive="both")
            val_mask   = grouped["date"].between(val_start, val_end, inclusive="both")
            test_mask  = grouped["date"].between(test_start, test_end, inclusive="both")

            # Headers
            cols = [
                "user_id:token", "item_id:token", "timestamp:float",
                "label:float", "interaction_count:float", "date"
            ]
            print(grouped.head(10))
            print(list(grouped.columns))
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

        else:
            users_ids = [inter["user_id"] for inter in new_interactions]
            item_ids = [inter["item_id"] for inter in new_interactions]
            timestamps = [inter["timestamp"] for inter in new_interactions]

            df_new_inter = pd.DataFrame({
                "user_id:token": users_ids if len(new_interactions) > 1 else [users_ids],
                "item_id:token": item_ids if len(new_interactions) > 1 else [item_ids],
                "timestamp:float": timestamps if len(new_interactions) > 1 else [timestamps],
            })

            try: # They must exists because of k = 0
                train_file = os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.train.inter")
                train_file_df = pd.read_csv(train_file, sep="\t")
                # Open val file
                val_file = os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.val.inter")
                val_file_df = pd.read_csv(val_file, sep="\t")
                # Open test file
                test_file = os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.test.inter")
                test_file_df = pd.read_csv(test_file, sep="\t")
            except Exception as e:
                raise(f"problem loading dataset file. Init recbole model function must run before")

            working_df = pd.concat([train_file_df, val_file_df, test_file_df, df_new_inter], ignore_index=True)

            working_df["date"] = (
            pd.to_datetime(working_df["timestamp:float"].astype(float), unit="s", utc=True)
                .dt.tz_convert("Europe/Rome")
                .dt.normalize()
            )

            # Monthly buckets
            working_df["month"] = working_df["date"].dt.to_period("M")
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
                # Train on all months from the beginning
                train_start_idx = 0
            else:
                # Fixed moving window of size `train_len` (in months)
                if train_len is None or train_len <= 0:
                    raise ValueError("train_window_months must be a positive integer when use_all_data=False.")
                window = min(train_len, train_end_idx + 1)
                train_start_idx = train_end_idx - window + 1

            train_start = months[train_start_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            train_end = months[train_end_idx].to_timestamp(how="end").tz_localize("Europe/Rome")
            val_start = months[val_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            val_end = months[val_idx].to_timestamp(how="end").tz_localize("Europe/Rome")
            test_start = months[test_idx].to_timestamp(how="start").tz_localize("Europe/Rome")
            test_end = months[test_idx].to_timestamp(how="end").tz_localize("Europe/Rome")

            grouped = (
                working_df
                .groupby(["user_id:token", "item_id:token", "timestamp:float"], as_index=False)
                .agg(
                    interaction_count=("item_id:token", "size"),
                    date=("date", "first"),
                )
            )
            grouped["label:float"] = 1.0 
            grouped["interaction_count:float"] = 1.0

            # Masks
            train_mask = grouped["date"].between(train_start, train_end, inclusive="both")
            val_mask   = grouped["date"].between(val_start, val_end, inclusive="both")
            test_mask  = grouped["date"].between(test_start, test_end, inclusive="both")

            cols = [
                "user_id:token", "item_id:token", "timestamp:float",
                "label:float", "interaction_count:float", "date"
            ]

            self.working_train_df = grouped.loc[train_mask, cols].sort_values(["user_id:token", "timestamp:float"])
            self.working_val_df   = grouped.loc[val_mask, cols].sort_values(["user_id:token", "timestamp:float"])
            self.working_test_df  = grouped.loc[test_mask, cols].sort_values(["user_id:token", "timestamp:float"])

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

    def init_choice_model(self) -> None:
        df_init = self.dataset_unrolled_cold_start.copy()

        self.user_choice_model = ChoiceModel(interaction_df=df_init, config=self.config)
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
        item_id_recbole = recbole_dataset.token2id(recbole_dataset.iid_field, selected_item_id)
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

        # ---- patch ----
        trainer._train_epoch = custom_train_epoch
        trainer._valid_epoch = custom_valid_epoch

        # ---- fit ----
        best_valid_result = trainer.fit(
            train_data=self.train_data,
            valid_data=self.valid_data,
            show_progress=show_progress,
            saved=False
        )

        # ---- assemble dataframes ----
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
            except Exception as e:
                traceback.print_exc()
                raise Exception(f"{e}")
        try:
            if curr_epoch not in self.top_k_users_scores:
                self.top_k_users_scores[curr_epoch] = {}
                self.all_users_scores[curr_epoch] = {}
            if user_id_recbole not in self.top_k_users_scores[curr_epoch]:
                shifted_scores = item_scores - torch.min(item_scores) 
                
                probabilities = shifted_scores / torch.sum(shifted_scores)
                top_k_indices = torch.topk(probabilities, K_horizon, dim=0)[1]
                
                top_k_probs = probabilities[top_k_indices]
                top_k_probs = top_k_probs / torch.sum(top_k_probs)
            
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

            local_selected_index = int(self.rng.choice(len(top_k_probs), p=top_k_probs.cpu().numpy()))
            selected = int(top_k_indices[local_selected_index])
            while selected == 0:
                local_selected_index = int(self.rng.choice(len(top_k_probs), p=top_k_probs.cpu().numpy()))
                selected = int(top_k_indices[local_selected_index])
            
        except Exception as e:
            traceback.print_exc()
            print(e)
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

            epoch_interactions = []
            epochs_interactions_df = []
            print(f"\n --- Epoch n° {epoch} with model: {self.model_name_config} --- \n")

            # Get dates from the pre-computed distribution
            epoch_dates = [start_simulation_date + relativedelta(days=i) for i in range((end_date-start_simulation_date).days+1)]

            print(f"\n Epoch {epoch} dates: {[d.strftime("%Y-%m-%d") for d in epoch_dates]} \n")
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
                start_users = time.time()
                for user, items in valid_users.items():
                    valid_items = items
                    
                    if len(valid_items) == 0:
                        continue
                    basket_size = len(valid_items)
                    user_id_recbole = self.recbole_dataset.token2id(self.recbole_dataset.uid_field, user)

                    use_recommender_mask = (self.rng.random(basket_size) < p)

                    for use_rec in use_recommender_mask:
                        if use_rec:
                            recbole_item_id = self.recom_recbole_model(curr_epoch=epoch, user_id_recbole=user_id_recbole, K_horizon=k_horizon)
                        else:
                            if self.config.user_strategy["model_name"] == "Custom choice model":
                                recbole_item_id = self.recom_choice_model(curr_epoch=epoch, user_id_recbole=user_id_recbole)
                            else:
                                raise Exception(f"\n Only choice model is supported")
                        if recbole_item_id == 0:
                            print(f"\n --- Recommendation not found for user: {user_id_recbole} --- \n")
                            print(f"\n --- The item suggested was the 0 (PAD) --- \n")
                            continue

                        our_item_id = self.recbole_dataset.id2token(self.recbole_dataset.iid_field, recbole_item_id)
                        
                        single_interaction_df = {
                            "user_id": user,
                            "item_id": our_item_id,
                            "date": date,
                            "timestamp": recbole_timestamp,
                        }
                        epochs_interactions_df.append(single_interaction_df)
                end_users = time.time()
                if (end_users - start_users) <= 60:
                    print(f"\n Finished users new interactions in {end_users - start_users} seconds --- \n")
                else:
                    print(f"\n Finished users new interactions in {(end_users - start_users)/60} minutes --- \n")
            start_tr = time.time()
            if (epoch % self.config.delta_training_epoch) == 0:
                start_try = time.time()
                self.user_choice_model.update(new_interactions=epochs_interactions_df, epoch=epoch)
                self._build_window_for_epoch(k=epoch, new_interactions=epochs_interactions_df)
                self.init_recbole_model()
                end_try = time.time()
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

                # re-training strategy
                self._fit_with_tracking(
                        trainer=trainer,
                        save_stem=save_stem,
                        show_progress=False,
                        save_plots=False 
                    )

            if (len(epochs_interactions_df) == 0):
                raise Exception(f"ZERO Interactions in epoch {epoch}")
            end_tr = time.time()

            if (end_tr - start_tr) <= 60:
                print(f"\n Update, evaluation and training of the model elapsed in {end_tr - start_tr} seconds --- \n")
            else:
                print(f"\n Update, evaluation and training of the model elapsed in {(end_tr - start_tr)/60} minutes --- \n")
            
            end_e = time.time()
            
            if self.config.experiment_mode == "p-validation":
                if (end_e - start_e) <= 60:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds with total new interactions: {len(epochs_interactions_df)} -- p = {p} --- K AV items = {k_horizon} \n")
                else:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes with total new interactions: {len(epochs_interactions_df)} -- p = {p} --- K AV items = {k_horizon} \n")
            elif self.config.experiment_mode == "compare-models":
                if (end_e - start_e) <= 60:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds with total new interactions: {len(epochs_interactions_df)} -- recom model = {self.config.recommender_model["model_name"]} --- p = {p} --- K AV items = {k_horizon} \n")
                else:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes with total new interactions: {len(epochs_interactions_df)} -- recom model = {self.config.recommender_model["model_name"]} --- p = {p} --- K AV items = {k_horizon} \n")
            elif self.config.experiment_mode == "recom_model_test":
                if (end_e - start_e) <= 60:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds with total new interactions: {len(epochs_interactions_df)} -- p = {p} --- K AV items = {k_horizon} \n")
                else:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes with total new interactions: {len(epochs_interactions_df)} -- p = {p} --- K AV items = {k_horizon} \n")
            elif self.config.experiment_mode == "k_items_evaluation":
                if (end_e - start_e) <= 60:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {end_e - start_e} seconds with total new interactions: {len(epochs_interactions_df)} -- recom model = {self.config.recommender_model["model_name"]} --- p = {p} --- K AV items = {k_horizon} \n")
                else:
                    print(f"\n --- Time elapsed in epoch n° {epoch}: {(end_e - start_e)/60} minutes with total new interactions: {len(epochs_interactions_df)} -- recom model = {self.config.recommender_model["model_name"]} --- p = {p} --- K AV items = {k_horizon} \n")
            
            print(f"\n The epoch worked from {start_simulation_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")} \n")

            start_simulation_date = (start_simulation_date.replace(day=1) + relativedelta(months=1)).replace(year=start_simulation_date.year + (start_simulation_date.month // 12))
            end_date = start_simulation_date + relativedelta(months=1) - relativedelta(days=1)

            epoch_interactions_df = pd.DataFrame(epochs_interactions_df)
            epoch_interactions_df.to_csv(os.path.join(results_path, f"epoch_{epoch}.csv"))

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