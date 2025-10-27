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
        df_dataset_total = self.initialization_dataset.unrolled_dataset_total
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
    def init_recbole_dataset(self) -> None:
        working_df = self.dataset_unrolled_cold_start.copy()
        working_df['date'] = pd.to_datetime(working_df['date'], format="%Y-%m-%d")
        working_df['timestamp'] = working_df.date.values.astype(np.int64) // 10 ** 9
        working_df = working_df[["user_id", "item_id", "timestamp", "date"]].rename(columns=
            {
                "user_id": "user_id:token",
                "item_id": "item_id:token",
                "timestamp": "timestamp:float",
            }
        )
        working_df = working_df[(working_df['user_id:token'].isin(self.users_ids)) & (working_df['item_id:token'].isin(self.items_ids))]

        # Define training: from start of the dataset to (initialization months)- 2 
        self.start_training_filter = datetime(self.initialization_dataset.y1, 1, 1)
        self.end_train_filter = datetime(self.initialization_dataset.y1, 1, 1) + relativedelta(months=self.config.cold_start_months-2) - relativedelta(days=1)
        train_filter_dates = (working_df['date'] >= self.start_training_filter) & (working_df['date'] <= self.end_train_filter)

        # Define validation: from end of training, plus 1 month more
        self.start_val_filter = self.end_train_filter + relativedelta(days=1)
        self.end_val_filter = self.start_val_filter + relativedelta(months=1) - relativedelta(days=1)
        val_filter_dates = (working_df['date'] >= self.start_val_filter) & (working_df['date'] <= self.end_val_filter)

        # Define test: from end of validation, plus 1 month
        self.start_test_filter = self.end_val_filter + relativedelta(days=1)
        self.end_test_filter = self.start_test_filter + relativedelta(months=1) - relativedelta(days=1)
        test_filter_dates = (working_df['date'] >= self.start_test_filter) & (working_df['date'] <= self.end_test_filter)

        # Filter data
        self.working_train_df = working_df[train_filter_dates]
        self.working_train_df = self.working_train_df[["user_id:token", "item_id:token", "timestamp:float"]]

        self.working_val_df = working_df[val_filter_dates]
        self.working_val_df = self.working_val_df[["user_id:token", "item_id:token", "timestamp:float"]]

        self.working_test_df = working_df[test_filter_dates]
        self.working_test_df = self.working_test_df[["user_id:token", "item_id:token", "timestamp:float"]]

        self.test_users_ids = self.working_test_df['user_id:token'].unique()
        self.test_items_ids = self.working_test_df['item_id:token'].unique()

        self.working_train_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part1.inter"), index=False, sep='\t') 
        self.working_val_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part2.inter"), index=False, sep='\t') 
        self.working_test_df.to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part3.inter"), index=False, sep='\t')
        return self.working_train_df, self.working_val_df, self.working_test_df

    def init_recbole_model(self):
        if self.model_name_config == "Individual Random":
            self.model_name_recbole = "Pop"
        elif self.model_name_config == "Individual Popularity":
            self.model_name_recbole = "Pop"
        elif self.model_name_config == "Collective Random":
            self.model_name_recbole = "Random"
        elif self.model_name_config == "Collective Popularity":
            self.model_name_recbole = "Pop"
        elif self.model_name_config == "CF_KNN_user":
            self.model_name_recbole = "ItemKNN"
        elif self.model_name_config == "CF_KNN_item":
            self.model_name_recbole = "ItemKNN"
        elif self.model_name_config == "MultiVAE":
            self.model_name_recbole = "MultiVAE"
        elif self.model_name_config == "BPR":
            self.model_name_recbole = "BPR"
        elif self.model_name_config == "LightGCN":
            self.model_name_recbole = "LightGCN"
        elif self.model_name_config == "SpectralCF":
            self.model_name_recbole = "SpectralCF"
        elif self.model_name_config == "NNCF":
            self.model_name_recbole = "NNCF"
        elif self.model_name_config == "NeuMF":
            self.model_name_recbole = "NeuMF"
        else:
            raise Exception(f"Model name not recognized")
        
        parameter_dict = {
            'data_path': self.tmp_folder,
            'checkpoint_dir': os.path.join(self.tmp_folder, "checkpoints"),
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'load_col': {
                'inter': ['user_id', 'item_id', 'timestamp']
            },
            'epochs': self.config.recommender_model.epochs,
            'eval_args': {
                'group_by': 'user_id',
                'order': 'TO',
                'split': {'LS': 'valid_and_test'},
                'mode': 'full'
            },
            'benchmark_filename': ['part1', 'part2', 'part3'],
            'reproducibility': True,
            'seed': self.master_seed,
            'metrics': ["Precision", "Recall", "Hit", "NDCG", "ItemCoverage", "MRR", "MAP", "AveragePopularity"],
            'topk': 10,
            'valid_metric': 'NDCG@10',
            'use_gpu': True,
            'gpu_id': 0,
            "learning_rate": self.config.recommender_model.learning_rate if hasattr(self.config.recommender_model, "learning_rate") else 0.005,
            "n_layers": self.config.recommender_model.n_layers if hasattr(self.config.recommender_model, "n_layers") else None,
            "reg_weight": self.config.recommender_model.reg_weight if hasattr(self.config.recommender_model, "reg_weight") else None,
            "k": self.config.recommender_model.K if hasattr(self.config.recommender_model, "K") else 100,
            "shrink": self.config.recommender_model.shrink if hasattr(self.config.recommender_model, "shrink") else 0.0,
            "user_embedding_size": self.config.recommender_model.user_embedding_size if hasattr(self.config.recommender_model, "user_embedding_size") else 64,
            "item_embedding_size": self.config.recommender_model.item_embedding_size if hasattr(self.config.recommender_model, "item_embedding_size") else 64,
            "user_hidden_size_list": self.config.recommender_model.user_hidden_size_list if hasattr(self.config.recommender_model, "user_hidden_size_list") else [64, 64],
            "item_hidden_size_list": self.config.recommender_model.item_hidden_size_list if hasattr(self.config.recommender_model, "item_hidden_size_list") else [64, 64],
            "inter_matrix_type": self.config.recommender_model.inter_matrix_type if hasattr(self.config.recommender_model, "inter_matrix_type") else "01",
            "mlp_hidden_size": self.config.recommender_model.mlp_hidden_size if hasattr(self.config.recommender_model, "mlp_hidden_size") else [64, 32, 16],
            "dropout_prob": self.config.recommender_model.dropout_prob if hasattr(self.config.recommender_model, "dropout_prob") else 0.1,
            "neigh_embedding_size": self.config.recommender_model.neigh_embedding_size if hasattr(self.config.recommender_model, "neigh_embedding_size") else 64,
            "num_conv_kernel": self.config.recommender_model.num_conv_kernel if hasattr(self.config.recommender_model, "num_conv_kernel") else 128
        }
        self.parameter_dict = parameter_dict   

        try:
            self.model_config = Config(model=self.model_name_recbole, dataset=f'experiment_dataset', config_dict=self.parameter_dict)
            init_seed(self.model_config['seed'], self.model_config['reproducibility'])
            init_logger(self.model_config)
            self.logger = getLogger()
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)
            self.logger.addHandler(c_handler)
        except Exception as e:
            raise Exception(f"Error during the configuration of the model -> {e}")
        
        try:
            self.recbole_dataset = create_dataset(self.model_config)
            self.train_data, self.valid_data, self.test_data = data_preparation(config=self.model_config, dataset=self.recbole_dataset)
            self.logger.info(self.train_data)
        except Exception as e:
            raise Exception(f"Error during the initialization of the dataset -> {e}")

        if self.model_name_config == "Individual Random":
            self.recbole_model = IndividualRandom(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "Collective Random":
            self.recbole_model = Random(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "Individual Popularity":
            self.recbole_model = IndividualPopularity(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "Collective Popularity":
            self.recbole_model = Pop(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "CF_KNN_item":
            self.recbole_model = ItemKNN(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "CF_KNN_user":
            self.recbole_model = UserKNN(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "MultiVAE":
            self.recbole_model = MultiVAE(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "BPR":
            self.recbole_model = BPR(config=self.model_config, dataset=self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "LightGCN":
            self.recbole_model = LightGCN(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "NeuMF":
            self.recbole_model = NeuMF(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "SpectralCF":
            self.recbole_model = SpectralCF(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        elif self.model_name_config == "NNCF":
            self.recbole_model = NNCF(self.model_config, self.recbole_dataset).to(self.model_config['device'])
        else:
            raise Exception(f"Something's wrong with the model name")

        trainer = Trainer(self.model_config, self.recbole_model)

        # Initialize lists to store results from all epochs
        all_validation_results = []
        all_test_results = []
        all_train_losses = []

        # Store original methods
        original_valid_epoch = trainer._valid_epoch
        original_train_epoch = trainer._train_epoch

        def custom_train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False):
            # Call original training epoch
            train_loss = original_train_epoch(train_data, epoch_idx, loss_func, show_progress)
            # Store training loss
            all_train_losses.append(train_loss)
            return train_loss

        def custom_valid_epoch(valid_data, show_progress=False):
            # Call original validation
            valid_result = original_valid_epoch(valid_data, show_progress)
            # Store validation results
            all_validation_results.append(valid_result)
            # Also evaluate on test data
            test_result = trainer.evaluate(self.test_data, load_best_model=False, show_progress=False)
            all_test_results.append(test_result)
            return valid_result

        # Replace the methods
        trainer._train_epoch = custom_train_epoch
        trainer._valid_epoch = custom_valid_epoch

        # Train the model
        best_valid_result = trainer.fit(
            train_data=self.train_data,
            valid_data=self.valid_data,
            show_progress=False,
            saved=False
        )

        # Combine all results into DataFrame
        data_for_df = []
        max_epochs = max(len(all_train_losses), len(all_validation_results), len(all_test_results))

        for epoch in range(max_epochs):
            row_data = {"epoch": epoch}
            
            # Add training loss
            if epoch < len(all_train_losses):
                row_data["train_loss"] = all_train_losses[epoch]
            
            # Add validation results
            if epoch < len(all_validation_results):
                valid_data = all_validation_results[epoch]
                if isinstance(valid_data, dict):
                    for key, value in valid_data.items():
                        row_data[f"valid_{key}"] = value
                elif isinstance(valid_data, tuple):
                    if len(valid_data) >= 2 and isinstance(valid_data[1], dict):
                        for key, value in valid_data[1].items():
                            row_data[f"valid_{key}"] = value
                    elif len(valid_data) >= 1:
                        row_data["valid_score"] = valid_data[0]
            
            # Add test results
            if epoch < len(all_test_results):
                for key, value in all_test_results[epoch].items():
                    row_data[f"test_{key}"] = value
            
            data_for_df.append(row_data)

        # Create DataFrame
        results_df = pd.DataFrame(data_for_df)
        results_df.set_index('epoch', inplace=True)

        # Separate DataFrames for different metrics
        validation_cols = [col for col in results_df.columns if col.startswith('valid_')]
        test_cols = [col for col in results_df.columns if col.startswith('test_')]

        # Create validation DataFrame (remove 'valid_' prefix)
        validation_df = results_df[validation_cols].copy()
        validation_df.columns = [col.replace('valid_', '') for col in validation_df.columns]

        # Create test DataFrame (remove 'test_' prefix)
        test_df = results_df[test_cols].copy()
        test_df.columns = [col.replace('test_', '') for col in test_df.columns]

        # Create loss DataFrame
        loss_df = results_df[['train_loss']].copy() if 'train_loss' in results_df.columns else pd.DataFrame()

        # self.best_valid_list = [{"epoch_index": 0, **best_valid_result}]

        # print(loss_df.head(20))
        # plt.figure(figsize=(10, 10))

        # plt.plot(validation_df.index.values, validation_df["ndcg@10"], label="Validation ndcg@10")

        # plt.plot(test_df.index.values, test_df["ndcg@10"], label="Test ndcg@10")

        # plt.xticks(range(0, len(all_train_losses) + 1, 2), fontsize=25)  # Every 2 epochs
        # plt.tick_params(axis="y", which="major", labelsize=25)
        # plt.xlabel("Epochs", fontsize=28)
        # plt.ylabel("ndcg@10", fontsize=28)
        # plt.title(f"Training phase: {self.model_name_config}", fontsize=30, fontweight="bold")
        # plt.grid(True, linestyle="--", alpha=0.5)
        # plt.legend(fontsize=26, loc="best")
        # plt.tight_layout()

        # plt.savefig(f"training_phase-model={self.model_name_config}_dataset=amazon.png", bbox_inches='tight')
        # plt.close()

        # plt.figure(figsize=(10, 10))

        # plt.plot(loss_df.index.values, loss_df["train_loss"], label="Loss")

        # plt.xticks(range(0, len(all_train_losses) + 1, 2), fontsize=25)  # Every 2 epochs
        # plt.tick_params(axis="y", which="major", labelsize=25)
        # plt.xlabel("Epochs", fontsize=28)
        # plt.ylabel("Train loss", fontsize=28)
        # plt.title(f"Training phase loss: {self.model_name_config}", fontsize=30, fontweight="bold")
        # plt.grid(True, linestyle="--", alpha=0.5)
        # plt.legend(fontsize=26, loc="best")
        # plt.tight_layout()

        # plt.savefig(f"training_phase_loss_model={self.model_name_config}_dataset=amazon.png", bbox_inches='tight')
        # plt.close()

        json_filename = f"test_metrics_model={self.model_name_config}_dataset=amazon.json"

        if not os.path.exists(json_filename) and all_test_results:
            final_test_metrics = all_test_results[-1]  # Get metrics from the last epoch
            
            # Create a dictionary with the metrics you want to track
            metrics_to_save = ["Precision", "Recall", "Hit", "NDCG", "ItemCoverage", "MRR", "MAP", "AveragePopularity"]
            
            # Filter and organize the metrics
            test_metrics_json = {}
            for metric_name in metrics_to_save:
                # Look for metrics with different possible formats (case variations, @k suffixes)
                found_metrics = {}
                for key, value in final_test_metrics.items():
                    # Check if the metric name is in the key (case insensitive)
                    if metric_name.lower() in key.lower():
                        found_metrics[key] = value
                
                if found_metrics:
                    test_metrics_json[metric_name] = found_metrics
                else:
                    test_metrics_json[metric_name] = None  # or skip this metric
            
            # Also add some metadata
            test_metrics_json["model_name"] = self.model_name_config
            test_metrics_json["dataset"] = "amazon"
            test_metrics_json["total_epochs"] = len(all_test_results)
            test_metrics_json["final_epoch"] = len(all_test_results) - 1
            
            with open(json_filename, 'w') as f:
                json.dump(test_metrics_json, f, indent=2, default=str)  # default=str handles non-serializable types
            
            print(f"Test metrics saved to: {json_filename}")
            
            # Optionally print the metrics
            print("\nFinal Test Metrics:")
            for metric_name, metric_values in test_metrics_json.items():
                if metric_name not in ["model_name", "dataset", "total_epochs", "final_epoch"]:
                    print(f"{metric_name}: {metric_values}")

        else:
            print("No test results available to save.")

    def tuning_hyperparameters(self) -> None:
        hyper_file = f"tuning_parameters/{self.model_name_config}.hyper"
        export_result_file = f"tuning_parameters/{self.model_name_config}.result"
        if self.model_name_config in ["Collective Random", "Collective Popularity"]:
            return None
        if os.path.exists(export_result_file):
            print(f"\n Tuning of the model {self.model_name_config} already in the folder. Skip \n")
            return None
        def _objective_function(params_dict=None, config_file_list=None):
            # config = Config(model=self.model_name_recbole, dataset=f'models_tmp/experiment_dataset', config_dict=params_dict)
            dataset = create_dataset(self.model_config)
            train_data, valid_data, test_data = data_preparation(config=self.model_config, dataset=dataset)
            model = self.recbole_model  
            trainer = Trainer(self.model_config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
            test_result = trainer.evaluate(test_data)

            return {
                'model': self.model_name_config,
                'best_valid_score': best_valid_score,
                'valid_score_bigger': self.model_config['valid_metric_bigger'],
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
    
    def update_train_val_test(self, new_interactions: list, epoch: int = 1):
        if epoch < 1:
            raise Exception(f"Current epoch must be > 1")

        subset_cols = ["user_id:token", "item_id:token", "timestamp:float"]

        users_ids = [inter["user_id"] for inter in new_interactions]
        item_ids = [inter["item_id"] for inter in new_interactions]
        timestamps = [inter["timestamp"] for inter in new_interactions]

        df_new_inter = pd.DataFrame({
            "user_id:token": users_ids if len(new_interactions) > 1 else [users_ids],
            "item_id:token": item_ids if len(new_interactions) > 1 else [item_ids],
            "timestamp:float": timestamps if len(new_interactions) > 1 else [timestamps]
        })

        # Define training set: old training plus old validation set
        # Open training file
        train_file = os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part1.inter")
        train_file_df = pd.read_csv(train_file, sep="\t")
        # Open val file
        val_file = os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part2.inter")
        val_file_df = pd.read_csv(val_file, sep="\t")
        # Open test file
        test_file = os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part3.inter")
        test_file_df = pd.read_csv(test_file, sep="\t")

        train_file_df = pd.concat([train_file_df[subset_cols], val_file_df[subset_cols]], ignore_index=True)
        val_file_df = test_file_df 
        test_file_df = df_new_inter

        train_file_df[subset_cols].to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part1.inter"), index=False, sep='\t')
        val_file_df[subset_cols].to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part2.inter"), index=False, sep='\t')
        test_file_df[subset_cols].to_csv(os.path.join(self.tmp_folder, self.tmp_dataset_folder, f"experiment_dataset.part3.inter"), index=False, sep='\t')
    
    def update_actual_dataset(self):
        """
            Get the .inter files with the tmp model folder and define a new dataset with those files.
            Also, re-init the model onto the dataset, training and evaluating it
        """

        try:
            self.model_config = Config(model=self.model_name_recbole, dataset=f'experiment_dataset', config_dict=self.parameter_dict)
        except Exception as e:
            raise Exception(f"Error during the configuration of the model -> {e}")
        try:
            self.recbole_dataset = create_dataset(self.model_config)

            self.train_data, self.valid_data, self.test_data = data_preparation(config=self.model_config, dataset=self.recbole_dataset)

        except Exception as e:
            raise Exception(f"Error during the initialization of the dataset -> {e}")
        
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
                
                for user, items in valid_users.items():
                    # Take only valid items
                    # The occurences even for the same items are preserved
                    # valid_items = [item for item in items if item in self.items_ids]
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
            
            start_update_dataset = time.time()
            self.update_train_val_test(new_interactions=epochs_interactions_df, epoch=epoch)

            self.user_choice_model.update(new_interactions=epochs_interactions_df, epoch=epoch)

            end_update_dataset = time.time()
            if (end_update_dataset - start_update_dataset) <= 60:
                print(f"\n File dataset updates took {end_update_dataset - start_update_dataset} seconds \n")
            else:
                print(f"\n File dataset update took {(end_update_dataset - start_update_dataset)/60} minutes \n") 

            start_tr = time.time()
            
            if (epoch % self.config.delta_training_epoch) == 0:
                self.update_actual_dataset()

                # Keep training the SAME model (continues learning with new data)
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