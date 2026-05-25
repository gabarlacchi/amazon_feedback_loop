import traceback
import hvplot.pandas
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import os
import isoweek
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
import sys
color_list = sns.color_palette()
sys.path.append('../feedback_loop_naive')
from feedback_loop.choice_model import ChoiceModel

from feedback_loop.coop_dataset import CoopDataset
from feedback_loop.lfm_dataset import LFMDataset
from feedback_loop.amazonecommerce_dataset import AmazonECommerceDataset
from feedback_loop.yambda_dataset import YambdaDataset

# from feedback_loop_v2.movielens_dataset import MovieLensDataset
# from feedback_loop_v2.amazon_kindle_dataset import AmazonKindleDataset
# from feedback_loop_v2.deezer_dataset import DeezerDataset

from feedback_loop.coop_feedback_loop import CoopFeedbackLoop
from feedback_loop.lfm_feedback_loop import LFMFeedbackLoop
from feedback_loop.amazonecommerce_feedback_loop import AmazonECommerceFeedbackLoop
from feedback_loop.yambda_feedback_loop import YambdaFeedbackLoop
# from feedback_loop.coop_feedback_loop_v2_timetestset import CoopFeedbackLoop
# from feedback_loop_v2.movielens_feedback_loop import MovieLensFeedbackLoop
# from feedback_loop_v2.amazon_kindle_feedback_loop import AmazonKindleFeedbackLoop

DATASETS = ["coop", "lfm", "amazon_e_commerce", "yambda"]

RESULTS_PATH = "./results"
RESULTS_PATH_LFM = "./results_lfm_albums"
RESULTS_PATH_AMAZON = "./results_amazon_category"
RESULTS_PATH_YAMADA = "./results_yambda"

class DotDict(dict):
    """A dictionary that supports dot notation."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def main(args: argparse):
    json_config_path = args.json_config
    if not os.path.exists(json_config_path):
        raise Exception(f"JSON config file not found")

    with open(json_config_path, 'r') as f:
        config = json.load(f)
    
    config = DotDict(config)

    if config.dataset.lower() not in DATASETS:
        raise Exception(f"\n Dataset not implemented yet or not recognized -> {config.dataset} \n")
    
    if config.dataset.lower() == "lfm":
        init_dataset = LFMDataset(config=config)
        init_dataset.setup()

        model_name = config.recommender_model["model_name"]
        usrstrategy_model_name = config.user_strategy["model_name"]

        if config.experiment_mode == "p-validation":
            if config.experiment_mode == "p-validation":
                results_path = os.path.join(RESULTS_PATH_LFM, "p-validation", "lfm")
            
            if config.use_rca:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_rcathr={config.rca_threshold}_coldStart={config.cold_start_months}"
            else:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_coldStart={config.cold_start_months}"

            # if config.use_rca:
            #     fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_rcathr={config.rca_threshold}_coldStart={config.cold_start_months}"
            # else:
            #     fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_coldStart={config.cold_start_months}"

            ps = [1, 0.8, 0.5, 0.2, 0]
            ps_names = [ f"{model_name}", "P=0.8", "P=0.5", "P=0.2", f"{usrstrategy_model_name}"]
            # ps = [0, 0.2, 0.5, 0.8, 1]
            # ps_names = [ f"{usrstrategy_model_name}", "P=0.2", "P=0.5", "P=0.8", f"{model_name}"]

            k_av_items = config.k_items

            for p, p_name in zip(ps, ps_names):
                to_create = fld_name.replace("probability", str(p)).replace("KITEMS", str(k_av_items))
                os.makedirs(os.path.join(results_path, to_create), exist_ok=True)
                with open(os.path.join(results_path, to_create, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                os.makedirs(os.path.join(results_path, to_create, "dataframe"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "recom_scores"), exist_ok=True)
                p = round(p, 2)

                feedback_loop_tool = LFMFeedbackLoop(config=config, initialization_dataset=init_dataset)
                feedback_loop_tool.init_experiment()
                inter_train, inter_val, inter_test = feedback_loop_tool.init_recbole_dataset()
                feedback_loop_tool.init_recbole_model()
                _ = feedback_loop_tool.tuning_hyperparameters()

                if usrstrategy_model_name == "Custom choice model":
                    feedback_loop_tool.init_choice_model()

                    # Set results path for user choice model
                    feedback_loop_tool.user_choice_model.results_path = os.path.join(results_path, to_create, "choice_model")
                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(results_path, to_create, "dataframe"), 
                    results_scores_path=os.path.join(results_path, to_create, "recom_scores"),
                    k_horizon=k_av_items
                )
                with open(os.path.join(results_path, to_create, "model_metrics.json"), 'w') as f:
                    json.dump(model_metrics, f, indent=4)

    if config.dataset.lower() == "yambda":
        init_dataset = YambdaDataset(config=config)
        init_dataset.setup()

        model_name = config.recommender_model["model_name"]
        usrstrategy_model_name = config.user_strategy["model_name"]

        if config.experiment_mode == "p-validation":
            results_path = os.path.join(RESULTS_PATH_YAMADA, "p-validation", "yambda")

            if config.use_rca:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_rcathr={config.rca_threshold}_coldStart={config.cold_start_months}"
            else:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_coldStart={config.cold_start_months}"
            
            
            ps = [0, 0.2, 0.5, 0.8, 1]
            ps_names = [f"{usrstrategy_model_name}", "P=0.2", "P=0.5", "P=0.8", f"{model_name}"]
            k_av_items = config.k_items

            # ps = [1, 0.8, 0.5, 0.2, 0]
            # ps_names = [f"{model_name}", "P=0.8", "P=0.5", "P=0.2", f"{usrstrategy_model_name}"]

            for p, p_name in zip(ps, ps_names):
                to_create = fld_name.replace("probability", str(p)).replace("KITEMS", str(k_av_items))
                os.makedirs(os.path.join(results_path, to_create), exist_ok=True)
                with open(os.path.join(results_path, to_create, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                os.makedirs(os.path.join(results_path, to_create, "dataframe"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "recom_scores"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "choice_model"), exist_ok=True)
                p = round(p, 2)

                feedback_loop_tool = YambdaFeedbackLoop(config=config, initialization_dataset=init_dataset)
                feedback_loop_tool.init_experiment()

                inter_train, inter_val, inter_test = feedback_loop_tool.init_recbole_dataset()

                feedback_loop_tool.init_recbole_model()
                _ = feedback_loop_tool.tuning_hyperparameters()

                if usrstrategy_model_name == "Custom choice model":
                    feedback_loop_tool.init_choice_model()

                    # Set results path for user choice model
                    feedback_loop_tool.user_choice_model.results_path = os.path.join(results_path, to_create, "choice_model")
                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(results_path, to_create, "dataframe"), 
                    results_scores_path=os.path.join(results_path, to_create, "recom_scores"),
                    k_horizon=k_av_items
                )

                with open(os.path.join(results_path, to_create, "model_metrics.json"), 'w') as f:
                    json.dump(model_metrics, f, indent=4)
            
    if config.dataset.lower() == "amazon_e_commerce":
        init_dataset = AmazonECommerceDataset(config=config)
        init_dataset.setup()
        
        model_name = config.recommender_model["model_name"]
        usrstrategy_model_name = config.user_strategy["model_name"]

        if config.experiment_mode == "p-validation":
            results_path = os.path.join(RESULTS_PATH_AMAZON, "p-validation", "amazon")

            if config.use_rca:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_rcathr={config.rca_threshold}_coldStart={config.cold_start_months}"
            else:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_coldStart={config.cold_start_months}"

            # ps = [0, 0.2, 0.5, 0.8, 1]
            # ps_names = [f"{usrstrategy_model_name}", "P=0.2", "P=0.5", "P=0.8", f"{model_name}"]

            ps = [0, 0.2, 0.5, 0.8, 1]
            ps_names = [f"{model_name}", "P=0.8", "P=0.5", "P=0.2", f"{usrstrategy_model_name}"]
            # ps = [0.2, 0.8]
            # ps_names = ["p=0.2", "p=0.8"]
            k_av_items = config.k_items

            for p, p_name in zip(ps, ps_names):
                to_create = fld_name.replace("probability", str(p)).replace("KITEMS", str(k_av_items))
                os.makedirs(os.path.join(results_path, to_create), exist_ok=True)
                with open(os.path.join(results_path, to_create, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                os.makedirs(os.path.join(results_path, to_create, "dataframe"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "recom_scores"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "choice_model"), exist_ok=True)
                p = round(p, 2)

                feedback_loop_tool = AmazonECommerceFeedbackLoop(config=config, initialization_dataset=init_dataset)
                feedback_loop_tool.init_experiment()

                inter_train, inter_val, inter_test = feedback_loop_tool.init_recbole_dataset()
                feedback_loop_tool.init_recbole_model()
                _ = feedback_loop_tool.tuning_hyperparameters()
                if usrstrategy_model_name == "Custom choice model":
                    feedback_loop_tool.init_choice_model()

                    # Set results path for user choice model
                    feedback_loop_tool.user_choice_model.results_path = os.path.join(results_path, to_create, "choice_model")

                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(results_path, to_create, "dataframe"), 
                    results_scores_path=os.path.join(results_path, to_create, "recom_scores"),
                    k_horizon=k_av_items
                )
                with open(os.path.join(results_path, to_create, "model_metrics.json"), 'w') as f:
                    json.dump(model_metrics, f, indent=4)

    if config.dataset.lower() == "coop":
        init_dataset = CoopDataset(config=config)
        init_dataset.setup()

        model_name = config.recommender_model["model_name"]
        usrstrategy_model_name = config.user_strategy["model_name"]
        
        if config.experiment_mode == "p-validation":
            results_path = os.path.join(RESULTS_PATH, "p-validation", "coop")
            # fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}-p=probability-Kitems=KITEMS-PushDiversity={config.push_diversity}"
            if config.use_rca:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_rcathr={config.rca_threshold}_coldStart={config.cold_start_months}"
            else:
                fld_name = f"recom={model_name}-usrstrategy={usrstrategy_model_name}_tau={config.user_strategy["tau"]}_candidate_set_size={config.user_strategy["candidate_set"]["size"]}-p=probability-Kitems=KITEMS-useRCA={config.use_rca}_coldStart={config.cold_start_months}"

            ps = [0, 0.2, 0.5, 0.8, 1]
            ps_names = [f"{usrstrategy_model_name}", "P=0.2", "P=0.5", "P=0.8", f"{model_name}"]

            # ps = [0.2, 0.8]
            # ps_names = ["p=0.2", "p=0.8"]
            k_av_items = config.k_items

            for p, p_name in zip(ps, ps_names):
                to_create = fld_name.replace("probability", str(p)).replace("KITEMS", str(k_av_items))
                os.makedirs(os.path.join(results_path, to_create), exist_ok=True)
                with open(os.path.join(results_path, to_create, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                os.makedirs(os.path.join(results_path, to_create, "dataframe"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "recom_scores"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "choice_model"), exist_ok=True)
                p = round(p, 2)

                feedback_loop_tool = CoopFeedbackLoop(config=config, initialization_dataset=init_dataset)
                feedback_loop_tool.init_experiment()

                inter_train, inter_val, inter_test = feedback_loop_tool.init_recbole_dataset()
                feedback_loop_tool.init_recbole_model()
                _ = feedback_loop_tool.tuning_hyperparameters()
                if usrstrategy_model_name == "Custom choice model":
                    feedback_loop_tool.init_choice_model()
                    # Set results path for user choice model
                    feedback_loop_tool.user_choice_model.results_path = os.path.join(results_path, to_create, "choice_model")
                
                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(results_path, to_create, "dataframe"), 
                    results_scores_path=os.path.join(results_path, to_create, "recom_scores"),
                    k_horizon=k_av_items
                )
                with open(os.path.join(results_path, to_create, "model_metrics.json"), 'w') as f:
                    json.dump(model_metrics, f, indent=4)

        if config.experiment_mode == "k_items_evaluation":
            results_path = os.path.join(RESULTS_PATH, "k-items-eval", "coop")
            fld_name = f"recom={model_name}-K_items=k_items"

            Ks = [5, 10, 20, 30, 40, 50, 70, 80, 90, 100, 200, 300, 400, 500]
            ks_names = ["5", "10", "20", "30", "40", "50", "70", "80", "90", "100", "200", "300", "400", "500"]

            for k, k_name in zip(Ks, ks_names):
                to_create = fld_name.replace("k_items", str(k))
                os.makedirs(os.path.join(results_path, to_create), exist_ok=True)
                with open(os.path.join(results_path, to_create, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                os.makedirs(os.path.join(results_path, to_create, "dataframe"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "recom_scores"), exist_ok=True)
                feedback_loop_tool = CoopFeedbackLoop(config=config, initialization_dataset=init_dataset)
                feedback_loop_tool.init_experiment()
                users_train, users_val, users_test = feedback_loop_tool.init_recbole_dataset()
                np.save(os.path.join(results_path, to_create, "users_train.npy"), users_train)
                np.save(os.path.join(results_path, to_create, "users_val.npy"), users_val)
                np.save(os.path.join(results_path, to_create, "users_test.npy"), users_test)

                p = 1.0

                feedback_loop_tool.init_recbole_model()
                _ = feedback_loop_tool.tuning_hyperparameters()
                
                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(results_path, to_create, "dataframe"), 
                    results_scores_path=os.path.join(results_path, to_create, "recom_scores"),
                    k_horizon=k
                )

                with open(os.path.join(results_path, to_create, "model_metrics.json"), 'w') as f:
                    json.dump(model_metrics, f, indent=4)

        if config.experiment_mode == "compare-models":
            results_path = os.path.join(RESULTS_PATH, "models-compare", "coop")
            fld_name = f"k_item={config.k_items}-model=recom_model-p=probability-usrstrategy=STRATEGYMODEL"

            m = [
                {
                    "model_name": "CF_KNN_item",
                    "epochs": 1,
                    "k": 100,
                    "shrink": 0.0
                },
                {
                    "model_name": "BPR",
                    "epochs": 25,
                    "learning_rate": 0.001
                },
                {
                    "model_name": "NeuMF",
                    "epochs": 25,
                    "dropout_prob": 0.2,
                    "learning_rate": 0.004,
                    "mlp_hidden_size": [
                        64,
                        32,
                        16
                    ]
                },
                {
                    "model_name": "NGCF",
                    "epochs": 25,
                    "learning_rate": 0.02,
                    "embedding_size": 32,
                    "hidden_size_list": [64, 64, 64],
                    "node_dropout": 0.2,
                    "message_dropout": 0.01,
                    "reg_weight": 0.002
                },
                {
                    "model_name": "SGL",
                    "epochs": 25,
                    "learning_rate": 0.03,
                    "embedding_size": 128,
                    "n_layers": 3,
                    "reg_weight": 0.002,
                    "ssl_tau": 0.12,
                    "ssl_weight": 0.03,
                    "drop_ratio": 0.2,
                    "type": "ND"
                },
                {
                    "model_name": "SpectralCF",
                    "epochs": 15,
                    "learning_rate": 0.0045,
                    "n_layers": 3,
                    "reg_weight" :0.0005
                },
                {
                    "model_name": "DeepFM",
                    "epochs": 25,
                    "learning_rate": 0.02,
                    "embedding_size": 32,
                    "mlp_hidden_size": [
                        64,
                        64
                    ],
                    "dropout_prob": 0.4
                },
                {
                    "model_name": "DCNV2",
                    "epochs": 25,
                    "learning_rate": 0.008,
                    "mlp_hidden_size": [
                        768,
                        768
                    ],
                    "dropout_prob": 0.1,
                    "reg_weight": 0.0005,
                    "mixed": False,
                    "structure": "stacked",
                    "cross_layer_num": 3,
                    "expert_num": 3,
                    "low_rank": 128,
                    "embedding_size": 16
                },
                {
                    "model_name": "FM",
                    "epochs": 25,
                    "learning_rate": 0.0001,
                    "embedding_size": 16
                }
            ]

            recom_models = [
                {
                    "model_name": "Collective Random",
                    "epochs": 1
                },
                {
                    "model_name": "CF_KNN_item",
                    "epochs": 1,
                    "k": 100,
                    "shrink": 0.0
                },
                {
                    "model_name": "CF_KNN_user",
                    "epochs": 1,
                    "k": 100,
                    "shrink": 0.0
                },
                {
                    "model_name": "Collective Popularity",
                    "epochs": 1
                },
                {
                    "model_name": "BPR",
                    "epochs": 25,
                    "learning_rate": 0.001
                },
                {
                    "model_name": "MultiVAE",
                    "epochs": 25,
                    "learning_rate": 0.01,
                    "dropout_prob": 0.16,
                    "latent_dimension": 256,
                    "mlp_hidden_size": [
                        300,
                        200
                    ]
                },
                {
                    "model_name": "LightGCN",
                    "epochs": 25,
                    "learning_rate": 0.0006,
                    "n_layers": 3,
                    "reg_weight": 0.0006
                },
                {
                    "model_name": "NeuMF",
                    "epochs": 25,
                    "dropout_prob": 0.2,
                    "learning_rate": 0.004,
                    "mlp_hidden_size": [
                        64,
                        32,
                        16
                    ]
                },
                {
                    "model_name": "NNCF",
                    "epochs": 10,
                    "learning_rate": 0.05,
                    "mlp_hidden_size" :[128,64,32,16],
                    "neigh_embedding_size": 32,
                    "num_conv_kernel": 128
                },
                {
                    "model_name": "NGCF",
                    "epochs": 25,
                    "learning_rate": 0.02,
                    "embedding_size": 32,
                    "hidden_size_list": [64, 64, 64],
                    "node_dropout": 0.2,
                    "message_dropout": 0.01,
                    "reg_weight": 0.002
                },
                {
                    "model_name": "SGL",
                    "epochs": 25,
                    "learning_rate": 0.03,
                    "embedding_size": 128,
                    "n_layers": 3,
                    "reg_weight": 0.002,
                    "ssl_tau": 0.12,
                    "ssl_weight": 0.03,
                    "drop_ratio": 0.2,
                    "type": "ND"
                },
                {
                    "model_name": "EASE",
                    "epochs": 25,
                    "learning_rate": 0.02,
                    "reg_weight": 250.0
                },
                {
                    "model_name": "SpectralCF",
                    "epochs": 15,
                    "learning_rate": 0.0045,
                    "n_layers": 3,
                    "reg_weight" :0.0005
                },
                {
                    "model_name": "DeepFM",
                    "epochs": 25,
                    "learning_rate": 0.02,
                    "embedding_size": 32,
                    "mlp_hidden_size": [
                        64,
                        64
                    ],
                    "dropout_prob": 0.4
                },
                {
                    "model_name": "xDeepFM",
                    "epochs": 25,
                    "learning_rate": 0.0001,
                    "embedding_size": 64,
                    "mlp_hidden_size": [
                        128,
                        128,
                        128
                    ],
                    "dropout_prob": 0.1,
                    "reg_weight" :0.0005,
                    "cin_layer_size": [
                        128,
                        128                    
                    ],
                    "direct": True
                },
                {
                    "model_name": "DCNV2",
                    "epochs": 25,
                    "learning_rate": 0.008,
                    "mlp_hidden_size": [
                        768,
                        768
                    ],
                    "dropout_prob": 0.1,
                    "reg_weight": 0.0005,
                    "mixed": False,
                    "structure": "stacked",
                    "cross_layer_num": 3,
                    "expert_num": 3,
                    "low_rank": 128,
                    "embedding_size": 16
                },
                {
                    "model_name": "FM",
                    "epochs": 25,
                    "learning_rate": 0.0001,
                    "embedding_size": 16
                },
            ]

            for model_data in recom_models:
                p = 0.5
                k_items = config.k_items
                model_name = model_data["model_name"]
                config.recommender_model = model_data
                config.costumer_choice_prob = p

                to_create = fld_name.replace("recom_model", str(model_name)).replace("probability", str(p)).replace("STRATEGYMODEL", config.user_strategy["model_name"])
                os.makedirs(os.path.join(results_path, to_create, "dataframe"), exist_ok=True)
                os.makedirs(os.path.join(results_path, to_create, "recom_scores"), exist_ok=True)
                with open(os.path.join(results_path, to_create, "config.json"), 'w') as f:
                    json.dump(config, f, indent=4)
                
                feedback_loop_tool = CoopFeedbackLoop(config=config, initialization_dataset=init_dataset)
                feedback_loop_tool.init_experiment()
                users_train, users_val, users_test = feedback_loop_tool.init_recbole_dataset()
                np.save(os.path.join(results_path, to_create, "users_train.npy"), users_train)
                np.save(os.path.join(results_path, to_create, "users_val.npy"), users_val)
                np.save(os.path.join(results_path, to_create, "users_test.npy"), users_test)

                feedback_loop_tool.init_recbole_model()
                _ = feedback_loop_tool.tuning_hyperparameters()
                
                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(results_path, to_create, "dataframe"), 
                    results_scores_path=os.path.join(results_path, to_create, "recom_scores"),
                    k_horizon=k_items
                )

                with open(os.path.join(results_path, to_create, "model_metrics.json"), 'w') as f:
                    json.dump(model_metrics, f, indent=4)
        
        


            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--json_config",
        type=str,
        required=True,
        default="json_config/feedback_loop.json",
        help="[str] Set for specific json config.",
    )

    args = parser.parse_args()

    _ = main(args)





