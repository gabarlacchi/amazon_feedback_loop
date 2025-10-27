import argparse
import os 
import json 
from copy import deepcopy

from utils import DotDict, make_rep_seed
from amazonecommerce_dataset import AmazonECommerceDataset
from amazonecommerce_feedback_loop import AmazonECommerceFeedbackLoop

RESULTS_PATH_AMAZON = "./results_amazon_category"


def main(args: argparse):
    if not os.path.exists(RESULTS_PATH_AMAZON):
        os.makedirs(RESULTS_PATH_AMAZON)

    json_config_path = args.json_config
    if not os.path.exists(json_config_path):
        raise Exception(f"JSON config file not found")

    with open(json_config_path, 'r') as f:
        config = json.load(f)

    config = DotDict(config)

    init_dataset = AmazonECommerceDataset(config=config)
    init_dataset.setup()

    model_name = config.recommender_model.model_name
    usrstrategy_model_name = config.user_strategy.model_name # Suppose to be a variant of our custom choice model

    if config.experiment_mode == "p-validation":
        results_path = os.path.join(RESULTS_PATH_AMAZON, "p-validation", "amazon")
        fld_name = (
            f"recom={model_name}-usrstrategy={usrstrategy_model_name}"
            f"_tau={config.user_strategy.tau}"
            f"_candidate_set_size={config.user_strategy.candidate_set.size}"
            f"-p=probability-Kitems=KITEMS_coldStart={config.cold_start_months}"
        )

        ps = [0, 0.2, 0.5, 0.8, 1]
        ps_names = [f"{model_name}", "P=0.8", "P=0.5", "P=0.2", f"{usrstrategy_model_name}"]
        k_av_items = config.k_items

        n_reps = getattr(config, "n_reps", 3)
        base_seed = int(getattr(config, "seed", 2025))

        for p, p_name in zip(ps, ps_names):
            to_create = fld_name.replace("probability", str(p)).replace("KITEMS", str(k_av_items))
            p_root = os.path.join(results_path, to_create)
            os.makedirs(p_root, exist_ok=True)

            with open(os.path.join(p_root, "config_template.json"), 'w') as f:
                json.dump(config, f, indent=4, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else o)

            for rep in range(n_reps):
                rep_seed = make_rep_seed(base_seed=base_seed, p=float(p), rep_idx=rep)
                config_run = deepcopy(config)
                config_run.seed = rep_seed

                rep_dir = os.path.join(p_root, f"rep={rep:03d}")
                os.makedirs(rep_dir, exist_ok=True)

                df_dir = os.path.join(rep_dir, "dataframe")
                scores_dir  = os.path.join(rep_dir, "recom_scores")
                choice_dir  = os.path.join(rep_dir, "choice_model")
                os.makedirs(df_dir, exist_ok=True)
                os.makedirs(scores_dir, exist_ok=True)
                os.makedirs(choice_dir, exist_ok=True)

                with open(os.path.join(rep_dir, "config.json"), 'w') as f:
                    json.dump(config_run, f, indent=4, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else o)

                # Load and process the dataset from the original file
                # Produce also implicit feedback dataset format
                init_dataset = AmazonECommerceDataset(config=config_run)
                init_dataset.setup()

                feedback_loop_tool = AmazonECommerceFeedbackLoop(config=config_run, initialization_dataset=init_dataset)

                # Set up the system, save usefull data and pre-compute  interactions distributions
                feedback_loop_tool.init_experiment()

                # Define training-validation-test set for the recom. models
                _ = feedback_loop_tool.init_recbole_dataset()
                # Traing and set up RecBole system
                feedback_loop_tool.init_recbole_model()

                # Tune parameters of the selected model
                # NOTE that it does not set up the found parameters. They are to be defined in the json config
                # A file will be output with the results
                _ = feedback_loop_tool.tuning_hyperparameters()

                exit()

                if usrstrategy_model_name == "Custom choice model":
                    feedback_loop_tool.init_choice_model()
                    feedback_loop_tool.user_choice_model.results_path = choice_dir
                else:
                    raise Exception("\n Choice model not implemented yet or not recognized \n")
                
                model_metrics = feedback_loop_tool.run_feedback_loop(
                    p=p, 
                    results_path=os.path.join(rep_dir, "dataframe"),
                    results_scores_path=os.path.join(rep_dir, "recom_scores"),
                    k_horizon=k_av_items
                )

    else:
        raise Exception(f"\n Experiment mode not implemented yet or not recognized -> {config.experiment_mode} \n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--json_config",
        type=str,
        required=True,
        default="default_amazon_ecommerce.json",
        help="[str] Set for specific json config.",
    )

    args = parser.parse_args()

    _ = main(args)