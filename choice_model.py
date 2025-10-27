import pandas as pd
import uuid
import os
import shutil
import numpy as np
from scipy import stats
import time 
import random
from collections import defaultdict
import matplotlib.pyplot as plt 
import seaborn as sns
import traceback

from utils import DotDict


class ChoiceModel():
    def __init__(self, interaction_df: pd.DataFrame, config: DotDict,  user_col: str = "user_id", item_col: str = "item_id", timestamp_col: str = "date", **kwargs):
        self.config = config
        
        self.df = interaction_df

        self.user_col = user_col
        self.item_col = item_col
        self.time_col = timestamp_col

        self.tau = config.user_strategy.tau

        self.cache_root = "./cache/choice_model"

        candidate_size = self.config.user_strategy.candidate_set.size
        self.tau_users_cache_path = f"exploration_score_users_{self.config.dataset}_{self.config.time_window}_INIT_{self.config.cold_start_months}MONTHS.csv"
        self.candidate_set_users_path = f"candidate_set_items_{self.config.dataset}_{self.config.time_window}_INIT_{self.config.cold_start_months}MONTHS_size={candidate_size}.csv"
        self.utilities_users_path = f"utilities_users_{self.config.dataset}_{self.config.time_window}_INIT_{self.config.cold_start_months}MONTHS.csv"

    def compute_gini(self, array):
        array = np.array(array)
        if np.amin(array) < 0:
            array -= np.amin(array)
        array = array + 1e-8  # avoid zeros
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def analyze_cumulative_curve(self, cumulative_curve, months):
        """
        Return exploration score (0.001 to 100).
        
        Uses multiple methods:
        1. Recent slope (how fast discovering new items recently)
        2. Overall trend (linear regression slope)
        3. Acceleration (is discovery rate increasing or decreasing)
        """
        
        y = np.array(cumulative_curve)
        x = np.arange(len(y)) # Months

        recent_window = min(3, len(y) - 1)
        if recent_window > 0:
            recent_slope = (y[-1] - y[-recent_window-1]) / recent_window
        else:
            recent_slope = 0

        if len(y) >= 2:
            overall_slope, _, r_value, _, _ = stats.linregress(x, y)
            trend_strength = abs(r_value)  # How linear is the trend
        else:
            overall_slope = 0
            trend_strength = 0

        # Method 3: Acceleration (second derivative)
        if len(y) >= 3:
            # Calculate first differences (monthly new items)
            first_diff = np.diff(y)
            # Calculate second differences (acceleration)
            second_diff = np.diff(first_diff)
            acceleration = np.mean(second_diff[-2:]) if len(second_diff) >= 2 else np.mean(second_diff)
        else:
            acceleration = 0

        
        components = {
            'recent_slope': recent_slope * 0.5,     
            'overall_slope': overall_slope * 0.25, 
            'acceleration': acceleration * 0.25, 
        }

        raw_score = sum(components.values())
        normalized_score = 2 / (1 + np.exp(-raw_score)) - 1  # Maps to [-1,1]
        normalized_score = max(0, normalized_score)

        # exploration_score = 0.00001 + normalized_score * 99.999
        # exploration_score = max(0.1, min(100.0, exploration_score))

        exploration_score = 1 + normalized_score * 999
        exploration_score = min(1000.0, max(1.0, exploration_score))

        return exploration_score

    def calculate_exploration_from_cumulative_curve(self, df):
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df['year_month'] = df[self.time_col].dt.to_period('M')

        user_scores = []
        for user_id in df[self.user_col].unique():
            user_data = df[df[self.user_col] == user_id].sort_values(self.time_col)
            
            # Build cumulative new items curve
            seen_items = set()
            cumulative_curve = []
            months = []

            for month in sorted(user_data['year_month'].unique()):
                month_data = user_data[user_data['year_month'] == month]
                month_items = set(month_data[self.item_col].unique())
                
                # Add new items discovered this month
                new_items_this_month = month_items - seen_items
                seen_items.update(new_items_this_month)
                
                cumulative_curve.append(len(seen_items))
                months.append(month)
            
            if len(cumulative_curve) < 3:
                # Not enough data
                exploration_score = 0.00001
            else:
                # Calculate exploration score using multiple methods
                exploration_score = self.analyze_cumulative_curve(cumulative_curve, months)
            
            user_scores.append({
                'user_id': user_id,
                'exploration_score': exploration_score,
                'cumulative_curve': cumulative_curve,
                'months': months
            })
            
        return pd.DataFrame(user_scores)

    def compute_utilities(self, df, candidate_sets=None):
        all_items = df["item_id"].unique()
        n_users = df['user_id'].nunique()

        # Precompute item interaction counts across all users
        global_item_counts = df.groupby("item_id").size()
        c_i_dict = (global_item_counts / n_users).to_dict()  # average interactions per user

        # Group once by user_id to avoid filtering each time
        grouped = df.groupby('user_id')

        results = {}

        for user_id, user_df in grouped:
            candidate_set = candidate_sets[user_id]
            user_df = user_df[user_df[self.item_col].isin(candidate_set)]
            user_item_counts = user_df.groupby(self.item_col).size()
            c_u = user_item_counts.mean()
            sigma_u = user_item_counts.std(ddof=0)

            # Compute utility for all items
            eta = np.random.normal(0, 0.01, size=len(all_items))
            c_i_values = np.array([c_i_dict.get(item, 0) for item in all_items])
            omega_ui = c_u + (sigma_u * c_i_values) + eta

            # Normalize utilities to [0, 1]
            min_u, max_u = omega_ui.min(), omega_ui.max()
            norm_utils = (omega_ui - min_u) / (max_u - min_u + 1e-8)

            results[user_id] = pd.Series(norm_utils, index=all_items)

        results = pd.DataFrame(results)

        return results

    def compute_utilities_v2(self, df, candidate_sets=None):
        lambda_rare_items = 0.1
        all_items = df["item_id"].unique()
        n_users = df['user_id'].nunique()

        # Precompute normalized global item popularity
        global_item_counts = df.groupby("item_id").size()
        max_c_i = global_item_counts.max()
        c_i_norm = (global_item_counts / n_users) / max_c_i
        c_i_dict = c_i_norm.to_dict()

        grouped = df.groupby('user_id')
        results = {}

        for user_id, user_df in grouped:
            candidate_set = candidate_sets[user_id]
            user_df = user_df[user_df[self.item_col].isin(candidate_set)]
            user_item_counts = user_df.groupby(self.item_col).size()
            c_u = user_item_counts.mean()

            gini_u = self.compute_gini(user_item_counts.values)
            gini_mod = 1 - gini_u

            eta = np.random.normal(0, 0.01, size=len(all_items))
            c_i_values = np.array([c_i_dict.get(item, 0) for item in all_items])

            omega_ui = (
                c_u +
                gini_mod * np.log1p(c_i_values) +
                lambda_rare_items * 1 / (1 + c_i_values) +
                eta
            )
            exp_omega = np.exp(omega_ui - omega_ui.max())
            norm_utils = exp_omega / (exp_omega.sum() + 1e-8)
            results[user_id] = pd.Series(norm_utils, index=all_items)

        results = pd.DataFrame(results)

        return results
    
    def candidate_set_items(self, df):
        """
            For each user, define the candidate set items for the given dataframe
        """
        candidate_set_settings = self.config.user_strategy.candidate_set
        results = {}
        try:
            all_items = df['item_id'].unique().tolist()
        except Exception as e:
            print("PORCODDIO")
            print(traceback.format_exc())
            raise Exception(f"{e}")

        if candidate_set_settings.size is None:
            K = len(all_items)
        else:
            K = candidate_set_settings.size
            
        p_global = candidate_set_settings.p_global
        p_user = candidate_set_settings.p_user
        p_random = candidate_set_settings.p_random

        if not (p_global+p_user+p_random):
            raise Exception(f"Sum of ps must be one")

        # TOP X GLOBAL FAMOUS ITEMS
        global_popular_items = df[self.item_col].value_counts().index.tolist()

        # Build a dictionary of user -> most frequent items (individual popular)
        user_item_counts = (
            df.groupby([self.user_col, self.item_col])
            .size()
            .reset_index(name='count')
        )
        
        individual_popular_dict = defaultdict(list)
        for user_id, group in user_item_counts.groupby('user_id'):
            sorted_items = group.sort_values('count', ascending=False)['item_id'].tolist()
            individual_popular_dict[user_id] = sorted_items

        user_candidates = dict()

        for user_id in df[self.user_col].unique():
            if K == len(all_items):
                user_candidates[user_id] = all_items
                continue
            n_global = int(K * p_global)
            n_individual = int(K * p_user)
            n_random = K - n_global - n_individual

            global_items = global_popular_items[:n_global]
            individual_items = individual_popular_dict.get(user_id, [])[:n_individual]

            missing_individual = n_individual - len(individual_items)
            selected_items = set(global_items) | set(individual_items)
            remaining_items = list(set(all_items) - selected_items)
            np.random.shuffle(remaining_items)

            n_random_total = n_random + max(0, missing_individual)
            random_items = remaining_items[:n_random_total]

            candidate_set = global_items + individual_items + random_items
            np.random.shuffle(candidate_set[:K])

            if len(candidate_set) != K:
                print(f"Candidate set items not equal as requested")

            user_candidates[user_id] = candidate_set            

        df_results = pd.DataFrame(user_candidates)
        return df_results
    
    def setup(self):
        """
            Compute evrytihing needed by the model later on to make prediction
            It uses cache within a tmp folder
        """
        df = self.df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df['year_month'] = df[self.time_col].dt.to_period('M')
        months = df['year_month'].unique()
        if len(months) < 3:
            raise Exception(f"The initialization dataset for the choice model must contains data for at least 3 months")

        if os.path.exists(os.path.join(self.cache_root, self.tau_users_cache_path)):
            results = pd.read_csv(os.path.join(self.cache_root, self.tau_users_cache_path))
        else:
            results = self.calculate_exploration_from_cumulative_curve(df=df)
            results.to_csv(os.path.join(self.cache_root, self.tau_users_cache_path), index=False)

        if os.path.exists(os.path.join(self.cache_root, self.candidate_set_users_path)) and self.config.user_strategy.candidate_set.size is not None:
            candidate_sets = pd.read_csv(os.path.join(self.cache_root, self.candidate_set_users_path))
        else:
            candidate_sets = self.candidate_set_items(df=df)
            candidate_sets.to_csv(os.path.join(self.cache_root, self.candidate_set_users_path), index=False)
        
        if os.path.exists(os.path.join(self.cache_root, self.utilities_users_path)):
            utilities_users = pd.read_csv(os.path.join(self.cache_root, self.utilities_users_path), index_col=0)
        else:
            utilities_users = self.compute_utilities_v2(df=df, candidate_sets=candidate_sets)
            utilities_users.to_csv(os.path.join(self.cache_root, self.utilities_users_path))
        
        self.exploration_rate_users = results
        self.candidate_sets = candidate_sets
        self.utilities_users = utilities_users

        # PLOTS
        # Utilities over population
        # utilities_users = self.compute_utilities_v2(df=df, candidate_sets=candidate_sets)
        # df = self.utilities_users

        # figsize = (20, 15)
        # dpi = 200

        # np.random.seed(5)
        # random_users = np.random.choice(df.columns, size=10, replace=False)

        # for user in random_users:
        #     user_utilities = df[user].dropna()
            
        #     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        #     sns.histplot(user_utilities, bins=30, kde=True, ax=ax)
            
        #     ax.set_xlabel(r'Utility', fontsize=45, color="#000000")
        #     ax.set_ylabel(r'Number of items', fontsize=45, color="#000000")
        #     ax.grid(True, linestyle='--', alpha=0.7)
        #     ax.tick_params(axis='both', which='major', labelsize=45, colors="#000000")
        #     ax.set_facecolor('#FFFFFF')
            
        #     # Remove "0" from y-axis ticks
        #     yticks = ax.get_yticks()
        #     yticks = [yt for yt in yticks if yt != 0]
        #     ax.set_yticks(yticks)
            
        #     # Remove x-axis ticks
        #     ax.set_xticks([])
            
        #     fig.savefig(f"utilities_distr_user={user}_epoch=0.png")
        #     plt.close(fig)

        # exit()

    def predict_for_a_user(self, user_id, tau=None):
        if tau is None:
            # tau = self.exploration_rate_users[self.exploration_rate_users[self.user_col]==user_id].exploration_score.values[0]
            tau = round(self.exploration_rate_users.exploration_score.mean(), 3)
        # fig, ax = plt.subplots(figsize=(10, 8))
        try:
            user_id = int(user_id)
            candidate_set = self.candidate_sets[user_id]
        except KeyError as e:
            user_id = str(user_id)
            candidate_set = self.candidate_sets[user_id]
        except ValueError as e:
            user_id = str(user_id)
            candidate_set = self.candidate_sets[user_id]
        try:
            utilities = self.utilities_users[user_id]
        except KeyError as e:
            utilities = self.utilities_users[str(user_id)]
        
        utilities = utilities.loc[candidate_set.values]

        logits = utilities / tau
        exp_util = np.exp(logits - np.max(logits))
        probs = exp_util / exp_util.sum()
        
        # PLOT score ranking variation respect to tau
        # candidate_set_size = self.config.user_strategy["candidate_set"]["size"]
        # sorted_probs = probs.sort_values(ascending=True)

        # sorted_probs.plot(kind='bar', ax=ax, color='skyblue', alpha=0.6)
        # # sorted_probs.plot(kind='kde', ax=ax, color='skyblue', linewidth=3)
        # # sorted_probs.plot(ax=ax, color='skyblue')

        # ax.set_ylabel("Choice Probability", fontsize=25, color="#000000")
        # ax.set_xlabel("Item ID", fontsize=25, color="#000000")
        # # ax.set_xticklabels(sorted_probs.index, rotation=45)
        # ax.set_xticks([])
        # ax.set_xticklabels([])

        # ax.grid(True, linestyle='-')
        # # legend = ax.legend(fontsize=20, facecolor='#FFFFFF')
        # # legend.get_frame().set_edgecolor('#046865')
        # for spine in ax.spines.values():
        #     spine.set_color('#000000')
        # ax.tick_params(axis='both', labelsize=18, colors="#000000")

        # fig.savefig(f"User={user_id}_tau={tau}.png")
        # plt.close()

        # exit()
        item_ids = probs.index.to_list()
        probs_arr = probs.values
        return item_ids, probs_arr
    
    def update(self, new_interactions, epoch):
        new_df = pd.DataFrame(new_interactions)
        new_df[self.time_col] = pd.to_datetime(new_df[self.time_col])
        new_df['year_month'] = new_df[self.time_col].dt.to_period('M')
        
        new_df = new_df[["user_id", "item_id", "date", "timestamp"]]
        df = pd.concat([self.df, new_df], ignore_index=True)
        df = df.sort_values(self.time_col).reset_index(drop=True)

        self.df = df # The main is updated with new interactions

        if self.results_path is None:
            raise Exception(f"The choice model has no access to results folder")

        # Candidate set items
        candidate_sets = self.candidate_set_items(df=self.df)
        target_folder = os.path.join(self.results_path, "candidate_set_items")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        candidate_sets.to_csv(os.path.join(target_folder, f"epoch_{epoch}.csv"), index=False)

        # Utilities
        utilities_users = self.compute_utilities_v2(df=self.df, candidate_sets=candidate_sets)
        target_folder = os.path.join(self.results_path, "utilities_users")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        utilities_users.to_csv(os.path.join(target_folder, f"epoch_{epoch}.csv"), index=False)

        # self.exploration_rate_users = results
        self.candidate_sets = candidate_sets
        self.utilities_users = utilities_users