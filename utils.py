import pandas as pd
import os 
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from recbole.trainer import Trainer

class DotDict(dict):
    """A dictionary supporting attribute-style (dot) access, including nested dicts."""

    def __getattr__(self, attr):
        if attr in self:
            value = self[attr]
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    @staticmethod
    def from_dict(obj):
        """Recursively convert dicts (and nested dicts/lists) to DotDicts."""
        if isinstance(obj, dict):
            return DotDict({k: DotDict.from_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [DotDict.from_dict(v) for v in obj]
        else:
            return obj

def make_rep_seed(base_seed: int, p: float, rep_idx: int) -> int:
    """Derive a stable seed from (base_seed, p, rep_idx)."""
    p_key = int(round(float(p) * 1_000_000))
    ss = np.random.SeedSequence(base_seed, spawn_key=[p_key, rep_idx])
    # produce one 32-bit int
    return int(ss.generate_state(1, dtype=np.uint32)[0])

def _setup_repro(master_seed: int):
    os.environ["PYTHONHASHSEED"] = str(master_seed)
    try:
        import torch
        torch.manual_seed(master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(master_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # modern NumPy RNG
    ss = np.random.SeedSequence(master_seed)
    rng = np.random.default_rng(ss)
    # seed Python's random from same seed space (optional, but consistent)
    import random as pyrandom
    child = ss.spawn(1)[0]
    pyrandom.seed(int(child.generate_state(1, dtype=np.uint32)[0]))
    return rng

def get_consistent_users(df, date_col="date", user_col="user_id", min_months_per_year=12):
        """
        Find users who have at least min_months_per_year interactions in every year from 2018 to 2022.
        """
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract year and month from date
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
    
        # Get the range of years in the dataset
        min_year = df['year'].min()
        max_year = df['year'].max()
        years_range = list(range(min_year, max_year + 1))
        
        # For each user, count distinct months in each year
        user_activity = df.groupby([user_col, 'year'])['month'].nunique().reset_index()
        
        # Identify users who have at least min_months_per_year distinct months in every year
        active_users = []
        
        for user in df[user_col].unique():
            user_data = user_activity[user_activity[user_col] == user]
            
            # Check if user has data for all years
            if set(user_data['year'].values) == set(years_range):
                # Check if user meets minimum months criteria for all years
                if all(user_data['month'] >= min_months_per_year):
                    active_users.append(user)
        
        return active_users