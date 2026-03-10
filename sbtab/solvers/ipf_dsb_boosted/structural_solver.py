import numpy as np
import pandas as pd
import networkx as nx
import torch
import math
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore
from catboost import CatBoostRegressor, Pool

from sbtab.bridge.timegrid import TimeGrid
from sbtab.models.field.neural.time_embedding import FourierTime
from sbtab.evaluation.metrics.statistical import sliced_wasserstein

class StructuralBoostedDSBSolver:
    def __init__(
        self,
        num_steps: int = 30,
        ipf_iters: int = 5,
        alpha_ou: float = 1.0,
        n_bins: int = 5,
        cat_params: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.num_steps = num_steps
        self.ipf_iters = ipf_iters
        self.alpha_ou = alpha_ou
        self.n_bins = n_bins
        self.seed = seed
        self.cat_params = cat_params or {"iterations": 500, "depth": 6, "learning_rate": 0.05, "verbose": 0}
        
        self.timegrid = TimeGrid(num_steps=num_steps)
        self.time_embedder = FourierTime(features=16)
        
        self.dag = None
        self.generation_order = []
        self.models = {} 
        self.feature_cols = []

    def _prepare_conditional_features(self, x_i: np.ndarray, t: np.ndarray, parents_data: np.ndarray) -> np.ndarray:
        """Input: [sign_value, embedding_time, parent values]"""
        t_tensor = torch.from_numpy(t.astype(np.float32))
        with torch.no_grad():
            t_emb = self.time_embedder(t_tensor).numpy()
        return np.concatenate([x_i.reshape(-1, 1), t_emb, parents_data], axis=1)

    def _train_conditional_bridge(self, col: str, df: pd.DataFrame):
        parents = list(self.dag.predecessors(col))
        x_data = df[col].values.astype(np.float32)
        p_data_clean = df[parents].values.astype(np.float32) if parents else np.empty((len(df), 0))
        n = len(x_data)

        f_net = CatBoostRegressor(**self.cat_params)
        b_net = CatBoostRegressor(**self.cat_params)

        gammas = self.timegrid.gammas().numpy()
        times = self.timegrid.times().numpy()

        # 1. OU Pretrain
        k_rand = np.random.randint(0, self.num_steps, size=n)
        t_k = times[k_rand]
        dt = gammas[k_rand]
        target_ou = x_data + dt * (-self.alpha_ou * x_data)
        f_net.fit(self._prepare_conditional_features(x_data, t_k, p_data_clean), target_ou)

        for it in range(self.ipf_iters):
            # Adding micro-noise to the parents to escape from Exposure Bias when sampling
            p_data = p_data_clean + np.random.randn(*p_data_clean.shape).astype(np.float32) * 0.01 if parents else p_data_clean

            curr_x = x_data.copy()
            xs_train, ts_train, ys_target = [], [], []

            for k in range(self.num_steps):
                t_val = np.full((n,), times[k], dtype=np.float32)
                # The time for B should be t_{k+1}, since B starts from the future
                t_val_next = np.full((n,), times[min(k+1, self.num_steps-1)], dtype=np.float32)
                
                f_feats = self._prepare_conditional_features(curr_x, t_val, p_data)
                mean_next = f_net.predict(f_feats)
                noise = np.random.randn(n) * np.sqrt(2.0 * gammas[k])
                next_x = mean_next + noise

                f_feats_next = self._prepare_conditional_features(next_x, t_val, p_data)
                target_b = next_x + (mean_next - f_net.predict(f_feats_next))

                xs_train.append(next_x)
                ts_train.append(t_val_next)
                ys_target.append(target_b)
                curr_x = next_x

            b_net.fit(self._prepare_conditional_features(np.hstack(xs_train), np.hstack(ts_train), np.tile(p_data, (self.num_steps, 1))), 
                      np.hstack(ys_target))


            # --- step 2: Track the movement from  (Backward simulation) ---
            curr_x = np.random.randn(n).astype(np.float32) 
            xs_train, ts_train, ys_target = [], [],[]

            for k in range(self.num_steps - 1, -1, -1):
                t_val = np.full((n,), times[k], dtype=np.float32)
                t_val_prev = np.full((n,), times[max(k-1, 0)], dtype=np.float32)
                
                b_feats = self._prepare_conditional_features(curr_x, t_val, p_data)
                mean_prev = b_net.predict(b_feats)
                noise = np.random.randn(n) * np.sqrt(2.0 * gammas[k])
                prev_x = mean_prev + noise

                b_feats_prev = self._prepare_conditional_features(prev_x, t_val, p_data)
                target_f = prev_x + (mean_prev - b_net.predict(b_feats_prev))

                xs_train.append(prev_x)
                ts_train.append(t_val_prev)
                ys_target.append(target_f)
                curr_x = prev_x

            f_net.fit(self._prepare_conditional_features(np.hstack(xs_train), np.hstack(ts_train), np.tile(p_data, (self.num_steps, 1))), 
                      np.hstack(ys_target))
            
            print(f"  Column {col} | IPF {it+1} complete")

        self.models[col] = {'B': b_net}

    def fit(self, df: pd.DataFrame):
        self.feature_cols = list(df.columns)
        self._learn_structure(df) 
        
        for col in self.generation_order:
            self._train_conditional_bridge(col, df)
        return self

    def _learn_structure(self, df: pd.DataFrame):
        """Hill Climbing Discovery"""
        print("Learning causal DAG structure...")
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
        df_binned = pd.DataFrame(discretizer.fit_transform(df), columns=df.columns).astype(int)
        hc = HillClimbSearch(df_binned)
        best_model = hc.estimate(scoring_method=BicScore(df_binned))
        
        G = nx.DiGraph(best_model.edges())
        G.add_nodes_from(df.columns)
        
        # Cycle protection just in case
        while not nx.is_directed_acyclic_graph(G):
            cycle = nx.find_cycle(G)
            G.remove_edge(cycle[-1][0], cycle[-1][1])
            
        self.dag = G
        self.generation_order = list(nx.topological_sort(G))
        print(f"Generation Order: {' -> '.join(self.generation_order)}")


    def sample(self, n: int) -> pd.DataFrame:
        """Sequential generation following the topological order."""
        gen_df = pd.DataFrame(index=range(n))
        gammas = self.timegrid.gammas().numpy()
        times = self.timegrid.times().numpy()
        
        for col in self.generation_order:
            parents = list(self.dag.predecessors(col))
            p_data = gen_df[parents].values.astype(np.float32) if parents else np.empty((n, 0))
            
            x_i = np.random.randn(n).astype(np.float32)
            
            for k in range(self.num_steps - 1, -1, -1):
                t_k = np.full((n,), times[k], dtype=np.float32)
                feats = self._prepare_conditional_features(x_i, t_k, p_data)
                x_i = self.models[col]['B'].predict(feats) + np.random.randn(n) * np.sqrt(2.0 * gammas[k])
                
            gen_df[col] = x_i
            
        return gen_df[self.feature_cols]