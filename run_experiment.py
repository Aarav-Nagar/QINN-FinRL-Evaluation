#!/usr/bin/env python
"""QINN-vs-ANN signals inside the official FinRL stock-trading environment.

The experiment is intentionally an ablation:

1. Fit parameter-matched ANN and matrix-product-state (MPS) regressors to the
   same 13 market features and next-day return target.
2. Add either prediction as one extra signal to the standard FinRL state.
3. Train the same PPO agent with the same costs, dates, assets, and seeds.
4. Evaluate chronologically on the untouched 2019-2023 trading period.

This is a classical tensor-network experiment. It does not run on quantum
hardware and does not claim quantum advantage.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FINRL_REPOSITORY = "https://github.com/AI4Finance-Foundation/FinRL.git"
FINRL_COMMIT = "2334a5fe6d30629157f13c3b0319e1637e15e123"
DATASET_REPOSITORY = "https://huggingface.co/datasets/benstaf/nasdaq_2013_2023"
DATA_FILES = {
    "train_data_2013_2018.csv": {
        "sha256": "92eb993137595e9c461091e9f3569295d0050f5536c75b707468ba6d2197657b",
        "url": f"{DATASET_REPOSITORY}/resolve/main/train_data_2013_2018.csv?download=true",
    },
    "trade_data_2019_2023.csv": {
        "sha256": "01587b66236b5563df8f871f0110bbf752f1c593427a346192c20e271efffd3b",
        "url": f"{DATASET_REPOSITORY}/resolve/main/trade_data_2019_2023.csv?download=true",
    },
}

TICKERS = [
    "AAPL",
    "AMD",
    "AMGN",
    "AMZN",
    "COST",
    "FANG",
    "GILD",
    "HON",
    "INTC",
    "MSFT",
    "NFLX",
    "NVDA",
    "PEP",
    "SBUX",
    "XEL",
]

FINRL_FEATURES = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "vix",
    "turbulence",
]

REPRESENTATION_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "price_to_sma_30",
    "price_to_sma_60",
    "macd_scaled",
    "bollinger_position",
    "bollinger_width",
    "rsi_scaled",
    "cci_scaled",
    "dx_scaled",
    "volume_change_5d",
    "vix_scaled",
]


@dataclass(frozen=True)
class ExperimentConfig:
    representation_seed: int = 2026
    ppo_seeds: tuple[int, ...] = (0, 1, 2)
    ppo_timesteps: int = 5_000
    ppo_update_epochs: int = 3
    transaction_cost: float = 0.001
    initial_amount: int = 1_000_000
    hmax: int = 100
    reward_scaling: float = 1e-4
    mps_bond_dimension: int = 4
    encoder_device: str = "auto"
    encoder_epochs: int = 60
    encoder_patience: int = 10
    encoder_batch_size: int = 512
    encoder_learning_rate: float = 2e-3
    representation_train_end: str = "2017-12-29"
    representation_validation_start: str = "2018-01-01"
    train_period: str = "2013-01-02 to 2018-12-28"
    test_period: str = "2019-01-02 to 2023-12-28"


def validate_config(config: ExperimentConfig) -> None:
    """Reject ambiguous or invalid experiment configurations before training."""
    if not config.ppo_seeds:
        raise ValueError("At least one PPO seed is required")
    if len(set(config.ppo_seeds)) != len(config.ppo_seeds):
        raise ValueError("PPO seeds must be unique")
    if any(seed < 0 for seed in config.ppo_seeds):
        raise ValueError("PPO seeds must be non-negative")
    if config.ppo_timesteps <= 0:
        raise ValueError("PPO timesteps must be positive")
    if config.mps_bond_dimension <= 0:
        raise ValueError("MPS bond dimension must be positive")
    if config.encoder_epochs <= 0:
        raise ValueError("Encoder epochs must be positive")
    if config.encoder_patience <= 0:
        raise ValueError("Encoder patience must be positive")
    if config.encoder_batch_size <= 0:
        raise ValueError("Encoder batch size must be positive")
    if config.encoder_learning_rate <= 0:
        raise ValueError("Encoder learning rate must be positive")
    if not 0 <= config.transaction_cost < 1:
        raise ValueError("Transaction cost must be in [0, 1)")
    if config.encoder_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Encoder device must be one of: auto, cpu, cuda")


def resolve_encoder_device(requested: str) -> torch.device:
    """Resolve an encoder device without silently ignoring explicit CUDA use."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA encoder requested, but this PyTorch build cannot access CUDA"
        )
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_sha256:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {destination.name} ...")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: {actual} != {expected_sha256}"
        )
    temporary.replace(destination)


def ensure_finrl(finrl_dir: Path) -> Path:
    env_file = finrl_dir / "finrl" / "meta" / "env_stock_trading" / "env_stocktrading.py"
    if not env_file.exists():
        finrl_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", FINRL_REPOSITORY, str(finrl_dir)],
            check=True,
        )
        subprocess.run(["git", "-C", str(finrl_dir), "checkout", FINRL_COMMIT], check=True)
    current = subprocess.check_output(
        ["git", "-C", str(finrl_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != FINRL_COMMIT:
        raise RuntimeError(f"Expected FinRL commit {FINRL_COMMIT}, found {current}")
    return env_file


def load_stock_trading_env(env_file: Path):
    spec = importlib.util.spec_from_file_location("pinned_finrl_stock_env", env_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FinRL environment from {env_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StockTradingEnv


def read_market_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    for filename, metadata in DATA_FILES.items():
        download_file(metadata["url"], data_dir / filename, metadata["sha256"])
    train = pd.read_csv(data_dir / "train_data_2013_2018.csv")
    trade = pd.read_csv(data_dir / "trade_data_2019_2023.csv")
    for frame in (train, trade):
        frame.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
        frame["date"] = pd.to_datetime(frame["date"])
    missing = sorted(set(TICKERS) - set(train["tic"].unique()))
    if missing:
        raise ValueError(f"Requested tickers are missing from the dataset: {missing}")
    train = train[train["tic"].isin(TICKERS)].copy()
    trade = trade[trade["tic"].isin(TICKERS)].copy()
    return train, trade


def engineer_representation_features(
    train: pd.DataFrame, trade: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.assign(_period="train"), trade.assign(_period="trade")],
        ignore_index=True,
    ).sort_values(["tic", "date"])
    pieces: list[pd.DataFrame] = []
    for _, group in combined.groupby("tic", sort=False):
        group = group.sort_values("date").copy()
        close = group["close"].astype(float)
        group["return_1d"] = close.pct_change()
        group["return_5d"] = close.pct_change(5)
        group["return_20d"] = close.pct_change(20)
        group["price_to_sma_30"] = close / group["close_30_sma"] - 1
        group["price_to_sma_60"] = close / group["close_60_sma"] - 1
        group["macd_scaled"] = group["macd"] / close.replace(0, np.nan)
        band_width = (group["boll_ub"] - group["boll_lb"]).replace(0, np.nan)
        group["bollinger_position"] = (close - group["boll_lb"]) / band_width
        group["bollinger_width"] = band_width / close.replace(0, np.nan)
        group["rsi_scaled"] = (group["rsi_30"] - 50.0) / 50.0
        group["cci_scaled"] = group["cci_30"] / 200.0
        group["dx_scaled"] = group["dx_30"] / 100.0
        group["volume_change_5d"] = np.log1p(group["volume"]).diff(5)
        group["vix_scaled"] = group["vix"] / 100.0
        group["target_return_1d"] = close.pct_change().shift(-1)
        pieces.append(group)
    combined = pd.concat(pieces, ignore_index=True)
    combined[REPRESENTATION_FEATURES] = combined[REPRESENTATION_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    combined[REPRESENTATION_FEATURES] = combined.groupby("tic")[
        REPRESENTATION_FEATURES
    ].transform(lambda values: values.ffill().fillna(0.0))
    train_out = combined[combined["_period"] == "train"].drop(columns="_period")
    trade_out = combined[combined["_period"] == "trade"].drop(columns="_period")
    return train_out, trade_out


class ANNRegressor(nn.Module):
    """369-parameter classical baseline for 13 input features."""

    def __init__(self, feature_count: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class MPSRegressor(nn.Module):
    """Classical matrix-product-state regressor with a quantum-style feature map."""

    def __init__(self, feature_count: int, bond_dimension: int):
        super().__init__()
        ranks = [1] + [bond_dimension] * (feature_count - 1) + [1]
        cores = []
        generator = torch.Generator().manual_seed(2026)
        for left_rank, right_rank in zip(ranks[:-1], ranks[1:]):
            scale = 1.0 / math.sqrt(2 * left_rank)
            core = torch.randn(
                left_rank, 2, right_rank, generator=generator
            ) * scale
            cores.append(nn.Parameter(core))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def local_feature_map(inputs: torch.Tensor) -> torch.Tensor:
        clipped = inputs.clamp(-3.0, 3.0) / 3.0
        angles = (math.pi / 4.0) * (clipped + 1.0)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.local_feature_map(inputs)
        state = torch.ones(
            (inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device
        )
        for site, core in enumerate(self.cores):
            state = torch.einsum("br,ris,bi->bs", state, core, local[:, site, :])
        return state.squeeze(-1) + self.bias


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    if np.std(target) > 0 and np.std(prediction) > 0:
        ic = float(np.corrcoef(target, prediction)[0, 1])
    else:
        ic = float("nan")
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "directional_accuracy": float(np.mean((target > 0) == (prediction > 0))),
        "information_coefficient": ic,
    }


def fit_encoder(
    name: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    started = time.perf_counter()
    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader_generator = torch.Generator().manual_seed(config.representation_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.encoder_batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    model = model.to(device)
    validation_x = torch.as_tensor(
        x_validation, dtype=torch.float32, device=device
    )
    validation_y = torch.as_tensor(
        y_validation, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.encoder_learning_rate, weight_decay=1e-5
    )
    loss_function = nn.HuberLoss(delta=1.0)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.encoder_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_samples += len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x)
            validation_loss = float(
                loss_function(validation_prediction, validation_y).item()
            )
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "train_huber": total_loss / total_samples,
                "validation_huber": validation_loss,
            }
        )
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.encoder_patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    history_frame = pd.DataFrame(history)
    history_frame.attrs["elapsed_seconds"] = time.perf_counter() - started
    return model, history_frame


def train_signal_models(
    train: pd.DataFrame, trade: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    representation_train_end = pd.Timestamp(config.representation_train_end)
    validation_start = pd.Timestamp(config.representation_validation_start)
    training_mask = (
        (train["date"] <= representation_train_end)
        & train["target_return_1d"].notna()
    )
    validation_mask = (
        (train["date"] >= validation_start) & train["target_return_1d"].notna()
    )
    means = train.loc[training_mask, REPRESENTATION_FEATURES].mean()
    scales = train.loc[training_mask, REPRESENTATION_FEATURES].std().replace(0, 1.0)
    target_mean = float(train.loc[training_mask, "target_return_1d"].mean())
    target_scale = float(train.loc[training_mask, "target_return_1d"].std())
    if target_scale <= 0:
        raise ValueError("Target scale must be positive")

    def standardized(frame: pd.DataFrame) -> np.ndarray:
        values = (frame[REPRESENTATION_FEATURES] - means) / scales
        return values.clip(-8, 8).to_numpy(dtype=np.float32)

    x_train = standardized(train.loc[training_mask])
    y_train = (
        (train.loc[training_mask, "target_return_1d"] - target_mean) / target_scale
    ).clip(-8, 8).to_numpy(dtype=np.float32)
    x_validation = standardized(train.loc[validation_mask])
    y_validation = (
        (train.loc[validation_mask, "target_return_1d"] - target_mean) / target_scale
    ).clip(-8, 8).to_numpy(dtype=np.float32)
    device = resolve_encoder_device(config.encoder_device)

    set_global_seed(config.representation_seed)
    ann = ANNRegressor(len(REPRESENTATION_FEATURES))
    set_global_seed(config.representation_seed)
    mps = MPSRegressor(
        len(REPRESENTATION_FEATURES), config.mps_bond_dimension
    )
    if config.mps_bond_dimension == 4 and parameter_count(ann) != parameter_count(mps):
        raise AssertionError(
            f"Parameter mismatch: ANN={parameter_count(ann)}, MPS={parameter_count(mps)}"
        )
    ann, ann_history = fit_encoder(
        "ANN", ann, x_train, y_train, x_validation, y_validation, config, device
    )
    mps, mps_history = fit_encoder(
        "QINN-MPS",
        mps,
        x_train,
        y_train,
        x_validation,
        y_validation,
        config,
        device,
    )
    ann_elapsed = float(ann_history.attrs["elapsed_seconds"])
    mps_elapsed = float(mps_history.attrs["elapsed_seconds"])

    inference_started = time.perf_counter()
    def add_signals(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        inputs = torch.as_tensor(
            standardized(frame), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            result["ann_signal"] = ann(inputs).cpu().numpy().clip(-8, 8)
            result["qinn_mps_signal"] = mps(inputs).cpu().numpy().clip(-8, 8)
        return result

    train = add_signals(train)
    trade = add_signals(trade)
    inference_elapsed = time.perf_counter() - inference_started
    signal_rows = []
    for split, frame, mask in (
        ("validation_2018", train, validation_mask),
        ("test_2019_2023", trade, trade["target_return_1d"].notna()),
    ):
        target = (
            (frame.loc[mask, "target_return_1d"] - target_mean) / target_scale
        ).to_numpy()
        for model_name, column in (
            ("ANN", "ann_signal"),
            ("QINN-MPS", "qinn_mps_signal"),
        ):
            row = {
                "split": split,
                "model": model_name,
                "parameter_count": parameter_count(ann)
                if model_name == "ANN"
                else parameter_count(mps),
            }
            row.update(regression_metrics(target, frame.loc[mask, column].to_numpy()))
            signal_rows.append(row)
    metadata = pd.DataFrame(
        {
            "feature": REPRESENTATION_FEATURES,
            "training_mean": means.values,
            "training_std": scales.values,
        }
    )
    metadata.attrs["target_mean"] = target_mean
    metadata.attrs["target_std"] = target_scale
    runtime = {
        "device": str(device),
        "ann_fit_seconds": ann_elapsed,
        "mps_fit_seconds": mps_elapsed,
        "signal_inference_seconds": inference_elapsed,
        "total_signal_pipeline_seconds": time.perf_counter() - started,
    }
    return (
        train,
        trade,
        pd.DataFrame(signal_rows),
        pd.concat([ann_history, mps_history], ignore_index=True),
        runtime,
    )


def finrl_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["date", "tic"]).copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.index = pd.factorize(result["date"], sort=True)[0]
    return result


def make_environment_factory(
    stock_env_class,
    frame: pd.DataFrame,
    indicators: list[str],
    config: ExperimentConfig,
    seed: int,
    record_terminal: bool = False,
) -> Callable[[], object]:
    stock_dim = len(TICKERS)
    state_space = 1 + 2 * stock_dim + len(indicators) * stock_dim

    def factory():
        environment = stock_env_class(
            df=frame,
            stock_dim=stock_dim,
            hmax=config.hmax,
            initial_amount=config.initial_amount,
            num_stock_shares=[0] * stock_dim,
            buy_cost_pct=[config.transaction_cost] * stock_dim,
            sell_cost_pct=[config.transaction_cost] * stock_dim,
            reward_scaling=config.reward_scaling,
            state_space=state_space,
            action_space=stock_dim,
            tech_indicator_list=indicators,
            turbulence_threshold=None,
            make_plots=False,
            print_verbosity=100_000,
        )
        environment.reset(seed=seed)
        return EpisodeRecorder(environment) if record_terminal else environment

    return factory


class EpisodeRecorder(gym.Wrapper):
    """Preserve terminal FinRL memories before DummyVecEnv automatically resets."""

    def __init__(self, env):
        super().__init__(env)
        self.last_episode: dict[str, object] | None = None
        self.gross_traded_notionals: list[float] = []
        self.daily_turnovers: list[float] = []

    def step(self, action):
        prices = np.asarray(
            self.env.state[1 : 1 + self.env.stock_dim], dtype=float
        )
        holdings = np.asarray(
            self.env.state[
                1 + self.env.stock_dim : 1 + 2 * self.env.stock_dim
            ],
            dtype=float,
        )
        beginning_asset = float(self.env.state[0] + np.dot(prices, holdings))
        previous_action_count = len(self.env.actions_memory)
        observation, reward, terminated, truncated, info = self.env.step(action)
        if len(self.env.actions_memory) > previous_action_count:
            executed_shares = np.asarray(self.env.actions_memory[-1], dtype=float)
            gross_notional = float(np.dot(np.abs(executed_shares), prices))
            self.gross_traded_notionals.append(gross_notional)
            self.daily_turnovers.append(
                gross_notional / beginning_asset if beginning_asset > 0 else 0.0
            )
        if terminated or truncated:
            self.last_episode = {
                "dates": list(self.env.date_memory),
                "account_values": list(self.env.asset_memory),
                "cost": float(self.env.cost),
                "trades": int(self.env.trades),
                "gross_traded_notionals": list(self.gross_traded_notionals),
                "daily_turnovers": list(self.daily_turnovers),
            }
        return observation, reward, terminated, truncated, info


def calculate_metrics(
    dates: Iterable[str],
    account_values: Iterable[float],
    costs: float,
    trades: int,
    gross_traded_notionals: Iterable[float] | None = None,
    daily_turnovers: Iterable[float] | None = None,
    transaction_cost_rate: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    curve = pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "account_value": np.asarray(list(account_values), dtype=float),
        }
    )
    step_count = max(len(curve) - 1, 0)

    def aligned_steps(values: Iterable[float] | None, name: str) -> np.ndarray:
        if values is None:
            return np.zeros(len(curve), dtype=float)
        array = np.asarray(list(values), dtype=float)
        if len(array) == step_count:
            return np.concatenate(([0.0], array))
        if len(array) == len(curve):
            return array
        raise ValueError(
            f"{name} must have one value per account observation or trading step"
        )

    curve["gross_traded_notional"] = aligned_steps(
        gross_traded_notionals, "gross_traded_notionals"
    )
    curve["daily_turnover"] = aligned_steps(daily_turnovers, "daily_turnovers")
    curve["transaction_cost"] = (
        curve["gross_traded_notional"] * transaction_cost_rate
    )
    curve["daily_return"] = curve["account_value"].pct_change().fillna(0.0)
    returns = curve["daily_return"]
    years = max((len(curve) - 1) / 252.0, 1 / 252.0)
    total_return = curve["account_value"].iloc[-1] / curve["account_value"].iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_volatility = float(returns.std(ddof=1) * math.sqrt(252))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
        if returns.std(ddof=1) > 0
        else float("nan")
    )
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std(ddof=1) * math.sqrt(252))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )
    drawdown = curve["account_value"] / curve["account_value"].cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "total_cost": float(costs),
        "trade_count": int(trades),
        "gross_traded_notional": float(curve["gross_traded_notional"].sum()),
        "cumulative_turnover": float(curve["daily_turnover"].sum()),
        "average_daily_turnover": float(
            curve["daily_turnover"].sum() / max(step_count, 1)
        ),
        "annualized_turnover": float(
            curve["daily_turnover"].sum() / max(step_count, 1) * 252
        ),
    }
    return metrics, curve


def run_ppo_condition(
    condition: str,
    indicators: list[str],
    seed: int,
    stock_env_class,
    train: pd.DataFrame,
    trade: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    started = time.perf_counter()
    set_global_seed(seed)
    train_factory = make_environment_factory(
        stock_env_class, train, indicators, config, seed
    )
    train_vector = DummyVecEnv([train_factory])
    normalized_train = VecNormalize(
        train_vector,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    model = PPO(
        "MlpPolicy",
        normalized_train,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=config.ppo_update_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=config.ppo_timesteps, progress_bar=False)
    training_diagnostics = ppo_training_diagnostics(model)

    test_factory = make_environment_factory(
        stock_env_class, trade, indicators, config, seed, record_terminal=True
    )
    test_vector = DummyVecEnv([test_factory])
    normalized_test = VecNormalize(
        test_vector,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    normalized_test.obs_rms = copy.deepcopy(normalized_train.obs_rms)
    observation = normalized_test.reset()
    done = np.array([False])
    while not bool(done[0]):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _ = normalized_test.step(action)
    recorder = normalized_test.venv.envs[0]
    if recorder.last_episode is None:
        raise RuntimeError("The test episode finished without a terminal recording")
    episode = recorder.last_episode
    metrics, curve = calculate_metrics(
        episode["dates"],
        episode["account_values"],
        float(episode["cost"]),
        int(episode["trades"]),
        episode["gross_traded_notionals"],
        episode["daily_turnovers"],
        config.transaction_cost,
    )
    metrics.update(
        {
            "condition": condition,
            "seed": seed,
            "state_features_per_asset": len(indicators),
            "ppo_timesteps": config.ppo_timesteps,
            "condition_elapsed_seconds": time.perf_counter() - started,
            **training_diagnostics,
        }
    )
    curve["condition"] = condition
    curve["seed"] = seed
    normalized_train.close()
    normalized_test.close()
    return metrics, curve


def ppo_training_diagnostics(model) -> dict[str, float]:
    """Extract the final Stable-Baselines3 training diagnostics as plain floats."""
    supported = {
        "train/approx_kl": "train_approx_kl",
        "train/clip_fraction": "train_clip_fraction",
        "train/entropy_loss": "train_entropy_loss",
        "train/explained_variance": "train_explained_variance",
        "train/learning_rate": "train_learning_rate",
        "train/loss": "train_loss",
        "train/policy_gradient_loss": "train_policy_gradient_loss",
        "train/value_loss": "train_value_loss",
    }
    values = getattr(getattr(model, "logger", None), "name_to_value", {})
    diagnostics: dict[str, float] = {}
    for source, destination in supported.items():
        value = values.get(source)
        if value is not None and np.isscalar(value):
            diagnostics[destination] = float(value)
    return diagnostics


def equal_weight_benchmark(
    trade: pd.DataFrame, config: ExperimentConfig
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    prices = trade.pivot(index="date", columns="tic", values="close")[TICKERS]
    daily_returns = prices.pct_change().fillna(0.0).mean(axis=1)
    account_values = np.empty(len(daily_returns), dtype=float)
    account_values[0] = config.initial_amount
    if len(account_values) > 1:
        account_values[1] = (
            config.initial_amount
            * (1 - config.transaction_cost)
            * (1 + daily_returns.iloc[1])
        )
        for index in range(2, len(account_values)):
            account_values[index] = account_values[index - 1] * (
                1 + daily_returns.iloc[index]
            )
    metrics, curve = calculate_metrics(
        prices.index.astype(str),
        account_values,
        config.initial_amount * config.transaction_cost,
        len(TICKERS),
        np.concatenate(([config.initial_amount], np.zeros(len(prices) - 1))),
        np.concatenate(([1.0], np.zeros(len(prices) - 1))),
        config.transaction_cost,
    )
    metrics.update(
        {
            "condition": "Equal-weight buy-and-hold",
            "seed": -1,
            "state_features_per_asset": 0,
            "ppo_timesteps": 0,
        }
    )
    curve["condition"] = "Equal-weight buy-and-hold"
    curve["seed"] = -1
    return metrics, curve


def add_robustness_metrics(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    train: pd.DataFrame,
    trade: pd.DataFrame,
) -> pd.DataFrame:
    threshold = float(train["vix"].quantile(0.75))
    vix_by_date = trade.groupby("date")["vix"].first()
    enriched = curves.copy()
    enriched["vix"] = enriched["date"].map(vix_by_date)
    enriched["high_vix"] = enriched["vix"] >= threshold
    rows = []
    for (condition, seed), group in enriched.groupby(["condition", "seed"]):
        high = group.loc[group["high_vix"], "daily_return"]
        low = group.loc[~group["high_vix"], "daily_return"]
        monthly = (
            group.set_index("date")["account_value"]
            .resample("ME")
            .last()
            .pct_change()
            .dropna()
        )
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "vix_train_75pct_threshold": threshold,
                "high_vix_mean_daily_return": float(high.mean()),
                "high_vix_daily_volatility": float(high.std(ddof=1)),
                "low_vix_mean_daily_return": float(low.mean()),
                "monthly_positive_frequency": float((monthly > 0).mean()),
            }
        )
    return metrics.merge(pd.DataFrame(rows), on=["condition", "seed"], how="left")


def add_monthly_outperformance(curves: pd.DataFrame) -> pd.DataFrame:
    monthly_rows = []
    benchmark = curves[
        curves["condition"] == "Equal-weight buy-and-hold"
    ].copy()
    benchmark_monthly = (
        benchmark.set_index("date")["account_value"]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
    )
    for (condition, seed), group in curves.groupby(["condition", "seed"]):
        monthly = (
            group.set_index("date")["account_value"]
            .resample("ME")
            .last()
            .pct_change()
            .dropna()
        )
        aligned = pd.concat(
            [monthly.rename("model"), benchmark_monthly.rename("benchmark")],
            axis=1,
        ).dropna()
        monthly_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "monthly_outperformance_frequency": float(
                    (aligned["model"] > aligned["benchmark"]).mean()
                )
                if len(aligned)
                else float("nan"),
            }
        )
    return pd.DataFrame(monthly_rows)


def period_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    """Calculate transparent calendar-year results for each condition and seed."""
    rows: list[dict[str, float | int | str]] = []
    for (condition, seed), group in curves.groupby(["condition", "seed"]):
        ordered = group.sort_values("date").copy()
        ordered["year"] = ordered["date"].dt.year
        for year, period in ordered.groupby("year"):
            returns = period["daily_return"].astype(float)
            wealth = (1.0 + returns).cumprod()
            drawdown = wealth / wealth.cummax() - 1.0
            volatility = float(returns.std(ddof=1) * math.sqrt(252))
            sharpe = (
                float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
                if len(returns) > 1 and returns.std(ddof=1) > 0
                else float("nan")
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "year": int(year),
                    "observations": int(len(period)),
                    "period_return": float(wealth.iloc[-1] - 1.0),
                    "annualized_volatility": volatility,
                    "sharpe": sharpe,
                    "max_drawdown": float(drawdown.min()),
                    "positive_day_frequency": float((returns > 0).mean()),
                    "gross_traded_notional": float(
                        period["gross_traded_notional"].sum()
                    ),
                    "total_cost": float(period["transaction_cost"].sum()),
                    "cumulative_turnover": float(period["daily_turnover"].sum()),
                    "annualized_turnover": float(
                        period["daily_turnover"].mean() * 252
                    ),
                }
            )
    return pd.DataFrame(rows)


def student_t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical value without adding a SciPy dependency."""
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if degrees_of_freedom <= 0:
        return float("nan")
    return table.get(degrees_of_freedom, 1.96)


def summarize_across_seeds(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    """Return descriptive seed means and explicitly exploratory t intervals."""
    rows: list[dict[str, float | int | str]] = []
    ppo = frame[frame["seed"] >= 0]
    for group_key, group in ppo.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        labels = dict(zip(group_columns, group_key))
        for metric in metric_columns:
            values = group[metric].dropna().astype(float)
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            standard_deviation = (
                float(values.std(ddof=1)) if n > 1 else float("nan")
            )
            if n > 1:
                margin = (
                    student_t_critical_95(n - 1)
                    * standard_deviation
                    / math.sqrt(n)
                )
            else:
                margin = float("nan")
            rows.append(
                {
                    **labels,
                    "metric": metric,
                    "n_seeds": n,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "ci95_lower": mean - margin,
                    "ci95_upper": mean + margin,
                    "minimum": float(values.min()) if n else float("nan"),
                    "maximum": float(values.max()) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_ann_vs_mps(
    curves: pd.DataFrame,
    seed: int = 2026,
    block_length: int = 20,
    samples: int = 2_000,
) -> dict[str, float | int]:
    ann = curves[curves["condition"] == "ANN signal"].pivot(
        index="date", columns="seed", values="daily_return"
    )
    mps = curves[curves["condition"] == "QINN-MPS signal"].pivot(
        index="date", columns="seed", values="daily_return"
    )
    ann_mean = ann.mean(axis=1)
    mps_mean = mps.mean(axis=1)
    aligned = pd.concat(
        [ann_mean.rename("ann"), mps_mean.rename("mps")], axis=1
    ).dropna()
    values = aligned.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for sample in range(samples):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend((start + offset) % n for offset in range(block_length))
        draw = values[np.asarray(indices[:n])]
        differences[sample] = (draw[:, 1].mean() - draw[:, 0].mean()) * 252
    point = float((values[:, 1].mean() - values[:, 0].mean()) * 252)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "annualized_mean_return_difference_mps_minus_ann": point,
        "bootstrap_95pct_lower": float(lower),
        "bootstrap_95pct_upper": float(upper),
        "bootstrap_probability_mps_gt_ann": float(np.mean(differences > 0)),
        "block_length_days": block_length,
        "bootstrap_samples": samples,
    }


def figure_output_path(
    figures: Path, preferred: str, compact: str
) -> Path:
    preferred_path = figures / preferred
    if len(str(preferred_path)) < 260:
        return preferred_path
    return figures / compact


def save_plots(
    output_dir: Path,
    curves: pd.DataFrame,
    encoder_history: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    for condition, group in curves.groupby("condition"):
        if condition == "Equal-weight buy-and-hold":
            representative = group
            plt.plot(
                representative["date"],
                representative["account_value"] / representative["account_value"].iloc[0],
                label=condition,
                linewidth=2.2,
                color="black",
            )
            continue
        pivot = group.pivot(index="date", columns="seed", values="account_value")
        normalized = pivot / pivot.iloc[0]
        mean = normalized.mean(axis=1)
        low = normalized.min(axis=1)
        high = normalized.max(axis=1)
        line = plt.plot(mean.index, mean, label=f"{condition} (seed mean)", linewidth=2)[0]
        plt.fill_between(mean.index, low, high, alpha=0.12, color=line.get_color())
    plt.title("FinRL PPO out-of-sample equity curves (2019-2023)")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figure_output_path(figures, "equity_curves.png", "equity.png"),
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    for model, group in encoder_history.groupby("model"):
        plt.plot(group["epoch"], group["validation_huber"], label=model)
    plt.title("Encoder validation loss (2018)")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figure_output_path(
            figures, "encoder_validation_loss.png", "encoder.png"
        ),
        dpi=180,
    )
    plt.close()

    ppo = metrics[metrics["seed"] >= 0]
    summary = ppo.groupby("condition")["sharpe"].agg(["mean", "std"])
    plt.figure(figsize=(8, 5))
    plt.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    plt.title("Out-of-sample Sharpe across PPO seeds")
    plt.ylabel("Annualized Sharpe (risk-free rate = 0)")
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        figure_output_path(
            figures, "sharpe_by_condition.png", "sharpe.png"
        ),
        dpi=180,
    )
    plt.close()


def write_run_manifest(
    output_dir: Path,
    config: ExperimentConfig,
    signal_metrics: pd.DataFrame,
    bootstrap: dict[str, float | int],
    runtime: dict[str, object],
    encoder_runtime: dict[str, object],
) -> None:
    manifest = {
        "experiment": asdict(config),
        "tickers": TICKERS,
        "finrl_features": FINRL_FEATURES,
        "representation_features": REPRESENTATION_FEATURES,
        "state_construction": {
            "base_formula": "cash + 15 prices + 15 holdings + 10 indicators x 15 assets",
            "base_dimension": 1 + 2 * len(TICKERS) + len(FINRL_FEATURES) * len(TICKERS),
            "signal_formula": "base state + one frozen encoder prediction per asset",
            "signal_dimension": 1
            + 2 * len(TICKERS)
            + (len(FINRL_FEATURES) + 1) * len(TICKERS),
            "signal_position": "appended as the final per-asset technical-indicator block",
            "signal_scale": "standardized next-day return prediction clipped to [-8, 8]",
        },
        "turnover_definition": {
            "daily_turnover": "sum(abs(executed shares) * pre-trade price) / beginning-of-step portfolio value",
            "cumulative_turnover": "sum of daily turnover over the evaluation period",
            "annualized_turnover": "mean daily turnover x 252",
            "interpretation": "gross one-way turnover; buys and sells both contribute to traded notional",
        },
        "ann_parameter_count": parameter_count(
            ANNRegressor(len(REPRESENTATION_FEATURES))
        ),
        "mps_parameter_count": parameter_count(
            MPSRegressor(len(REPRESENTATION_FEATURES), config.mps_bond_dimension)
        ),
        "finrl_repository": FINRL_REPOSITORY,
        "finrl_commit": FINRL_COMMIT,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_sha256": {
            filename: metadata["sha256"] for filename, metadata in DATA_FILES.items()
        },
        "signal_test_metrics": signal_metrics[
            signal_metrics["split"] == "test_2019_2023"
        ].to_dict(orient="records"),
        "paired_block_bootstrap": bootstrap,
        "runtime": runtime,
        "encoder_runtime": encoder_runtime,
        "limitations": [
            "Classical tensor-network simulation; no quantum hardware was used.",
            "One historical train/test split and one 15-stock Nasdaq subset.",
            f"{len(config.ppo_seeds)} PPO seeds quantify policy instability but do not establish broad statistical generality.",
            "ANN and MPS parameter counts are reported explicitly; equality applies only to configurations with matching counts.",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def runtime_metadata(
    config: ExperimentConfig,
    started_at: datetime,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    resolved_device = resolve_encoder_device(config.encoder_device)
    metadata: dict[str, object] = {
        "started_at_utc": started_at.isoformat(),
        "status": "completed" if elapsed_seconds is not None else "running",
        "git_commit": current_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "encoder_device_requested": config.encoder_device,
        "encoder_device_resolved": str(resolved_device),
        "ppo_device": "cpu",
    }
    if torch.cuda.is_available():
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
    if elapsed_seconds is not None:
        metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
        metadata["elapsed_seconds"] = elapsed_seconds
    return metadata


def write_run_status(
    output_dir: Path, config: ExperimentConfig, runtime: dict[str, object]
) -> None:
    payload = {"experiment": asdict(config), "runtime": runtime}
    (output_dir / "run_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def finalize_existing_results(output_dir: Path) -> None:
    """Finalize complete tabular artifacts after a post-training interruption."""
    required_paths = {
        "status": output_dir / "run_status.json",
        "signal": output_dir / "signal_metrics.csv",
        "history": output_dir / "encoder_training_history.csv",
        "metrics": output_dir / "ppo_backtest_metrics.csv",
        "curves": output_dir / "equity_curves.csv",
        "annual": output_dir / "annual_period_metrics.csv",
        "condition_summary": output_dir / "condition_seed_summary.csv",
        "annual_summary": output_dir / "annual_period_seed_summary.csv",
        "bootstrap": output_dir / "ann_vs_mps_block_bootstrap.csv",
    }
    missing = [
        name for name, path in required_paths.items() if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Cannot finalize; missing completed artifacts: {sorted(missing)}"
        )
    if (output_dir / "run_manifest.json").exists():
        raise RuntimeError("Run manifest already exists; refusing recovery overwrite")

    status = json.loads(required_paths["status"].read_text(encoding="utf-8"))
    if status.get("runtime", {}).get("status") != "running":
        raise RuntimeError("Recovery requires a run status marked running")
    config = ExperimentConfig(**status["experiment"])
    validate_config(config)

    metrics = pd.read_csv(required_paths["metrics"])
    curves = pd.read_csv(required_paths["curves"])
    signal_metrics = pd.read_csv(required_paths["signal"])
    encoder_history = pd.read_csv(required_paths["history"])
    bootstrap_rows = pd.read_csv(required_paths["bootstrap"])
    expected_keys = {
        (condition, seed)
        for condition in ("Base FinRL", "ANN signal", "QINN-MPS signal")
        for seed in config.ppo_seeds
    }
    observed_keys = set(
        metrics.loc[metrics["seed"] >= 0, ["condition", "seed"]]
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise RuntimeError("Completed metrics do not match configured conditions/seeds")
    curve_keys = set(
        curves.loc[curves["seed"] >= 0, ["condition", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if curve_keys != expected_keys:
        raise RuntimeError("Completed curves do not match configured conditions/seeds")
    if len(bootstrap_rows) != 1:
        raise RuntimeError("Expected exactly one block-bootstrap result")
    bootstrap = bootstrap_rows.iloc[0].to_dict()

    run_lock = RunLock(output_dir)
    run_lock.acquire()
    try:
        save_plots(output_dir, curves, encoder_history, metrics)
        runtime = dict(status["runtime"])
        started_at = datetime.fromisoformat(str(runtime["started_at_utc"]))
        latest_artifact = max(
            path.stat().st_mtime
            for name, path in required_paths.items()
            if name != "status"
        )
        completed_at = datetime.fromtimestamp(latest_artifact, UTC)
        runtime.update(
            {
                "status": "completed",
                "completed_at_utc": completed_at.isoformat(),
                "elapsed_seconds": max(
                    0.0, (completed_at - started_at).total_seconds()
                ),
                "finalization_git_commit": current_git_commit(),
                "completion_basis": (
                    "Recovered after post-training plot failure; completion "
                    "time is the latest required tabular artifact timestamp."
                ),
            }
        )
        encoder_runtime = {
            "status": "not_persisted_before_post_training_failure",
            "note": (
                "Encoder timing was not recoverable from completed tabular "
                "artifacts and is not estimated."
            ),
        }
        write_run_manifest(
            output_dir,
            config,
            signal_metrics,
            bootstrap,
            runtime,
            encoder_runtime,
        )
        write_run_status(output_dir, config, runtime)
    finally:
        run_lock.release()


def json_config(config: ExperimentConfig) -> dict[str, object]:
    """Return the configuration in its stable JSON representation."""
    return json.loads(json.dumps(asdict(config)))


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    """Prevent concurrent writers from corrupting one experiment directory."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / ".run.lock"
        self.pid = os.getpid()
        self.acquired = False

    def acquire(self) -> None:
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError:
                try:
                    owner = int(self.path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    owner = -1
                if pid_is_running(owner):
                    raise RuntimeError(
                        f"Output directory is already in use by process {owner}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"{self.pid}\n")
            self.acquired = True
            return
        raise RuntimeError(f"Could not acquire run lock: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            owner = int(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = -1
        if owner == self.pid:
            self.path.unlink(missing_ok=True)
        self.acquired = False


def load_partial_results(
    output_dir: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    """Load policy checkpoints only when both partial artifacts are consistent."""
    compact_paths = (
        output_dir / "metrics.partial.csv",
        output_dir / "curves.partial.csv",
        output_dir / "config.partial.json",
    )
    legacy_paths = (
        output_dir / "ppo_backtest_metrics.partial.csv",
        output_dir / "equity_curves.partial.csv",
        output_dir / "partial_config.json",
    )
    metrics_path, curves_path, config_path = (
        compact_paths if any(path.exists() for path in compact_paths) else legacy_paths
    )
    if not metrics_path.exists() and not curves_path.exists():
        return [], []
    if not metrics_path.exists() or not curves_path.exists() or not config_path.exists():
        raise RuntimeError(
            "Partial metrics, equity curves, and configuration must exist together"
        )
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config != json_config(config):
        raise RuntimeError("Partial results belong to a different configuration")
    metrics = pd.read_csv(metrics_path)
    curves = pd.read_csv(curves_path)
    required_metric_columns = {"condition", "seed"}
    required_curve_columns = {"condition", "seed", "date", "account_value"}
    if not required_metric_columns.issubset(metrics.columns):
        raise RuntimeError("Partial metrics are missing run identity columns")
    if not required_curve_columns.issubset(curves.columns):
        raise RuntimeError("Partial curves are missing run identity columns")
    curves["date"] = pd.to_datetime(curves["date"], format="mixed", errors="raise")
    metric_keys = set(zip(metrics["condition"], metrics["seed"], strict=True))
    curve_keys = set(zip(curves["condition"], curves["seed"], strict=True))
    if metric_keys != curve_keys:
        raise RuntimeError("Partial metrics and curves contain different runs")
    curve_groups = [
        group.copy()
        for _, group in curves.groupby(["condition", "seed"], sort=False)
    ]
    return metrics.to_dict(orient="records"), curve_groups


def write_partial_results(
    output_dir: Path,
    config: ExperimentConfig,
    metric_rows: list[dict[str, object]],
    curve_rows: list[pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.partial.csv"
    curves_path = output_dir / "curves.partial.csv"
    config_path = output_dir / "config.partial.json"
    temporary_metrics = output_dir / ".metrics.tmp"
    temporary_curves = output_dir / ".curves.tmp"
    temporary_config = output_dir / ".config.tmp"
    try:
        pd.DataFrame(metric_rows).to_csv(temporary_metrics, index=False)
        pd.concat(curve_rows, ignore_index=True).to_csv(
            temporary_curves, index=False
        )
        temporary_config.write_text(
            json.dumps(json_config(config), indent=2), encoding="utf-8"
        )
        temporary_metrics.replace(metrics_path)
        temporary_curves.replace(curves_path)
        temporary_config.replace(config_path)
    finally:
        temporary_metrics.unlink(missing_ok=True)
        temporary_curves.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--finrl-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bond-dimension", type=int, default=4)
    parser.add_argument("--encoder-epochs", type=int, default=60)
    parser.add_argument("--encoder-patience", type=int, default=10)
    parser.add_argument("--encoder-batch-size", type=int, default=512)
    parser.add_argument(
        "--encoder-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Finalize complete tabular artifacts after a post-training failure.",
    )
    return parser.parse_args()


def main() -> None:
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()
    args = parse_args()
    if args.finalize_existing:
        finalize_existing_results(args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(args.output_dir)
    run_lock.acquire()
    atexit.register(run_lock.release)
    config = ExperimentConfig(
        ppo_seeds=tuple(args.seeds),
        ppo_timesteps=args.timesteps,
        mps_bond_dimension=args.bond_dimension,
        encoder_epochs=args.encoder_epochs,
        encoder_patience=args.encoder_patience,
        encoder_batch_size=args.encoder_batch_size,
        encoder_device=args.encoder_device,
    )
    validate_config(config)
    write_run_status(
        args.output_dir, config, runtime_metadata(config, started_at)
    )
    env_file = ensure_finrl(args.finrl_dir)
    stock_env_class = load_stock_trading_env(env_file)
    train, trade = read_market_data(args.data_dir)
    train, trade = engineer_representation_features(train, trade)
    train, trade, signal_metrics, encoder_history, encoder_runtime = train_signal_models(
        train, trade, config
    )
    train_finrl = finrl_frame(train)
    trade_finrl = finrl_frame(trade)

    conditions = {
        "Base FinRL": FINRL_FEATURES,
        "ANN signal": FINRL_FEATURES + ["ann_signal"],
        "QINN-MPS signal": FINRL_FEATURES + ["qinn_mps_signal"],
    }
    partial_metrics, partial_curves = load_partial_results(args.output_dir, config)
    metric_rows: list[dict[str, object]] = partial_metrics
    curve_rows: list[pd.DataFrame] = partial_curves
    completed = {
        (str(row["condition"]), int(row["seed"])) for row in metric_rows
    }
    for condition, indicators in conditions.items():
        for seed in config.ppo_seeds:
            if (condition, seed) in completed:
                print(f"Resuming after {condition}, seed={seed}", flush=True)
                continue
            print(f"Training {condition}, seed={seed} ...", flush=True)
            metrics, curve = run_ppo_condition(
                condition,
                indicators,
                seed,
                stock_env_class,
                train_finrl,
                trade_finrl,
                config,
            )
            metric_rows.append(metrics)
            curve_rows.append(curve)
            # Preserve completed policy runs if a long benchmark is interrupted.
            write_partial_results(
                args.output_dir, config, metric_rows, curve_rows
            )
    benchmark_metrics, benchmark_curve = equal_weight_benchmark(trade, config)
    metric_rows.append(benchmark_metrics)
    curve_rows.append(benchmark_curve)
    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    metrics = add_robustness_metrics(metrics, curves, train, trade)
    metrics = metrics.merge(
        add_monthly_outperformance(curves),
        on=["condition", "seed"],
        how="left",
    )
    annual_metrics = period_metrics(curves)
    condition_summary = summarize_across_seeds(
        metrics,
        ["condition"],
        [
            "total_return",
            "annual_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "annualized_turnover",
            "total_cost",
        ],
    )
    annual_summary = summarize_across_seeds(
        annual_metrics,
        ["condition", "year"],
        [
            "period_return",
            "sharpe",
            "max_drawdown",
            "annualized_turnover",
            "total_cost",
        ],
    )
    bootstrap = block_bootstrap_ann_vs_mps(curves)

    signal_metrics.to_csv(args.output_dir / "signal_metrics.csv", index=False)
    encoder_history.to_csv(args.output_dir / "encoder_training_history.csv", index=False)
    metrics.to_csv(args.output_dir / "ppo_backtest_metrics.csv", index=False)
    curves.to_csv(args.output_dir / "equity_curves.csv", index=False)
    annual_metrics.to_csv(args.output_dir / "annual_period_metrics.csv", index=False)
    condition_summary.to_csv(args.output_dir / "condition_seed_summary.csv", index=False)
    annual_summary.to_csv(
        args.output_dir / "annual_period_seed_summary.csv", index=False
    )
    for checkpoint_name in (
        "metrics.partial.csv",
        "curves.partial.csv",
        "config.partial.json",
        "ppo_backtest_metrics.partial.csv",
        "equity_curves.partial.csv",
        "partial_config.json",
    ):
        (args.output_dir / checkpoint_name).unlink(missing_ok=True)
    pd.DataFrame([bootstrap]).to_csv(
        args.output_dir / "ann_vs_mps_block_bootstrap.csv", index=False
    )
    save_plots(args.output_dir, curves, encoder_history, metrics)
    elapsed_seconds = time.perf_counter() - started_clock
    runtime = runtime_metadata(config, started_at, elapsed_seconds)
    write_run_manifest(
        args.output_dir,
        config,
        signal_metrics,
        bootstrap,
        runtime,
        encoder_runtime,
    )
    write_run_status(args.output_dir, config, runtime)
    print(metrics.to_string(index=False))
    print(json.dumps(bootstrap, indent=2))
    run_lock.release()


if __name__ == "__main__":
    main()
