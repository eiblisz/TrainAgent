# scripts/train_agent.py
# ---- UNIFIED FULL VERSION (legacy features restored + Eval fix) ----

import os
import sys
import time
import json
import math
import argparse
import datetime as dt
import subprocess
import warnings

# --- third-party ---
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # fej nélküli/worker-safe render
import matplotlib.pyplot as plt
import torch as th

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from scripts.finalize_training import save_baseline_snapshot

# (EvalCallback, Monitor csak akkor kell, ha tényleg használod – különben hagyd ki)
# --- Discord always-import (thread/subproc safe) ---
from core.utils.discord import send_discord_message, send_discord_file
from core.rl.curriculum_scheduler import get_scheduler

# --- Override: local save_baseline_snapshot will shadow imported one ---
from scripts import finalize_training
finalize_training.save_baseline_snapshot = save_baseline_snapshot

# ---------------------------------------------------------------------------
# Helper paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "core", "rl", "envs", "reward_patch_config.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

from core.utils.discord import send_discord_message

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        reward_cfg = json.load(f)
    runtime_cfg = reward_cfg.get("runtime", {})
    validation_freq = runtime_cfg.get("validation_interval", 100_000)
    checkpoint_interval = runtime_cfg.get("checkpoint_interval", 300_000)
    reload_every_steps = runtime_cfg.get("reload_every_steps", 100_000)
    device_mode = runtime_cfg.get("device", "cuda")

    # --- VecNormalize kapcsoló (több forrásból összevont) ---
    use_vecnorm = False

    # 1️⃣ Reward patch JSON (Reward Tuner)
    try:
        reward_patch_path = os.path.join("core", "rl", "envs", "reward_patch_config.json")
        if os.path.exists(reward_patch_path):
            with open(reward_patch_path, "r", encoding="utf-8") as f:
                reward_patch = json.load(f)
            if reward_patch.get("vecnorm", False):
                use_vecnorm = True
                print("[CFG] VecNormalize aktiválva a reward_patch_config.json alapján.")
    except Exception as e:
        print(f"[WARN] VecNormalize betöltés hiba (reward_patch): {e}")

    # 2️⃣ Reward config runtime szekció
    if runtime_cfg.get("vecnorm", False):
        use_vecnorm = True
        print("[CFG] VecNormalize aktiválva a reward_patch_config runtime szekció alapján.")

    # 3️⃣ Globális live_params.json fallback
    try:
        live_path = os.path.join(ROOT_DIR, "config", "live_params.json")
        if os.path.exists(live_path):
            with open(live_path, "r", encoding="utf-8") as f:
                live_cfg = json.load(f)
            if live_cfg.get("vecnorm", False) or live_cfg.get("runtime", {}).get("vecnorm", False):
                use_vecnorm = True
                print("[CFG] VecNormalize aktiválva a live_params.json alapján.")
    except Exception as e:
        print(f"[WARN] VecNormalize betöltés hiba (live_params): {e}")

except Exception:
    validation_freq = 100_000
    checkpoint_interval = 300_000
    reload_every_steps = 100_000
    device_mode = "cuda"
    use_vecnorm = False
  
from core.rl.curriculum_scheduler import get_scheduler


# ======================================================================
# VALIDATION CALLBACK (Unified Async + Resume-safe)
# ======================================================================
class ValidationCallback(BaseCallback):
    """
    Teljes async validáció tréning közben, resume-safe módon.
    """

    def __init__(
        self,
        freq=50_000,
        verbose=0,
        symbol=None,
        timeframe=None,
        timesteps=None,
        reward_mode=None,
        device="cpu",
        start_step=None,
    ):
        super().__init__(verbose)
        self.freq = int(freq)
        self.symbol = symbol
        self.timeframe = timeframe
        self.timesteps = timesteps
        self.reward_mode = reward_mode
        self.device = device
        # ✅ Resume-safe: az első validáció csak a következő stride-nál indul
        self.last_trigger = int(start_step or 0)
    
    def _on_step(self) -> bool:
        try:
            cur = int(self.num_timesteps)
            
            # 🔹 ha több periódust is átugrott, mindet pótolja
            while cur - self.last_trigger >= self.freq:
                self.last_trigger += self.freq  # nem ugrik el a current-re
                
                # --- Discord értesítés ---
                try:
                    send_discord_message(
                        "TRAINING",
                        f"[VALID] Async validation indul... ({cur:,} lépésnél)"
                    )
                except Exception:
                    pass
                
                # --- Async validáció indítása külön processzben ---
                try:
                    cmd = [
                        sys.executable, "-m", "scripts.run_validation",
                        "--symbol", str(self.symbol),
                        "--timeframe", str(self.timeframe),
                        "--device", str(self.device),
                        "--reward", str(self.reward_mode or "realistic"),
                        "--timesteps", str(cur),
                        "--mode", "full",  # ✅ a "fast" helyett teljes validációt kérünk
                        "--dataset", "val",  # ✅ ha van külön validációs halmaz
                        "--auto_close_eval", "true",
                        "--deterministic", "true"
                    ]
                    
                    # Ha szeretnéd, hogy mindig lássad a visszajelzést a konzolon/logban:
                    log_path = os.path.join("reports", f"validation_{self.symbol}_{self.timeframe}_{cur}.log")
                    os.makedirs("reports", exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as lf:
                        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                    
                    send_discord_message(
                        "TRAINING",
                        f"✅ Teljes async validation elindítva ({self.symbol} {self.timeframe}, {cur:,} lépés)."
                    )
                
                except Exception as e:
                    print(f"[WARN] Validation subprocess hiba: {e}")
        
        except Exception as e:
            print(f"[WARN] ValidationCallback hiba: {e}")
        
        return True  # tréning folytatódjon


# --- Compact validation summary handling (class-on kívül) ---
try:
    val_path = os.path.join("data/reports", "validation_results.json")
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        win = data.get("winrate", "n/a")
        sharpe = data.get("sharpe", "n/a")
        avg_rew = data.get("avg_reward", "n/a")
        print(f"[VALID SUMMARY] Winrate: {win}% | Sharpe: {sharpe} | AvgReward: {avg_rew}")
    else:
        print("[WARN] Validation results JSON not found.")
except Exception as e:
    print(f"[VALID ERROR] {e}")

    
# === Diagnosztikai segédfüggvény (biztonságos, Discord nélkül) ===
def send_diag(msg: str):
    import time
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[DIAG-SUPPRESSED {now}] {msg}")


# === Diagnosztikai Discord üzenetküldő (ideiglenesen csak konzolos log) ===
def send_diag(msg: str):
    """
    Külön DIAG Discord csatornára küldene üzenetet,
    de jelenleg le van tiltva a flood megelőzése érdekében.
    A log konzolra továbbra is kiíródik időbélyeggel.
    """
    import time
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[DIAG-SUPPRESSED {now_str}] {msg}")
    
    
def generate_final_analysis_chart(log_callback, symbol, timeframe, done, MODELS_DIR):
    """
    Deep Training Analysis (improved)
    Hárompaneles grafikon: Lossok / Entropy / Performance overlay
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    chart_path = os.path.join(MODELS_DIR, f"deep_analysis_{symbol}_{timeframe}_latest.png")
    
    try:
        # --- adatok ---
        timesteps = np.array(getattr(log_callback, "timesteps", []))
        pol = np.array(getattr(log_callback, "policy_losses", []))
        val = np.array(getattr(log_callback, "value_losses", []))
        ent = np.array(getattr(log_callback, "entropies", []))
        expv = np.array(getattr(log_callback, "explained_variances", []))
        rew = np.array(getattr(log_callback, "rewards", []))
        win = np.array(getattr(log_callback, "winrates", []))
        
        # --- biztonsági ellenőrzés ---
        if len(timesteps) < 5:
            print("[WARN] Nincs elég adat a deep chart-hoz.")
            return None
        
        # --- smoothing segédfüggvény ---
        def smooth(arr, n=50):
            arr = np.array(arr, dtype=float)
            if len(arr) < n:
                return arr
            return np.convolve(arr, np.ones(n) / n, mode="valid")
        
        # --- smoothing mindenre ---
        pol_s, val_s, ent_s, expv_s = map(smooth, [pol, val, ent, expv])
        rew_s, win_s = map(smooth, [rew, win])
        
        # --- length align safeguard (avoid x/y mismatch) ---
        try:
            min_len = min(
                len(timesteps),
                len(pol_s),
                len(val_s),
                len(ent_s),
                len(expv_s),
                len(rew_s),
                len(win_s),
            )
            timesteps = timesteps[:min_len]
            pol_s, val_s, ent_s, expv_s, rew_s, win_s = (
                pol_s[:min_len],
                val_s[:min_len],
                ent_s[:min_len],
                expv_s[:min_len],
                rew_s[:min_len],
                win_s[:min_len],
            )
        except Exception as e:
            print(f"[WARN] DeepAnalysis length-align hiba: {e}")
        
        # --- plotting ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # 1️⃣ Policy vs Value Loss
        axes[0].plot(timesteps, pol_s, label="Policy Loss", color="#0077cc", alpha=0.8)
        axes[0].plot(timesteps, val_s, label="Value Loss", color="#ff9900", alpha=0.8)
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[0].legend(loc="upper right")
        
        # 2️⃣ Entropy + Explained Var
        axes[1].plot(timesteps, ent_s, label="Entropy", color="#9933cc", alpha=0.8)
        axes[1].plot(timesteps, expv_s, label="Explained Var", color="#2ecc71", alpha=0.7)
        axes[1].set_ylabel("Entropy / ExpVar")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        axes[1].legend(loc="upper right")
        
        # 3️⃣ Performance overlay (Reward + Winrate)
        ax3 = axes[2]
        ax3.plot(timesteps, rew_s, label="Reward (avg)", color="#ffaa00", alpha=0.8)
        ax3.set_ylabel("Reward", color="#ffaa00")
        ax3.tick_params(axis="y", labelcolor="#ffaa00")
        ax3.grid(True, linestyle="--", alpha=0.3)
        
        ax4 = ax3.twinx()
        ax4.plot(timesteps, win_s * 100, label="Winrate %", color="#00cc66", alpha=0.7)
        ax4.set_ylabel("Winrate %", color="#00cc66")
        ax4.tick_params(axis="y", labelcolor="#00cc66")
        
        # --- cím és mentés ---
        axes[-1].set_xlabel("Timesteps")
        fig.suptitle(f"🧠 Deep Training Analysis — {symbol} {timeframe}", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"[OK] Deep analysis chart mentve: {chart_path}")
        return chart_path
    
    except Exception as e:
        print(f"[WARN] Deep analysis chart generálás hiba: {e}")
        return None


# === Egységes tréninggrafikon (Reward + Winrate + uPnL + DeepMetrics overlay) ===
def generate_training_chart(log_callback, symbol, timeframe, done, MODELS_DIR, phase="checkpoint"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    chart_path = os.path.join(MODELS_DIR, f"training_chart_{symbol}_{timeframe}_{phase}_{done}.png")
    try:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()  # jobb tengely a Winrate-nek / uPnL-nek
        
        timesteps = np.array(getattr(log_callback, "timesteps", []))
        rewards = np.array(getattr(log_callback, "rewards", []))
        winrates = np.array(getattr(log_callback, "winrates", []))
        upnls = np.array(getattr(log_callback, "upnls", [0] * len(timesteps)))
        
        # --- Deep metrikák (opcionális) ---
        policy_loss = np.array(getattr(log_callback, "policy_losses", []))
        value_loss = np.array(getattr(log_callback, "value_losses", []))
        entropy = np.array(getattr(log_callback, "entropies", []))
        explained_var = np.array(
            getattr(log_callback, "explained_variances", getattr(log_callback, "explained_vars", []))
        )
        
        # --- Hossz-igazítás minden metrikára ---
        min_len = min([
            len(timesteps),
            len(rewards),
            len(winrates),
            len(upnls),
            len(policy_loss) if len(policy_loss) else len(timesteps),
            len(value_loss) if len(value_loss) else len(timesteps),
            len(entropy) if len(entropy) else len(timesteps),
            len(explained_var) if len(explained_var) else len(timesteps),
        ])
        timesteps = timesteps[:min_len]
        rewards = rewards[:min_len]
        winrates = winrates[:min_len]
        upnls = upnls[:min_len]
        policy_loss = policy_loss[:min_len] if len(policy_loss) else policy_loss
        value_loss = value_loss[:min_len] if len(value_loss) else value_loss
        entropy = entropy[:min_len] if len(entropy) else entropy
        explained_var = explained_var[:min_len] if len(explained_var) else explained_var
        
        # --- Reward (bal Y) ---
        ax1.plot(timesteps, rewards, label="EpReward", color="tab:blue", alpha=0.4)
        if len(rewards) >= 50:
            smoothed = np.convolve(rewards, np.ones(50) / 50, mode="valid")
            ax1.plot(timesteps[49:], smoothed, color="tab:orange", label="EpReward (avg50)")
        ax1.set_ylabel("Reward")
        
        # --- Winrate (jobb Y) ---
        ax2.plot(timesteps, winrates, color="tab:green", alpha=0.5, label="Winrate %")
        if len(winrates) >= 50:
            sm_wr = np.convolve(winrates, np.ones(50) / 50, mode="valid")
            ax2.plot(timesteps[49:], sm_wr, color="tab:red", label="Winrate (avg50)")
        ax2.set_ylabel("Winrate % / uPnL")
        
        # --- uPnL overlay ---
        if len(upnls) == len(timesteps):
            ax2.plot(timesteps, upnls, color="gray", alpha=0.3, linestyle="--", label="uPnL (floating)")
        
        # --- Deep Metrics overlay (külön skálán, halvány színekkel) ---
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        if len(policy_loss):
            ax3.plot(timesteps[:len(policy_loss)], policy_loss, color="#3b83bd", alpha=0.3, label="PolicyLoss")
        if len(value_loss):
            ax3.plot(timesteps[:len(value_loss)], value_loss, color="#ffb000", alpha=0.3, label="ValueLoss")
        if len(entropy):
            ax3.plot(timesteps[:len(entropy)], entropy, color="#9b59b6", alpha=0.3, label="Entropy")
        if len(explained_var):
            ax3.plot(timesteps[:len(explained_var)], explained_var, color="#2ecc71", alpha=0.3, label="Expl.Var")
        ax3.set_ylabel("Deep Metrics")
        
        # --- Skálázás automatikusan ---
        if len(upnls) > 0:
            ax2.set_ylim(min(upnls) * 1.1, max(upnls) * 1.1)
        if len(rewards) > 0:
            ax1.set_ylim(min(rewards) * 1.1, max(rewards) * 1.1)
        
        # --- Átlagok szövege (safe mean/std számítás) ---
        try:
            mean_wr = 100 * np.nanmean(winrates) if len(winrates) > 0 else 0.0
            mean_rw = np.nanmean(rewards) if len(rewards) > 0 else 0.0
            mean_upnl = np.nanmean(upnls) if len(upnls) > 0 else 0.0
            std_rw = np.nanstd(rewards) if len(rewards) > 1 else 0.0
        except Exception as e:
            print(f"[WARN] Átlag számítás hiba: {e}")
            mean_wr, mean_rw, mean_upnl, std_rw = 0.0, 0.0, 0.0, 0.0
        
        ax1.text(
            0.01, 0.97,
            f"Avg WR={mean_wr:.2f}% | Avg Rew={mean_rw:.4f} | Avg uPnL={mean_upnl:.2f} | Rσ={std_rw:.2f}",
            transform=ax1.transAxes,
            fontsize=9, color="white", backgroundcolor="black"
        )
        
        # --- Közös legenda ---
        lines, labels = ax1.get_legend_handles_labels()
        for ax in [ax2, ax3]:
            l, lb = ax.get_legend_handles_labels()
            lines += l
            labels += lb
        ax1.legend(lines, labels, loc="best", fontsize=8)
        
        ax1.set_xlabel("Timesteps")
        plt.title(f"Final Training Chart – {symbol} {timeframe}")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        
        print(f"[OK] Training chart saved: {chart_path}")
        return chart_path
    
    except Exception as e:
        print(f"[WARN] Chart generálás hiba: {e}")
        return None
    
    
    # ---------------------------------------------------------------------------
# Callbacks
from stable_baselines3.common.callbacks import BaseCallback


class OnBestCallback(BaseCallback):
    """
    Discord értesítés, ha új legjobb modell mentve, és időnként async validációt indít.
    """

    def __init__(self, freq=100000, symbol=None, timeframe=None, reward_mode=None, device="cpu"):
        super().__init__()
        self.freq = freq
        self.symbol = symbol
        self.timeframe = timeframe
        self.reward_mode = reward_mode
        self.device = device
        self.last_trigger = 0

    def _on_step(self) -> bool:
        try:
            # --- 1️⃣ Értesítés, ha új legjobb modell készült ---
            try:
                send_discord_message("TRAINING", "[EVAL] Új legjobb modell mentve.")
            except Exception as e:
                print(f"[WARN] Discord hiba OnBestCallback alatt: {e}")
            
            # --- 2️⃣ Periodikus async validáció (pl. 50k vagy 100k lépésenként) ---
            cur = int(self.num_timesteps)
            
            # ha több periódust is átugrott, mindet pótolja
            while cur - self.last_trigger >= self.freq:
                self.last_trigger += self.freq  # nem a current stepre ugrik
                
                try:
                    send_discord_message(
                        "TRAINING",
                        f"[VALID] Indul a validation… ({cur:,} lépésnél)"
                    )
                except Exception:
                    pass
                
                try:
                    cmd = [
                        sys.executable, "-m", "scripts.run_validation",
                        "--symbol", str(self.symbol),
                        "--timeframe", str(self.timeframe),
                        "--device", str(self.device),
                        "--reward", str(self.reward_mode or "realistic"),
                        "--timesteps", str(cur),
                        "--mode", "full",  # ✅ teljes validáció
                        "--dataset", "val",
                        "--auto_close_eval", "true",
                        "--deterministic", "true"
                    ]
                    
                    # 🔹 LOG fájl hozzáadása – így látod a run_validation console outputját
                    os.makedirs("reports", exist_ok=True)
                    log_path = os.path.join(
                        "reports",
                        f"validation_{self.symbol}_{self.timeframe}_{int(cur)}.log"
                    )
                    
                    with open(log_path, "w", encoding="utf-8") as lf:
                        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                    
                    print(f"[DEBUG] Validation subprocess indítva → log: {log_path}")
                    
                    send_discord_message(
                        "TRAINING",
                        f"✅ Teljes async validation elindítva ({self.symbol} {self.timeframe}, {cur:,} lépés)."
                    )
                
                except Exception as e:
                    print(f"[WARN] Validation trigger hiba: {e}")

        except Exception as e:
            print(f"[WARN] OnBestCallback hiba: {e}")

        return True  # ⚙️ tréning folytatódjon

class TrainAndLogCallback(BaseCallback):
    """Fő logging callback: tréning közbeni metrikák gyűjtése és Discord logolás."""
    def __init__(self, log_every: int = 512, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.timesteps = []
        self.rewards = []
        self.winrates = []
        self.upnls = []
        self.last_ppo_cfg = {}
        self.target_total_steps = None
        self.end_summary_stride = 250_000
        self.last_logged = 0
        self.stat_window = 100
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.explained_variances = []

    # --- Kötelező callback metódus ---
    def _on_step(self) -> bool:
        """Ezt minden BaseCallback alosztálynál kötelező implementálni."""
        try:
            # Ha akarod, ide rakhatsz valami egyszerű logot:
            # print(f"[ROLL] Step={self.num_timesteps}")
            pass
        except Exception as e:
            print(f"[WARN] TrainAndLogCallback hiba: {e}")
        return True

    # --- belső segéd: SB3 logger metrikák egy sorban ---
    def _print_sb3_train_stats(self):
        try:
            lg = getattr(self, "logger", None)
            ntv = getattr(lg, "name_to_value", {}) if lg else {}
            if not ntv:
                return
            ent = ntv.get("train/entropy_loss") if "train/entropy_loss" in ntv else ntv.get("train/entropy", None)
            vloss = ntv.get("train/value_loss")
            pgloss = ntv.get("train/policy_gradient_loss")
            kl = ntv.get("train/approx_kl")
            clipf = ntv.get("train/clip_fraction")
            parts = []
            if ent is not None:
                parts.append(f"ent={float(ent):.4f}")
            if vloss is not None:
                parts.append(f"v={float(vloss):.4f}")
            if pgloss is not None:
                parts.append(f"pg={float(pgloss):.4f}")
            if kl is not None:
                parts.append(f"kl={float(kl):.5f}")
            if clipf is not None:
                parts.append(f"clip={float(clipf):.3f}")
            if parts:
                print("[PPO] " + " | ".join(parts))
        except Exception as e:
            print(f"[PPO DIAG ERR] {e}")
    
    def _on_step(self) -> bool:
        steps = int(self.num_timesteps)
        if steps - self.last_logged >= self.log_every:
            # Biztonságos env lekérés mind SubprocVecEnv, mind DummyVecEnv esetén
            env_u = None
            try:
                if hasattr(self.training_env, "get_attr"):
                    env_u = self.training_env.get_attr("unwrapped")[0]
                elif hasattr(self.training_env, "envs"):
                    env_u = self.training_env.envs[0].unwrapped
            except Exception:
                env_u = None
            
            winrate = float(getattr(env_u, "last_winrate", 0.0) if env_u else 0.0)
            ep_reward = float(getattr(env_u, "last_ep_reward", 0.0) if env_u else 0.0)
            last_pnl = float(getattr(env_u, "last_realized_pnl", 0.0) if env_u else 0.0)
            upnl = float(getattr(env_u, "last_unrealized_pnl", 0.0) if env_u else 0.0)
            pos = int(getattr(env_u, "position", 0) if env_u else 0)
            
            if steps % 5000 == 0 and hasattr(self.model, "logger"):
                print("[DBG] logger keys:", list(self.model.logger.name_to_value.keys())[:10])
            
            # csúszó ablak statisztikák a rewardra
            self.timesteps.append(steps)
            self.rewards.append(ep_reward)
            self.winrates.append(winrate)
            self.upnls.append(upnl)
            self.last_logged = steps
            
            r_mu = r_sigma = r_var = 0.0
            n = min(len(self.rewards), max(2, self.stat_window))
            if n >= 2:
                window = np.array(self.rewards[-n:], dtype=np.float64)
                r_mu = float(np.mean(window))
                r_sigma = float(np.std(window, ddof=1))
                r_var = float(np.var(window, ddof=1))
            
            print(
                f"[ROLL] Steps={steps} | Pos={pos} | Winrate={100 * winrate:.2f}% | "
                f"LastPnL={last_pnl:.4f} | uPnL={upnl:.4f} | EpReward={ep_reward:.4f} | "
                f"Rμ={r_mu:.4f} | Rσ={r_sigma:.4f} | RVar={r_var:.4f}"
            )
            
            # PPO metrikák egy sorban (ha elérhető)
            self._print_sb3_train_stats()
            
            # --- Mély tréning-metrikák logolása (Deep Analysis támogatás, CPU+CUDA fix) ---
            try:
                if hasattr(self, "model") and hasattr(self.model, "logger"):
                    ntv = self.model.logger.name_to_value
                    # SB3 kulcsnevek platformtól függően változhatnak
                    p_loss = ntv.get("train/policy_gradient_loss", ntv.get("train/policy_loss", np.nan))
                    v_loss = ntv.get("train/value_loss", np.nan)
                    entropy = ntv.get("train/entropy_loss", ntv.get("train/entropy", np.nan))
                    exp_var = ntv.get("train/explained_variance", np.nan)
                    
                    self.policy_losses.append(float(p_loss))
                    self.value_losses.append(float(v_loss))
                    self.entropies.append(float(entropy))
                    self.explained_variances.append(float(exp_var))
            except Exception as e:
                print(f"[WARN] DeepMetrics log error: {e}")
            
            # --- Fallback PPO buffer-alapú mentés (ha logger üres) ---
            try:
                if hasattr(self.model, "rollout_buffer"):
                    buf = self.model.rollout_buffer
                    if hasattr(buf, "advantages") and buf.advantages is not None:
                        adv_mean = float(np.nanmean(buf.advantages))
                        if np.isfinite(adv_mean):
                            self.policy_losses.append(adv_mean)
                    if hasattr(buf, "returns") and buf.returns is not None:
                        val_mean = float(np.nanmean(buf.returns))
                        if np.isfinite(val_mean):
                            self.value_losses.append(val_mean)
            except Exception as e:
                print(f"[WARN] DeepMetrics fallback log error: {e}")
        
        # --- Fallback metrikák biztos gyűjtése ---
        try:
            if hasattr(self.model, "rollout_buffer"):
                buf = self.model.rollout_buffer
                if hasattr(buf, "advantages") and buf.advantages is not None:
                    adv_mean = float(np.nanmean(buf.advantages))
                    self.policy_losses.append(adv_mean)
                if hasattr(buf, "returns") and buf.returns is not None:
                    ret_mean = float(np.nanmean(buf.returns))
                    self.value_losses.append(ret_mean)
                if hasattr(buf, "values") and buf.values is not None:
                    val_mean = float(np.nanmean(buf.values))
                    self.explained_variances.append(val_mean)
                if hasattr(buf, "log_probs") and buf.log_probs is not None:
                    ent_mean = -float(np.nanmean(buf.log_probs))
                    self.entropies.append(ent_mean)
        except Exception as e:
            print(f"[WARN] DeepMetrics extended fallback error: {e}")
        
        # --- DIAG minták és statok ritkábban ---
        if self.n_calls % 1000 == 0:
            try:
                rewards = self.locals.get("rewards")
                if isinstance(rewards, np.ndarray) and rewards.size > 0:
                    print(f"[DIAG] Reward mean={rewards.mean():.6f}, std={rewards.std():.6f}")
                    print(f"[DIAG] Reward sample={rewards[:10]}")
                values = self.locals.get("values")
                if isinstance(values, np.ndarray) and values.size > 0:
                    print(f"[DIAG] Value pred mean={values.mean():.6f}, std={values.std():.6f}")
                    print(f"[DIAG] Value sample={values[:10]}")
                advantages = self.locals.get("advantages")
                if isinstance(advantages, np.ndarray) and advantages.size > 0:
                    print(f"[DIAG] Adv mean={advantages.mean():.6f}, std={advantages.std():.6f}")
                logs = self.locals.get("log_probs")
                if isinstance(logs, np.ndarray) and logs.size > 0:
                    print(f"[DIAG] LogProb mean={logs.mean():.6f}, std={logs.std():.6f}")
            except Exception as e:
                print(f"[DIAG ERR] {e}")
            
            if steps % 10000 == 0:
                try:
                    print(
                        f"[DBG-DEEP] steps={steps:,} | "
                        f"policy={len(self.policy_losses)} (μ={np.nanmean(self.policy_losses):.4f}), "
                        f"value={len(self.value_losses)} (μ={np.nanmean(self.value_losses):.4f}), "
                        f"entropy={len(self.entropies)} (μ={np.nanmean(self.entropies):.4f}), "
                        f"explVar={len(self.explained_variances)} (μ={np.nanmean(self.explained_variances):.4f})"
                    )
                except Exception as e:
                    print(f"[DBG-DEEP-ERR] {e}")
                    # --- DeepMetrics Discord summary (100k-onként) ---
                    if steps % 100000 == 0 and steps > 0:
                        try:
                            pol_mean = float(np.nanmean(self.policy_losses)) if len(self.policy_losses) else 0.0
                            val_mean = float(np.nanmean(self.value_losses)) if len(self.value_losses) else 0.0
                            ent_mean = float(np.nanmean(self.entropies)) if len(self.entropies) else 0.0
                            exp_mean = float(np.nanmean(self.explained_variances)) if len(
                                self.explained_variances) else 0.0
                            
                            msg = (
                                f"🧠 **Deep Metrics Summary @ {steps:,} steps**\n"
                                f"• Policy loss μ = {pol_mean:.6f}\n"
                                f"• Value loss μ = {val_mean:.6f}\n"
                                f"• Entropy μ = {ent_mean:.6f}\n"
                                f"• Explained Var μ = {exp_mean:.6f}"
                            )
                            send_discord_message("TRAINING", msg)
                            print(f"[DISCORD] Deep metrics summary sent ({steps:,})")
                        except Exception as e:
                            print(f"[WARN] DeepMetrics Discord summary error: {e}")
        return True
        
    def _on_rollout_end(self) -> None:
        """Runtime PPO paraméter frissítés JSON-ból (floodmentes, tiszta verzió)."""
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    patch = json.load(f)
                
                if "ppo_params" in patch:
                    ppo_cfg = patch["ppo_params"]
                    
                    # Csak akkor frissít, ha tényleges változás történt
                    if ppo_cfg != self.last_ppo_cfg:
                        updated = []
                        
                        if "learning_rate" in ppo_cfg:
                            lr = float(ppo_cfg["learning_rate"])
                            self.model.learning_rate = lr
                            self.model.lr_schedule = get_schedule_fn(lr)
                            updated.append(f"lr={lr}")
                        
                        if "ent_coef" in ppo_cfg:
                            self.model.ent_coef = float(ppo_cfg["ent_coef"])
                            updated.append(f"ent={ppo_cfg['ent_coef']}")
                        
                        if "clip_range" in ppo_cfg:
                            val = float(ppo_cfg["clip_range"])
                            self.model.clip_range = lambda _: val
                            updated.append(f"clip={val}")
                        
                        if "gamma" in ppo_cfg:
                            self.model.gamma = float(ppo_cfg["gamma"])
                            updated.append(f"gamma={ppo_cfg['gamma']}")
                        
                        # Logolás, ha tényleg volt változás
                        if updated:
                            print(f"[PPO] Paraméter frissítve: {', '.join(updated)}")
                        
                        self.last_ppo_cfg = dict(ppo_cfg)
        
        except Exception as e:
            print(f"[WARN] PPO param frissítés hiba: {e}")
        
        # --- Clip ratio frissítése, ha az env tartalmazza ---
        try:
            if hasattr(self, "env") and hasattr(self.env, "envs"):
                env0 = self.env.envs[0]
                if hasattr(env0, "_info_cache") and "clip_ratio" in env0._info_cache:
                    clip_ratio = env0._info_cache.get("clip_ratio", 0.0)
        except Exception:
            pass
        
        # --- Exploration Burst Monitor ---
        # IDE hints to silence unresolved reference warnings
        steps: int = getattr(self, "steps", 0)
        ep_reward: float = locals().get("ep_reward", 0.0)
        r_sigma: float = locals().get("r_sigma", 0.0)
        
        if steps % 5000 == 0:
            try:
                # Reward és szórás stagnál -> növeljük az entropiát (felfedezés)
                if abs(ep_reward) < 0.001 and r_sigma < 1.0:
                    old_ent = getattr(self.model.policy, "ent_coef", None)
                    if old_ent is not None:
                        self.model.policy.ent_coef = min(float(old_ent) * 2.0, 0.02)
                        print(
                            f"[BOOST] Exploration burst aktiválva! ent_coef {old_ent:.4f} → {self.model.policy.ent_coef:.4f}")
                # Ha újra van mozgás -> visszaállítjuk
                elif r_sigma > 10.0 and getattr(self.model.policy, "ent_coef", 0) > 0.011:
                    old_ent = self.model.policy.ent_coef
                    self.model.policy.ent_coef = max(float(old_ent) * 0.8, 0.01)
                    print(f"[BOOST] Exploration burst leállítva, ent_coef visszaáll {self.model.policy.ent_coef:.4f}")
            except Exception as e:
                print(f"[BOOST] Exploration monitor hiba: {e}")
        
        # --- következő eredeti sor a fájlban ---
        self._print_sb3_train_stats()
    
    def _on_training_end(self) -> None:
        
        # --- Flood-védelem: csak 100k-nként vagy a legvégén fusson ---
        try:
            cur = int(getattr(self.model, "num_timesteps", 0))
            total = int(getattr(self, "target_total_steps", 0))
            stride = int(getattr(self, "chart_mod_stride", 100_000))
            # ha NEM a legvége és NEM 100k többszöröse, akkor kilépünk
            if total and cur < total and (cur % stride != 0):
                return
        except Exception:
            # hiba esetén inkább visszalépünk, mint floodoljunk
            return
        
        # Végső modell/grafikon/összefoglaló – RITKÍTVE
        try:
            cur = int(self.model.num_timesteps)
            # csak akkor fusson, ha 250k-nál vagy a teljes tréning végén vagyunk
            is_final_total = (self.target_total_steps is not None and cur >= int(self.target_total_steps))
            if not is_final_total:
                if self.end_summary_stride is None or self.end_summary_stride <= 0 or (
                        cur % int(self.end_summary_stride) != 0):
                    return  # túl sűrű lenne, kihagyjuk
            
            # ✅ Biztonságos environment-lekérés (SubprocVecEnv és DummyVecEnv kompatibilis)
            try:
                if hasattr(self.training_env, "get_attr"):
                    env_u = self.training_env.get_attr("unwrapped")[0]
                elif hasattr(self.training_env, "envs"):
                    env_u = self.training_env.envs[0].unwrapped
                else:
                    env_u = None
            except Exception:
                env_u = None
            # ✅ Szimbólum és timeframe lekérés biztonságosan
            sym = getattr(env_u, "symbol", "UNK") if env_u else "UNK"
            tf = getattr(env_u, "timeframe", "UNK") if env_u else "UNK"
            
            # NEM mentünk itt modellt (a végén a fő függvény úgyis ment) – csak grafikon + összefoglaló
            if self.timesteps and self.rewards:
                chart_path = os.path.join(MODELS_DIR, f"final_training_chart_{sym}_{tf}.png")
                try:
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    
                    # 🎯 Reward + Winrate görbék
                    ax1.plot(self.timesteps, self.rewards, label="EpReward", alpha=0.4, color="tab:blue")
                    if len(self.rewards) >= 50:
                        smoothed = np.convolve(self.rewards, np.ones(50) / 50, mode="valid")
                        ax1.plot(self.timesteps[49:], smoothed, label="EpReward (avg50)", color="tab:orange")
                    ax1.plot(self.timesteps, self.winrates, label="Winrate %", alpha=0.4, color="tab:green")
                    if len(self.winrates) >= 50:
                        sm_wr = np.convolve(self.winrates, np.ones(50) / 50, mode="valid")
                        ax1.plot(self.timesteps[49:], sm_wr, label="Winrate (avg50)", color="tab:red")
                    
                    ax1.set_xlabel("Timesteps")
                    ax1.set_ylabel("Reward / Winrate")
                    
                    # 💚 uPnL (floating profit) – második tengely
                    ax2 = None
                    try:
                        if hasattr(self, "upnls") and len(self.upnls) == len(self.timesteps):
                            ax2 = ax1.twinx()
                            ax2.plot(self.timesteps, self.upnls, color="limegreen", alpha=0.5, label="uPnL (floating)")
                            ax2.set_ylabel("uPnL", color="limegreen")
                            ax2.tick_params(axis="y", labelcolor="limegreen")
                    except Exception as e:
                        print(f"[WARN] uPnL plot skipped (final): {e}")
                    
                    # 🧩 Közös legenda az összes görbéhez
                    lines, labels = ax1.get_legend_handles_labels()
                    if ax2 is not None:
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        lines += lines2
                        labels += labels2
                    ax1.legend(lines, labels, loc="best")
                    
                    # 🏷️ Cím az átláthatóságért
                    plt.title(f"Final Training Chart – {sym} {tf}")
                    plt.tight_layout()
                    plt.savefig(chart_path)
                    plt.close()
                
                except Exception as e:
                    print(f"[WARN] uPnL/final plot error: {e}")
                
                # --- ez a rész már a try után legyen, ne alatta! ---
                avg_entropy = 0.0
                try:
                    lg = getattr(self, "logger", None)
                    ntv = getattr(lg, "name_to_value", {}) if lg else {}
                    ent_vals = [v for k, v in ntv.items() if "entropy_loss" in k or k.endswith("/entropy")]
                    if ent_vals:
                        avg_entropy = float(np.mean(ent_vals))
                except Exception:
                    pass
                
                summary = f"""📊 [TRAIN-END]
                • Timesteps: {cur}
                • Utolsó Winrate: {100 * self.winrates[-1]:.2f}%
                • Utolsó EpReward: {self.rewards[-1]:.4f}
                • Átlag Winrate: {100 * np.mean(self.winrates):.2f}%
                • Átlag EpReward: {np.mean(self.rewards):.4f}
                • Átlag Entropy: {avg_entropy:.2f}
                """
                try:
                    send_discord_file(channel="TRAINING", file_path=chart_path, content=summary)
                except Exception as e:
                    send_discord_message("TRAINING", f"[WARN] Grafikon mentés/küldés hiba: {e}")
            
            else:
                send_discord_message("TRAINING", "[WARN] Training-end: nincs adat a grafikonhoz.")
        
        except Exception as e:
            send_discord_message("TRAINING", f"[WARN] Training-end hiba: {e}")
        
        # --- Post-Training Checklist automatikus futtatás ---
        try:
            send_discord_message("TRAINING", "[INFO] Post-training checklist automatikusan indul...")
            checklist_path = os.path.join(os.getcwd(), "post_training_checklist.bat")
            
            # --- Failsafe: ha nincs docs/checklist, automatikusan létrehozza ---
            docs_dir = os.path.join(os.getcwd(), "docs")
            checklist_txt = os.path.join(docs_dir, "post_training_checklist.txt")
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir, exist_ok=True)
            if not os.path.exists(checklist_txt):
                with open(checklist_txt, "w", encoding="utf-8") as f:
                    f.write("✅ Quant_AI Post-Training Checklist automatikusan létrehozva.\n")
                print(f"[AUTO] Hiányzó checklist.txt létrehozva: {checklist_txt}")
            
            if os.path.exists(checklist_path):
                # automatikus futtatás, új ablakban
                subprocess.Popen(["cmd", "/c", checklist_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
                print(f"[POST] Post-training checklist futtatva: {checklist_path}")
            else:
                print(f"[WARN] post_training_checklist.bat nem található a gyökérben.")
                send_discord_message("TRAINING", "[WARN] Checklist nem található a gyökérkönyvtárban.")
        except Exception as e:
            print(f"[WARN] Checklist futtatási hiba: {e}")
            send_discord_message("TRAINING", f"[WARN] Checklist futtatási hiba: {e}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_agent(
    timeframe: str,
    symbol: str,
    timesteps: int,
    reward_mode: str,
    resume_path: str | None = None,
    device: str = "cpu",
    num_envs: int = 4,  # 🧩 új paraméter, SubprocVecEnv környezetek száma
):
    global os  # <--- ez oldja meg a shadowing problémát
    model = None  # safety init (prevents UnboundLocalError on failed resume)
    
    import torch
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"[INIT] CUDA device lock: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # --- Thread limitek a stabil FPS miatt ---
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    try:
        import torch as th
        th.set_num_threads(1)
    except Exception:
        pass
    
    # --- LATE IMPORTS: csak a main process tölti be, worker nem ---
    from core.utils.env import load_env_once
    from core.utils.discord import send_discord_message, send_discord_file
    from core.rl.envs.mtf_env_full import MTFEnvFull
    
    load_env_once()
    send_discord_message("TRAINING", f"[INIT] Training start: {symbol} {timeframe} ({reward_mode})")
    
    # Load data
    df = pd.read_csv(f"data/ohlcv/ohlcv_{timeframe}_{symbol}.csv")
    
    # --- OS-aware, pickle-safe env factory ---
    import platform
    
    # fontos: fájlútvonal, hogy a subproc újra be tudja olvasni
    df_path = f"data/ohlcv/ohlcv_{timeframe}_{symbol}.csv"
    
    def make_env_factory(df_path: str, reward_mode: str, symbol: str, timeframe: str):
        def _init():
            from core.rl.envs.mtf_env_full import MTFEnvFull
            return MTFEnvFull(
                csv_path=df_path,
                reward_mode=reward_mode,
                symbol=symbol,
                timeframe=timeframe,
                debug=True,  # ✅ Debug mód: CHUNK betöltés logok láthatók lesznek
            )
        
        return _init
    
    # --- Environment setup (SubprocVecEnv / DummyVecEnv auto) ---
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    
    if num_envs > 1:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            env = SubprocVecEnv([
                make_env_factory(df_path, reward_mode, symbol, timeframe)
                for _ in range(num_envs)
            ], start_method='spawn')
            print(f"[ENV] SubprocVecEnv ({num_envs} szál) aktiválva (spawn).")
        except Exception as e:
            from stable_baselines3.common.vec_env import DummyVecEnv
            env = DummyVecEnv([make_env_factory(df_path, reward_mode, symbol, timeframe)])
            print(f"[ENV WARN] SubprocVecEnv hiba ({type(e).__name__}: {e}) → DummyVecEnv fallback.")
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_env_factory(df_path, reward_mode, symbol, timeframe)])
        print("[ENV] DummyVecEnv aktiválva (single-thread).")
    
    # === VecNormalize setup (egységesített, QA-verzió, JSON + ENV támogatással) ===
    from stable_baselines3.common.vec_env import VecNormalize
    
    # 1️⃣ JSON config betöltése (ha van)
    cfg_path = os.path.join("core", "rl", "envs", "reward_patch_config.json")
    use_vecnorm = globals().get("use_vecnorm", True)
    clip_obs = 10.0
    norm_reward = True
    
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            vcfg = cfg.get("vecnorm", {})
            # --- Extra fallback: Reward Tuner által létrehozott 'vecnormalize' blokk is támogatott ---
            if not vcfg and "vecnormalize" in cfg:
                vcfg = cfg.get("vecnormalize", {})
            if isinstance(vcfg, dict):
                use_vecnorm = bool(vcfg.get("enabled", True))
                clip_obs = float(vcfg.get("clip_obs", 10.0))
                norm_reward = bool(vcfg.get("norm_reward", True))
            elif isinstance(vcfg, bool):
                use_vecnorm = vcfg
            print(
                f"[CFG] VecNormalize config betöltve: enabled={use_vecnorm}, clip_obs={clip_obs}, norm_reward={norm_reward}")
        except Exception as e:
            print(f"[WARN] live_params.json olvasási hiba: {e}")
        
        # --- [SYNC] VecNormalize flag összehangolása reward_patch_config.json-nal ---
        try:
            import json
            patch_path = os.path.join("core", "rl", "envs", "reward_patch_config.json")
            if os.path.exists(patch_path):
                with open(patch_path, "r", encoding="utf-8") as f:
                    patch_cfg = json.load(f)
                if reward_mode in patch_cfg and "vecnormalize" in patch_cfg[reward_mode]:
                    use_vecnorm = bool(patch_cfg[reward_mode]["vecnormalize"])
                    print(f"[SYNC] VecNormalize flag átvéve reward_patch_config-ból → {use_vecnorm}")
        except Exception as e:
            print(f"[SYNC-ERR] Reward patch config sync hiba: {e}")
    
    # 2️⃣ CLI/ENV override
    if os.getenv("VECNORM", "").lower() in ("1", "true", "on"):
        use_vecnorm = True
        print("[CFG] VecNormalize kényszerítve CLI/ENV alapján.")
    
    # 3️⃣ Betöltés vagy új VecNormalize létrehozása
    vec_path = os.path.join(MODELS_DIR, f"vecnormalize_{symbol}_{timeframe}.pkl")
    
    if use_vecnorm:
        if os.path.exists(vec_path):
            try:
                env = VecNormalize.load(vec_path, env)
                env.training = True
                env.norm_reward = norm_reward
                print(f"[VECNORM] Betöltve meglévő statok: {vec_path}")
            except Exception as e:
                print(f"[WARN] VecNormalize betöltés sikertelen: {e}")
                env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs)
                print("[VECNORM] Új VecNormalize létrehozva (fallback).")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs)
            print("[VECNORM] Új VecNormalize létrehozva (norm_obs + norm_reward).")
    else:
        print("[VECNORM] Kikapcsolva (JSON/CLI alapján) – raw környezet fut.")
    
    # 4️⃣ VecNormalize mentése checkpointnál
    def save_vecnorm_state(env, step=None):
        """VecNormalize statok mentése checkpointtal együtt"""
        try:
            if isinstance(env, VecNormalize):
                suffix = f"_{step}" if step else ""
                vec_path = os.path.join(MODELS_DIR, f"vecnormalize_{symbol}_{timeframe}{suffix}.pkl")
                env.save(vec_path)
                print(f"[OK] VecNormalize statok mentve: {vec_path}")
        except Exception as e:
            print(f"[WARN] VecNormalize mentés sikertelen: {e}")
        
        # Példa hívás checkpoint mentésnél:
        # save_vecnorm_state(env, step=cur_step)
        
        # --- Curriculum Scheduler init (env után, csak ha szükséges) ---
        scheduler = None
        if str(reward_mode).lower().startswith("curriculum"):
            try:
                from core.rl.curriculum_scheduler import get_scheduler
                scheduler = get_scheduler(send_fn=send_discord_message)
                scheduler.update_if_needed(env, 0)
                print("[CURRICULUM] Scheduler aktív.")
            except Exception as e:
                print(f"[CURRICULUM] Betöltési hiba: {e}")
        else:
            print(f"[INFO] Static reward mode active → {reward_mode}.json")
        
        # --- Curriculum toggling (example) ---
    prefer_trend_cutoff = 50_000
    if timesteps >= prefer_trend_cutoff:
        try:
            env.envs[0].prefer_trend = True
            print("[CURRICULUM] prefer_trend = True (első 50k timesteps)")
        except Exception:
            pass
    
    # --- Device & cuDNN optimalizáció ---
    import torch as th
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    if device == "cuda":
        th.backends.cudnn.benchmark = True  # 🔥 dinamikus kernel-optimalizáció (FPS boost)
        th.backends.cudnn.deterministic = False
        print("[DEBUG] cuDNN benchmark ON — optimal kernel search enabled.")
    
    # --- Device beállítás (GPU ha elérhető) ---
    import torch as th
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"[INIT] Training device: {device.upper()}")
    
    # --- Model init / resume unified handler ---
    if resume_path and os.path.exists(resume_path):
        import os
        print(f"[DEBUG] Resume path: {resume_path}")
        
        # --- Plateau stop.flag ellenőrzés és automatikus törlés ---
        if os.path.exists("stop.flag"):
            try:
                os.remove("stop.flag")
                print("[CLEANUP] Régi stop.flag törölve (resume safe).")
            except Exception as e:
                print(f"[WARN] stop.flag törlés sikertelen: {e}")
                
        # --- Model betöltése ---
        try:
            model = PPO.load(resume_path, env=env, device=device)
            print(f"[DEBUG] model.num_timesteps (load után): {model.num_timesteps}")
            
            # --- Timesteps metaadat visszatöltése ---
            meta_path = resume_path.replace(".zip", "_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    tsteps = int(meta.get("timesteps", 0))
                    model.num_timesteps = tsteps
                
                    # --- PPO belső timesteps fix (deep resume) ---
                    import time
                    if hasattr(model, "logger"):
                        try:
                            model.logger.start_time = time.time()  # reset training clock
                            model._total_timesteps = tsteps  # internal SB3 counter
                            model._num_timesteps = tsteps
                            print(f"[PATCH] PPO belső timesteps beállítva: {tsteps}")
                        except Exception as e:
                            print(f"[WARN] Belső timesteps fix sikertelen: {e}")
                    
                    print(f"[RESUME] Folytatás betöltve: {resume_path} (steps={tsteps})")
                
                except Exception as e:
                    print(f"[WARN] Metaadat olvasás sikertelen ({e})")
            else:
                print(f"[WARN] Metaadat fájl nem található ({meta_path})")
        
        except Exception as e:
            print(f"[WARN] Model betöltése sikertelen ({e}) → új modell inicializálva.")
            model = PPO(
                "MlpPolicy", env, verbose=1,
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.0, vf_coef=0.5,
                device=device,
            )
                    
            # --- Biztonsági guard (NEM hoz létre új modellt) ---
            if model is None:
                print("[WARN] Model objektum None maradt — de újraindítás nélkül folytatni nem lehet.")
                return
        
        
        # --- Restore timestep counter if available ---
        if hasattr(model, "num_timesteps") and model.num_timesteps == 0:
            old_steps = 0
            try:
                import re
                # Elfogad bármilyen mintát: _ckpt_123456, _final_123456, _resume_123456, _fixed_123456.zip stb.
                m = re.search(r"_(\d+)(?=[^0-9]*\.zip$)", os.path.basename(resume_path))
                if m:
                    old_steps = int(m.group(1))
                    print(f"[RESUME] Lépésszám kinyerve fájlnévből: {old_steps:,}")
                else:
                    print(f"[WARN] Nem sikerült lépésszámot kinyerni a fájlnévből: {resume_path}")
            except Exception as e:
                print(f"[WARN] Regex parse hiba: {e}")
            
            model._total_timesteps = old_steps
            model.num_timesteps = old_steps
            print(f"[RESUME] Korábbi timesteps visszaállítva: {old_steps:,}")
        else:
            print(f"[RESUME] Model már tartalmaz timesteps értéket: {getattr(model, 'num_timesteps', 'N/A')}")
        
        # --- Discord log ---
        try:
            send_discord_message("TRAINING", f"[RESUME] Folytatás betöltve: {resume_path}")
            print(f"[OK] Discord üzenet elküldve a TRAINING csatornára.")
        except Exception as e:
            print(f"[WARN] Discord üzenet küldése sikertelen: {e}")
    
    else:
        # --- Ha nincs modell (új indulás) ---
        print(f"[WARN] Model file not found or resume disabled → új modell inicializálva.")
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5,
            device=device,
        )
        print("[OK] Fresh PPO model created.")
        send_discord_message("TRAINING", "[MODEL] Új PPO modell inicializálva (no resume).")
    
    # ✅ GPU optimalizációs beállítások
    import torch
    torch.backends.cudnn.benchmark = True
    print("[DEBUG] cuDNN benchmark ON — optimal kernel search enabled.")
    print("[DEBUG] Policy device:", next(model.policy.parameters()).device)
    
    # --- EVAL ENV (ugyanolyan típus, mint a training env) ---
    def make_eval_env():
        from core.rl.envs.mtf_env_full import MTFEnvFull
        def _init():
            # fontos: csv_path-ot használunk, hogy a subprocess tudjon olvasni
            env_e = MTFEnvFull(
                csv_path=df_path,
                reward_mode=reward_mode,
                symbol=symbol,
                timeframe=timeframe,
            )
            return Monitor(env_e)
        
        return _init
    
    if num_envs > 1:
        eval_env = SubprocVecEnv([make_eval_env() for _ in range(num_envs)], start_method="spawn")
    else:
        eval_env = DummyVecEnv([make_eval_env()])
    
    # --- 🔧 VecNormalize sync javítás ---
    from stable_baselines3.common.vec_env import VecNormalize
    try:
        if "env" in locals() and isinstance(env, VecNormalize):
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("[EVAL] VecNormalize wrap létrehozva az eval_env-hez is.")
        else:
            print("[EVAL] VecNormalize wrap kihagyva (train env nem normalizált).")
    except Exception as e:
        print(f"[EVAL WARN] VecNormalize wrap sikertelen az eval_env-nél: {e}")
    
    # Eval callback
    eval_best_dir = os.path.join(MODELS_DIR, "best")
    eval_log_dir = os.path.join(REPORTS_DIR, "eval")
    os.makedirs(eval_best_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    import threading
    
    # async Discord wrapper
    def async_discord(channel, msg):
        threading.Thread(target=send_discord_message, args=(channel, msg), daemon=True).start()
    
    # =====================================================================
    # === EvalCallback + Discord-integrált részleges validáció riport ===
    # =====================================================================
    
    eval_best_dir = os.path.join(MODELS_DIR, "best")
    eval_log_dir = os.path.join(REPORTS_DIR, "eval")
    os.makedirs(eval_best_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    import threading
    from stable_baselines3.common.callbacks import EvalCallback
    from core.utils.discord import send_discord_message
    
    # --- Async Discord wrapper (ne blokkoljon tréning közben) ---
    def async_discord(channel, msg):
        threading.Thread(target=send_discord_message, args=(channel, msg), daemon=True).start()
    
    # --- Discord-integrált EvalCallback osztály ---
    class DiscordEvalCallback(EvalCallback):
        """EvalCallback kiegészítve Discord-riporttal (WR + mean reward)"""
        
        def __init__(self, *args, symbol=None, timeframe=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.symbol = symbol
            self.timeframe = timeframe
        
        def _on_step(self) -> bool:
            result = super()._on_step()
            # csak ha ténylegesen lefutott az eval és van értelmes reward
            if (
                    self.last_mean_reward is not None
                    and self.last_mean_reward != 0
                    and self.n_calls % self.eval_freq == 0
            ):
                try:
                    msg = (
                        f"📊 **[INTERMEDIATE VALIDATION]**\n"
                        f"Symbol: {self.symbol} | TF: {self.timeframe}\n"
                        f"WR≈{self.last_mean_reward:+.4f} | "
                        f"Eval every {self.eval_freq:,} steps"
                    )
                    async_discord("TRAINING", msg)
                    print(f"[EVAL-CB] Discord validáció üzenet elküldve ({self.symbol} {self.timeframe})")
                except Exception as e:
                    print(f"[WARN] DiscordEvalCallback send error: {e}")
            return result
    
    # --- EvalCallback inicializálása (Discord verzió) ---
    eval_callback = DiscordEvalCallback(
        eval_env,
        best_model_save_path=f"models/best_{symbol}_{timeframe}",
        log_path=f"models/logs_{symbol}_{timeframe}",
        eval_freq=50_000,  # minden 50k lépés után validál
        deterministic=True,
        render=False,
        callback_on_new_best=OnBestCallback(),
        symbol=symbol,
        timeframe=timeframe,
    )
    
    # --- Validation callback (Smart Validation számára) ---
    validation_cb = ValidationCallback(
        freq=50_000,
        symbol=symbol,
        timeframe=timeframe,
        timesteps=timesteps,
        reward_mode=reward_mode,
        device=device,
        start_step=getattr(model, "num_timesteps", 0)  # ✅ ez marad!
    )
    
    # ---------------- EVAL_MODE flag: ensure eval episodes can end ----------------
    os.environ["EVAL_MODE"] = "1"
    
    
    # --- Training loop + checkpoints + checkpoint chart ---
    total_steps = timesteps
    chunk = 2048
    done = 0
    ckpt_path = None  # <-- add ide
    chart_path = None  # <-- EZ A FONTOS
    
    # --- Inicializálás a statisztikákhoz (prevent undefined vars) ---
    mean_wr = 0.0
    mean_rw = 0.0
    mean_upnl = 0.0
    std_rw = 0.0
    
    # Validation callback (async validáció 100k-nként)
    validation_cb = ValidationCallback(
        freq=100_000,
        symbol=symbol,
        timeframe=timeframe,
        timesteps=total_steps,
        reward_mode=reward_mode,
        device=device
    )
    
    # Logging callback (console + chart + ppo reload)
    log_callback = TrainAndLogCallback(log_every=512)
    log_callback.target_total_steps = int(total_steps)  # a teljes cél (pl. 1_000_000)
    log_callback.chart_mod_stride = 100_000  # csak minden 100k-nál chartoljon
    callbacks = CallbackList([eval_callback, validation_cb, log_callback])
    
    # 📊 validációs és grafikon gyakoriság a reward configból
    validation_stride = validation_freq  # pl. 100_000
    chart_stride = validation_freq * 2  # pl. 200_000
    next_validation = validation_stride  # első validáció 100k-nál
    next_chart = chart_stride  # első chart 200k-nál
    
    print(f"[INIT] Stabil tréningciklus indítása — cél: {total_steps:,} steps")
    
    try:
        while int(model.num_timesteps) < total_steps:
            
            # === Validation + Chart trigger (ritkítva) ===
            cur = int(model.num_timesteps)
            
            # Ha több stride‑ot is átugrott, mindet pótoljuk
            while cur >= next_validation:
                print(f"[VALID] Validation trigger {cur:,} lépésnél…")
                try:
                    cmd = [
                        sys.executable, "-m", "scripts.run_validation",
                        "--symbol", str(self.symbol),
                        "--timeframe", str(self.timeframe),
                        "--device", str(self.device),
                        "--reward", str(self.reward_mode or "realistic"),
                        "--timesteps", str(cur),
                        "--mode", "full",  # ✅ teljes validáció
                        "--dataset", "val",  # ✅ ha van külön validációs halmaz
                        "--auto_close_eval", "true",
                        "--deterministic", "true"
                    ]
                    
                    # 🔹 LOG fájl létrehozása, hogy semmi ne vesszen el
                    os.makedirs("reports", exist_ok=True)
                    log_path = os.path.join(
                        "reports",
                        f"validation_{self.symbol}_{self.timeframe}_{int(cur)}.log"
                    )
                    
                    # 🔹 Subprocess indítása a logfájllal -- így minden output megmarad
                    with open(log_path, "w", encoding="utf-8") as lf:
                        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                    
                    # 🔹 Konzol és Discord visszajelzések
                    print(f"[DEBUG] Validation subprocess indítva → log: {log_path}")
                    send_discord_message(
                        "TRAINING",
                        f"✅ Teljes async validation elindítva ({self.symbol} {self.timeframe}, {cur:,} lépés)"
                    )
                
                except Exception as e:
                    print(f"[WARN] Validation trigger hiba: {e}")
                    send_discord_message(
                        "TRAINING",
                        f"⚠️ Validation trigger hiba: {e}"
                    )
                
                # *mindig léptesd tovább, hogy a következő 100 k‑nál újra fusson*
                next_validation += validation_stride
            
            # ✅ Első biztonsági stop — ha elérte a limitet
            if int(model.num_timesteps) >= total_steps:
                print(f"[STOP] Elérte a megadott {total_steps:,} lépést — tréning lezárva.")
                break
                
            # --- Periodikus validáció + checkpoint + Discord summary ---
            if int(model.num_timesteps) >= next_chart and int(model.num_timesteps) < total_steps:
                try:
                    current_steps = int(model.num_timesteps)
                    print(f"[POST] {current_steps:,} lépés után grafikon + validáció...")
                    send_discord_message("TRAINING", f"📊 {current_steps:,} lépés után grafikon + validáció...")
                    
                    # --- Validation futtatás ---
                    cmd = [
                        sys.executable, "-m", "scripts.run_validation",
                        "--symbol", str(self.symbol),
                        "--timeframe", str(self.timeframe),
                        "--device", str(self.device),
                        "--reward", str(self.reward_mode or "realistic"),
                        "--timesteps", str(cur),
                        "--mode", "full",  # ✅ a "fast" helyett teljes validációt kérünk
                        "--dataset", "val",  # ✅ ha van külön validációs halmaz
                        "--auto_close_eval", "true",
                        "--deterministic", "true"
                    ]
                    os.system(" ".join(cmd))
                    
                    send_discord_message("TRAINING", f"✅ Validation lefutott {current_steps:,} lépés után.")
                    
                    # --- Validation után részletes Discord összesítő ---
                    try:
                        mean_wr = 100 * np.mean(getattr(log_callback, "winrates", [0]))
                        mean_rw = np.mean(getattr(log_callback, "rewards", [0]))
                        mean_upnl = np.mean(getattr(log_callback, "upnls", [0]))
                        std_rw = np.std(getattr(log_callback, "rewards", [0]))
                        
                        msg = (
                                f"📊 [VALIDATION SUMMARY]\n"
                                f"• Symbol: {symbol}\n"
                                f"• Timeframe: {timeframe}\n"
                                f"• Steps: {current_steps:,}\n"
                                f"• Winrate: {mean_wr:.2f}%\n"
                                f"• AvgReward: {mean_rw:.4f}\n"
                                f"• Avg uPnL: {mean_upnl:.2f}\n"
                                f"• Reward σ: {std_rw:.4f}"
                        )
                        send_discord_message("TRAINING", msg)
                        print(msg)
                    except Exception as e:
                        print(f"[WARN] Validation summary Discord hiba: {e}")
                    
                    # --- Checkpoint mentés 50k-nként vagy a végén ---
                    if current_steps % 50_000 == 0 or current_steps >= total_steps:
                        ckpt_path = os.path.join(
                            MODELS_DIR,
                            f"ppo_{symbol}_{timeframe}_ckpt_{current_steps}.zip"
                        )
                        try:
                            model.save(ckpt_path)
                            print(f"[CKPT ✅] Checkpoint mentve: {ckpt_path}")
                        except Exception as e:
                            print(f"[CKPT ⚠️] Mentés sikertelen: {e}")
                    
                    # --- VecNormalize állapot mentése ---
                    try:
                        from stable_baselines3.common.vec_env import VecNormalize
                        
                        # --- Multi-level env lookup ---
                        venv = getattr(env, "venv", getattr(env, "env", env))
                        if isinstance(venv, VecNormalize):
                            vec_path_final = os.path.join("models", f"vecnormalize_{symbol}_{timeframe}.pkl")
                            venv.save(vec_path_final)
                            print(f"[QA] VecNormalize final mentve: {vec_path_final}")
                        else:
                            print(f"[QA] VecNormalize inaktív ({type(venv).__name__}) – nincs mentés.")
                    except Exception as e:
                        print(f"[QA-ERROR] VecNormalize mentés sikertelen: {e}")
                    
                    # --- Timesteps metaadat mentése ---
                    try:
                        meta_path = ckpt_path.replace(".zip", "_meta.json")
                        with open(meta_path, "w") as f:
                            json.dump({"timesteps": int(model.num_timesteps)}, f)
                        print(f"[META] Timesteps metaadat mentve: {meta_path}")
                    except Exception as e:
                        print(f"[WARN] Metaadat mentés hiba: {e}")
                    
                    # --- Következő validáció lépés beállítása ---
                    next_chart += validation_freq
                
                except Exception as e:
                    print(f"[WARN] Post-validation blokk hiba: {e}")
                    
            # --- Tanulási ciklus chunkokra bontva ---
            remaining = total_steps - int(model.num_timesteps)
            this_chunk = min(chunk, remaining)
            model.learn(total_timesteps=this_chunk, reset_num_timesteps=False, callback=callbacks)
            
            current_steps = int(model.num_timesteps)
            print(f"[LOOP] Aktuális timesteps: {current_steps:,} / {total_steps:,}")
            
            # 🔁 Curriculum frissítés fázisonként (csak ha aktív)
            try:
                if 'scheduler' in locals() and scheduler is not None:
                    scheduler.update_if_needed(env, current_steps)
            except Exception as e:
                print(f"[CURRICULUM] update_if_needed hiba: {e}")
            
            # === QA Diagnosztikai monitor ===
            try:
                # --- Loop drift detektor ---
                drift = abs(done - int(model.num_timesteps))
                if drift > 512:
                    send_diag(
                        f"⚠️ LOOP DRIFT DETECTED: done={done:,}, model.num_timesteps={model.num_timesteps:,}"
                    )
            except Exception as e:
                print(f"[DIAG] Loop drift check hiba: {e}")
                
                # --- NaN/inf metrika detektor ---
                if any(np.isnan(x).any() for x in [
                    np.array(getattr(log_callback, "rewards", [])),
                    np.array(getattr(log_callback, "winrates", []))
                ]):
                    send_diag("⚠️ NaN érték észlelve reward vagy winrate metrikában!")
                
                # --- Entropy trend monitor ---
                if hasattr(log_callback, "entropies") and len(log_callback.entropies) > 20:
                    ent_first = np.nanmean(log_callback.entropies[:10])
                    ent_last = np.nanmean(log_callback.entropies[-10:])
                    delta_ent = ent_last - ent_first
                    if delta_ent < -0.1:
                        send_diag(f"⚠️ ENTROPY DROP: Δ={delta_ent:.4f}  (lehetséges overfit trend)")
                
                # --- Variancia monitor ---
                rewards = np.array(getattr(log_callback, "rewards", []))
                if len(rewards) > 50:
                    r_std = np.nanstd(rewards[-50:])
                    if r_std > 3 * np.nanmean(np.abs(rewards[-50:])):
                        send_diag(f"⚠️ Reward variance spike: σ={r_std:.4f}")
            
            except Exception as e:
                print(f"[DIAG WARN] QA monitor hiba: {e}")
            
            # ✅ Második biztonsági stop közvetlen a learn után
            if current_steps >= total_steps:
                print(f"[STOP] Elérte a megadott {total_steps:,} lépést — tréning lezárva.")
                send_discord_message("TRAINING", f"✅ Tréning befejezve ({total_steps:,} steps).")
                break
                
                # --- Discord küldés (async thread) ---
                import threading
            
            def _send_chart():
                from core.utils.discord import send_discord_file, send_discord_message
                try:
                    summary_msg = (
                        f"📊 [TRAINING]\n"
                        f"• Timesteps: {current_steps:,}\n"
                        f"• Átlag Winrate: {mean_wr:.2f}%\n"
                        f"• Átlag EpReward: {mean_rw:.4f}\n"
                        f"• Átlag uPnL: {mean_upnl:.2f}\n"
                        f"• Reward szórás (Rσ): {std_rw:.4f}"
                    )
                    send_discord_message(
                        "TRAINING",
                        f"📈 Checkpoint {current_steps:,} lépésnél ({symbol} {timeframe})"
                    )
                    send_discord_file("TRAINING", chart_path, summary_msg)
                except Exception as e:
                    print(f"[WARN] Discord chart küldés hiba: {e}")
            
            # indítás külön threadben
            threading.Thread(target=_send_chart, daemon=True).start()
            
            # --- Plateau stop detection (auto_plateau_check integration) ---
            if os.path.exists("stop.flag"):
                print("[STOP] Plateau flag detected → training halted.")
                send_discord_message(
                    channel="TRAINING",
                    content="🛑 Plateau stop.flag detected — training halted."
                )
                return  # azonnal kilép a függvényből
            
            # --- Validation subprocess (async, checkpoint után) ---
            try:
                cmd = [
                    sys.executable, "-m", "scripts.run_validation",
                    "--timeframe", timeframe,
                    "--symbol", symbol,
                    "--timesteps", str(current_steps),
                    "--reward", reward_mode,
                    "--device", "cpu",
                    "--dataset", "val",
                    "--mode", "full",  # ✅ teljes validáció
                    "--auto_close_eval", "true",  # ✅ automatikus bezárás
                    "--deterministic", "true",  # ✅ reprodukálható környezet
                    "--resume", ckpt_path
                ]
                
                # 🔹 logfájl létrehozása, hogy minden output megmaradjon
                os.makedirs("reports", exist_ok=True)
                log_path = os.path.join(
                    "reports",
                    f"validation_ckpt_{symbol}_{timeframe}_{int(current_steps)}.log"
                )
                
                # 🔹 a subprocess kimenetét logfájlba irányítjuk
                with open(log_path, "w", encoding="utf-8") as lf:
                    subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                
                # 🔹 konzolos és discordos visszajelzés
                print(f"[DEBUG] Checkpoint validation subprocess indítva → log: {log_path}")
                send_discord_message(
                    "TRAINING",
                    f"✅ Teljes validation elindítva {current_steps:,} lépés után (async).\n📄 Log: {log_path}"
                )
            
            except Exception as e:
                print(f"[WARN] Validation/Checkpoint hiba: {e}")
                # extra védelem: Discordon is jelzés, ha valami elhasal
                try:
                    send_discord_message(
                        "TRAINING",
                        f"⚠️ Validation/Checkpoint hiba történt: {e}"
                    )
                except Exception:
                    pass
            
            # --- Final validation at end of training (DISABLED — async validation handles everything) ---
            """
            try:
                timesteps = int(getattr(model, "num_timesteps", 0))
            except Exception:
                timesteps = 0
            try:
                print("[FINAL] Training cycle complete. Running final validation...")

                # --- QA Final VecNormalize save ---
                try:
                    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
                    venv = getattr(env, "venv", getattr(env, "env", env))

                    if isinstance(venv, VecNormalize):
                        target = venv
                    elif hasattr(env, "venv") and isinstance(env.venv, VecNormalize):
                        target = env.venv
                    elif isinstance(env, VecNormalize):
                        target = env
                    else:
                        target = None

                    if target is not None:
                        vec_path = os.path.join(MODELS_DIR, f"vecnormalize_{symbol}_{timeframe}_final.pkl")
                        target.save(vec_path)
                        print(f"[QA] VecNormalize final mentve: {vec_path}")
                    else:
                        print(f"[QA] VecNormalize nem található – nincs mentés ({type(env).__name__}).")
                except Exception as e:
                    print(f"[QA-WARN] VecNormalize mentés sikertelen a tréning végén: {e}")

                send_discord_message(
                    "TRAINING",
                    "📊 [FINAL] Training completed — running final validation..."
                )

                cmd = [
                    sys.executable, "-m", "scripts.run_validation",
                    "--timeframe", timeframe,
                    "--symbol", symbol,
                    "--timesteps", "10000",
                    "--reward", reward_mode,
                    "--device", device,
                    "--dataset", "val"
                ]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                send_discord_message("TRAINING",
                                     f"✅ Post-training validáció elindítva async módban ({timeframe}, {symbol}).")

            except Exception as e:
                send_discord_message(
                    "TRAINING",
                    f"⚠️ [FINAL] Validation failed: {e}"
                )
            """
            
            # --- Tréning vége ---
            final_steps = int(model.num_timesteps)
            print(f"[STOP] Elérte a megadott {final_steps:,} lépést — tréning lezárva.")
            send_discord_message("TRAINING", f"✅ Tréning befejezve ({final_steps:,} steps).")
        
        # --- Deep Analysis chart automatikus generálása és Discord feltöltése ---
        try:
            deep_path = generate_final_analysis_chart(
                log_callback, symbol, timeframe, int(model.num_timesteps), MODELS_DIR
            )
            if deep_path and os.path.exists(deep_path):
                send_discord_file(
                    "TRAINING",
                    deep_path,
                    f"🧠 Deep Analysis (final {int(model.num_timesteps):,} steps)"
                )
            else:
                send_discord_message("TRAINING", "[WARN] Deep analysis chart nem jött létre vagy üres volt.")
        except Exception as e:
            send_discord_message("TRAINING", f"[WARN] Deep analysis chart hiba: {e}")
        
        # --- Smart checkpoint mentés a tréning végén ---
        try:
            current_steps = int(model.num_timesteps)
            ckpt_path = os.path.join(
                MODELS_DIR,
                f"ppo_{symbol}_{timeframe}_final_{current_steps}.zip"
            )
            
            # --- Safety re-init before final validation ---
            if "model_path_final" not in locals() or model_path_final is None:
                model_path_final = os.path.join(
                    MODELS_DIR,
                    f"ppo_{symbol}_{timeframe}_final_{int(model.num_timesteps)}.zip"
                )
                print(f"[SAFETY] model_path_final újrainicializálva: {model_path_final}")
            
            if not os.path.exists(model_path_final):
                print(f"[SAFETY] ⚠️ Nem található a final modell → latest fallback használat.")
                latest_fallback = os.path.join(MODELS_DIR, f"ppo_{symbol}_{timeframe}_final_latest.zip")
                if os.path.exists(latest_fallback):
                    model_path_final = latest_fallback
                    print(f"[SAFETY] ✅ Fallback modell kiválasztva: {model_path_final}")
            
            # --- Safety patch: model_path változók újrainicializálása, ha elvesztek ---
            if "model_path_progress" not in locals() or model_path_progress is None:
                model_path_progress = os.path.join(MODELS_DIR,
                                                   f"ppo_{symbol}_{timeframe}_progress_{int(model.num_timesteps)}.zip")
            if "model_path_final" not in locals() or model_path_final is None:
                model_path_final = os.path.join(MODELS_DIR,
                                                f"ppo_{symbol}_{timeframe}_final_{int(model.num_timesteps)}.zip")
            
            # --- Extra biztonság: készíts másolatot "latest" néven is ---
            model.save(ckpt_path)
            latest_copy = os.path.join(MODELS_DIR, f"ppo_{symbol}_{timeframe}_final_latest.zip")
            import shutil
            try:
                shutil.copy(ckpt_path, latest_copy)
                print(f"[CKPT] 🧩 Latest copy frissítve: {latest_copy}")
            except Exception as e:
                print(f"[CKPT WARN] Latest copy sikertelen: {e}")
            
            print(f"[OK] Checkpoint mentve a tréning végén: {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Checkpoint mentés hiba: {e}")
        
        # --- Deep Metrics JSON export a tréning végén ---
        try:
            metrics_path = os.path.join(MODELS_DIR, f"deep_metrics_{symbol}_{timeframe}.json")
            data = {
                "timesteps": getattr(log_callback, "timesteps", []),
                "policy_losses": getattr(log_callback, "policy_losses", []),
                "value_losses": getattr(log_callback, "value_losses", []),
                "entropies": getattr(log_callback, "entropies", []),
                "explained_variances": getattr(log_callback, "explained_variances", []),
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[DEEP] Metrikák JSON-ba mentve: {metrics_path}")
        except Exception as e:
            print(f"[DEEP-ERR] JSON mentés sikertelen: {e}")
        
        # --- QA VecNormalize path fallback ---
        vec_path = os.path.join("models", f"vecnormalize_{symbol}_{timeframe}.pkl")
        if not isinstance(vec_path, (str, bytes, os.PathLike)):
            vec_path = f"models/vecnormalize_{symbol}_{timeframe}.pkl"
        
        # --- VecNormalize statok mentése (ha aktív) ---
        try:
            if "env" in locals() and isinstance(env, VecNormalize):
                from stable_baselines3.common.vec_env import VecNormalize
                vec_path = os.path.join(MODELS_DIR, f"vecnormalize_{symbol}_{timeframe}.pkl")
                if os.path.exists(vec_path):
                    eval_env = VecNormalize.load(vec_path, eval_env)
                    eval_env.training = False
                    eval_env.norm_reward = False
                    print(f"[EVAL] VecNormalize statok betöltve: {vec_path}")
                else:
                    print("[EVAL] VecNormalize fájl nem található, raw eval_env használva.")
        
        # --- Always executed cleanup / save ---
        finally:
            print("[FINAL] Tréning ciklus lezárva.")
        
        # --- Biztonsági inicializálás (ha a korábbi checkpoint skip-elve lett) ---
        ckpt_path = None
        
        # --- Modell végső mentése ---
        final_alias = os.path.join(
            MODELS_DIR,
            f"ppo_{symbol}_{timeframe}_final_{int(model.num_timesteps)}.zip"
        )
        try:
            model.save(final_alias)
            print(f"[OK] Final model saved -> {final_alias}")
        except Exception as e:
            print(f"[WARN] Final model save failed: {e}")
        
        # --- Final chart és Discord küldés ---
        chart_path = generate_training_chart(
            log_callback, symbol, timeframe, int(model.num_timesteps), MODELS_DIR, phase="final"
        )
        if chart_path:
            send_discord_file("TRAINING", chart_path, "🏁 Final training chart")
        
        # --- Deep analysis chart (final QA) ---
        try:
            deep_path = generate_final_analysis_chart(
                log_callback, symbol, timeframe, int(model.num_timesteps), MODELS_DIR
            )
            if deep_path:
                import threading
                threading.Thread(
                    target=lambda: send_discord_file(
                        "TRAINING",
                        deep_path,
                        f"🧠 Deep Analysis (final {int(model.num_timesteps):,} steps)"
                    ),
                    daemon=True
                ).start()
        except Exception as e:
            print(f"[WARN] Deep analysis chart hiba: {e}")
        
        # --- Baseline snapshot (ha nagy tréning volt) ---
        if total_steps >= 1_000_000:
            save_baseline_snapshot(
                symbol=symbol,
                timeframe=timeframe,
                reward_mode=reward_mode,
                timesteps=int(model.num_timesteps),
                model_path=final_alias,
                metrics={
                    "winrate": float(np.mean(log_callback.winrates)),
                    "reward_mean": float(np.mean(log_callback.rewards)),
                    "entropy": float(np.mean(getattr(log_callback, "entropies", [0]))),
                },
                charts={
                    "training_chart": chart_path,
                    "deep_analysis": deep_path if 'deep_path' in locals() else None,
                }
            )
        # --- Automatikus Central AI trigger baseline mentés után ---
        try:
            print("[AUTO] Central AI trigger indul...")
            os.system(f"{sys.executable} -m core.central.training.central_trainer --auto")
        except Exception as e:
            print(f"[WARN] Central AI auto-trigger sikertelen: {e}")
            
    finally:
        pass

    
# === Baseline snapshot mentés (1M+ tréningekhez, FIXED) ===
def save_baseline_snapshot(symbol, timeframe, reward_mode, timesteps, model_path, metrics, charts):
    """
    Létrehozza a QuantAI tudásbázis snapshotot (baseline feed) a Kontroll-AI számára.
    Mentési hely: core/central/knowledge_base/baseline_snapshots/
    Automatikusan korrigálja az UNKNOWN_UNKNOWN névhibákat.
    """
    import json, os, datetime
    from core.utils.discord import send_discord_message

    kb_dir = os.path.join("core", "central", "knowledge_base", "baseline_snapshots")
    os.makedirs(kb_dir, exist_ok=True)

    # --- Fallback névfix: ha symbol/timeframe üres, próbálja kinyerni ---
    try:
        if not symbol or symbol == "UNKNOWN":
            symbol = os.environ.get("TRAIN_SYMBOL", "UNKNOWN")
        if not timeframe or timeframe == "UNKNOWN":
            timeframe = os.environ.get("TRAIN_TIMEFRAME", "UNKNOWN")

        # Ha továbbra is hiányzik, próbáljuk a model_path alapján
        if (not symbol or symbol == "UNKNOWN") or (not timeframe or timeframe == "UNKNOWN"):
            try:
                model_name = os.path.basename(model_path or "")
                parts = model_name.replace(".zip", "").split("_")
                if len(parts) >= 3:
                    symbol = parts[1]
                    timeframe = parts[2]
            except Exception:
                pass

        # Ha még így sincs, állíts be biztonságos defaultot
        if not symbol:
            symbol = "UNKNOWN"
        if not timeframe:
            timeframe = "UNKNOWN"

        print(f"[BASELINE] Snapshot name verified → {symbol}_{timeframe}")
    except Exception as e:
        print(f"[WARN] Baseline névfix hiba: {e}")

    # --- Snapshot objektum összeállítása ---
    snapshot = {
        "symbol": symbol,
        "timeframe": timeframe,
        "reward_mode": reward_mode,
        "timesteps": timesteps,
        "metrics": metrics,
        "model_path": model_path,
        "charts": charts,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    # --- Mentés ---
    out_path = os.path.join(kb_dir, f"{symbol}_{timeframe}_baseline.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=4)
        print(f"[OK] Baseline snapshot mentve: {out_path}")
        send_discord_message(channel="CENTRAL", content=f"📘 Baseline snapshot frissítve → {symbol} {timeframe}")
        
        # --- Auto Meta mentés (WR + Reward + Time + Symbol + TF) ---
        try:
            meta_path = os.path.join("models", f"ppo_{symbol}_{timeframe}_meta.json")
            meta = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timesteps": int(timesteps),
                "winrate": float(metrics.get("winrate", 0.0)),
                "avg_reward": float(metrics.get("avg_reward", 0.0)),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)
            
            print(f"[META+WR] Meta frissítve WR={meta['winrate']:.2f} R={meta['avg_reward']:.3f}")
        
        except Exception as e:
            print(f"[META ERR] Meta mentési hiba: {e}")
        
        # --- 🔧 Auto Meta mentés (WR + Reward + Time + Symbol + TF + Checkpoint) ---
        try:
            ckpt_id = int(timesteps)
            meta_path = os.path.join("models", f"ppo_{symbol}_{timeframe}_ckpt_{ckpt_id}_meta.json")
            
            meta = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timesteps": ckpt_id,
                "winrate": float(metrics.get("winrate", 0.0)),
                "avg_reward": float(metrics.get("avg_reward", 0.0)),
                "reward_mode": reward_mode,
                "model_path": model_path if 'model_path' in locals() else None,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)
            
            print(
                f"[META+CKPT] Meta mentve: {os.path.basename(meta_path)} | WR={meta['winrate']:.2f}% | R={meta['avg_reward']:.3f}")
            
            send_discord_message(
                channel="CENTRAL",
                content=f"📘 Meta frissítve ({symbol} {timeframe}) — WR={meta['winrate']:.2f}% R={meta['avg_reward']:.3f} @ {ckpt_id:,} steps"
            )
        
        except Exception as e:
            print(f"[META ERR] Meta mentési hiba (ckpt): {e}")
    
    # 🔻 EZ volt a hiányzó except a legkülső try-hoz:
    except Exception as e:
        print(f"[WARN] Baseline snapshot mentési hiba: {e}")
    
    # --- Safe global model reference ---
    model = globals().get("model", None)
    
    file_name = f"{symbol}_{timeframe}_{reward_mode}_{timesteps // 1000}k.json"
    file_path = os.path.join(kb_dir, file_name)
    
    # --- Update baseline snapshot with final timesteps (safe) ---
    try:
        snapshot["timesteps"] = int(getattr(model, "num_timesteps", timesteps))
        snapshot["last_update"] = datetime.datetime.utcnow().isoformat() + "Z"
    except Exception as e:
        print(f"[WARN] Timesteps update in snapshot failed: {e}")
    
    # --- Safety pre-definitions (avoid unresolved references) ---
    file_path = os.path.join(MODELS_DIR, "baseline_snapshot.json")
    snapshot = {}
    symbol = globals().get("symbol", "UNKNOWN")
    timeframe = globals().get("timeframe", "UNKNOWN")
    timesteps = 0  # model később tölti be a valós értéket
    
    # === 1️⃣ Lokális (TRAINING) snapshot mentés ===
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=4)
        print(f"[OK] Baseline snapshot saved -> {file_path}")
        send_discord_message(
            channel="TRAINING",
            content=f"📘 Baseline snapshot saved ({symbol} {timeframe}, {timesteps} steps)"
        )
    except Exception as e:
        print(f"[WARN] Snapshot save failed (TRAINING): {e}")
    
    # === 2️⃣ Central AI snapshot mentés (külön könyvtár) ===
    try:
        kb_dir = os.path.join("core", "central", "knowledge_base", "baseline_snapshots")
        os.makedirs(kb_dir, exist_ok=True)
        kb_file = os.path.join(kb_dir, f"{symbol}_{timeframe}_baseline.json")
        
        # adjunk hozzá minimális adatokat, ha hiányoznak
        snapshot["symbol"] = symbol
        snapshot["timeframe"] = timeframe
        snapshot["timesteps"] = snapshot.get("timesteps", timesteps)
        snapshot["last_update"] = datetime.datetime.utcnow().isoformat() + "Z"
        
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=4)
        print(f"[OK] Baseline snapshot saved -> {kb_file}")
        
        send_discord_message(
            channel="CENTRAL",
            content=f"📘 Baseline snapshot updated — {symbol} {timeframe} @ {snapshot['timesteps']:,} steps"
        )
    
    except Exception as e:
        print(f"[WARN] Snapshot save failed (CENTRAL): {e}")
    
    
    
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--reward", default="realistic")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose logs")
    parser.add_argument("--curriculum", type=int, default=0, help="Enable curriculum-phase training (1=on)")
    args = parser.parse_args()
    
    # --- Auto-resume path resolver ---
    import os
    
    if args.resume:
        resume_arg = args.resume.strip().replace('"', '').replace("'", "")
        if not resume_arg.lower().endswith(".zip"):
            # csak az ID van megadva pl. 606208 → megkeressük a models mappában
            pattern = f"ppo_{args.symbol}_{args.timeframe}_final_{resume_arg}.zip"
            candidate = os.path.join("models", pattern)
            if os.path.exists(candidate):
                args.resume = candidate
                print(f"[AUTO-RESUME] Fájl megtalálva: {candidate}")
            else:
                # keresés részleges egyezéssel, ha pl. csak 606 vagy 507-et ad meg
                for fn in os.listdir("models"):
                    if fn.startswith(f"ppo_{args.symbol}_{args.timeframe}_final_") and resume_arg in fn:
                        args.resume = os.path.join("models", fn)
                        print(f"[AUTO-RESUME] Részleges egyezés: {args.resume}")
                        break
        else:
            if not os.path.exists(args.resume):
                print(f"[WARN] Resume path nem létezik: {args.resume}")
    
    print("[BOOT] train_agent main entry ✅")
    
    train_agent(
        timeframe=args.timeframe,
        symbol=args.symbol,
        timesteps=args.timesteps,
        reward_mode=args.reward,
        resume_path=args.resume,
        device=args.device,
        num_envs=args.num_envs,
    )
