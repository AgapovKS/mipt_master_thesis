import re
import os

# ── Переменные окружения NCCL / коммуникация между GPU ──────────────────────
# Увеличиваем таймаут heartbeat до 2 часов — нужно для длинных шагов GRPO
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"
# Блокирующее ожидание при синхронизации — предотвращает потерю задач
os.environ["NCCL_BLOCKING_WAIT"] = "1"
# Отключаем InfiniBand (в Kaggle / облаке его нет)
os.environ["NCCL_IB_DISABLE"] = "1"
# Параллельная токенизация ломается при мультипроцессинге — отключаем
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import math
import torch
import json
import random
import gc
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    IntervalStrategy,
    SchedulerType,
)
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers.trainer_utils import SaveStrategy

from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as st_load
from accelerate import Accelerator


# ── Глобальные метрики (общие для всех уровней) ──────────────────────────────
# Накапливаем историю обучения по всему курикулуму, чтобы строить единые графики
GLOBAL_HISTORY = {
    "step": [], "loss": [], "reward": [], "kl": [], "vram": [], "level": []
}
GLOBAL_STEP_COUNTER = 0


def _fmt(x):
    """Форматирует число до 4 знаков после запятой, или «—» если None."""
    return f"{x:.4f}" if isinstance(x, (float, int)) else "—"


def extract_boxed(text):
    """
    Извлекает содержимое \\boxed{...} с учётом произвольной вложенности скобок.
    Возвращает строку внутри boxed или None, если маркер не найден.
    """
    marker = r'\boxed{'
    idx = text.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    # Идём посимвольно, отслеживая глубину вложенности
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    # Инициализируем Accelerate для мульти-GPU обучения
    accelerator = Accelerator()
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            accelerator.local_process_index if hasattr(accelerator, "local_process_index") else 0
        )
    )
    device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # bfloat16 предпочтительнее float16 на современных GPU (A100, H100, T4 v2)
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_supported = False
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    print(f"[Accelerate] local_rank={local_rank} | устройство={device_str} | dtype={compute_dtype}")

    # ── Загрузка и подготовка модели ─────────────────────────────────────────
    def get_model_and_tokenizer(
        level,
        prev_adapter_path=None,
        accelerator=None,
        device_str="cuda:0",
        compute_dtype=torch.float16,
    ):
        """
        Загружает базовую модель с 4-битной квантизацией (QLoRA / NF4)
        и подключает LoRA-адаптер предыдущего уровня (если передан).
        При ошибке загрузки адаптера создаёт новый LoRA с нуля.
        """
        import torch.distributed as dist

        if dist.is_initialized():
            local_rank = dist.get_rank()
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"\n[Ранг {local_rank}] Загрузка модели для уровня {level}...")

        # Явно назначаем CUDA-устройство данному процессу
        try:
            torch.cuda.set_device(local_rank)
            print(f"[Ранг {local_rank}] Активное устройство: {torch.cuda.current_device()}")
        except Exception as _e:
            print(f"[Предупреждение] set_device({local_rank}) пропущен: {_e}")

        torch.cuda.empty_cache()
        gc.collect()

        # NF4 — нормально-плавающая 4-битная квантизация, лучше обычного int4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # двойная квантизация — чуть меньше памяти
        )

        # Важно: НЕ обнуляем config.rope_scaling!
        # Qwen2.5 использует YaRN rope_scaling — без него позиционные эмбеддинги ломаются
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        config._attn_implementation = "eager"  # flash attention пока нестабилен с GRPO
        config.use_cache = False  # кэш KV несовместим с gradient checkpointing

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # У Qwen2.5 есть собственный pad_token, но страхуемся на случай других чекпоинтов
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left-padding — обязателен для генерации с батчем разных длин
        tokenizer.padding_side = "left"

        print(f"[Ранг {local_rank}] Загрузка весов модели...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Отключаем KV-кэш везде — нужно для обратного прохода
        model.config.use_cache = False
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False

        # Запрещаем tie_word_embeddings, чтобы lm_head мог учиться независимо
        model.config.tie_word_embeddings = False

        # Если словарь токенайзера расширился — ресайзим embedding-слой
        try:
            emb = model.get_input_embeddings()
            if emb is not None and len(tokenizer) != emb.weight.size(0):
                print(f"[Модель] Ресайз embeddings до {len(tokenizer)}")
                model.resize_token_embeddings(len(tokenizer))
        except Exception as ex:
            print(f"[Предупреждение] resize_token_embeddings: {ex}")

        # prepare_model_for_kbit_training: переводит нормализации в float32 и настраивает grad
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except Exception as ex:
            print(f"[Предупреждение] prepare_model_for_kbit_training: {ex}")

        # ── Загрузка адаптера предыдущего уровня ─────────────────────────────
        if prev_adapter_path and os.path.exists(prev_adapter_path):
            print(f"[Модель] Загрузка LoRA-адаптера: {prev_adapter_path}")

            adapter_config_path = os.path.join(prev_adapter_path, "adapter_config.json")
            temp_config_path = os.path.join(prev_adapter_path, "adapter_config_backup.json")
            modules_to_save_backup = None

            try:
                # Временно удаляем modules_to_save из конфига, чтобы избежать
                # ошибки при загрузке адаптера с другой архитектурой головы
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, "r") as f:
                        adapter_config = json.load(f)

                    if "modules_to_save" in adapter_config:
                        modules_to_save_backup = adapter_config["modules_to_save"]
                        print(f"[Правка] Временно убираем modules_to_save: {modules_to_save_backup}")
                        shutil.copy(adapter_config_path, temp_config_path)
                        del adapter_config["modules_to_save"]
                        with open(adapter_config_path, "w") as f:
                            json.dump(adapter_config, f, indent=2)

                model = PeftModel.from_pretrained(model, prev_adapter_path, is_trainable=True)
                print("[Модель] Адаптер загружен успешно")

            except Exception as ex:
                print(f"[ОШИБКА] Не удалось загрузить адаптер: {ex}")
                print("[Модель] Создаём новый LoRA с нуля...")
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    # Qwen2.5 использует те же имена проекций, что и LLaMA
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                )
                model = get_peft_model(model, peft_config)

            finally:
                # В любом случае восстанавливаем оригинальный конфиг адаптера
                if os.path.exists(temp_config_path):
                    shutil.move(temp_config_path, adapter_config_path)
                    print("[Правка] adapter_config.json восстановлен")

        elif level == 1:
            # Уровень 1 — SFT с нуля, адаптера ещё нет
            print("[Модель] Создание LoRA для уровня 1 (SFT)...")
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                # Qwen2.5 использует те же имена проекций — совместимо без изменений
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, peft_config)

        # Приводим все float32-параметры к bfloat16, чтобы не было dtype-конфликтов
        print("[Правка] Приведение всех параметров к bfloat16...")
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        model.config.use_cache = False

        # Явно включаем градиенты только для LoRA-параметров и lm_head
        lora_count = 0
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                lora_count += 1
            if any(k in name for k in ["lm_head", "modules_to_save"]):
                param.requires_grad = True

        print(f"[Модель] Количество LoRA-параметров: {lora_count}")

        # enable_input_require_grads нужен для grad-flow через квантизованные слои
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(
            f"Обучаемые: {sum(p.numel() for p in trainable):,} | "
            f"Всего: {sum(p.numel() for p in model.parameters()):,}"
        )

        model.config.use_cache = False
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False
        print(f"[Проверка] model.config.use_cache = {model.config.use_cache}")

        # ── Обёртка для безопасного приведения dtype ─────────────────────────
        # При смешанной точности входной тензор может не совпасть по dtype с весами —
        # этот модуль исправляет это прозрачно
        class CastInputToWeightDtype(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def _get_target_dtype(self):
                """Определяет dtype весов обёртываемого модуля."""
                try:
                    for p in self.module.parameters():
                        return p.dtype
                except Exception:
                    pass
                try:
                    if hasattr(self.module, "weight"):
                        return self.module.weight.dtype
                except Exception:
                    pass
                return torch.bfloat16

            def forward(self, x, *args, **kwargs):
                target_dtype = self._get_target_dtype()
                # Целочисленные тензоры (индексы токенов) не трогаем
                if isinstance(x, torch.Tensor):
                    if x.dtype not in [torch.long, torch.int, torch.int32, torch.int64]:
                        if x.dtype != target_dtype:
                            x = x.to(target_dtype)
                return self.module(x, *args, **kwargs)

        # Оборачиваем embedding-слой
        try:
            emb = model.get_input_embeddings()
            if emb:
                model.set_input_embeddings(CastInputToWeightDtype(emb))
                print("[Обёртка] embed_tokens — успешно")
        except Exception as ex:
            print(f"[Предупреждение] embed_tokens: {ex}")

        # Оборачиваем lm_head
        try:
            lm = getattr(model, "lm_head", None)
            if lm:
                setattr(model, "lm_head", CastInputToWeightDtype(lm))
                print("[Обёртка] lm_head — успешно")
        except Exception as ex:
            print(f"[Предупреждение] lm_head: {ex}")

        # Синхронизируем все ранги перед началом обучения
        if dist.is_initialized():
            print(f"[Ранг {local_rank}] Ожидание остальных рангов...")
            dist.barrier()
            print(f"[Ранг {local_rank}] Все ранги готовы")

        return model, tokenizer

    # ── Глобальные константы обучения ─────────────────────────────────────────
    MODEL_NAME           = "Qwen/Qwen2.5-1.5B"
    DATASET_GSM          = "openai/gsm8k"
    DATASET_GSM_CONFIG   = "socratic"
    DATASET_S1K          = "simplescaling/s1K-1.1"
    DATASET_S1K_CONFIG   = "train"
    DATASET_COMP_MATH    = "qwedsacf/competition_math"

    MAX_STEPS_PER_LEVEL  = 150
    MAX_SAMPLES_PER_LEVEL = 5000
    STARTING_LEVEL       = 6          # Продолжаем с уровня 6 (адаптер уровня 5 уже есть)
    MAX_CURRICULUM_LEVELS = 6
    # Путь к сохранённому адаптеру предыдущего уровня (для возобновления обучения)
    RESUME_ADAPTER_PATH  = '/kaggle/input/datasets/ksagapov/new-qwen-adapter/level5/models_saved/level_5'
    STABILITY_THRESHOLD  = 0.8
    SEED                 = 42

    # Фиксируем все источники случайности для воспроизводимости
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ── Callback: логирование и накопление метрик ─────────────────────────────
    class GlobalPlottingCallback(TrainerCallback):
        """
        Перехватывает логи тренера на каждом шаге и складывает метрики
        в глобальный словарь GLOBAL_HISTORY для последующей визуализации.
        """
        def __init__(self):
            self.history = {"step": [], "loss": [], "reward": [], "kl": []}

        def on_log(self, args, state, control, logs=None, **kwargs):
            global GLOBAL_STEP_COUNTER, GLOBAL_HISTORY
            if not logs:
                return
            print(f"\n[ЛОГИ]: {logs}")

            # Отладочный вывод ключей на первых шагах — помогает разобраться со схемой логов
            if GLOBAL_STEP_COUNTER < 5:
                print(f"\n[Отладка] Доступные ключи логов: {list(logs.keys())}")

            loss_val = logs.get("loss")

            # Reward может приходить под разными именами в зависимости от версии TRL
            reward_val = (
                logs.get("reward") or
                logs.get("rewards/mean") or
                logs.get("train/reward") or
                logs.get("objective/scores") or
                logs.get("scores/mean")
            )

            # KL-дивергенция — мера отклонения политики от референсной модели
            kl_val = (
                logs.get("kl") or
                logs.get("policy_kl") or
                logs.get("objective/kl") or
                logs.get("kl/mean") or
                logs.get("train/kl")
            )

            if loss_val is not None or reward_val is not None:
                GLOBAL_STEP_COUNTER += 1

            if reward_val is not None or kl_val is not None:
                print(f"[Шаг {GLOBAL_STEP_COUNTER}] Reward: {_fmt(reward_val)} | KL: {_fmt(kl_val)}")

            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

            GLOBAL_HISTORY["step"].append(GLOBAL_STEP_COUNTER)
            GLOBAL_HISTORY["loss"].append(loss_val)
            GLOBAL_HISTORY["reward"].append(reward_val)
            GLOBAL_HISTORY["kl"].append(kl_val)
            GLOBAL_HISTORY["vram"].append(vram_gb)
            GLOBAL_HISTORY["level"].append(kwargs.get("current_level_tag", 0))

    def save_plots():
        """Сохраняет графики динамики обучения (loss/reward и KL) в папку plots/."""
        if not GLOBAL_HISTORY["step"]:
            return
        steps = GLOBAL_HISTORY["step"]
        os.makedirs("plots", exist_ok=True)

        # График 1: Loss (SFT) + Reward (RL) на одной оси
        fig, ax1 = plt.subplots(figsize=(12, 6))
        valid_loss = [(s, l) for s, l in zip(steps, GLOBAL_HISTORY["loss"]) if l is not None]
        if valid_loss:
            s_loss, v_loss = zip(*valid_loss)
            ax1.plot(s_loss, v_loss, color="tab:blue", label="SFT Loss", alpha=0.6)
            ax1.set_ylabel("Loss", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        valid_rew = [(s, r) for s, r in zip(steps, GLOBAL_HISTORY["reward"]) if r is not None]
        if valid_rew:
            s_rew, v_rew = zip(*valid_rew)
            ax2.plot(s_rew, v_rew, color="tab:orange", label="RL Reward", linewidth=2)
            ax2.set_ylabel("Средний Reward", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("Динамика обучения: SFT → RL")
        plt.savefig("plots/training_dynamics.png", dpi=300)
        plt.close()

        # График 2: KL-дивергенция — важен для мониторинга стабильности GRPO
        valid_kl = [(s, k) for s, k in zip(steps, GLOBAL_HISTORY["kl"]) if k is not None]
        if valid_kl:
            s_kl, v_kl = zip(*valid_kl)
            plt.figure(figsize=(10, 5))
            plt.plot(s_kl, v_kl, color="purple", label="KL Divergence")
            plt.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Целевая зона")
            plt.title("Стабильность политики (KL-дивергенция)")
            plt.xlabel("Шаги")
            plt.ylabel("KL")
            plt.legend()
            plt.savefig("plots/training_kl.png", dpi=300)
            plt.close()

        print("Графики сохранены в папку /plots")

    # ── Подготовка данных: загрузка и разбивка по уровням ────────────────────
    def preprocess_and_bucket(split="train", max_samples=MAX_SAMPLES_PER_LEVEL):
        """
        Загружает три датасета (GSM8K, S1K, competition_math), распределяет задачи
        по 6 уровням сложности и возвращает словарь {уровень: {train, eval}}.

        Принцип разбивки:
          Уровни 1-3 — GSM8K (простые, средние, сложные задачи)
          Уровень 4  — сложный GSM8K + лёгкий competition_math
          Уровень 5  — competition_math (уровень 4) + короткий S1K + GSM8K replay
          Уровень 6  — competition_math (уровень 5) + длинный S1K + GSM8K replay (меньше)

        GSM8K replay на уровнях 5-6 выступает якорем: сохраняет базовые навыки,
        снижает риск коллапса награды (frac_reward_zero_std) при GRPO.
        """
        MIN_SAMPLES_RL = 600
        MAX_EVAL       = 30
        PCT_TEST       = 0.05
        # Ограничения на количество задач per level (подобраны под VRAM T4×2)
        TRAIN_LIMITS   = {1: 800, 2: 800, 3: 1000, 4: 700, 5: 800, 6: 680}

        print(f"Загрузка {DATASET_GSM}...")
        ds_gsm = load_dataset(DATASET_GSM, name=DATASET_GSM_CONFIG, split=split)

        print(f"Загрузка {DATASET_S1K}...")
        ds_s1k = load_dataset(DATASET_S1K, split=DATASET_S1K_CONFIG)

        print(f"Загрузка {DATASET_COMP_MATH}...")
        ds_comp = load_dataset(DATASET_COMP_MATH, split="train")

        level_data = {
            i: {"prompt": [], "history": [], "expert_action": [], "source": []}
            for i in range(1, 7)
        }

        # ── 1. GSM8K → уровни 1–4 (строгое непересекающееся разбиение) ───────
        print("Обработка GSM8K (строгое разбиение по уровням)...")
        for question, answer in zip(ds_gsm["question"], ds_gsm["answer"]):
            full_solution = answer.split("####")[0].strip()
            steps = [s.strip() for s in full_solution.split("\n") if len(s.strip()) > 5]
            if not steps:
                continue

            # Число арифметических операций <<...>> используем как прокси сложности
            ops_count = len(re.findall(r"<<.*?>>", full_solution))
            if   ops_count <= 2: lvl = 1  # тривиальные задачи
            elif ops_count <= 3: lvl = 2  # лёгкие
            elif ops_count <= 6: lvl = 3  # средние
            else:                lvl = 4  # сложные (только новые, без реплея)

            total_steps = len(steps)
            # Выбираем случайный шаг k как целевое действие эксперта (teacher-forcing)
            k = random.randint(1, total_steps)
            history = "\n".join(steps[: k - 1])

            level_data[lvl]["prompt"].append(question)
            level_data[lvl]["history"].append(history)
            level_data[lvl]["expert_action"].append(steps[k - 1])
            level_data[lvl]["source"].append("gsm8k")

        # ── GSM8K Replay Pool — только самые лёгкие задачи (ops_count <= 2) ──
        # Используется на уровнях 5–6 как якорь стабильности GRPO
        gsm_replay_pool = []
        for question, answer in zip(ds_gsm["question"], ds_gsm["answer"]):
            full_solution = answer.split("####")[0].strip()
            steps = [s.strip() for s in full_solution.split("\n") if len(s.strip()) > 5]
            if not steps:
                continue
            ops_count = len(re.findall(r"<<.*?>>", full_solution))
            if ops_count <= 2:
                k = random.randint(1, len(steps))
                gsm_replay_pool.append({
                    "prompt": question,
                    "history": "\n".join(steps[:k-1]),
                    "expert_action": steps[k-1],
                    "source": "gsm8k_replay"
                })
        random.shuffle(gsm_replay_pool)

        # ── 2. Competition Math → пулы по сложности (Level 1–5) ──────────────
        print("Обработка competition_math...")
        comp_pools = {1: [], 2: [], 3: [], 4: [], 5: []}

        for item in ds_comp:
            prob_level = item.get("level", "")
            problem    = item.get("problem", "").strip()
            solution   = item.get("solution", "").strip()
            if not problem or not solution:
                continue

            # Уровень сложности закодирован как "Level N" в строке
            m = re.search(r"Level (\d)", prob_level)
            if m:
                l_idx = int(m.group(1))
                if l_idx in comp_pools:
                    comp_pools[l_idx].append({
                        "prompt": problem,
                        "history": "",
                        "expert_action": solution,
                        "source": f"comp_math_l{l_idx}"
                    })

        for k in comp_pools:
            random.shuffle(comp_pools[k])

        print(f"  CompMath уровни 1-3 (лёгкие): {sum(len(comp_pools[i]) for i in [1,2,3])} задач")
        print(f"  CompMath уровень 4: {len(comp_pools[4])} задач")
        print(f"  CompMath уровень 5: {len(comp_pools[5])} задач")

        # Уровень 4: сложный GSM8K (уже добавлен выше) + лёгкий CompMath (1-3)
        TARGET_COMP_EASY_L4 = 600
        comp_easy_pool = comp_pools[1] + comp_pools[2] + comp_pools[3]
        random.shuffle(comp_easy_pool)

        for entry in comp_easy_pool[:TARGET_COMP_EASY_L4]:
            level_data[4]["prompt"].append(entry["prompt"])
            level_data[4]["history"].append(entry["history"])
            level_data[4]["expert_action"].append(entry["expert_action"])
            level_data[4]["source"].append(entry["source"])

        # ── 3. S1K → уровни 5-6 (короткие задачи — в 5-й, длинные — в 6-й) ──
        print("Обработка S1K...")
        s1k_pool = []
        for item in ds_s1k:
            if not isinstance(item, dict):
                continue
            q   = item.get("question", "") or ""
            sol = item.get("solution",  "") or ""
            # Отсекаем очень длинные примеры: риск OOM при генерации
            if len(q) + len(sol) < 6000:
                s1k_pool.append({
                    "prompt": q, "expert_action": sol, "len": len(q) + len(sol)
                })

        # Сортируем по возрастанию суммарной длины — прокси «сложности»
        s1k_pool.sort(key=lambda x: x["len"])

        # Делим по медиане: первая половина → уровень 5, вторая → уровень 6
        mid_idx = len(s1k_pool) // 2
        s1k_for_l5 = s1k_pool[:mid_idx]
        s1k_for_l6 = s1k_pool[mid_idx:]

        random.shuffle(s1k_for_l5)
        random.shuffle(s1k_for_l6)

        # Уровень 5: CompMath (lvl 4) + S1K короткий + GSM8K replay (~20%)
        TARGET_COMP_L5      = 450
        TARGET_S1K_L5       = 200
        TARGET_GSM_REPLAY_L5 = 150

        for entry in comp_pools[4][:TARGET_COMP_L5]:
            level_data[5]["prompt"].append(entry["prompt"])
            level_data[5]["history"].append(entry["history"])
            level_data[5]["expert_action"].append(entry["expert_action"])
            level_data[5]["source"].append(entry["source"])

        for item in s1k_for_l5[:TARGET_S1K_L5]:
            level_data[5]["prompt"].append(item["prompt"])
            level_data[5]["history"].append("")
            level_data[5]["expert_action"].append(item["expert_action"])
            level_data[5]["source"].append("s1k_short")

        # GSM8K replay — якорь стабильности GRPO: не даёт наградам коллапсировать
        for entry in gsm_replay_pool[:TARGET_GSM_REPLAY_L5]:
            level_data[5]["prompt"].append(entry["prompt"])
            level_data[5]["history"].append(entry["history"])
            level_data[5]["expert_action"].append(entry["expert_action"])
            level_data[5]["source"].append("gsm8k_replay")

        # Уровень 6: CompMath (lvl 5) + S1K длинный + GSM8K replay (~13%, угасает)
        TARGET_COMP_L6      = 300
        TARGET_S1K_L6       = 300
        TARGET_GSM_REPLAY_L6 = 80

        for entry in comp_pools[5][:TARGET_COMP_L6]:
            level_data[6]["prompt"].append(entry["prompt"])
            level_data[6]["history"].append(entry["history"])
            level_data[6]["expert_action"].append(entry["expert_action"])
            level_data[6]["source"].append(entry["source"])

        for item in s1k_for_l6[:TARGET_S1K_L6]:
            level_data[6]["prompt"].append(item["prompt"])
            level_data[6]["history"].append("")
            level_data[6]["expert_action"].append(item["expert_action"])
            level_data[6]["source"].append("s1k_long")

        # Берём следующий срез replay-пула (без пересечения с уровнем 5)
        for entry in gsm_replay_pool[TARGET_GSM_REPLAY_L5:TARGET_GSM_REPLAY_L5 + TARGET_GSM_REPLAY_L6]:
            level_data[6]["prompt"].append(entry["prompt"])
            level_data[6]["history"].append(entry["history"])
            level_data[6]["expert_action"].append(entry["expert_action"])
            level_data[6]["source"].append("gsm8k_replay")

        # ── Формирование финальных buckets с train/eval сплитом ──────────────
        final_buckets = {}
        print("\nИтоговое распределение (без дубликатов):")
        for lvl, data in level_data.items():
            if not data["prompt"]:
                continue

            full_ds     = Dataset.from_dict(data)
            level_limit = TRAIN_LIMITS.get(lvl, max_samples)
            limit       = min(len(full_ds), level_limit)
            if limit == 0:
                continue
            full_ds = full_ds.shuffle(seed=SEED).select(range(limit))

            # Выделяем 5% под eval, но не больше MAX_EVAL примеров
            desired_test = max(1, math.floor(len(full_ds) * PCT_TEST))
            desired_test = min(desired_test, MAX_EVAL)
            if desired_test >= len(full_ds):
                desired_test = max(1, len(full_ds) // 10)

            split_ds = full_ds.train_test_split(test_size=desired_test, seed=SEED)
            final_buckets[lvl] = {"train": split_ds["train"], "eval": split_ds["test"]}

            sources = full_ds["source"]
            counts = {src: sources.count(src) for src in set(sources)}
            count_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])

            print(f"  Уровень {lvl}: Train={len(split_ds['train'])}, Eval={len(split_ds['test'])} | ({count_str})")

        return final_buckets

    print("Загрузка и разбивка датасета...")
    buckets = preprocess_and_bucket(split="train")

    # ── Функция вознаграждения (Reward) ──────────────────────────────────────
    def combined_reward(prompts, completions, expert_action, **kwargs):
        """
        Многоуровневая функция вознаграждения для GRPO.

        Порядок проверок (от точного к приближённому):
          1. Сравнение через \\boxed{...} — точное, численное, Jaccard
          2. Числовое сравнение для GSM8K-ответов
          3. Строковое совпадение (exact match)
          4. Jaccard-fallback по токенам — мягкая оценка частичного совпадения

        Возвращает список float ∈ [0, 1] для каждого примера батча.
        """
        rewards = []
        for completion, ref in zip(completions, expert_action):
            # Пробуем извлечь ответ из тега <solution>...</solution>
            m    = re.search(r"<solution>(.*?)</solution>", completion, re.DOTALL)
            pred = m.group(1).strip() if m else completion.strip()

            # Очищаем эталонный ответ от разметки
            if "**" in ref:
                ref_clean = ref.split("**")[-1].strip()
            else:
                ref_clean = ref.strip()
            ref_clean = re.sub(r"</?solution>", "", ref_clean).strip()

            # ── Путь 1: через \boxed{...} ─────────────────────────────────────
            boxed_ref = extract_boxed(ref_clean)
            if boxed_ref:
                boxed_pred = extract_boxed(pred)

                # Точное совпадение строк внутри boxed
                if boxed_pred and boxed_pred.strip() == boxed_ref.strip():
                    rewards.append(1.0)
                    continue

                # Эталонное значение встречается в тексте предсказания
                if boxed_ref.strip() in pred:
                    rewards.append(1.0)
                    continue

                # Численное сравнение — нужно для "1,000" vs "1000" и т.п.
                try:
                    ref_val = float(boxed_ref.replace(",", "").strip())
                    if boxed_pred:
                        try:
                            pred_val = float(boxed_pred.replace(",", "").strip())
                            if abs(ref_val - pred_val) < 1e-6:
                                rewards.append(1.0)
                                continue
                        except ValueError:
                            pass
                    # Ищем любое число в предсказании близкое к эталону
                    pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred)
                    if any(abs(ref_val - float(n)) < 1e-6 for n in pred_nums
                           if n not in ("", ".")):
                        rewards.append(0.6)
                        continue
                except (ValueError, TypeError):
                    pass

                # Частичное совпадение через Jaccard по токенам boxed-ответа
                if boxed_pred:
                    pred_toks = set(re.findall(r"[a-zA-Z0-9]+|\\[a-zA-Z]+", boxed_pred))
                    ref_toks  = set(re.findall(r"[a-zA-Z0-9]+|\\[a-zA-Z]+", boxed_ref))
                    if pred_toks or ref_toks:
                        box_jaccard = len(pred_toks & ref_toks) / max(1, len(pred_toks | ref_toks))
                        rewards.append(round(box_jaccard * 0.5, 4))
                    else:
                        rewards.append(0.25)
                    continue

                rewards.append(0.0)
                continue

            # ── Путь 2: числовое сравнение (GSM8K без boxed) ─────────────────
            try:
                ref_num   = float(ref_clean)
                pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
                if pred_nums:
                    try:
                        is_match = abs(ref_num - float(pred_nums[-1])) < 1e-6
                    except Exception:
                        is_match = False
                else:
                    is_match = False
                if is_match:
                    rewards.append(1.0)
                    continue
            except ValueError:
                pass

            # ── Путь 3: точное строковое совпадение ──────────────────────────
            if len(ref_clean) < 20:
                # Для коротких ответов — word-boundary match
                pattern = r"\b" + re.escape(ref_clean) + r"\b"
                if re.search(pattern, pred):
                    rewards.append(1.0)
                    continue
            else:
                if ref_clean in pred:
                    rewards.append(1.0)
                    continue

            # ── Путь 4: Jaccard-fallback по всем токенам ──────────────────────
            pred_toks    = set(re.findall(r"[a-zA-Z0-9]+|\\[a-zA-Z]+", pred))
            ref_toks_all = re.findall(r"[a-zA-Z0-9]+|\\[a-zA-Z]+", ref_clean)
            ref_toks     = set(ref_toks_all[:100])  # ограничиваем для скорости
            jaccard = (
                len(pred_toks & ref_toks) / max(1, len(pred_toks | ref_toks))
                if (pred_toks or ref_toks) else 0.0
            )
            rewards.append(jaccard)

        return rewards

    # ── Callback: остановка при стабилизации метрик ───────────────────────────
    class StabilityStopCallback(TrainerCallback):
        """Дублирует метрики в GLOBAL_HISTORY. Остановка реализована в RLEarlyStoppingCallback."""
        def __init__(self, threshold):
            self.threshold = threshold

        def on_log(self, args, state, control, logs=None, **kwargs):
            global GLOBAL_STEP_COUNTER, GLOBAL_HISTORY
            if not logs:
                return
            loss_val   = logs.get("loss")
            reward_val = logs.get("rewards/mean", logs.get("reward"))
            kl_val     = logs.get("kl", logs.get("policy_kl"))

            if loss_val is not None or reward_val is not None:
                GLOBAL_STEP_COUNTER += 1
                GLOBAL_HISTORY["step"].append(GLOBAL_STEP_COUNTER)
                GLOBAL_HISTORY["loss"].append(loss_val)
                GLOBAL_HISTORY["reward"].append(reward_val)
                GLOBAL_HISTORY["kl"].append(kl_val)
                if kl_val is not None or reward_val is not None:
                    print(f"[Шаг {GLOBAL_STEP_COUNTER}] Reward: {_fmt(reward_val)} | KL: {_fmt(kl_val)}")

    # ── Callback: ранняя остановка RL ─────────────────────────────────────────
    class RLEarlyStoppingCallback(TrainerCallback):
        """
        Запускает генерацию на eval-выборке и вычисляет reward.
        Останавливает обучение, если за `patience` шагов улучшения < threshold.
        Синхронизирует метрику через dist.broadcast, чтобы все GPU остановились одновременно.
        """
        def __init__(self, model, eval_dataset, reward_fn, tokenizer,
                     patience=3, threshold=0.001, max_new_tokens=128):
            self.model          = model
            self.eval_dataset   = eval_dataset
            self.reward_fn      = reward_fn
            self.tokenizer      = tokenizer
            self.max_new_tokens = max_new_tokens
            self.patience       = patience
            self.threshold      = threshold
            self.best_reward    = -float("inf")
            self.wait           = 0

        def _generate_batch(self, model, prompts):
            """Генерирует ответы модели на список промптов (жадная декодировка)."""
            self.model.eval()
            inputs         = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            input_ids      = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # Декодируем только сгенерированную часть (без промпта)
            return self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            import torch.distributed as dist

            reward_tensor   = torch.zeros(1).cuda()
            is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

            if is_main_process:
                print(f"\n[Eval] Генерируем ответы на {len(self.eval_dataset)} примерах (ранг 0)...")
                prompts = list(self.eval_dataset["prompt"])
                refs    = list(self.eval_dataset["expert_action"])
                preds   = self._generate_batch(self.model, prompts)
                rewards = self.reward_fn(prompts, preds, refs)
                mean_reward = float(np.mean(rewards)) if rewards else 0.0
                reward_tensor[0] = mean_reward
                if metrics is not None:
                    metrics["eval_reward"] = mean_reward

            # Транслируем метрику на все ранги — они должны принять одно решение
            if dist.is_initialized():
                dist.broadcast(reward_tensor, src=0)

            current_metric = reward_tensor.item()

            if is_main_process:
                print(f"[Eval] Reward (синхр.): {current_metric:.4f}")

            if current_metric > self.best_reward + self.threshold:
                self.best_reward = current_metric
                self.wait = 0
                if is_main_process:
                    print("[EarlyStop] Новый лучший результат!")
            else:
                self.wait += 1
                if is_main_process:
                    print(f"[EarlyStop] Без улучшений {self.wait}/{self.patience}")
                if self.wait >= self.patience:
                    control.should_training_stop = True
                    if is_main_process:
                        print("[EarlyStop] Остановка обучения на всех GPU.")

    # ── Logits Processor: clamp логитов ──────────────────────────────────────
    class ClampLogitsProcessor(LogitsProcessor):
        """
        Ограничивает логиты в диапазоне [min_val, max_val].
        Предотвращает численное переполнение при генерации, особенно с bfloat16.
        """
        def __init__(self, min_val=-100.0, max_val=100.0):
            self.min_val = min_val
            self.max_val = max_val

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            return torch.clamp(scores, min=self.min_val, max=self.max_val)

    # ── Callback: eval-reward через генерацию (для SFT-фазы) ─────────────────
    class InjectEvalRewardCallback(TrainerCallback):
        """
        Запускается при каждой eval-итерации тренера.
        Генерирует ответы только на ранге 0, затем транслирует метрику на остальные.
        Используется в SFT-фазе, где стандартный тренер не считает reward.
        """
        def __init__(self, tokenizer, reward_fn, model, eval_dataset, max_new_tokens=128):
            self.tokenizer      = tokenizer
            self.reward_fn      = reward_fn
            self.model          = model
            self.eval_dataset   = eval_dataset
            self.max_new_tokens = max_new_tokens

        def _generate_batch(self, model, prompts):
            model.eval()
            unwrapped_model = model.module if hasattr(model, "module") else model

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(unwrapped_model.device)

            input_ids      = inputs["input_ids"].to(unwrapped_model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(unwrapped_model.device)

            with torch.no_grad():
                out = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([ClampLogitsProcessor()]),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_lens = inputs["input_ids"].shape[1]
            texts = []
            for seq in out:
                gen_part = seq[input_lens:].cpu().numpy().tolist()
                texts.append(self.tokenizer.decode(gen_part, skip_special_tokens=True).strip())

            model.train()
            return texts

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            import torch.distributed as dist

            reward_tensor   = torch.zeros(1).cuda()
            is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

            if is_main_process:
                print(f"\n[Eval Callback] Генерация на {len(self.eval_dataset)} примерах (ранг 0)...")
                prompts = list(self.eval_dataset["prompt"])
                refs    = list(self.eval_dataset["expert_action"])
                preds   = self._generate_batch(self.model, prompts)
                rewards = self.reward_fn(prompts, preds, refs)
                mean_reward = float(np.mean(rewards)) if rewards else 0.0
                reward_tensor[0] = mean_reward
                if metrics is not None:
                    metrics["eval_reward"] = mean_reward
                print(f"[Eval Callback] Eval Reward: {mean_reward:.4f}")

            if dist.is_initialized():
                dist.broadcast(reward_tensor, src=0)

    # ── Callback: логирование градиентов ─────────────────────────────────────
    class GradientLoggerCallback(TrainerCallback):
        """
        Периодически выводит нормы весов и градиентов для LoRA-параметров.
        Помогает диагностировать затухающие/взрывающиеся градиенты.
        """
        def __init__(self, log_every_n_steps=50):
            self.log_every_n_steps = log_every_n_steps

        def on_backward(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.log_every_n_steps == 0:
                print(f"\n[Backward] Шаг {state.global_step} | Нормы градиентов/весов:")
                has_grads = False
                grad_info = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        p_norm    = param.norm().item()
                        grad_norm = param.grad.norm().item() if param.grad is not None else None
                        if grad_norm is not None:
                            has_grads = True
                        if "lora" in name or "lm_head" in name:
                            grad_info.append(f"  {name} | param={p_norm:.6f} | grad={grad_norm}")

                for info in grad_info[:10]:
                    print(info)
                if len(grad_info) > 10:
                    print(f"  ... и ещё {len(grad_info) - 10} параметров")
                if not has_grads:
                    print("  ⚠️  ВНИМАНИЕ: Градиенты не вычислены!")
                print("-" * 50)

    def print_gpu_memory():
        """Выводит текущее и зарезервированное потребление VRAM по всем GPU."""
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved  = torch.cuda.memory_reserved(i)  / 1024 ** 3
            print(f"GPU {i}: Выделено={allocated:.2f}ГБ | Зарезервировано={reserved:.2f}ГБ")

    # ── Подготовка датасета для конкретного уровня ────────────────────────────
    def prepare_dataset_for_level(ds_split, level, include_think):
        """
        Форматирует сырые данные уровня в промпты и эталонные ответы.

        include_think=True (уровни 2–6): добавляем тег <think> для цепочки рассуждений.
        include_think=False (уровень 1, SFT): простой формат «вопрос → ответ».

        Источники форматируются по-разному:
          gsm8k / gsm8k_replay — пошаговое решение с историей
          comp_math           — олимпиадная задача, ответ в \\boxed{}
          s1k                 — сложная задача, свободный формат решения
        """
        prompts     = []
        expects     = []
        completions = []
        sources     = []

        # Системный промпт (на английском — влияет на поведение модели)
        system_text = "You are a math problem solver. Think step by step and provide a clear solution."

        for pb, hist, ex_step, src in zip(
            ds_split["prompt"], ds_split["history"],
            ds_split["expert_action"], ds_split["source"]
        ):
            if src in ("gsm8k", "gsm8k_replay"):
                if include_think:
                    # Сократовский формат: вопрос → шаг рассуждения → ответ
                    if "**" in ex_step:
                        parts     = ex_step.split("**", 1)
                        socratic_q, math_a = parts[0].strip(), parts[1].strip()
                    else:
                        socratic_q, math_a = "Solve the next step", ex_step
                    full_prompt = (
                        f"{system_text}\n\nProblem:\n{pb}\nHistory:\n{hist}\n"
                        f"Next step objective: {socratic_q}\n<think>"
                    )
                    expert = f"{socratic_q}</think>\n<solution>{math_a}</solution>"
                else:
                    # SFT-формат: без цепочки рассуждений
                    full_prompt = (
                        f"{system_text}\n\nProblem:\n{pb}\nHistory:\n{hist}\n"
                        f"Next step:\n<solution>"
                    )
                    expert = f"{ex_step}</solution>"

            elif src == "comp_math":
                # Олимпиадная задача: ответ обязательно в \boxed{}
                full_prompt = (
                    f"{system_text}\n\n"
                    f"Solve the following competition math problem. "
                    f"Show your reasoning step by step, then give the final answer in \\boxed{{}}.\n\n"
                    f"Problem:\n{pb}\n\n<think>"
                )
                expert = ex_step

            else:
                # s1k — длинные рассуждения, свободный формат
                full_prompt = (
                    f"{system_text}\n\nSolve the following complex math problem.\n"
                    f"Problem:\n{pb}\n\nAnalyze and provide the solution.\n<think>"
                )
                expert = ex_step

            prompts.append(full_prompt)
            expects.append(expert)
            completions.append(expert)
            sources.append(src)

        return Dataset.from_dict({
            "prompt":        prompts,
            "expert_action": expects,
            "completion":    completions,
            "source":        sources,
        })

    # ── Главный цикл обучения по уровням ─────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    callbacks = [
        GlobalPlottingCallback(),
        GradientLoggerCallback(log_every_n_steps=50),
    ]

    current_level = STARTING_LEVEL

    # Если есть сохранённый адаптер — копируем во временную директорию
    # (защита от изменений в Kaggle input, который read-only)
    if RESUME_ADAPTER_PATH and os.path.exists(RESUME_ADAPTER_PATH):
        _tmp_adapter = "/tmp/resume_adapter"
        if os.path.exists(_tmp_adapter):
            shutil.rmtree(_tmp_adapter)
        shutil.copytree(RESUME_ADAPTER_PATH, _tmp_adapter, dirs_exist_ok=True)
        print(f"[Возобновление] Адаптер скопирован в {_tmp_adapter}")
        prev_adapter_path = _tmp_adapter
    else:
        prev_adapter_path = None
        print(f"[Возобновление] Старт с уровня {STARTING_LEVEL}, адаптер не загружается")

    while current_level <= MAX_CURRICULUM_LEVELS:
        print(f"\n{'=' * 40} Уровень {current_level} {'=' * 40}")

        ds_splits = buckets.get(current_level)
        if not ds_splits or len(ds_splits["train"]) == 0:
            print(f"[Пропуск] Уровень {current_level}: данных нет")
            current_level += 1
            continue

        # На уровне 1 (SFT) тег <think> не нужен — модель ещё не обучена рассуждать
        include_think = current_level >= 2
        train_ds      = prepare_dataset_for_level(ds_splits["train"], current_level, include_think)
        eval_ds_full  = prepare_dataset_for_level(ds_splits["eval"],  current_level, include_think)

        # Для SFT достаточно 64 примеров на eval; для GRPO — 16 (генерация дорогая)
        if current_level == 1:
            MAX_EVAL_SFT = 64
            eval_ds = eval_ds_full.select(range(min(MAX_EVAL_SFT, len(eval_ds_full))))
        else:
            MAX_EVAL_RL = 16
            eval_ds = eval_ds_full.select(range(min(MAX_EVAL_RL, len(eval_ds_full))))

        model, tokenizer = get_model_and_tokenizer(
            current_level,
            prev_adapter_path=prev_adapter_path,
            accelerator=accelerator,
            device_str=device_str,
            compute_dtype=compute_dtype,
        )

        output_dir = f"tmp_output/level_{current_level}"
        os.makedirs(output_dir, exist_ok=True)

        # ── Гиперпараметры GRPO: подобраны под каждый уровень сложности ───────
        # Чем выше уровень — больше длина генерации и температура, меньше шагов
        if current_level == 2:
            max_steps     = 100
            grad_accum    = 8
            eval_steps    = 12
            num_gens      = 8
            max_compl_len = 128
            temperature   = 1.0
        elif current_level == 3:
            max_steps     = 73
            grad_accum    = 8
            eval_steps    = 999
            num_gens      = 8
            max_compl_len = 192
            temperature   = 1.0
        elif current_level == 4:
            max_steps     = 35
            grad_accum    = 8
            eval_steps    = 999
            num_gens      = 8
            max_compl_len = 512
            temperature   = 1.1
        elif current_level == 5:
            max_steps     = 15
            grad_accum    = 8
            eval_steps    = 999
            num_gens      = 4
            max_compl_len = 512
            temperature   = 1.1
        elif current_level == 6:
            max_steps     = 35
            grad_accum    = 4
            eval_steps    = 999
            num_gens      = 4
            max_compl_len = 512
            temperature   = 1.2
        else:
            # Запасной вариант на случай расширения курикулума
            max_steps     = 35
            grad_accum    = 4
            eval_steps    = 999
            num_gens      = 4
            max_compl_len = 512
            temperature   = 1.0

        current_callbacks = callbacks.copy()

        if current_level == 1:
            # ── Уровень 1: SFT (Supervised Fine-Tuning) ───────────────────────
            print(">>> Режим: SFT")
            args = SFTConfig(
                output_dir=output_dir,
                max_steps=50,
                per_device_train_batch_size=4,
                learning_rate=2e-4,
                gradient_checkpointing=True,
                gradient_accumulation_steps=2,
                lr_scheduler_type="cosine",
                optim="adamw_torch",
                logging_steps=5,
                warmup_steps=10,
                do_eval=True,
                eval_strategy="steps",
                eval_steps=10,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                fp16=not bf16_supported,
                bf16=bf16_supported,
                ddp_find_unused_parameters=True,
                dataloader_pin_memory=True,
                local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            )
            try:
                args.max_seq_length = 512
                args.packing        = False
            except Exception:
                pass

            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=tokenizer,
                callbacks=current_callbacks,
            )

        else:
            # ── Уровни 2–6: GRPO (Group Relative Policy Optimization) ─────────
            print(f">>> Режим: GRPO | Генераций: {num_gens} | Длина: {max_compl_len}")

            # RLEarlyStoppingCallback следит за reward и останавливает при плато
            rl_early_stop = RLEarlyStoppingCallback(
                model=model,
                eval_dataset=eval_ds,
                reward_fn=combined_reward,
                tokenizer=tokenizer,
                patience=3,
                threshold=0.01,
            )
            current_callbacks.append(rl_early_stop)

            grpo_config = GRPOConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=grad_accum,
                learning_rate=5e-6,
                num_generations=num_gens,
                max_completion_length=max_compl_len,
                temperature=temperature,
                beta=0.02,          # коэффициент KL-штрафа (малый = меньше регуляризации)
                logging_steps=5,
                do_eval=False,
                eval_strategy="no",
                eval_steps=eval_steps,
                save_strategy="no",
                report_to="none",
                fp16=not bf16_supported,
                bf16=bf16_supported,
                ddp_timeout=7200,
                dataloader_drop_last=True,
                ddp_find_unused_parameters=True,
                dataloader_pin_memory=True,
                local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            )

            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=combined_reward,
                args=grpo_config,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                callbacks=current_callbacks,
            )

            print("Запуск обучения...")

        # ── Диагностика: число обучаемых параметров ───────────────────────────
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Диагностика] Обучаемых параметров: {total_trainable:,}")

        # Отладочный хук на первый LoRA-параметр (отключён в продакшне)
        DEBUG_GRADIENTS = False
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_A" in name:
                if DEBUG_GRADIENTS:
                    def gradient_hook(grad, nm=name):
                        print(f"  ✓ [Hook] grad {nm}: norm={grad.norm().item():.6f}")
                        return grad
                    param.register_hook(gradient_hook)
                    print(f"[Отладка] Хук зарегистрирован на: {name}")
                break

        # ── Обучение ──────────────────────────────────────────────────────────
        try:
            trainer.train()

            import torch.distributed as dist
            if dist.is_initialized():
                print(f"[Ранг {local_rank}] Обучение завершено, синхронизация рангов...")
                dist.barrier()

        except Exception as e:
            print(f"!!! ОШИБКА ПРИ ОБУЧЕНИИ: {e}")
            import traceback
            traceback.print_exc()
            break

        # ── Сохранение адаптера ───────────────────────────────────────────────
        final_path = f"models_saved/level_{current_level}"
        os.makedirs(final_path, exist_ok=True)

        try:
            # Пробуем сохранить через trainer (предпочтительно — корректно обрабатывает PEFT)
            if hasattr(trainer, "model") and isinstance(trainer.model, PeftModel):
                trainer.model.save_pretrained(final_path)
            elif isinstance(model, PeftModel):
                model.save_pretrained(final_path)
            else:
                trainer.save_model(final_path)
            print(f"[Сохранение] PEFT-адаптер → {final_path}")
        except Exception as e:
            print(f"[Сохранение] Основной способ не сработал, пробуем trainer.save_model(): {e}")
            trainer.save_model(final_path)

        tokenizer.save_pretrained(final_path)
        prev_adapter_path = final_path

        print(f"Уровень {current_level} завершён. Адаптер: {final_path}")
        save_plots()
        print_gpu_memory()

        # Освобождаем VRAM перед следующим уровнем
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        current_level += 1

    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")


if __name__ == "__main__":
    main()
