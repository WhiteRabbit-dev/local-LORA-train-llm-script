#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc
import os
import shutil
import traceback
import time

# ЛОГИРОВАНИЕ
SCRIPT_DIR = Path(__file__).parent.absolute()
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "logs.txt"

def log_print(message):
    """print() + log в файл"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] merge.py: {message}\n"
    
    # КОНСОЛЬ (как раньше)
    print(message)
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def log_error(message):
    """print() + ОШИБКА в лог"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] ОШИБКА merge.py: {message}\n"
    
    # КОНСОЛЬ (как раньше)
    print(message)
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def show_error(e):
    """Показывает ошибку + логирует"""
    error_msg = f"{str(e)}\n{traceback.format_exc()}"
    log_error(error_msg)

try:
    log_print("=== LoRA СЛИЯНИЕ ===")

    # ПАРАМЕТРЫ ИЗ АРГУМЕНТА
    if len(sys.argv) < 2:
        log_print("Использование: python merge.py 'json_params'")
        sys.exit(1)

    params_json = sys.argv[1]
    params = json.loads(params_json)

    MODEL_PATH = params['model_path']
    LORA_PATH = params['lora_path'] 
    MERGED_PATH = params['merged_path']

    log_print(f"Базовая модель: {MODEL_PATH}")
    log_print(f"LoRA: {LORA_PATH}")
    log_print(f"Сохранение: {MERGED_PATH}")

    # ПРОВЕРКИ
    assert os.path.exists(MODEL_PATH), f"БАЗОВАЯ МОДЕЛЬ НЕ НАЙДЕНА: {MODEL_PATH}"
    assert os.path.exists(LORA_PATH), f"LoRA НЕ НАЙДЕНА: {LORA_PATH}"

    # МОДЕЛЬ
    log_print("Загрузка базовой модели (CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # ТОКЕНИЗАТОР
    log_print("Токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # СЛИЯНИЕ LoRA
    log_print("Применяем LoRA веса...")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    log_print("LoRA успешно загружена")

    log_print("MERGE...")
    model = model.merge_and_unload()
    log_print("Merge завершен")

    # СОХРАНЕНИЕ
    log_print("СОХРАНЕНИЕ...")
    Path(MERGED_PATH).mkdir(exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=False)
    tokenizer.save_pretrained(MERGED_PATH)

    # КОПИЯ tokenizer.model для GGUF
    orig_tokenizer = os.path.join(MODEL_PATH, 'tokenizer.model')
    merged_tokenizer = os.path.join(MERGED_PATH, 'tokenizer.model')
    if os.path.exists(orig_tokenizer):
        shutil.copy2(orig_tokenizer, merged_tokenizer)
        log_print("tokenizer.model скопирован")
    else:
        log_print("tokenizer.model НЕ НАЙДЕН")

    # ПРОВЕРКА ФАЙЛОВ
    required = ['config.json', 'tokenizer.json', 'tokenizer.model', 'model.safetensors']
    log_print("GGUF-ready файлы:")
    for filename in required:
        filepath = os.path.join(MERGED_PATH, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            log_print(f"  OK {filename} {size_mb:.1f}MB")
        else:
            log_print(f"  MISSING {filename}")

    log_print(f"СЛИЯНИЕ ОК: {MERGED_PATH}")

    # ОЧИСТКА
    del model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_print("Память очищена")

    log_print("=== ОК! ===")
   # input("Нажмите ENTER для закрытия...")

except Exception as e:
    show_error(e)
    input("Нажмите ENTER для закрытия...")
