#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import os
import subprocess
import traceback
import time

# ЛОГИРОВАНИЕ
SCRIPT_DIR = Path(__file__).parent.absolute()
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "logs.txt"

def log_print(msg):
    """print() + log в файл"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] convert.py: {msg}\n"
    
    # КОНСОЛЬ (как раньше)
    print(msg)
    sys.stdout.flush()
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def log_error(msg):
    """print() + ОШИБКА в лог"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] ОШИБКА convert.py: {msg}\n"
    
    # КОНСОЛЬ (как раньше)
    print(msg)
    sys.stdout.flush()
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def show_error(e):
    """Показывает ошибку + логирует"""
    error_msg = f"{str(e)}\n{traceback.format_exc()}"
    log_error(error_msg)
    print("\n" + "="*60)
    traceback.print_exc()
    print("="*60)
    print("\nСкопируйте ошибку выше!")
    print("\nПОЛНАЯ ОШИБКА В logs/logs.txt")

print("HF -> GGUF КОНВЕРТАЦИЯ")

try:
    if len(sys.argv) < 2:
        raise ValueError("Использование: python convert.py 'json_params'")
    
    log_print("Чтение параметров...")
    params_json = sys.argv[1]
    params = json.loads(params_json)
    
    HF_PATH = params['hf_path']
    GGUF_PATH = params['gguf_path']
    
    log_print(f"HF модель: {HF_PATH}")
    log_print(f"GGUF выход: {GGUF_PATH}")
    
    if not os.path.exists(HF_PATH):
        raise FileNotFoundError(f"HF модель не найдена: {HF_PATH}")
    
    os.makedirs(os.path.dirname(GGUF_PATH), exist_ok=True)
    
    convert_script = 'convert_hf_to_gguf.py'
    if not os.path.exists(convert_script):
        log_print("Скачиваем convert_hf_to_gguf.py...")
        subprocess.run([
            'powershell', '-Command', 
            f'Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py" -OutFile "{convert_script}"'
        ], check=True)
    
    # ФИКС: используем VENV_PYTHON для convert_hf_to_gguf.py
    venv_python = Path(sys.executable).parent.parent / "Scripts" / "python.exe"
    log_print(f"Используем Python: {venv_python}")
    
    log_print("Конвертация HF -> GGUF...")
    result = subprocess.run([
        str(venv_python), convert_script,
        HF_PATH,
        '--outfile', GGUF_PATH,
        '--outtype', 'f16'
    ], capture_output=True, text=True)
    
    log_print("STDOUT: " + result.stdout[:300])
    if result.stderr:
        log_print("STDERR: " + result.stderr)
    
    if os.path.exists(GGUF_PATH):
        size_gb = os.path.getsize(GGUF_PATH) / (1024**3)
        log_print(f"GGUF готов: {GGUF_PATH} ({size_gb:.1f}GB)")
    else:
        raise RuntimeError("Ошибка конвертации!")
    
    log_print("КОНВЕРТАЦИЯ ОК")

except Exception as e:
    show_error(e)

print("\n" + "="*50)
print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА")
# input("Нажмите ENTER для закрытия...")
