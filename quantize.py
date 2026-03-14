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
    log_line = f"[{timestamp}] quantize.py: {msg}\n"
    
    # КОНСОЛЬ (как раньше)
    print(msg)
    sys.stdout.flush()
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def log_error(msg):
    """print() + ОШИБКА в лог"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] ОШИБКА quantize.py: {msg}\n"
    
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

def fix_windows_path(path):
    """Экранирование Windows путей"""
    return str(path).replace('\\\\', '/').strip()

print("GGUF КВАНТОВАНИЕ")

try:
    if len(sys.argv) < 2:
        raise ValueError("Использование: python quantize.py 'json_params'")
    
    log_print("Чтение параметров...")
    params_json = sys.argv[1]
    params = json.loads(params_json)
    
    F16_PATH = fix_windows_path(params['f16_path'])
    Q_PATH = fix_windows_path(params['q_path'])
    QUANT_METHOD = params['quant_method']
    
    log_print(f"F16 GGUF: {F16_PATH}")
    log_print(f"Q выход: {Q_PATH}")
    log_print(f"Метод: {QUANT_METHOD}")
    
    if not os.path.exists(params['f16_path']):
        raise FileNotFoundError(f"F16 GGUF не найден: {params['f16_path']}")
    
    file_size = os.path.getsize(params['f16_path']) / (1024**3)
    log_print(f"F16 GGUF ({file_size:.1f}GB) готов")
    
    # Поиск llama-quantize.exe
    log_print("Поиск llama-quantize.exe...")
    quantize_paths = [
        'llama.cpp/build/bin/llama-quantize.exe',
        'llama.cpp/llama-quantize.exe', 
        'llama-quantize.exe',
        './llama-quantize.exe',
        'bin/llama-quantize.exe',
        'llama.cpp/build/bin/Release/llama-quantize.exe'
    ]
    
    quantize_exe = None
    quantize_exe_path = None
    for rel_path in quantize_paths:
        full_path = Path(rel_path).resolve()
        if full_path.exists():
            quantize_exe = str(full_path)
            quantize_exe_path = full_path.parent
            break
    
    if not quantize_exe:
        raise FileNotFoundError("llama-quantize.exe не найден! Скачай llama.cpp")
    
    log_print(f"Найден: {quantize_exe}")
    log_print(f"CWD: {quantize_exe_path}")
    
    # Тестовый запуск
    log_print("Тест llama-quantize.exe...")
    test_result = subprocess.run([quantize_exe, "--help"], 
                                capture_output=True, text=True)
    log_print("llama-quantize.exe готов!")
    
    os.makedirs(os.path.dirname(params['q_path']), exist_ok=True)
    
    log_print(f"Квантование F16 -> {QUANT_METHOD}...")
    log_print(f"Команда: {quantize_exe} \"{params['f16_path']}\" \"{params['q_path']}\" {QUANT_METHOD}")
    
    # ПРАВИЛЬНЫЙ вызов с КАВЫЧКАМИ и абсолютными путями
    result = subprocess.run([
        quantize_exe, 
        str(params['f16_path']), 
        str(params['q_path']),    
        QUANT_METHOD
    ], capture_output=True, text=True, 
      cwd=quantize_exe_path)
    
    log_print("STDOUT: " + result.stdout[:300])
    if result.stderr:
        log_print("STDERR: " + result.stderr[:300])
    log_print(f"Код возврата: {result.returncode}")
    
    if os.path.exists(params['q_path']):
        size_gb = os.path.getsize(params['q_path']) / (1024**3)
        log_print(f"{QUANT_METHOD} готов: {params['q_path']} ({size_gb:.1f}GB)")
    else:
        raise RuntimeError("Квантование не удалось!")
    
    log_print("КВАНТОВАНИЕ ОК")

except Exception as e:
    show_error(e)

print("\n" + "="*50)
print("КВАНТОВАНИЕ ЗАВЕРШЕНО")
# input("Нажмите ENTER для закрытия...")
