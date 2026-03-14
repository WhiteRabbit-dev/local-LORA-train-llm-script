#!/usr/bin/env python3
#
import sys
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import gc
import os
import time
import traceback

# ЛОГИРОВАНИЕ
SCRIPT_DIR = Path(__file__).parent.absolute()
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "logs.txt"

def log_print(message):
    """print() + log в файл"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] train.py: {message}\n"
    
    # КОНСОЛЬ (как раньше)
    print(message)
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def log_error(message):
    """print() + ОШИБКА в лог"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    log_line = f"[{timestamp}] ОШИБКА train.py: {message}\n"
    
    # КОНСОЛЬ (как раньше)
    print(message)
    
    # ФАЙЛ
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

def safe_run():
    try:
        # GPU ТЕСТ + FALLBACK
        device = "cuda" 
        try:
            test = torch.randn(10,10, device="cuda")
            del test
            log_print("GPU тест OK")
        except:
            device = "cpu"
            log_print("GPU недоступен -> CPU")

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if device == "cuda": 
            torch.cuda.empty_cache()
        gc.collect()

        log_print("=== LoRA ОБУЧЕНИЕ (VENV) ===")

        # ЧИТАЕМ ПАРАМЕТРЫ ИЗ АРГУМЕНТОВ
        if len(sys.argv) < 2:
            log_print("Использование: python train_lora.py 'json_params'")
            return

        params_json = sys.argv[1]
        params = json.loads(params_json)

        MODEL_PATH = params['model_path']
        DATASET_PATH = params['dataset_path'] 
        LORA_PATH = params['lora_path']
        EPOCHS = params['epochs']
        LR = params['lr']
        BATCH_SIZE = params['batch_size']
        LORA_R = params['lora_r']
        LORA_ALPHA = params['lora_alpha']
        MAX_LEN = params['max_len']
        LORA_TARGETS = params['lora_targets']

        log_print(f"Device: {device}")
        log_print(f"Модель: {MODEL_PATH}")
        log_print(f"Датасет: {DATASET_PATH}")
        log_print(f"LoRA: {LORA_PATH}")

        # Токенизатор
        log_print("Токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log_print("Токенизатор OK")

        # Модель - с CPU fallback
        log_print("Модель...")
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            dtype=dtype,         
            low_cpu_mem_usage=True, 
            device_map=None,              
            trust_remote_code=True         
        )
        model = model.to(device)    
        model.config.use_cache = False       
        model.gradient_checkpointing_enable()
        log_print("Модель OK")

        # LoRA - ИСПРАВЛЕНО!
        log_print("LoRA config...")
        lora_config = LoraConfig(
            r=LORA_R, 
            lora_alpha=LORA_ALPHA, 
            lora_dropout=0.1,
            target_modules=LORA_TARGETS, 
            bias="none", 
            task_type=TaskType.CAUSAL_LM
        )
        peft_model = get_peft_model(model, lora_config)
        model = peft_model
        log_print("LoRA применен")

        # Dataset
        class CustomDataset(Dataset):
            def __init__(self, file_path, tokenizer, max_length):
                log_print("КЭШИРУЕМ ДАТАСЕТ В RAM...")
                self.texts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            text = f"<|im_start|>system\n{data['instruction']}<|im_end|>\n<|im_start|>user\n{data['input']}<|im_end|>\n<|im_start|>assistant\n{data['output']}<|im_end|>"
                            self.texts.append(text)
                        except Exception as e:
                            log_error(f"Строка {i}: {e}")
                
                self.tokenizer = tokenizer
                self.max_length = max_length
                log_print(f"ДАТАСЕТ: {len(self.texts)} примеров")
            
            def __len__(self): 
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text, truncation=True, max_length=self.max_length,
                    padding='max_length', return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": encoding["input_ids"].flatten()
                }

        log_print("Датасет...")
        dataset = CustomDataset(DATASET_PATH, tokenizer, MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=device == "cuda")

        # Обучение
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        model.train()

        total_steps = len(dataloader) * EPOCHS
        step_count = 0
        start_time = time.time()

        log_print(f"=== ОБУЧЕНИЕ ({total_steps} шагов) ===")

        for epoch in range(EPOCHS):
            log_print(f"Эпоха {epoch+1}/{EPOCHS}")
            for batch_idx, batch in enumerate(dataloader):
                step_count += 1
                progress = (step_count / total_steps) * 100
                
                loss_value = None
                
                try:
                    # УНИВЕРСАЛЬНЫЙ autocast
                    with torch.amp.autocast(device):
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        labels = batch["labels"].to(device, non_blocking=True)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        loss_value = loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                except RuntimeError as e:
                    log_error(f"ОШИБКА шаг {step_count}: {e}")
                    break
                
                # Очистка - универсальная
                try:
                    del input_ids, attention_mask, labels, outputs, loss
                except:
                    pass
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Статус
                elapsed = time.time() - start_time
                speed = step_count / elapsed if elapsed > 0 else 0
                time_left = (total_steps - step_count) / speed if speed > 0 else 0
                log_print(f"Шаг {step_count}/{total_steps} ({progress:.1f}%) | "
                         f"Loss: {loss_value:.3f} | "
                         f"Скорость: {speed:.2f} шаг/сек | "
                         f"Осталось: {time_left/60:.1f}мин")

        log_print("\n" + "="*60)
        log_print("ОБУЧЕНИЕ ОКОНЧЕНО")
        log_print(f"LoRA: {LORA_PATH}")

        log_print("Сохранение...")
        Path(LORA_PATH).mkdir(exist_ok=True)
        model.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
        log_print(f"LoRA СОХРАНЕН: {LORA_PATH}")
        log_print("ОК!")
        
    except Exception as e:
        error_msg = f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}\n{traceback.format_exc()}"
        log_error(error_msg)
        print("ПОЛНАЯ ОШИБКА В logs/logs.txt")

   # print("\n" + "="*50)
   # print("Нажмите ENTER для закрытия...")
   # input()

if __name__ == "__main__":
    safe_run()
