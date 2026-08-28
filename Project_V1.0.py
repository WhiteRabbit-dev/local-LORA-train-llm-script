import tkinter as tk
from tkinter import ttk
from pathlib import Path
from tkinter import filedialog
import subprocess
import re
import sys
import threading
import time
import json
import os
import gc


#Основные цвета формы:
bg_color = "bisque2"
bg_color_cons = "black"
weight_span = 3
color_span = "saddle brown"
bg_face = "bisque"
color_face = "green"
font_cons = "lime"
space_in = 5
space_out = 3

#системные
SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_DIR = SCRIPT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
LOGS_DIR = SCRIPT_DIR / "logs"
LOG_FILE = SCRIPT_DIR / "logs" / f"logs[{time.strftime("%Y-%m-%d %H.%M.%S")}].txt"
PROCESS_DIR = SCRIPT_DIR / "process"
PROCESS_DIR.mkdir(exist_ok=True)
LORA_PATH = PROCESS_DIR / "lora_adapter"
MERGED_PATH = PROCESS_DIR / "merged_f16"
GGUF_PATH = PROCESS_DIR / "model_f16.gguf"
LLAMA_CPP_DIR = SCRIPT_DIR / "llama.cpp"
FACES_DIR = SCRIPT_DIR / "faces"
FACE_STAND = FACES_DIR / "face_stand.txt"
FACE_TALK = FACES_DIR / "face_talk.txt"
FACE_NOW = FACE_STAND
#К самой модели
MODEL_PATH = ""  # модель Q4
text_monitor = None
mon_device = "CPU"
device = "cpu"

def handle_ctrl_key(event):
    kc = event.keycode
    #Ctrl+V (русская М) - Вставить
    if kc == 86:
        event.widget.event_generate('<<Paste>>')
        return 'break'
    #Ctrl+C (русская С) - Копировать
    elif kc == 67:
        event.widget.event_generate('<<Copy>>')
        return 'break'
    #Ctrl+X (русская Ч) - Вырезать
    elif kc == 88:
        event.widget.event_generate('<<Cut>>')
        return 'break'
    #Ctrl+Z (русская Я) - Отменить
    elif kc == 90:
        event.widget.event_generate('<<Undo>>')
        return 'break'

def activate_venv():
    global VENV_DIR, mon_device
    if not VENV_PYTHON.exists():
        log_to_file("создание .venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        log_to_file(".venv создан")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    try:
        if sys.executable != str(VENV_PYTHON):
            log_to_file(f"Перезапуск в venv: {VENV_PYTHON}")
            # Передаём все аргументы и переменные окружения
            subprocess.Popen(
                [str(VENV_PYTHON)] + sys.argv,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            sys.exit(0)  # Выходим из текущего процесса
        else:
            log_to_file(f"venv уже прописан: {VENV_PYTHON}")
    except Exception as e:
        log_to_file(f"не удалось добавить директорию venv: {e}", True)
    try:
        try:
            import numpy
        except:
            subprocess.run(
                [str(VENV_PIP), "install", "numpy"], check=True)
    except Exception as e:
        log_to_file(f"не удалось установить библиотеку numpy: {e}", True)
    try:
        try:
            import huggingface_hub
        except:
            subprocess.run(
                [str(VENV_PIP), "install", "huggingface_hub"], check=True)
    except Exception as e:
        log_to_file(f"не удалось установить библиотеку huggingface_hub: {e}", True)
    try:
        import psutil
    except:
        try:
            subprocess.run(
                [str(VENV_PIP), "install", "psutil"],
                check=True)
            add_text_cons(f"Установлена библиотека для мониторинга состояния CPU")
        except Exception as e:
            add_text_cons(f"не удалось установить psutil: {e}")
    try:
        import PIL
        import requests
    except Exception as e:
        log_to_file(f"Ошибка pillow и requests: {e}")
        try:
            subprocess.run(
                [str(VENV_PIP), "install", "pillow ","requests"],
                check=True)
            add_text_cons(f"Установлена библиотека для отображения артов")
        except Exception as e:
            add_text_cons(f"не удалось установить pillow и requests: {e}")

    try:
        import torch
    except:
        install_torch_with_cuda()
    log_to_file("библиотеки подготовлены")

def install_cuda(index):
    try:
        subprocess.run(
            [str(VENV_PIP), "install", "torch", "torchvision", "torchaudio",
             "--index-url", f"https://download.pytorch.org/whl/{index}"],
            check=True)
        log_to_file(f"PyTorch установлен версии {index}")
    except Exception as e:
        log_to_file(f"не удалось установить PyTorch: {e}", True)

def install_torch_with_cuda():
    cuda_ver = None
    #AMD
    try:
        # Проверяем наличие hipInfo.exe (если найдем его то значит rcom установлен)
        result = subprocess.run(
                ["where", "hipInfo.exe"],
                capture_output=True, text=True, timeout=5, shell=True
        )
        if result.returncode == 0 and result.stdout.strip():
            hip_info = result.stdout.splitlines()[0].strip()
            #Пробуем запустить его, если ошибок не будет то видеокарту можно использовать
            result = subprocess.run(
                [hip_info], capture_output=True, text=True, timeout=10
            )
            if "device#" in result.stdout or "Name:" in result.stdout:
                log_to_file("Обнаружена видеокарта с поддержкой Rcom.")
                cuda_ver = "AMD"
                log_to_file("Устанавливаю PyTorch для AMD (ROCm)...")
                try:
                    subprocess.run(
                        [str(VENV_PIP), "install", "--pre", "torch", "torchvision", "torchaudio",
                         "--index-url", "https://download.pytorch.org/whl/nightly/rocm6.2"],
                        check=True
                    )
                    log_to_file("PyTorch для AMD (ROCm) установлен успешно!")


                except Exception as e:
                    log_to_file(f"Ошибка установки PyTorch для AMD: {e}")

                # Если в выводе есть ошибка или нет устройств - ROCm не работает
                if "no ROCm-capable device" in result.stdout.lower():
                    log_to_file("Rocm не найден/видеокарта AMD не распознана.")
        else:
            log_to_file("Rocm не найден/видеокарта AMD не распознана.")

    except Exception as e:
        log_to_file("Rocm не найден/видеокарта AMD не распознана.")

    if cuda_ver != "AMD" :
        # NVIDIA
        try:
            output = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True ).stdout
            match = re.search(r"CUDA.*Version:\s*([0-9.]+)", output)
            if match:
                cuda_ver = match.group(1)
                log_to_file(f"Обнаружена CUDA: {cuda_ver}")
            else:
                log_to_file("CUDA не найдена, ставлю CPU-версию")
                cuda_ver = None
        except:
            log_to_file("nvidia-smi не сработал, ставлю CPU-версию")
            cuda_ver = None

        cuda_to_index = {
            "13.3": "cu130",
            "13.2": "cu130",
            "13.1": "cu130",
            "13.0": "cu130",
            "12.8": "cu124",
            "12.6": "cu124",
            "12.4": "cu124",
            "12.1": "cu121",
            "11.8": "cu118",
        }
        if cuda_ver :

            index = cuda_to_index.get(cuda_ver)
            if index == None: index = "cu124"
            log_to_file(f"пробуем установить PyTorch {index}...")
            try:
                install_cuda(index)
            except:
                log_to_file(f"при попытке установить PyTorch {index} что пошло не по плану...")
                log_to_file(f"пробуем установить (наиболее стабильную) PyTorch cu124...")
                index = "cu124"
                install_cuda(index)
        else:
            log_to_file("Устанавливаю CPU-версию PyTorch")
            index = "cpu"
            install_cuda(index)
            log_to_file("Установлена CPU-версия PyTorch")

def default_face():
    global FACE_NOW,FACE_STAND
    FACE_NOW = FACE_STAND

model_ready = False
def browse_model_folder():
    global model_ready
    model_ready = False
    folder_path = filedialog.askdirectory(
        title="Выберите папку с моделью",
        initialdir=str(SCRIPT_DIR)
    )
    if folder_path:
        models_file.delete(0, "end")
        models_file.insert(0, folder_path)
    thread = threading.Thread(target=load_model_layers, args=(folder_path,))
    thread.daemon = True
    thread.start()

def browse_dataset_file():
    filename = filedialog.askopenfilename(
        title="Выберите файл датасета",
        filetypes=[
            ("jsonl датасет", "*.jsonl *.json"),
            #("jsonl датасет", "*.jsonl"),
            ("Все файлы", "*.*")
        ],
        initialdir=str(SCRIPT_DIR)  # Начать с папки проекта
    )
    if filename:
        dataset_file.delete(0, "end")
        dataset_file.insert(0, filename)

    check_paths()

def load_model_layers(model_path):
    global device, checkboxes
    try:
        from transformers import AutoModelForCausalLM
        import torch
        # Очистка виджетов
        for widget in block6_top.winfo_children():
            widget.destroy()
        waiting = tk.Label(block6_top, text="Анализ слоев модели...", justify="left", anchor="w", bg=bg_color)
        waiting.grid(row=0, column=0, sticky="nsew")
        block6_top.update()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="meta",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers.append(name)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        unique_names = set()
        for name in layers:
            short = name.split(".")[-1]
            unique_names.add(short)
        unique_names = sorted(unique_names)
        root.after(0, lambda: create_layer_checkboxes(list(unique_names)))
        root.after(0, lambda: add_text_cons(f"Найдено {len(unique_names)} типов слоев"))
    except Exception as e:
        add_text_cons(f"не удалось проанализировать выбранную модель: {e}")

def create_layer_checkboxes(layer_names):
    global checkboxes

    # Стандартные слои
    default_layers = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}

    # Очистка виджетов
    for widget in block6_top.winfo_children():
        widget.destroy()

    checkboxes = []

    # Создаем Canvas для прокрутки

    canvas = tk.Canvas(block6_top, highlightthickness=0, bg=bg_color)
    scrollbar = ttk.Scrollbar(block6_top, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=bg_color)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    block6_top.grid_rowconfigure(0, weight=1)
    block6_top.grid_columnconfigure(0, weight=1)

    sorted_names = sorted(layer_names)

    # Создаем чекбоксы
    for i, short_name in enumerate(sorted_names):
        var = tk.BooleanVar()
        if short_name in default_layers:
            var.set(True)

        cb = tk.Checkbutton(
            scrollable_frame,
            text=short_name,
            variable=var,
            anchor="w",
            bg=bg_color
        )
        cb.grid(row=i, column=0, sticky="w", padx=4, pady=0)
        checkboxes.append((var, short_name))
    global model_ready
    model_ready = True

    check_paths()

def check_paths():
    model_path = models_file.get().strip()
    dataset_path = dataset_file.get().strip()
    global model_ready
    if os.path.isdir(model_path) and (dataset_path.endswith('.json') or dataset_path.endswith('.jsonl')) and model_ready:
        Btn_start_llama.config(state="normal")
    else:
        Btn_start_llama.config(state="disabled")

def make_tooltip(widget, text):
    tip = None
    timer_id = None  # для хранения ID таймера

    def hide_tip():
        nonlocal tip, timer_id
        if tip:
            tip.destroy()
            tip = None
        if timer_id:
            root.after_cancel(timer_id)
            timer_id = None

    def show(e):
        nonlocal tip, timer_id
        if tip:
            hide_tip()
        tip = tk.Toplevel()
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{e.x_root + 15}+{e.y_root + 10}")
        tk.Label(tip, text=text, bg=bg_color, highlightbackground=color_span, highlightthickness=weight_span).pack()
        timer_id = root.after(10000, hide_tip)
        tip.bind("<Button-1>", lambda event: hide_tip())

    def hide(e):
        nonlocal timer_id
        if timer_id:
            root.after_cancel(timer_id)
        timer_id = root.after(100, hide_tip)

    widget.bind('<Enter>', show)
    widget.bind('<Leave>', hide)
    widget.bind('<Button-1>', lambda e: hide_tip())

def log_to_file(message, is_error=False):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    level = "ОШИБКА" if is_error else "ИНФО"
    log_line = f"[{timestamp}] {level}: {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())
    if is_error:
        on_closing()

def add_text_cons(text):
    global FACE_NOW, FACE_TALK, FACE_STAND
    FACE_NOW = FACE_TALK
    Console_main.config(state="normal")
    Console_main.insert("end", text +"\n")
    Console_main.see("end")
    Console_main.config(state="disabled")
    log_to_file("usr_console: " +text)
    root.after(2000, default_face)

def chk_system():
    global device, mon_device
    try:
        add_text_cons("=== ПРОВЕРКА ПК ===")
        add_text_cons("Пожалуйста дождитесь сообщения об окончании проверки.")
        # 1. Процессор
        cpu = subprocess.run(['wmic', 'cpu', 'get', 'Name'], capture_output=True, text=True, encoding='cp866')
        cpu_name = cpu.stdout.splitlines()
        cpu_name = [x.strip() for x in cpu_name if x.strip()]
        cpu_name = cpu_name[-1]

        add_text_cons(f"ПРОЦЕССОР: {cpu_name}")
        creative_output_sys(f"Процессор: {cpu_name}", lable_CPU)
        device = "cpu"
        # 2. Оперативная память
        add_text_cons("ОПЕРАТИВНАЯ ПАМЯТЬ:")
        ram = subprocess.run(['wmic', 'memorychip', 'get', 'Capacity'], capture_output=True, text=True, encoding='cp866')
        lines = ram.stdout.strip().split('\n')
        total = 0
        for line in lines[1:]:
            if line.strip():
                total += int(line.strip()) / (1024 ** 3)
        add_text_cons(f"Всего: {total:.2f} ГБ")
        root.after(300, creative_output_sys,f"Оперативная память: {total:.2f} ГБ", lable_RAM)
    except Exception as e:
        add_text_cons(f"не удалось проверить конфигурацию ЦП и ОЗУ: {e}")

    # 3. CUDA драйвера
    try:
        import torch

        # ПРОВЕРКА NVIDIA
        if torch.cuda.is_available() and torch.version.cuda is not None:
            try:
                import GPUtil
            except:
                mon_device = "NVIDIA"
                try:
                    subprocess.run(
                        [str(VENV_PIP), "install", "GPUtil"],
                        check=True)
                    add_text_cons(f"Установлена библиотека для мониторинга состояния CPU")
                except Exception as e:
                    add_text_cons(f"не удалось установить GPUtil: {e}")
            add_text_cons(f"Видеокарта: {torch.cuda.get_device_name(0)}")
            add_text_cons(f"Видео память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
            add_text_cons(f"CUDA: {torch.version.cuda}")
            root.after(600, creative_output_sys, f"Видеокарта: {torch.cuda.get_device_name(0)}", lable_GPU)
            root.after(900, creative_output_sys, f"Видео память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ", lable_GPU_ram)
            root.after(1200, creative_output_sys, f"CUDA: {torch.version.cuda}", lable_CUDA)
            device = "cuda"
            mon_device = "NVIDIA"

        # ПРОВЕРКА AMD (ROCm)
        elif torch.cuda.is_available() and torch.version.cuda is None:
            try:
                import pyamdgpuinfo
            except:
                mon_device = "AMD"
                try:
                    subprocess.run(
                        [str(VENV_PIP), "install", "pyamdgpuinfo"],
                        check=True)
                    log_to_file(f"Установлена библиотека для мониторинга состояния CPU")
                except Exception as e:
                    log_to_file(f"не удалось установить pyamdgpuinfo: {e}", True)
            if pyamdgpuinfo.detect_gpus() > 0:
                gpu = pyamdgpuinfo.get_gpu(0)
                gpu_name = gpu.name if hasattr(gpu, 'name') else "AMD GPU"
                add_text_cons(f"Видеокарта: {gpu_name}")
                add_text_cons(f"Видео память: {gpu.memory_info['total'] / (1024 ** 3):.2f} ГБ")
                add_text_cons("ROCm: обнаружен")
                root.after(600, creative_output_sys, f"Видеокарта: {gpu_name}", lable_GPU)
                root.after(900, creative_output_sys, f"Видео память: {gpu.memory_info['total'] / (1024 ** 3):.2f} ГБ", lable_GPU_ram)
                root.after(1200, creative_output_sys, "ROCm: обнаружен", lable_CUDA)
                device = "cuda"
                mon_device = "AMD"
        # ЕСЛИ НЕТ CUDA И НЕТ ROCm
        else:
            root.after(600, creative_output_sys, "Видеокарта: НЕ удалось определить", lable_GPU)
            root.after(900, creative_output_sys, "Видео память: НЕ удалось определить", lable_GPU_ram)
            root.after(1200, creative_output_sys, "CUDA/ROCm: НЕ удалось определить", lable_CUDA)

    except Exception as e:
        log_to_file(f"не удалось проверить видеокарту и cuda: {e}", True)
    chk_lib()

def chk_lib():
    add_text_cons ("\n")
    add_text_cons ("=== Проверка и установка библиотек ===")
    add_text_cons ("Для корректной работы приложения пожалуйста дождитесь окончания проверки.")
    try: import transformers
    except:
        add_text_cons("Идет установка библиотеки: transformers")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "transformers>=4.36.0"], check=True)
    try: import peft
    except:
        add_text_cons("Идет установка библиотеки: peft")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "peft"], check=True)

    add_text_cons("Базовые библиотеки установлены корректно!")
    add_text_cons("Можно начинать обучение.")
    notebook.tab(second_tab, state="normal")
    notebook.tab(fourth_tab, state="normal")

def download_llama_cpp():
    global LLAMA_CPP_DIR

    if (LLAMA_CPP_DIR / "llama-quantize.exe").exists() and (LLAMA_CPP_DIR / "conversion").exists():
        add_text_cons("llama.cpp уже скачан")
        return

    try:
        add_text_cons("Устанавливаем llama-cpp-pydist...")

        subprocess.run([
            str(VENV_PYTHON), "-m", "pip", "install", "llama-cpp-pydist"
        ], check=True, capture_output=True)

        add_text_cons("llama-cpp-pydist установлен!")


    except Exception as e:
        add_text_cons(f"Ошибка загрузки llama-cpp-pydist: {e}")
        return

def Start_Lora_threat():
    notebook.select(main_tab)
    add_text_cons("\n")
    add_text_cons(f"=== Предварительная проверка Lora ===")
    try:
        import gguf
        import google.protobuf
        import llama_cpp
        add_text_cons("Библиотеки для конвертации подготовлены")
    except:
        add_text_cons("Установка библиотек для конвертации модели")
        time.sleep(0.1)
        try:
            subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "gguf", "protobuf", "llama-cpp-pydist"], check=True)
        except Exception as e:
            add_text_cons(f"Во время загрузки библиотек произошла ошибка: {e}")
        add_text_cons("Библиотеки для конвертации модели успешно скачаны")
    try:
        import sentencepiece
    except:
        add_text_cons("Идет установка библиотеки: sentencepiece")
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "sentencepiece"], check=True)
    add_text_cons(f"Все библиотеки прошли проверку.")
    Lora_learn()

def Start_Lora():
    threading.Thread(target=Start_Lora_threat, daemon=True).start()

def Lora_learn():
    try:
        add_text_cons("\n")
        add_text_cons(f"=== Запуск обучения Lora ===")
        global device, LORA_PATH,MODEL_PATH
        import torch
        # GPU ТЕСТ + FALLBACK

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        #Мера предосторожности))
        if device == "cuda":
            try:
                test = torch.randn(10,10, device="cuda")
                del test
                add_text_cons("GPU тест OK")
            except:
                device = "cpu"
                add_text_cons("GPU недоступен -> CPU")
            torch.cuda.empty_cache()

        if device == "cpu":
            import multiprocessing
            cores = multiprocessing.cpu_count()
            torch.set_num_threads(cores)
            torch.set_num_interop_threads(cores)
            os.environ["OMP_NUM_THREADS"] = str(cores)
            os.environ["MKL_NUM_THREADS"] = str(cores)
            os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
            os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
            torch.backends.mkldnn.enabled = True
            try:
                import cpuinfo
            except:
                subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "py-cpuinfo"], check=True)
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                flags = cpu_info.get('flags', [])
                if 'avx2' not in flags:
                    os.environ["MKL_DISABLE_AVX2"] = "1"
                    log_to_file("Обнаружен CPU без AVX2, оптимизировано для старого оборудования")
            except:
                pass
            add_text_cons(f"Используется {cores} ядер CPU")
            add_text_cons("GPU недоступен -> CPU")

        if device == "cuda" and torch.cuda.is_available():
            # Проверяем, ROCm это или CUDA
            if torch.version.cuda is None:  # Это ROCm
                # Отключаем некоторые оптимизации, которые могут не работать
                os.environ["PYTORCH_ROCM_ALLOC_CONF"] = "garbage_collection_threshold:0.8"

        gc.collect()

        MODEL_PATH = str(models_file.get())
        DATASET_PATH = str(dataset_file.get())
        EPOCHS = int(entry_l_epoch.get())
        LR = float(entry_l_LR.get())
        BATCH_SIZE = int(entry_l_bath.get())
        LORA_R = int(entry_l_lora_r.get())
        LORA_ALPHA = int(entry_l_alpha.get())
        MAX_LEN = int(entry_l_max_len.get())
        selected_targets = [name for var, name in checkboxes if var.get()]
        LORA_TARGETS = selected_targets
        DROPOUT = float(entry_l_Dropout.get())

        add_text_cons(f"Device: {device}")
        add_text_cons(f"Модель: {MODEL_PATH}")
        add_text_cons(f"Датасет: {DATASET_PATH}")
        add_text_cons(f"Новые веса LorA: {LORA_PATH}")


        # Токенизатор
        from transformers import AutoTokenizer, AutoModelForCausalLM
        add_text_cons("Токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        add_text_cons("Токенизатор OK")

        # Модель - с CPU fallback
        add_text_cons("Модель...")
        dtype = torch.float32
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
        add_text_cons("Модель подготовлена")

        from peft import LoraConfig, get_peft_model, TaskType
        add_text_cons("LoRA конфиг...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        peft_model = get_peft_model(model, lora_config)
        model = peft_model
        add_text_cons("LoRA конфиг применен")

        from torch.utils.data import Dataset, DataLoader
        # Dataset
        class CustomDataset(Dataset):
            def __init__(self, file_path, tokenizer, max_length):
                add_text_cons("Загружаем датасет в оперативную память...")
                self.texts = []

                # проверка по первой строке
                format_type = "jsonl"  # по умолчанию
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            data = json.loads(first_line)
                            if 'conversations' in data or 'conversation' in data:
                                format_type = "sharegpt"
                            elif 'instruction' in data or 'input' in data or 'output' in data:
                                format_type = "jsonl"
                            else:
                                format_type = "jsonl"  # пока оставлю как заглушку
                    add_text_cons(f"Формат датасета: {format_type}")
                except:
                    add_text_cons("Не удалось определить формат, пробуем как jsonl")
                    format_type = "jsonl"

                # ЧИТАЕМ ФАЙЛ
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            if format_type == "sharegpt":
                                conversations = data.get('conversations') or data.get('conversation', [])
                                if not conversations:
                                    continue

                                # Ищем первое сообщение пользователя и ассистента
                                user_msg = None
                                ass_msg = None
                                for msg in conversations:
                                    role = msg.get('from', '').lower()
                                    if role in ['human', 'user'] and user_msg is None:
                                        user_msg = msg.get('value', '')
                                    elif role in ['gpt', 'assistant'] and ass_msg is None:
                                        ass_msg = msg.get('value', '')

                                if user_msg and ass_msg:
                                    parts = []
                                    # Добавляем system только если он есть в данных
                                    system = data.get('system', '')
                                    if system:
                                        parts.append(f"<|im_start|>system: {system}<|im_end|>")
                                    parts.append(f"<|im_start|>user: {user_msg}<|im_end|>")
                                    parts.append(f"<|im_start|>assistant: {ass_msg}<|im_end|>")
                                    text = "\n".join(parts)
                                    self.texts.append(text)
                                else:
                                    add_text_cons(f"Строка {i}: нет пары user/assistant в ShareGPT")

                            else:
                                # JSONL формат (instruction/input/output)
                                instruction = data.get('instruction', '')
                                input_text = data.get('input', '')
                                output_text = data.get('output', '')

                                if output_text:
                                    if input_text:
                                        text = f"<|im_start|>system: {instruction}<|im_end|>\n<|im_start|>user: {input_text}<|im_end|>\n<|im_start|>assistant: {output_text}<|im_end|>"
                                    else:
                                        text = f"<|im_start|>system: {instruction}<|im_end|>\n<|im_start|>assistant: {output_text}<|im_end|>"
                                    self.texts.append(text)
                                else:
                                    add_text_cons(f"Строка {i}: нет output в JSONL")

                        except json.JSONDecodeError as e:
                            add_text_cons(f"Строка {i}: JSON ошибка - {e}")
                            return
                        except Exception as e:
                            add_text_cons(f"Строка {i}: ошибка - {e}")
                            return

                self.tokenizer = tokenizer
                self.max_length = max_length
                add_text_cons(f"Загружено примеров: {len(self.texts)}")

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "labels": encoding["input_ids"].flatten()
                }

        add_text_cons("Датасет...")
        dataset = CustomDataset(DATASET_PATH, tokenizer, MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=device == "cuda")

        # Обучение
        if device == "cuda":
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        model.train()

        total_steps = len(dataloader) * EPOCHS
        step_count = 0
        start_time = time.time()

        add_text_cons(f"\n=== ОБУЧЕНИЕ ({total_steps} шагов) ===")

        for epoch in range(EPOCHS):
            add_text_cons(f"Эпоха {epoch+1}/{EPOCHS}")
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
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                except RuntimeError as e:
                    add_text_cons(f"ОШИБКА шаг {step_count}: {e}")
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
                add_text_cons(f"Шаг {step_count}/{total_steps} ({progress:.1f}%) | "
                         f"Loss: {loss_value:.3f} | "
                         f"Скорость: {speed:.2f} шаг/сек | "
                         f"Осталось: {time_left/60:.1f}мин")

        add_text_cons("\n=== Обучение завершено успешно ===")
        add_text_cons("Сохранение...")
        Path(LORA_PATH).mkdir(exist_ok=True)
        model.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
        add_text_cons(f"Новые веса LorA сохранены: {LORA_PATH}")
        merge()
        gc.collect()
    except Exception as e:
        add_text_cons(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")

def merge():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        global device, MODEL_PATH, LORA_PATH, MERGED_PATH
        add_text_cons("\n")
        add_text_cons("=== LoRA СЛИЯНИЕ ===")

        add_text_cons(f"Базовая модель: {MODEL_PATH}")
        add_text_cons(f"Новые веса LoRA: {LORA_PATH}")
        add_text_cons(f"Слитая модель: {MERGED_PATH}")
        add_text_cons(f"Устройство: {device}")

        # ТОКЕНИЗАТОР
        add_text_cons("Токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # МОДЕЛЬ - грузим на CPU
        add_text_cons(f"Загрузка базовой модели...")
        dtype = torch.float16 if device == "cuda" else torch.bfloat16

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # СЛИЯНИЕ LoRA
        add_text_cons("Применяем LoRA веса...")
        model = PeftModel.from_pretrained(base, LORA_PATH)
        add_text_cons("LoRA успешно загружена")

        add_text_cons("Слияние...")
        model = model.merge_and_unload()
        add_text_cons("Слияние завершено")

        add_text_cons("Сохранение...")
        Path(MERGED_PATH).mkdir(exist_ok=True, parents=True)
        model.save_pretrained(MERGED_PATH, safe_serialization=False)
        tokenizer.save_pretrained(MERGED_PATH)

        # ПРОВЕРКА ФАЙЛОВ
        required = ['config.json', 'tokenizer.json', 'tokenizer.model', 'model.safetensors']
        add_text_cons("GGUF-ready файлы:")
        for filename in required:
            filepath = os.path.join(MERGED_PATH, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 ** 2)
                add_text_cons(f"  OK {filename} {size_mb:.1f}MB")
            else:
                add_text_cons(f"  MISSING {filename}")

        add_text_cons(f"СЛИЯНИЕ ОК: {MERGED_PATH}")

        # ОЧИСТКА
        del model, base
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        add_text_cons("Память очищена")
        convert()

    except Exception as e:
        add_text_cons(f"Произошла ошибка при слиянии весов: {e}")
        return

def convert():
    if convert_l_chkbox.get():
        add_text_cons("\n")
        add_text_cons("=== HF -> GGUF КОНВЕРТАЦИЯ ===")
        global MERGED_PATH, VENV_PYTHON, PROCESS_DIR, GGUF_PATH
        try:
            from llama_cpp import convert_hf_to_gguf

            add_text_cons(f"HF модель: {MERGED_PATH}")
            add_text_cons(f"GGUF выход: {GGUF_PATH}")

            if not os.path.exists(MERGED_PATH):
                add_text_cons(f"HF модель не найдена: {MERGED_PATH}")
                return

            success, result_message = convert_hf_to_gguf(
                model_path_or_name=str(MERGED_PATH),
                output_dir=str(PROCESS_DIR),
                output_filename=str(GGUF_PATH),
                outtype='f16'
            )

            if success:
                add_text_cons(f"Конвертация успешна! Файл: {result_message}")
            else:
                add_text_cons(f"Ошибка конвертации: {result_message}")
                return
            quant()
            gc.collect()
        except Exception as e:
            add_text_cons(f"При конвертации произошла ошибка: {e}")
            return
    else: return

def quant():
    global GGUF_PATH
    quant_var = str(selected_quant.get())
    if quant_var == "f16":
        add_text_cons(f"\nМодель сохранена в f16: {GGUF_PATH}")
        return

    GGUF_QUANT = PROCESS_DIR / f"model_{quant_var}.gguf"

    try:
        file_size = os.path.getsize(GGUF_PATH) / (1024 ** 3)
        add_text_cons(f"F16 GGUF ({file_size:.1f}GB) готов")
        add_text_cons(f"Q выход: {GGUF_QUANT}")
        add_text_cons(f"Метод: {quant_var}")

        import shutil
        quantize_exe = None
        exe_path = LLAMA_CPP_DIR / "llama-quantize.exe"
        if exe_path.exists():
            quantize_exe = str(exe_path)
        if not quantize_exe:
            quantize_exe = shutil.which("llama-quantize.exe")
        if not quantize_exe:
            exe_path = VENV_DIR / "Scripts" / "llama-quantize.exe"
            if exe_path.exists():
                quantize_exe = str(exe_path)
        if not quantize_exe:
            log_to_file("llama-quantize.exe не найден, ищем в ZIP...")
            zip_dirs = [
                VENV_DIR / "Lib" / "site-packages" / "llama_cpp" / "binaries",
                VENV_DIR / "Lib" / "site-packages" / "llama_cpp_pydist" / "binaries",
            ]
            for zip_dir in zip_dirs:
                if zip_dir.exists():
                    for zip_file in zip_dir.glob("*.zip"):
                        if "bin-win" in zip_file.name.lower():
                            log_to_file(f"Распаковываем {zip_file.name}...")
                            import zipfile
                            with zipfile.ZipFile(str(zip_file), 'r') as zf:
                                zf.extractall(str(LLAMA_CPP_DIR))
                            quantize_exe = str(LLAMA_CPP_DIR / "llama-quantize.exe")
                            break
                if quantize_exe:
                    break
        if not quantize_exe:
            log_to_file("llama-quantize.exe не найден!")
            add_text_cons("Произошла непредвиденная ошибка квантования. Сообщите создателю.")
            return
        log_to_file(f"Найден: {quantize_exe}")
        os.makedirs(os.path.dirname(GGUF_QUANT), exist_ok=True)
        add_text_cons(f"Квантование F16 -> {quant_var}...")
        result = subprocess.run([
            quantize_exe,
            str(GGUF_PATH),
            str(GGUF_QUANT),
            quant_var
        ], capture_output=True, text=True)

        if result.stdout:
            log_to_file("STDOUT: " + result.stdout[:300])
        if result.stderr:
            log_to_file("STDERR: " + result.stderr[:300])
        log_to_file(f"Код возврата: {result.returncode}")

        if result.returncode == 0 and os.path.exists(GGUF_QUANT):
            size_gb = os.path.getsize(GGUF_QUANT) / (1024 ** 3)
            add_text_cons(f"Модель {quant_var} готова: {GGUF_QUANT} ({size_gb:.1f}GB)")
            add_text_cons(f"\n")
            add_text_cons(f"Обучение завершено. Ваша модель: {GGUF_QUANT}")
        else:
            add_text_cons("Квантование не удалось!")
        gc.collect()
    #Врятли сюда канешн кто-то полезет)  знали бы вы как я зае... (ну вы поняли) переделывать квантование.... такое ощущение что процентов 60 всего времени работы занял именно этт метод...
    except Exception as e:
        add_text_cons(f"Во время квантования произошла ошибка: {e}")
        gc.collect()

def search_models(search_term: str, param_filter: str = None):
    from huggingface_hub import HfApi
    api = HfApi()
    models = api.list_models(
        search=search_term,
        num_parameters=param_filter,
        sort="downloads",
        limit=100
    )
    return [
        {
            "id": model.modelId,
            "author": model.author,
            "downloads": model.downloads,
            "pipeline_tag": model.pipeline_tag,
            "tags": model.tags,
            "last_modified": model.lastModified
        }
        for model in models
    ]

def perform_search(event):
    global result_listbox
    search_term = search_entry.get()
    from_size = from_size_combobox.get()
    to_size = to_size_combobox.get()
    listbox = result_listbox
    listbox.delete(0, tk.END)
    param_filter = None
    if from_size != "Любой" and to_size != "Любой":
        param_filter = f"min:{from_size},max:{to_size}"
    elif from_size != "Любой":
        param_filter = f"min:{from_size}"
    elif to_size != "Любой":
        param_filter = f"max:{to_size}"

    try:
        results = search_models(search_term, param_filter)
        if not results:
            listbox.insert(tk.END, "Модели не найдены.")
        else:
            for model in results:
                display_text = f"{model['id']} | Загрузки: {model['downloads']:,}"
                listbox.insert(tk.END, display_text)
    except Exception as e:
        listbox.insert(tk.END, f"Ошибка: {e}")

def download_selected_model():
    threading.Thread(target=download_selected_model_threat, daemon=True).start()

def download_selected_model_threat():
    try:
        from huggingface_hub import snapshot_download
        selection = result_listbox.curselection()
        if not selection:
            add_text_cons("Выберите модель из списка!")
            return

        selected_text = result_listbox.get(selection[0])
        model_id = selected_text.split(" |")[0].strip()

        add_text_cons(f"\nСкачиваем: {model_id}")

        model_folder_name = model_id.replace("/", "_")
        model_path = PROCESS_DIR / model_folder_name

        if model_path.exists():
            add_text_cons(f"Модель уже существует: {model_path}")
            return
        os.environ["HF_TOKEN"] = ""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Включает более быстрый протокол
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            max_workers=8,  # Увеличиваем потоки
            resume_download=True,  # Возобновление при обрыве
            local_files_only=False,
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin"],  # Загружаем только нужные файлы
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],  # Игнорируем ненужные форматы
        )
        add_text_cons(f"Модель скачана: {model_path}")

    except Exception as e:
        add_text_cons(f"Ошибка скачивания: {e}")

def browse_model_gguf_q():
    filename = filedialog.askopenfilename(
        title="Выберите файл датасета",
        filetypes=[
            ("gguf модель", "*.gguf"),
            ("Все файлы", "*.*")
        ],
        initialdir=str(SCRIPT_DIR)  # Начать с папки проекта
    )
    if filename:
        models_file_q.delete(0, "end")
        models_file_q.insert(0, filename)

    check_paths_q()

def browse_model_result_q():
    filename = filedialog.asksaveasfilename(
        title="Выберите файл датасета",
        defaultextension=".gguf",
        filetypes=[
            ("gguf модель", "*.gguf"),
            ("Все файлы", "*.*")
        ],
        initialdir=str(SCRIPT_DIR),
        initialfile="model.gguf"
    )
    if filename:
        result_file_q.delete(0, "end")
        result_file_q.insert(0, filename)

    check_paths_q()

def only_quant_start():
    threading.Thread(target=only_quant_start_threat, daemon=True).start()

def only_quant_start_threat():
    notebook.select(main_tab)
    add_text_cons("\n")
    add_text_cons(f"=== Предварительная проверка библиотек ===")
    try:
        import gguf
        import google.protobuf
        import sentencepiece
        import llama_cpp
        add_text_cons("Библиотеки для квантования подготовлены")
    except:
        add_text_cons("Установка библиотек для квантования модели")
        time.sleep(0.1)
        try:
            subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run(
                [str(VENV_PYTHON), "-m", "pip", "install", "sentencepiece", "gguf", "protobuf", "llama-cpp-pydist"], check=True)
        except Exception as e:
            add_text_cons(f"Во время загрузки библиотек произошла ошибка: {e}")
        add_text_cons("Библиотеки для конвертации модели успешно скачаны")

    add_text_cons(f"Все библиотеки прошли проверку.")
    only_quant()

def only_quant():
    quant_var = str(selected_quant.get())
    GGUF_PATH = str(models_file_q.get())
    GGUF_QUANT = str(result_file_q.get())
    try:
        file_size = os.path.getsize(GGUF_PATH) / (1024 ** 3)
        add_text_cons(f"F16 GGUF ({file_size:.1f}Gb)")
        add_text_cons(f"Q выход: {GGUF_QUANT}")
        add_text_cons(f"Метод: {quant_var}")
        import shutil
        quantize_exe = None
        exe_path = LLAMA_CPP_DIR / "llama-quantize.exe"
        if exe_path.exists():
            quantize_exe = str(exe_path)
        if not quantize_exe:
            quantize_exe = shutil.which("llama-quantize.exe")
        if not quantize_exe:
            exe_path = VENV_DIR / "Scripts" / "llama-quantize.exe"
            if exe_path.exists():
                quantize_exe = str(exe_path)
        if not quantize_exe:
            log_to_file("llama-quantize.exe не найден, ищем в ZIP...")
            zip_dirs = [
                VENV_DIR / "Lib" / "site-packages" / "llama_cpp" / "binaries",
                VENV_DIR / "Lib" / "site-packages" / "llama_cpp_pydist" / "binaries",
            ]
            for zip_dir in zip_dirs:
                if zip_dir.exists():
                    for zip_file in zip_dir.glob("*.zip"):
                        if "bin-win" in zip_file.name.lower():
                            log_to_file(f"Распаковываем {zip_file.name}...")
                            import zipfile
                            with zipfile.ZipFile(str(zip_file), 'r') as zf:
                                zf.extractall(str(LLAMA_CPP_DIR))
                            quantize_exe = str(LLAMA_CPP_DIR / "llama-quantize.exe")
                            break
                if quantize_exe:
                    break
        if not quantize_exe:
            log_to_file("llama-quantize.exe не найден!")
            add_text_cons("Произошла непредвиденная ошибка квантования. Сообщите создателю.")
            return
        log_to_file(f"Найден: {quantize_exe}")
        os.makedirs(os.path.dirname(GGUF_QUANT), exist_ok=True)
        add_text_cons(f"Квантование F16 -> {quant_var}...")
        result = subprocess.run([
            quantize_exe,
            str(GGUF_PATH),
            str(GGUF_QUANT),
            quant_var
        ], capture_output=True, text=True)

        if result.stdout:
            log_to_file("STDOUT: " + result.stdout[:300])
        if result.stderr:
            log_to_file("STDERR: " + result.stderr[:300])
        log_to_file(f"Код возврата: {result.returncode}")

        if result.returncode == 0 and os.path.exists(GGUF_QUANT):
            size_gb = os.path.getsize(GGUF_QUANT) / (1024 ** 3)
            add_text_cons("\n")
            add_text_cons(f"Модель {quant_var} готова: {GGUF_QUANT} ({size_gb:.1f}GB)")
        else:
            add_text_cons("Квантование не удалось!")
        gc.collect()
    except Exception as e:
        add_text_cons(f"Во время квантования произошла ошибка: {e}")
        gc.collect()

def check_paths_q():
    model_gguf = models_file_q.get().strip()
    result_model = result_file_q.get().strip()

    global model_ready
    if model_gguf.endswith('.gguf') and result_model.endswith('.gguf'):
        Btn_start_quant.config(state="normal")
    else:
        Btn_start_quant.config(state="disabled")

GOOGLE_DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1FtUSLFaLxhSo4GxnvwrkF2UpNk225LB9?usp=sharing"

def get_file_ids_from_public_folder(folder_id):
    try:
        import requests
        import re
        import json
        # Используем внутренний API Google Drive (работает для публичных папок)
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        # Ищем JSON данные в HTML
        # Google Drive хранит данные в переменной _DRIVE_ivd
        pattern = r'var _DRIVE_ivd = ({.*?});'
        match = re.search(pattern, response.text)
        if match:
            try:
                data = json.loads(match.group(1))
                # Ищем файлы в структуре
                if 'items' in data:
                    file_ids = [item['id'] for item in data['items'] if 'id' in item]
                    return file_ids
            except:
                pass
        # Альтернативный способ - ищем через data-id
        pattern = r'data-id="([a-zA-Z0-9_-]+)"'
        ids = re.findall(pattern, response.text)
        # Фильтруем мусор
        forbidden = ['AIzaSy', 'ANONYMOUS', 'ABCDEFGHIJKLMNOP', 'M120-240v']
        ids = [id for id in ids if len(id) >= 19 and not any(bad in id for bad in forbidden)]
        return list(set(ids))
    except Exception as e:
        log_to_file(f"Ошибка получения списка: {e}")
        return []

def load_art_images():
    try:
        import requests
        from io import BytesIO
        from PIL import Image, ImageTk
        import random
        import re
        # Извлекаем ID папки
        folder_id_match = re.search(r'folders/([a-zA-Z0-9_-]+)', GOOGLE_DRIVE_FOLDER_LINK)
        if not folder_id_match:
            log_to_file("Ошибка: неверная ссылка на папку Google Drive")
            return
        folder_id = folder_id_match.group(1)
        # Получаем список ID файлов
        file_ids = get_file_ids_from_public_folder(folder_id)
        if not file_ids:
            log_to_file("Не найдено файлов в папке")
            return
        log_to_file(f"Найдено {len(file_ids)} артов. Загрузка...")
        images_data = []
        for file_id in file_ids:
            try:
                # Прямая ссылка на скачивание
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                img_response = requests.get(download_url, timeout=15)
                if img_response.status_code == 200:
                    try:
                        img = Image.open(BytesIO(img_response.content))
                        if img.width > 50 and img.height > 50:
                            images_data.append(img)
                            log_to_file(f"✓ Загружен: {file_id[:10]}... ({img.width}x{img.height})")
                    except:
                        pass
                else:
                    log_to_file(f"✗ Ошибка {img_response.status_code}: {file_id[:10]}...")
            except:
                pass
        if not images_data:
            log_to_file("Не удалось загрузить ни одного изображения")
            return
        log_to_file(f"Загружено {len(images_data)} артов. Размещение...")
        place_arts_on_canvas(images_data)
    except Exception as e:
        log_to_file(f"Ошибка загрузки артов: {e}")

def place_arts_on_canvas(images):
    # я труба шатал возиться графикой и координатами так что этот метод 100% писала нейронка) я просто комментарии расставил чтобы понимать что где))
    # а что вы мне сделаете я в другом городе)))
    import random
    from PIL import Image, ImageTk, ImageDraw
    # Получаем размеры холста
    canvas_width = art_canvas.winfo_width()
    canvas_height = art_canvas.winfo_height()
    if canvas_width < 100 or canvas_height < 100:
        # Если холст еще не отрисовался, ждем
        art_canvas.after(100, lambda: place_arts_on_canvas(images))
        return
    # Максимальные размеры арта
    max_width = canvas_width // 3
    max_height = canvas_height // 2
    placed_rects = []  # Храним центры и размеры размещенных артов
    placed_count = 0
    # Перемешиваем изображения для случайного порядка
    random.shuffle(images)
    for img in images:
        # Случайный масштаб (от 0.3 до 1.0 от максимального)
        scale = random.uniform(0.7, 1.0)
        # Вычисляем размеры с сохранением пропорций
        img_width, img_height = img.size
        # Масштабируем чтобы вписаться в max_width и max_height
        scale_w = (max_width * scale) / img_width
        scale_h = (max_height * scale) / img_height
        final_scale = min(scale_w, scale_h, 1.0)  # Не увеличиваем, только уменьшаем
        new_width = int(img_width * final_scale)
        new_height = int(img_height * final_scale)
        # Если картинка слишком маленькая - используем оригинальный размер
        if new_width < 50 or new_height < 50:
            new_width = img_width
            new_height = img_height
            # Но все равно проверяем что не вылезает за границы
            if new_width > max_width:
                ratio = max_width / new_width
                new_width = max_width
                new_height = int(new_height * ratio)
            if new_height > max_height:
                ratio = max_height / new_height
                new_height = max_height
                new_width = int(new_width * ratio)
        # Пытаемся найти свободное место
        found_position = False
        x = 0
        y = 0
        for attempt in range(20):  # попытки найти свободное место (иначе комп взлетит)
            x = random.randint(0, canvas_width - new_width)
            y = random.randint(0, canvas_height - new_height)
            # Центр арта
            cx = x + new_width // 2
            cy = y + new_height // 2
            # Проверяем перекрытие с другими артами
            overlap = False
            for placed_cx, placed_cy, placed_w, placed_h in placed_rects:
                min_dist_x = (new_width + placed_w) // 2
                min_dist_y = (new_height + placed_h) // 2
                dist_x = abs(cx - placed_cx)
                dist_y = abs(cy - placed_cy)
                if dist_x < min_dist_x and dist_y < min_dist_y:
                    overlap = True
                    break
            if not overlap:
                found_position = True
                placed_rects.append((cx, cy, new_width, new_height))
                placed_count += 1
                break
        # Если свободное место не найдено - ставим куда попало
        if not found_position:
            x = random.randint(-new_width // 4, canvas_width - new_width + new_width // 4)
            y = random.randint(-new_height // 4, canvas_height - new_height + new_height // 4)
            # Все равно добавляем в список, чтобы новые арты не перекрывали его если есть место
            cx = x + new_width // 2
            cy = y + new_height // 2
            placed_rects.append((cx, cy, new_width, new_height))
            placed_count += 1
        # Масштабируем изображение
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        bordered_img = resized_img.copy()
        draw = ImageDraw.Draw(bordered_img)
        # Рисуем черную обводку толщиной 2 пикселя
        bordered_img = resized_img.copy()  # создаем копию
        draw = ImageDraw.Draw(bordered_img)
        draw.rectangle(
            [(0, 0), (new_width - 1, new_height - 1)],
            outline="black",
            width=2
        )
        # Случайный поворот от -25 до +25 градусов
        angle = random.uniform(-25, 25)
        rotated_img = bordered_img.rotate(angle, expand=True,fillcolor=(255, 255, 255, 0))  # поворачиваем bordered_img
        # Конвертируем для Tkinter
        tk_img = ImageTk.PhotoImage(rotated_img)
        # Сохраняем ссылку чтобы изображение не удалил сборщик мусора
        if not hasattr(art_canvas, 'art_images'):
            art_canvas.art_images = []
        art_canvas.art_images.append(tk_img)
        # Размещаем на холсте
        rotated_w, rotated_h = rotated_img.size
        display_x = x + (new_width - rotated_w) // 2
        display_y = y + (new_height - rotated_h) // 2
        art_canvas.create_image(display_x, display_y, image=tk_img, anchor="nw")
    log_to_file(f"Размещено {placed_count} артов (из них {len(images) - placed_count} перекрываются)")

def load_arts():
    art_canvas.delete("all")
    if hasattr(art_canvas, 'art_images'):
        art_canvas.art_images = []
    threading.Thread(target=load_art_images, daemon=True).start()

def on_tab_selected(event):
    selected = notebook.index(notebook.select())
    if selected == 4:  # Индекс 5-й вкладки (нумерация с 0)
        # Ждем пока холст отрисуется
        art_canvas.after(100, load_arts)

root = tk.Tk()
root.bind_class('all', '<Control-KeyPress>', handle_ctrl_key)
root.title("Дообучение LLM")
root.geometry("800x500")
root.minsize(800, 500)
root.resizable(True, True)

notebook = ttk.Notebook(root)  #панель вкладок
notebook.pack(fill='both', expand=True)
style = ttk.Style()
style.configure('TNotebook', background=bg_color)
style.configure('TFrame', background=bg_color)

# region === вкладка 1 ===
# ВКЛАДКА 1
main_tab = ttk.Frame(notebook)
notebook.add(main_tab, text="Главная")

main_tab.grid_columnconfigure(0, weight=2)
main_tab.grid_columnconfigure(1, weight=1)
main_tab.grid_rowconfigure(0, weight=1)

# Блок 1: слева 2/3
# , relief='solid', bd=1
block1 = tk.LabelFrame(main_tab,text="Консоль",bd=0, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
block1.grid_propagate(False)
block1.grid(row=0, column=0, sticky='nsew', padx=space_out, pady=space_out)
block1.grid_rowconfigure(0, weight=1)
block1.grid_rowconfigure(1, weight=0)
block1.grid_columnconfigure(0, weight=1)
block1.grid_columnconfigure(1, weight=1)
block1.grid_columnconfigure(2, weight=1)

#типа консоль
Console_main = tk.Text(block1,state="disabled", bg=bg_color_cons, fg=font_cons, wrap="word", width=1, height=1)
Console_main.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=space_in, pady=space_in)
#Console_main.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=space_in, pady=(space_in,0)) #тут отступ для низа (чтобы не забыть если добавлю что то)  )))

# Правая колонка: ПРАВО 1/3
#, relief='solid', bd=1
right_column = tk.Frame(main_tab, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
right_column.grid(row=0, column=1, sticky='nsew', padx=space_out, pady=space_out)
right_column.grid_rowconfigure(0, weight=1)
right_column.grid_rowconfigure(1, weight=1)
right_column.grid_columnconfigure(0, weight=1)

# Блок 2: ВЕРХ 2/3
# , relief='solid', bd=2
block2 = tk.LabelFrame(right_column,text="Характеристиики ПК:",bd=0,bg=bg_color)
block2.grid_propagate(False)
block2.grid(row=0, column=0, sticky='nsew', padx=space_in, pady=space_in)

block2.grid_rowconfigure(0, weight=0)
block2.grid_rowconfigure(1, weight=0)
block2.grid_rowconfigure(2, weight=0)
block2.grid_rowconfigure(3, weight=0)
block2.grid_rowconfigure(4, weight=0)
block2.grid_rowconfigure(5, weight=1)
block2.grid_rowconfigure(6, weight=0)
block2.grid_columnconfigure(0, weight=1)

lable_CPU = tk.Label(block2,justify="left", anchor="w", bg=bg_color)
lable_CPU.grid(row=0, column=0, sticky="nsew")
lable_RAM = tk.Label(block2,justify="left", anchor="w", bg=bg_color)
lable_RAM.grid(row=1, column=0, sticky="nsew")
lable_GPU = tk.Label(block2,justify="left", anchor="w", bg=bg_color)
lable_GPU.grid(row=2, column=0, sticky="nsew")
lable_GPU_ram = tk.Label(block2,justify="left", anchor="w", bg=bg_color)
lable_GPU_ram.grid(row=3, column=0, sticky="nsew")
lable_CUDA = tk.Label(block2,justify="left", anchor="w", bg=bg_color)
lable_CUDA.grid(row=4, column=0, sticky="nsew")

def creative_output_sys(text,lable):
    def run():
        current = ""
        delay = 0.02
        for char in text:
            current += char
            lable.config(text=current)
            lable.update()
            time.sleep(delay)
    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

#Console_prompt = tk.Text(block2, bg=bg_color_cons, fg=font_cons, wrap="word", width=1, height=1)
#Console_prompt.grid(row=0, column=0, columnspan=2, sticky="nsew")

lable_stat_1 = tk.Label(block2,justify="left", fg=font_cons, bg=bg_color_cons)
lable_stat_1.grid(row=6, column=0, sticky="nsew")


# Блок 3: НИЗ 1/3
# , relief='solid', bd=2
block3 = tk.Frame(right_column, bg=bg_color)
block3.grid_propagate(False)
block3.grid(row=1, column=0, sticky='nsew', padx=space_in, pady=space_in)
block3.grid_rowconfigure(0, weight=1)
block3.grid_columnconfigure(0, weight=1)
label_face = tk.Label(block3,justify="left", fg=font_cons,bg=bg_color_cons)
label_face.grid(row=0, column=0, sticky="nsew")

def resize_label(self):
    try:
        text = open(FACE_NOW, "r", encoding="utf-8").read()
        num_lines= text.count("\n")+1
        lines = text.split("\n")
        max_line_len = max(len(line) for line in lines)
        pixel_on_row = (root.winfo_width() // 3 )// max_line_len
        pixel_on_line = (root.winfo_height() // 2 ) // (num_lines+4)
        font_size = min (int(pixel_on_line  // 1.33) , int(pixel_on_row // 0.9))
        label_face.config(font=("Courier", font_size))
    except Exception as e:
        log_to_file(f"Ошибка подгонки размера шрифта: {e}")
        block3.unbind("<Configure>")

block3.bind("<Configure>", resize_label)
# endregion

# region === вкладка 2 ===
# ВКЛАДКА 2
second_tab = ttk.Frame(notebook)
notebook.add(second_tab, text="Lora-метод")
notebook.tab(second_tab, state="disabled")

second_tab.grid_rowconfigure(0, weight=0)
second_tab.grid_rowconfigure(1, weight=1)
second_tab.grid_rowconfigure(2, weight=1)
second_tab.grid_columnconfigure(0, weight=1)
second_tab.grid_columnconfigure(1, weight=1)

block4 = tk.Frame(second_tab, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
#block4.grid_propagate(False)
block4.grid(row=0, column=0,columnspan=2, sticky="nsew", padx=space_out, pady=space_out)
block4.grid_rowconfigure(0, weight=1)
block4.grid_rowconfigure(1, weight=1)
block4.grid_columnconfigure(0, weight=1)
block4.grid_columnconfigure(1, weight=0)

models_file = tk.Entry(block4, bg="white")
models_file.grid(row=0, column=0, sticky="nsew", padx=(space_in,0), pady=(space_in,0))
models_file.insert(0, "Выберите папку с моделью...")
models_file.bind("<KeyRelease>", lambda e: check_paths())
Btn_browse = tk.Button(block4, text="Обзор", command=browse_model_folder)
Btn_browse.grid(row=0, column=1, sticky="nsew", padx=(0, space_in), pady=(space_in,0))
dataset_file = tk.Entry(block4, bg="white")
dataset_file.grid(row=1, column=0, sticky="nsew", padx=(space_in,0), pady=(0,space_in))
dataset_file.insert(0, "Выберите файл датасета...")
dataset_file.bind("<KeyRelease>", lambda e: check_paths())
Btn_browse_dat = tk.Button(block4, text="Обзор", command=browse_dataset_file)
Btn_browse_dat.grid(row=1, column=1, sticky="nsew", padx=(0, space_in), pady=(0,space_in))

block5 = tk.Frame(second_tab, highlightbackground=color_span, highlightthickness=weight_span,bg=bg_color)
block5.grid_propagate(False)
block5.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=space_out, pady=space_out)
block5.grid_rowconfigure(0, weight=0)
block5.grid_rowconfigure(1, weight=0)
block5.grid_rowconfigure(2, weight=0)
block5.grid_rowconfigure(3, weight=0)
block5.grid_rowconfigure(4, weight=0)
block5.grid_rowconfigure(5, weight=0)
block5.grid_rowconfigure(6, weight=0)
block5.grid_rowconfigure(7, weight=0)
block5.grid_rowconfigure(8, weight=0)
block5.grid_rowconfigure(9, weight=0)
block5.grid_rowconfigure(10, weight=0)
block5.grid_rowconfigure(11, weight=0)
block5.grid_rowconfigure(12, weight=0)
block5.grid_rowconfigure(13, weight=0)
block5.grid_rowconfigure(14, weight=0)
block5.grid_columnconfigure(0, weight=1)
block5.grid_columnconfigure(1, weight=1)
block5.grid_columnconfigure(2, weight=4)

entr_width = 5
tk.Label(block5, text="Настройки Lora:", bg=bg_color).grid(row=0, column=0, columnspan=3, sticky="nsew")
tk.Label(block5, text="Эпохи обучения:",  anchor="w", bg=bg_color).grid(row=1, column=0, sticky="nsew")
entry_l_epoch = tk.Entry(block5, bg="white", width=entr_width)
entry_l_epoch.grid(row=1, column=1, sticky="ns")
tk.Label(block5, text="Размер Батча:",  anchor="w", bg=bg_color).grid(row=2, column=0, sticky="nsew")
entry_l_bath = tk.Entry(block5, bg="white", width=entr_width)
entry_l_bath.grid(row=2, column=1, sticky="ns")
tk.Label(block5, text="Lora rank (r):",  anchor="w", bg=bg_color).grid(row=3, column=0, sticky="nsew")
entry_l_lora_r = tk.Entry(block5, bg="white", width=entr_width)
entry_l_lora_r.grid(row=3, column=1, sticky="ns")
tk.Label(block5, text="Скорость обучения (LR):",  anchor="w", bg=bg_color).grid(row=4, column=0, sticky="nsew")
lr_options = ["1e-5", "2e-5", "5e-5", "1e-4", "2e-4"]
selected_lr = tk.StringVar()
entry_l_LR = ttk.Combobox( block5, textvariable=selected_lr, values=lr_options, width=entr_width*2, state="readonly")
entry_l_LR.grid(row=4, column=1, sticky="ns")
tk.Label(block5, text="Усиление Lora (alpha):",  anchor="w", bg=bg_color).grid(row=5, column=0, sticky="nsew")
entry_l_alpha = tk.Entry(block5, bg="white", width=entr_width)
entry_l_alpha.grid(row=5, column=1, sticky="ns")
tk.Label(block5, text="Параметр Dropout:",  anchor="w", bg=bg_color).grid(row=6, column=0, sticky="nsew")
entry_l_Dropout = tk.Entry(block5, bg="white", width=entr_width)
entry_l_Dropout.grid(row=6, column=1, sticky="ns")
tk.Label(block5, text="Максимальная длинна контекста:",  anchor="w", bg=bg_color).grid(row=7, column=0, sticky="nsew")
entry_l_max_len = tk.Entry(block5, bg="white", width=entr_width)
entry_l_max_len.grid(row=7, column=1, sticky="ns")

entry_l_epoch.insert(0, "1")
entry_l_bath.insert(0, "2")
entry_l_lora_r.insert(0, "8")
selected_lr.set("5e-5")
entry_l_alpha.insert(0, "16")
entry_l_Dropout.insert(0, "0.05")
entry_l_max_len.insert(0, "1024")

tk.Label(block5, text="Конвертировать в .gguf? ",  anchor="w", bg=bg_color).grid(row=8, column=0, sticky="nsew")
convert_l_chkbox = tk.BooleanVar()
convert_l_chkbox.set(True)
chkbox1 = tk.Checkbutton(block5, variable=convert_l_chkbox, bg=bg_color)
chkbox1.grid(row=8, column=1, sticky="ns")

quant_options = [
    # === Рекомендуемые K-Quants (лучший баланс) ===
    "f16",       # без квантования, модель сохраниться d 16бит
    "Q2_K",      # 2.5-3.5 бит, минимальный размер, заметная потеря качества
    "Q3_K_S",    # 3-4 бит, подходит для маленьких моделей
    "Q3_K_M",    # 3-4 бит, сбалансированный
    "Q3_K_L",    # 3-4 бит, крупный, лучше качество
    "Q4_K_S",    # 4-5 бит, маленький, хорошая скорость
    "Q4_K_M",    # Рекомендуемый, лучший баланс размера и качества
    "Q5_K_S",    # 5-6 бит, очень хорошее качество, маленький
    "Q5_K_M",    # Более высокое качество, немного больше размер
    "Q6_K",      # 6-7 бит, отличное качество, минимальная потеря
    "Q8_K",      # 8 бит, почти оригинальное качество
    # === Устаревшие (Legacy Quants) ===
    "Q4_0",      # Простое 4-битное квантование, не рекомендуется
    "Q4_1",      # Простое 4-битное квантование, чуть лучше Q4_0
    "Q5_0",      # Простое 5-битное квантование
    "Q5_1",      # Простое 5-битное квантование, чуть лучше Q5_0
    "Q8_0",      # 8-битное, почти без потерь, но большой размер
]

tk.Label(block5, text="Квантование: ",  anchor="w", bg=bg_color).grid(row=9, column=0, sticky="nsew")
selected_quant = tk.StringVar()
selected_quant.set("f16")
entry_quant = ttk.Combobox(
    block5,
    textvariable=selected_quant,
    values=quant_options,
    width=entr_width*2,
    state="readonly"
)
entry_quant.grid(row=9, column=1, sticky="ns")
Btn_start_llama = tk.Button(block5, text="Запуск обучения lora", state="disabled")
Btn_start_llama.grid(row=10, column=0, rowspan=3,sticky="nsew", padx=(0,space_in), pady=(0,space_in))
Btn_start_llama.config(command=Start_Lora)

make_tooltip(entry_l_epoch, "Epochs (Эпохи): Количество полных проходов модели через весь ваш набор данных.")
make_tooltip(entry_l_bath, "Batch Size (Размер батча): Количество примеров, которое модель обрабатывает перед одним обновлением весов.")
make_tooltip(entry_l_lora_r, "Rank (r): Размерность матриц адаптеров. Чем выше ранг, тем больше параметров обучается \nи тем выше выразительная способность модели, но и больше расход памяти. \nРекомендуемые значения: от 8 до 64")
make_tooltip(entry_l_LR, "Learning Rate (Скорость обучения): Определяет размер шага, с которым модель обновляет свои веса на основе ошибки. \n1e-5 - Очень медленное (стабильное) \n2e-5 - Медленное (безопасное) \n5e-5 - Среднее (рекомендуемое)  \n1e-4 - Быстрое (рискованное) \n2e-4 - Очень быстрое (агрессивное)")
make_tooltip(entry_l_alpha, "Alpha (lora_alpha): Масштабирующий коэффициент для обновлений LoRA. Он определяет, \nнасколько сильно влияние новых обучаемых весов. Часто его устанавливают \nравным рангу или в два раза больше")
make_tooltip(entry_l_Dropout, 'Dropout (lora_dropout): Вероятность случайного "выключения" нейронов в слоях LoRA \nдля предотвращения переобучения. Значение 0.0 часто работает хорошо, \nно можно попробовать 0.05–0.1 для регуляризации')
make_tooltip(entry_l_max_len, "Длина контекста (MAX_LEN, max_length) — это максимальное количество токенов (кусочков текста), \nкоторое модель может обработать за один раз.")
make_tooltip(models_file, "Укажите папку с уже скачанной моделью.")
make_tooltip(dataset_file, "Укажите путь к файлу датасета.")
make_tooltip(chkbox1, "Конвертирование в формат .gguf. \nБез конвертации квантование будет недоступно")
make_tooltip(entry_quant, "Квантование не обязательная вещь но она способна сильно облегчить модель и ее вычисления за счет потери точности. \nПо умолчанию установлен режим: f16. (16бит) - модель в исходной форме.\n=== Рекомендуемые K-Quants (лучший баланс) ===\nQ2_K - 2.5-3.5 бит, минимальный размер, заметная потеря качества [citation:1]\nQ3_K_S - 3-4 бит, подходит для маленьких моделей [citation:1]\nQ3_K_M - 3-4 бит, сбалансированный [citation:1]\nQ3_K_L - 3-4 бит, крупный, лучше качество [citation:1]\nQ4_K_S - 4-5 бит, маленький, хорошая скорость [citation:1]\nQ4_K_M - Рекомендуемый, лучший баланс размера и качества [citation:1][citation:4]\nQ5_K_S - 5-6 бит, очень хорошее качество, маленький [citation:1]\nQ5_K_M - Более высокое качество, немного больше размер [citation:1]\nQ6_K - 6-7 бит, отличное качество, минимальная потеря [citation:1]\nQ8_K - 8 бит, почти оригинальное качество [citation:1]\n=== Устаревшие (Legacy Quants) ===\nQ4_0 - Простое 4-битное квантование, не рекомендуется [citation:1]\nQ4_1 - Простое 4-битное квантование, чуть лучше Q4_0 [citation:1]\nQ5_0 - Простое 5-битное квантование [citation:1]\nQ5_1 - Простое 5-битное квантование, чуть лучше Q5_0 [citation:1]\nQ8_0 - 8-битное, почти без потерь, но большой размер [citation:1]")

def donation():
    try:
        import webbrowser
        webbrowser.open("https://www.donationalerts.com/r/whiterabbit_")
    except Exception as e:
        log_to_file(f"Ошибка открытия браузера: {e}")

block6_top = tk.LabelFrame(second_tab, text="Затрагиваемые уровни модели:",bd=0, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
block6_top.grid_propagate(False)
block6_top.grid(row=1, column=1, sticky="nsew", padx=space_out, pady=space_out)
make_tooltip(block6_top, "Target Modules (target_modules): Список слоев модели (например, q_proj, v_proj), к которым применяется LoRA. \nОбычно это все основные линейные слои внимания")

block6_top.grid_rowconfigure(0, weight=1)
block6_top.grid_rowconfigure(1, weight=0)
block6_top.grid_columnconfigure(0, weight=1)


block6_bottom = tk.Frame(second_tab, highlightbackground=color_span, highlightthickness=weight_span, bg=bg_color)
block6_bottom.grid_propagate(False)
block6_bottom.grid(row=2, column=1, sticky="nsew", padx=space_out, pady=space_out)

block6_bottom.grid_rowconfigure(0, weight=1)
block6_bottom.grid_rowconfigure(1, weight=0)
block6_bottom.grid_columnconfigure(0, weight=1)

tk.Label(block6_bottom, text="Автор старался!\nАвтор хороший!\nАвтора можно поддержать!\n\n\n\nПозднее этот блок будет использован под \nболее важные вещи но пока так))", bg=bg_color).grid(row=0, column=0, sticky="nsew")
btn_save = tk.Button(block6_bottom, text="Поддержать автора!", command=donation)
btn_save.grid(row=1, column=0, sticky="ew")
# endregion

# region === вкладка 3 ===
# ВКЛАДКА 3

third_tab = ttk.Frame(notebook)
notebook.add(third_tab, text="Поиск моделей")

third_tab.grid_columnconfigure(0, weight=1)
third_tab.grid_rowconfigure(1, weight=1)
filter_frame = ttk.Frame(third_tab)
filter_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
filter_frame.grid_columnconfigure(1, weight=1)

ttk.Label(filter_frame, text="Поиск:").grid(row=0, column=0, padx=2)
search_entry = ttk.Entry(filter_frame)
search_entry.grid(row=0, column=1, sticky="ew", padx=2)
search_entry.bind("<Return>", perform_search)

# 2. Фильтр "от"
ttk.Label(filter_frame, text="Размер (от):").grid(row=0, column=2, padx=2)
from_size_combobox = ttk.Combobox(
    filter_frame,
    values=["Любой", "1B", "3B", "6B", "13B", "30B", "70B"],
    state="readonly",
    width=6
)
from_size_combobox.grid(row=0, column=3, padx=2)
from_size_combobox.set("Любой")
# 3. Фильтр "до"
ttk.Label(filter_frame, text="до:").grid(row=0, column=4, padx=2)
to_size_combobox = ttk.Combobox(
    filter_frame,
    values=["Любой", "1B", "3B", "6B", "13B", "30B", "70B"],
    state="readonly",
    width=6
)
to_size_combobox.grid(row=0, column=5, padx=2)
to_size_combobox.set("Любой")

search_button = ttk.Button(filter_frame, text="Найти", command=lambda: perform_search(None))
search_button.grid(row=0, column=6, padx=5)

download_btn = ttk.Button(third_tab, text="Скачать выбранную модель", command=download_selected_model)
download_btn.grid(row=2, column=0, pady=10)

# Список для результатов
result_listbox = tk.Listbox(third_tab, font=("Segoe UI", 10))
result_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

# Добавляем скроллбар
scrollbar = ttk.Scrollbar(third_tab, orient="vertical", command=result_listbox.yview)
scrollbar.grid(row=1, column=1, sticky="ns", pady=5)
result_listbox.config(yscrollcommand=scrollbar.set)


#endregion

# region === вкладка 4 ===
# ВКЛАДКА 4
fourth_tab = ttk.Frame(notebook)
notebook.add(fourth_tab, text="Квантование")
notebook.tab(fourth_tab, state="disabled")

fourth_tab.grid_rowconfigure(0, weight=1)
fourth_tab.grid_rowconfigure(1, weight=0)
fourth_tab.grid_rowconfigure(2, weight=0)
fourth_tab.grid_rowconfigure(3, weight=0)
fourth_tab.grid_rowconfigure(4, weight=0)
fourth_tab.grid_rowconfigure(5, weight=1)
fourth_tab.grid_columnconfigure(0, weight=2)
fourth_tab.grid_columnconfigure(1, weight=3)
fourth_tab.grid_columnconfigure(2, weight=1)
fourth_tab.grid_columnconfigure(3, weight=2)

models_file_q = tk.Entry(fourth_tab, bg="white")
models_file_q.grid(row=1, column=1, sticky="nsew", padx=(space_in,0), pady=(space_in,0))
models_file_q.insert(0, "Выберите модель в формате.gguf ...")
models_file_q.bind("<KeyRelease>", lambda e: check_paths_q())
Btn_browse_q = tk.Button(fourth_tab, text="Обзор", command=browse_model_gguf_q)
Btn_browse_q.grid(row=1, column=2, sticky="nsew", padx=(0, space_in), pady=(space_in,0))
result_file_q = tk.Entry(fourth_tab, bg="white")
result_file_q.grid(row=2, column=1, sticky="nsew", padx=(space_in,0), pady=(0,space_in))
result_file_q.insert(0, "Выберите место сохранения квантованной модели...")
result_file_q.bind("<KeyRelease>", lambda e: check_paths_q())
Btn_browse_res_q = tk.Button(fourth_tab, text="Обзор", command=browse_model_result_q)
Btn_browse_res_q.grid(row=2, column=2, sticky="nsew", padx=(0, space_in), pady=(0,space_in))
quant_options_q = [
    # === Рекомендуемые K-Quants (лучший баланс) ===
    "Q2_K",      # 2.5-3.5 бит, минимальный размер, заметная потеря качества
    "Q3_K_S",    # 3-4 бит, подходит для маленьких моделей
    "Q3_K_M",    # 3-4 бит, сбалансированный
    "Q3_K_L",    # 3-4 бит, крупный, лучше качество
    "Q4_K_S",    # 4-5 бит, маленький, хорошая скорость
    "Q4_K_M",    # Рекомендуемый, лучший баланс размера и качества
    "Q5_K_S",    # 5-6 бит, очень хорошее качество, маленький
    "Q5_K_M",    # Более высокое качество, немного больше размер
    "Q6_K",      # 6-7 бит, отличное качество, минимальная потеря
    "Q8_K",      # 8 бит, почти оригинальное качество
    # === Устаревшие (Legacy Quants) ===
    "Q4_0",      # Простое 4-битное квантование, не рекомендуется
    "Q4_1",      # Простое 4-битное квантование, чуть лучше Q4_0
    "Q5_0",      # Простое 5-битное квантование
    "Q5_1",      # Простое 5-битное квантование, чуть лучше Q5_0
    "Q8_0",      # 8-битное, почти без потерь, но большой размер
]
tk.Label(fourth_tab, text="Тип квантования: ",  anchor="w", bg=bg_color).grid(row=3, column=1, sticky="nsew")
selected_quant_q = tk.StringVar()
selected_quant_q.set("Q4_K_M")
entry_quant = ttk.Combobox(
    fourth_tab,
    textvariable=selected_quant_q,
    values=quant_options_q,
    width=entr_width*2,
    state="readonly"
)
entry_quant.grid(row=3, column=2, sticky="ns")

Btn_start_quant = tk.Button(fourth_tab, text="Запуск квантования", state="disabled")
Btn_start_quant.grid(row=4, column=1, columnspan=2,sticky="nsew", padx=0, pady=0)
Btn_start_quant.config(command=only_quant_start)

make_tooltip(models_file_q, "Укажите путь к исходной модели fn16/fn23/fd16/и т.д.")
make_tooltip(result_file_q, "Укажите путь для сохранения модели после квантования")
make_tooltip(chkbox1, "Конвертирование в формат .gguf. \nБез конвертации квантование будет недоступно")
make_tooltip(entry_quant, "Квантование не обязательная вещь но она способна сильно облегчить модель и ее вычисления за счет потери точности. \nПо умолчанию установлен режим: f16. (16бит) - модель в исходной форме.\n=== Рекомендуемые K-Quants (лучший баланс) ===\nQ2_K - 2.5-3.5 бит, минимальный размер, заметная потеря качества [citation:1]\nQ3_K_S - 3-4 бит, подходит для маленьких моделей [citation:1]\nQ3_K_M - 3-4 бит, сбалансированный [citation:1]\nQ3_K_L - 3-4 бит, крупный, лучше качество [citation:1]\nQ4_K_S - 4-5 бит, маленький, хорошая скорость [citation:1]\nQ4_K_M - Рекомендуемый, лучший баланс размера и качества [citation:1][citation:4]\nQ5_K_S - 5-6 бит, очень хорошее качество, маленький [citation:1]\nQ5_K_M - Более высокое качество, немного больше размер [citation:1]\nQ6_K - 6-7 бит, отличное качество, минимальная потеря [citation:1]\nQ8_K - 8 бит, почти оригинальное качество [citation:1]\n=== Устаревшие (Legacy Quants) ===\nQ4_0 - Простое 4-битное квантование, не рекомендуется [citation:1]\nQ4_1 - Простое 4-битное квантование, чуть лучше Q4_0 [citation:1]\nQ5_0 - Простое 5-битное квантование [citation:1]\nQ5_1 - Простое 5-битное квантование, чуть лучше Q5_0 [citation:1]\nQ8_0 - 8-битное, почти без потерь, но большой размер [citation:1]")


# endregion

# region === вкладка 5 ===
# ВКЛАДКА 5 - ART GALLERY
fifth_tab = ttk.Frame(notebook)
notebook.add(fifth_tab, text="Арты") # Решил не нумеровать. малоли припрет еще что-то вставить) лень переименовывать будет)

fifth_tab.grid_rowconfigure(0, weight=1)
fifth_tab.grid_rowconfigure(1, weight=0)
fifth_tab.grid_columnconfigure(0, weight=1)

# Белый блок для артов
art_canvas = tk.Canvas(fifth_tab, bg="white", highlightthickness=0)
art_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

def arts_fold(event):
    try:
        import webbrowser
        webbrowser.open(GOOGLE_DRIVE_FOLDER_LINK)
    except Exception as e:
        log_to_file(f"Ошибка открытия браузера: {e}")

arts_message = tk.Label(fifth_tab, text="Ваши арты. (спасибо за актив)", bg=bg_color)
arts_message.grid(row=1, column=0, sticky="nsew")
arts_message.bind("<Button-1>", arts_fold)
make_tooltip(arts_message, "Решил добавить в программу холст для ваших артов) \nне переживайте они все находятся в гугл папке и не грузятся на ваш пк. \nЕсли хотите посмотреть в хорошем качестве нажмите на эту надпись и откроется гугл папка.")

notebook.bind('<<NotebookTabChanged>>', on_tab_selected)

# endregion

text_monitor = ""

def on_closing():
    #log_to_file("Приложение закрыто")
    root.destroy()

def thread_animation_loop():
    try:
        global FACE_NOW
        global text_monitor
        face_now = str(FACE_NOW)
        face = open(face_now, "r", encoding="utf-8").read()
        i = 0
        ch = -1
        line = 0
        while True:
            if not face_now == FACE_NOW:
                face = open(FACE_NOW, "r", encoding="utf-8").read()
                face_now = FACE_NOW
            if line == 0 or line == 6:
                ch *= -1
            if line == 0:
                root.after(0, lambda: label_face.config(text=face))
            if line == 1:
                root.after(0, lambda: label_face.config(text=face))
            elif line == 2:
                root.after(0, lambda: label_face.config(text="\n" + face))
            elif line == 3:
                root.after(0, lambda: label_face.config(text="\n\n" + face))
            elif line == 4:
                root.after(0, lambda: label_face.config(text="\n\n\n" + face))
            elif line == 5:
                root.after(0, lambda: label_face.config(text="\n\n\n\n" + face))
            elif line == 6:
                root.after(0, lambda: label_face.config(text="\n\n\n\n" + face))
            # lines.insert(0, "# Настройки программы\n")
            line += ch
            if text_monitor is not None :
                if i >= len(text_monitor): i = 0
                lable_stat_1.config(text=text_monitor[i:] + " | " + text_monitor[:i])
                i += 1

            time.sleep(0.3)
    except Exception as e:
        log_to_file(f"Произошла ошибка: {e}")

def get_cpu_temperature():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            temp = float(result.stdout.strip())
            return temp
    except Exception as e:
        try:
            import psutil
            return list(psutil.sensors_temperatures().values())[0][0].current
        except Exception as e:
            log_to_file(f"Ошибка сбора температуры процессора: {e}")
    return None

def monitor_text ():
    error=False
    global text_monitor, mon_device
    try:
        import psutil
    except Exception as e:
        log_to_file(f"Ошибка при сборе мониторинга: {e}")
        error = True
    while not error:
        try:
            if mon_device == "NVIDIA" :
                import GPUtil
                text_monitor = (
                    f"CPU temp: {get_cpu_temperature()}°C | CPU load: {psutil.cpu_percent()}% | GPU mem: {GPUtil.getGPUs()[0].memoryUsed:.0f} МБ / {GPUtil.getGPUs()[0].memoryTotal:.0f} МБ ({GPUtil.getGPUs()[0].memoryUtil * 100:.1f}%)"
                    f" | GPU temp: {GPUtil.getGPUs()[0].temperature}°C | GPU load: {GPUtil.getGPUs()[0].load * 100:.1f}%")
            if mon_device == "CPU" :
                text_monitor = (
                    f"CPU temp: {get_cpu_temperature()}°C | CPU load: {psutil.cpu_percent()}%")
            if mon_device == "AMD":
                import pyamdgpuinfo
                gpu = pyamdgpuinfo.get_gpu(0)
                text_monitor = (
                    f"CPU temp: {get_cpu_temperature()}°C | CPU load: {psutil.cpu_percent()}% | "
                    f"GPU mem: {gpu.memory_info['used'] / (1024 ** 2):.0f} МБ / {gpu.memory_info['total'] / (1024 ** 2):.0f} МБ ({gpu.query_vram_usage():.1f}%) | "
                    f"GPU temp: {gpu.query_temperature()}°C | GPU load: {gpu.query_load() / 100:.1f}%")
        except Exception as e:
            log_to_file(f"Ошибка при сборе мониторинга: {e}")
            error = True
        time.sleep(2)

root.after(100, lambda: threading.Thread(target=thread_animation_loop, daemon=True).start())
root.after(100, lambda: threading.Thread(target=monitor_text, daemon=True).start())

root.protocol("WM_DELETE_WINDOW", on_closing)
log_to_file("Приложение запущено")

activate_venv()

log_to_file("Запуск GUI")
root.after(100, resize_label,None)

root.after(200, lambda: threading.Thread(target=chk_system, daemon=True).start())
root.mainloop()