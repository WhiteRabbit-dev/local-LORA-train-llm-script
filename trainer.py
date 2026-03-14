import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import sys
import os
from pathlib import Path
import threading
import time
import traceback
import importlib.util  # для venv импортов

SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
PROCESS_DIR = SCRIPT_DIR / "process"
PROCESS_DIR.mkdir(exist_ok=True)
LOG_FILE = SCRIPT_DIR / "logs" / "logs.txt" 
LOGS_DIR = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# VENV ЛОГИКА
VENV_DIR = SCRIPT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"

def ensure_venv():
    """Создать venv если нет"""
    if not VENV_PYTHON.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        log_to_file(".venv создан")

def run_in_venv(args, **kwargs):
    cmd = [str(VENV_PYTHON)] + args
    kwargs.setdefault('capture_output', True)   # ← ЛОВИМ ВСЕГДА
    kwargs.setdefault('text', True)
    kwargs.setdefault('encoding', 'cp1251')  # ← RUSSIAN Windows!
    kwargs.setdefault('errors', 'ignore')    # ← Игнор битых символов
    return subprocess.run(cmd, **kwargs)

def test_lib_in_venv(lib_name):
    try:
        if lib_name == "protobuf":
            code = (
                "try:\n"
                "    import google.protobuf\n"
                "    print('OK')\n"
                "except Exception as e:\n"
                "    print('FAIL')"
            )
        else:
            code = (
                "try:\n"
                f"    import {lib_name}\n"
                "    print('OK')\n"
                "except Exception as e:\n"
                "    print('FAIL')"
            )
        result = run_in_venv(["-c", code])
        return 'OK' in result.stdout
    except:
        return False


# Логирование
def log_to_file(message, is_error=False):
    """Все логи в logs.txt"""
    timestamp = time.strftime("%Y-%m-%d %H.%M.%S")
    level = "❌ ОШИБКА" if is_error else "INFO" 
    log_line = f"[{timestamp}] {level}: {message}\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())


class TrainerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLM Тренер v1.0")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Переменные
        self.sys_list = None
        self.libs_list = None
        self.model_text = None
        self.dataset_text = None
        self.quant_model_text = None
        self.download_btn = None
        self.next_btn = None
        
        # Пути
        self.current_model = None
        self.current_dataset = None
        self.lora_output = PROCESS_DIR / "lora_adapter"
        self.merged_output = PROCESS_DIR / "merged_f16"
        self.gguf_output = PROCESS_DIR / "model_f16.gguf"
        self.quant_output = PROCESS_DIR / "model_quant.gguf"
        ensure_venv()
        self.setup_tabs()
        self.notebook.select(0)
        log_to_file("GUI запущен (.venv)")
        self.root.mainloop()

    def log(self, message):
        log_to_file(message)
        try:
            self.root.after(0, lambda: self.status_update(message))
        except:
            pass
    
    def status_update(self, message):
        self.root.title(f"LLM Тренер v1.0 (.venv) - {message[-60:]}")
    
    def safe_execute(self, func, *args, **kwargs):
        def wrapper():
            try:
                func(*args, **kwargs)
            except Exception as e:
                error_msg = f"ОШИБКА в {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                log_to_file(error_msg, is_error=True)  # ← В logs.txt
                messagebox.showerror("Ошибка", f"{func.__name__}(): {str(e)}")
            
        threading.Thread(target=wrapper, daemon=True).start()
    
    def update_libs_list(self):
        self.libs_list.delete(0, 'end')
        self.libs_list.insert(0, ".venv БИБЛИОТЕКИ")
        
        REQUIRED_LIBS = ['torch', 'transformers', 'peft', 'huggingface_hub', 
                        'psutil', 'sentencepiece', 'protobuf', 'accelerate', 'gguf']
        
        for lib in REQUIRED_LIBS:
            status = "yes" if test_lib_in_venv(lib) else "❌"
            self.libs_list.insert('end', f"{status} {lib}")
        
        # ПРЯМАЯ проверка pip list
        pip_result = run_in_venv(["-m", "pip", "list"])
        if pip_result.returncode == 0 and pip_result.stdout:
            self.libs_list.insert('end', f"Всего пакетов: {len(pip_result.stdout.splitlines())}")
        
        venv_status = "yes" if VENV_PYTHON.exists() else "no"
        self.libs_list.insert('end', f"{venv_status} .venv активен")
        self.libs_list.insert('end', "Готов к LoRA!")
    
    def analyze_system(self):
        self.sys_list.delete(0, 'end')
        self.sys_list.insert(0, "СИСТЕМА")
        
        # ДИАГНОСТИКА venv
        pip_list = run_in_venv(["-m", "pip", "list"])
        self.sys_list.insert('end', f".venv пакетов: {len(pip_list.stdout.splitlines()) if pip_list.stdout else 0}")
        
        # NVIDIA
        nvidia_ok = subprocess.run("nvidia-smi", capture_output=True).returncode == 0
        self.sys_list.insert('end', f"NVIDIA: {'yes' if nvidia_ok else '❌'}")

        
        # CPU/RAM через venv
        psu_result = run_in_venv(["-c", 
            "try:\n"
            "    import psutil\n"
            "    print(f'CPU: {psutil.cpu_count(logical=False)}/{psutil.cpu_count()}')\n"
            "    print(f'RAM: {psutil.virtual_memory().total/1e9:.1f}GB')\n"
            "except Exception as e:\n"
            "    print(f'psutil ОШИБКА: {str(e)}')"])
        if psu_result.returncode == 0 and psu_result.stdout:
            for line in psu_result.stdout.strip().split('\n'):
                self.sys_list.insert('end', f"{line}")
        else:
            self.sys_list.insert('end', f"psutil ОШИБКА: {psu_result.stderr[:100] if psu_result.stderr else 'не найден'}")

        # GPU через venv - ПОЛНАЯ ДИАГНОСТИКА
        gpu_result = run_in_venv(["-c", 
            "try:\n"
            "    import torch\n"
            "    print(f'torch версия: {torch.__version__}')\n"
            "    if torch.cuda.is_available():\n"
            "        gpu = torch.cuda.get_device_name(0)\n"
            "        vram = torch.cuda.get_device_properties(0).total_memory/1e9\n"
            "        cuda_ver = torch.version.cuda\n"
            "        print(f'GPU: {gpu}')\n"
            "        print(f'VRAM: {vram:.1f}GB')\n"
            "        print(f'CUDA версия: {cuda_ver}')\n"
            "        print('CUDA: yes')\n"
            "    else:\n"
            "        print('CUDA: no')\n"
            "        print(f'CUDA устройства: {torch.cuda.device_count()}')\n"
            "except Exception as e:\n"
            "    print(f'GPU ОШИБКА: {str(e)[:100]}')"])
            
        if gpu_result.returncode == 0 and gpu_result.stdout:
            for line in gpu_result.stdout.strip().split('\n'):
                if line.strip():
                    self.sys_list.insert('end', f" {line}")
        else:
            error_full = gpu_result.stderr[:200] if gpu_result.stderr else 'неизвестная ошибка'
            self.sys_list.insert('end', f" GPU ФЕЙЛ: {error_full}")


        self.update_libs_list()
        self.log(" Анализ готов")

        
    def install_libs(self):
        self.progress_var.set(0)
        self.progress_label.config(text=" pip...")

        simple_packages = [
            "psutil",
            "transformers>=4.36.0",
            "huggingface_hub",
            "sentencepiece",
            "protobuf<5.0",
            "accelerate",
            "gguf",
        ]

        complex_specs = [
            (["torch", "torchvision", "torchaudio"], ["--index-url", "https://download.pytorch.org/whl/cu121"]),
            (["peft"], ["--no-build-isolation", "--no-deps"]),
        ]

        total_steps = len(simple_packages) + len(complex_specs) 

        def next_pkg(i=0):
            if i >= total_steps:
                self.progress_var.set(100)
                self.progress_label.config(text=" ГОТОВО")
                self.root.after(2000, self.update_libs_list)
                return

            # Прогресс
            self.progress_var.set(i / total_steps * 100)

            if i < len(simple_packages):
                pkg = simple_packages[i]
                cmd = ["-m", "pip", "install", "--upgrade", pkg]
                self.libs_list.delete(0, 'end')
                self.libs_list.insert(0, f" {i+1}/{total_steps} {pkg}")
                result = run_in_venv(cmd)
                status = "yes" if result.returncode == 0 else "no"
                self.libs_list.insert('end', f"{status} {pkg}")
            else:
                idx = i - len(simple_packages)
                pkg_names, extra_args = complex_specs[idx]
                cmd = ["-m", "pip", "install", "--upgrade"] + extra_args + pkg_names
                self.libs_list.delete(0, 'end')
                self.libs_list.insert(0, f" {i+1}/{total_steps} {' '.join(pkg_names)}")
                result = run_in_venv(cmd)
                status = "yes" if result.returncode == 0 else "no"
                self.libs_list.insert('end', f"{status} {' '.join(pkg_names)}")

            self.root.after(2000, lambda: next_pkg(i+1))

        next_pkg() 

    def setup_tabs(self):
        self.main_frame = ttk.Frame(self.notebook)
        self.setup_main_tab()
        self.notebook.add(self.main_frame, text="Главная")
        
        self.params_frame = ttk.Frame(self.notebook)
        self.setup_params_tab()
        self.notebook.add(self.params_frame, text="Параметры")
        
        self.quant_frame = ttk.Frame(self.notebook)
        self.setup_quant_tab()
        self.notebook.add(self.quant_frame, text=" Квантование")
    
    def setup_main_tab(self):
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill='x', padx=10, pady=(0,10))
        ttk.Label(status_frame, textvariable=tk.StringVar(value="Готов")).pack()
        
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        sys_frame = ttk.LabelFrame(top_frame, text="Система + .venv", padding=10)
        sys_frame.pack(side='left', fill='both', expand=True, padx=(0,5))
        self.sys_list = tk.Listbox(sys_frame, height=12)
        self.sys_list.pack(fill='both', expand=True)
        self.sys_list.insert(0, "Нажми 'Анализ системы'")
        
        libs_frame = ttk.LabelFrame(top_frame, text=".venv Библиотеки", padding=10)
        libs_frame.pack(side='right', fill='both', expand=True, padx=(5,0))
        self.libs_list = tk.Listbox(libs_frame, height=12)
        self.libs_list.pack(fill='both', expand=True)
        
        btn_frame1 = ttk.Frame(self.main_frame)
        btn_frame1.pack(pady=15)
        ttk.Button(btn_frame1, text="Анализ системы", 
                  command=lambda: self.safe_execute(self.analyze_system)).pack(side='left', padx=20)
        ttk.Button(btn_frame1, text="Установить в .venv", 
                  command=lambda: self.safe_execute(self.install_libs)).pack(side='right', padx=20)
        
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.pack(fill='x', pady=10)        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=400, mode='determinate')
        self.progress_bar.pack()
        self.progress_label = ttk.Label(progress_frame, text="Готов")
        self.progress_label.pack()
        
        input_frame = ttk.Frame(self.main_frame)
        input_frame.pack(fill='x', padx=10, pady=10)
        input_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Модель (HF/путь f16):").grid(row=0, column=0, sticky='w', pady=5)
        self.model_text = tk.Text(input_frame, width=60, height=1, wrap='none', relief="solid", bd=1)
        self.model_text.grid(row=0, column=1, padx=(10, 5), pady=5, sticky='ew')
        self.model_text.bind("<Control-KeyPress>", self.ru_paste)
        ttk.Button(input_frame, text="Выбор папки", command=self.select_model_folder).grid(row=0, column=2, padx=5)
        self.download_btn = ttk.Button(input_frame, text="Скачать", command=lambda: self.safe_execute(self.download_model))
        self.download_btn.grid(row=0, column=3, padx=5)
        self.model_status = ttk.Label(input_frame, text="Модель:")
        self.model_status.grid(row=0, column=4, padx=(5, 0), sticky='w')

        ttk.Label(input_frame, text="Датасет jsonl:").grid(row=1, column=0, sticky='w', pady=5)
        self.dataset_text = tk.Text(input_frame, width=60, height=1, wrap='none', relief="solid", bd=1)
        self.dataset_text.grid(row=1, column=1, padx=(10, 5), pady=5, sticky='ew')
        self.dataset_text.bind("<Control-KeyPress>", self.ru_paste)
        ttk.Button(input_frame, text="Выбор", command=self.select_dataset).grid(row=1, column=2, padx=5)
        
        comment_label = ttk.Label(input_frame, 
            text="Дальнейшее выполнение требует ПАПКУ МОДЕЛИ + ФАЙЛ ДАТАСЕТА (jsonl - файл, пример есть в папке с программой) перед 'Проверить ресурсы'!", 
            foreground="red", font=('Arial', 9, 'bold'))
        comment_label.grid(row=2, column=0, columnspan=5, sticky='w', pady=(10,5))  

        ttk.Button(input_frame, text="Проверить ресурсы", command=lambda: self.safe_execute(self.check_resources)).grid(row=3, column=1, pady=15)

        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.pack(pady=20)
        self.next_btn = ttk.Button(nav_frame, text="Далее", command=self.to_params, state='disabled')
        self.next_btn.pack(side='right', padx=10)
        ttk.Button(nav_frame, text="Назад", state='disabled').pack(side='right')
   
    def ru_paste(self, event):
        kc = event.keycode
        if kc == 86:  # V или М
            event.widget.event_generate("<<Paste>>")
        elif kc == 67:  # C или С  
            event.widget.event_generate("<<Copy>>")
        elif kc == 88:  # X или Х
            event.widget.event_generate("<<Cut>>")
        return "break"
        
    def setup_params_tab(self):
        # ОСНОВНЫЕ ПАРАМЕТРЫ
        main_params = ttk.LabelFrame(self.params_frame, text="ОСНОВНЫЕ", padding=10)
        main_params.pack(fill='x', padx=10, pady=10)
        
        row = 0
        self.epochs = self.create_param(main_params, "Эпох:", "1", row, 0, "Количество проходов по датасету (3-5)")
        self.batch_size = self.create_param(main_params, "Батч:", "1", row, 2, "Размер батча (1-4)")
        row += 1
        
        self.lora_r = self.create_param(main_params, "r:", "8", row, 0, "LoRA rank - сила адаптера (8-64)")
        ttk.Label(main_params, text="LR:").grid(row=row, column=2*2, sticky='e', padx=5, pady=5)
        self.lr_combo = ttk.Combobox(main_params, values=["1e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3"], width=8, state="readonly")
        self.lr_combo.set("2e-4")  # дефолт
        self.lr_combo.grid(row=row, column=2*2+1, padx=5, pady=5, sticky='w')

        # Подсказка
        self.create_tooltip(self.lr_combo, "Скорость обучения:\n1e-5=медленно\n2e-4=стандарт\n5e-4=быстро")
        row += 1
        
        self.max_len = self.create_param(main_params, "Контекст:", "256", row, 0, "Макс. длина текста (512-4096)")
        
        # ПРОДВИНУТЫЕ
        adv_params = ttk.LabelFrame(self.params_frame, text="ПРОДВИНУТЫЕ", padding=10)
        adv_params.pack(fill='x', padx=10, pady=10)
        
        row = 0
        self.lora_alpha = self.create_param(adv_params, "α:", "16", row, 0, "Усиление LoRA (обычно r*2)")
        self.dropout = self.create_param(adv_params, "Dropout:", "0.1", row, 2, "Регуляризация (0.05-0.2)")
        row += 1
        
        self.grad_acc = self.create_param(adv_params, "Grad acc:", "4", row, 0, "Накопление градиента")
        
        # LoRA СЛОИ (ОРИГИНАЛЬНЫЙ НАБОР)
        ttk.Label(adv_params, text="LoRA слои:").grid(row=row+1, column=0, sticky='w', pady=(10,5))
        self.lora_targets = ttk.Combobox(adv_params, values=[
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",  # ← оригинал (ВСЕ)
            "q_proj,k_proj,v_proj,o_proj",  # Быстрые
            "q_proj,v_proj",  # Легкие
            "c_attn"  # самая легкя для моделей GPT-chat 
        ], width=40, state="readonly")
        self.lora_targets.set("q_proj,v_proj")  # дефолт
        self.lora_targets.grid(row=row+1, column=1, sticky='w', padx=10, columnspan=2)
        
        # АВТОПАЙПЛАЙН
        auto_frame = ttk.LabelFrame(self.params_frame, text="АВТОМАТИЗАЦИЯ", padding=10)
        auto_frame.pack(fill='x', padx=10, pady=10)
        
        self.gguf_var = tk.BooleanVar(value=True)
        self.merge_var = tk.BooleanVar(value=True)
        self.quant_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(auto_frame, text="Конвертация в GGUF f16 (только вместе со слиянием)", variable=self.gguf_var).grid(row=0, column=0, sticky='w', padx=10)
        ttk.Checkbutton(auto_frame, text="Слияние новых весов LoRA", variable=self.merge_var).grid(row=0, column=3, sticky='w', padx=10)
        ttk.Checkbutton(auto_frame, text="Квантовать в Q4_K_M (только вместе с ковертцией)", variable=self.quant_var).grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        # КНОПКИ
        nav_frame = ttk.Frame(self.params_frame)
        nav_frame.pack(pady=20, fill='x')
        
        ttk.Button(nav_frame, text="ЗАПУСК ПАЙПЛАЙНА", 
                  command=lambda: self.safe_execute(self.start_training), 
                  style="Accent.TButton").pack(side='right', padx=10, ipadx=20)
        ttk.Button(nav_frame, text="⬅️ Назад", command=self.to_main).pack(side='right')
        
                # ТЕСТ КНОПКИ - ПРЯМО ВЫЗЫВАЮТ ОСНОВНЫЕ
        test_frame = ttk.Frame(self.params_frame)
        test_frame.pack(pady=10, fill='x')

        ttk.Separator(test_frame, orient='horizontal').pack(fill='x', pady=(0,10))

        ttk.Button(test_frame, text="ТОЛЬКО СЛИЯНИЕ", 
                  command=lambda: self.safe_execute(self.merge_lora)).pack(side='left', padx=10, ipadx=15)
        ttk.Button(test_frame, text="ТОЛЬКО GGUF", 
                  command=lambda: self.safe_execute(self.convert_to_gguf)).pack(side='left', padx=10, ipadx=15)
        ttk.Button(test_frame, text="СЛИЯНИЕ+GGUF", 
                  command=lambda: self.safe_execute(self.pipeline_merge_gguf)).pack(side='left', padx=10, ipadx=15)
        
        # ИНФО
        info_frame = ttk.LabelFrame(self.params_frame, text="РЕКОМЕНДАЦИИ", padding=8)
        info_frame.pack(fill='x', padx=10, pady=(0,10))
        
        info_text = """Установленные в форме данные это минимальные возможные для обучения нейросети, если поставить сверх мощности ПК скрипт просто крашнется 
        оригинальные данные не будут испорчены."""
        
        ttk.Label(info_frame, text=info_text, font=('Consolas', 9), 
                  foreground='blue', justify='left').pack(anchor='w')

        
    def create_param(self, parent, label, default, row, col, tooltip_text, validator=None):
        """Простое поле ввода с валидацией"""
        ttk.Label(parent, text=f"{label}").grid(row=row, column=col*2, sticky='e', padx=5, pady=5)
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col*2+1, padx=5, pady=5, sticky='w')
        
        entry = ttk.Entry(frame, width=8, justify='right')
        entry.pack(side='left')
        entry.insert(0, default)
        
        # Валидатор
        if validator:
            def validate_entry(P):
                if P == "":
                    return True
                try:
                    validator(P)
                    entry.config(foreground='black')
                    return True
                except ValueError:
                    entry.config(foreground='red')
                    return False
            
            vcmd = (self.root.register(validate_entry), '%P')
            entry.config(validate='key', validatecommand=vcmd)
        
        self.create_tooltip(frame, tooltip_text)
        return entry

    # НОВЫЕ ВАЛИДАТОРЫ
    def _validate_int(value): return int(float(value))
    def _validate_float(value): return float(value)
    def _validate_power2(value): 
        val = int(float(value))
        if val not in [1,2,4,8,16,32,64,128,256,512,1024]:
            raise ValueError("только степени 2: 1,2,4,8,16,32,64...")
        return val
    def _validate_positive(value):
        val = float(value)
        if val <= 0:
            raise ValueError("только > 0")
        return val
    
    def setup_quant_tab(self):
        ttk.Label(self.quant_frame, text="КВАНТОВАНИЕ", font=('Arial', 16, 'bold')).pack(pady=20)
        ttk.Label(self.quant_frame, text="Модель f16/gguf:").pack(pady=(0,5))
        self.quant_model_text = tk.Text(self.quant_frame, width=80, height=1, wrap='none', relief="solid", bd=1)
        self.quant_model_text.pack(pady=5)
        self.quant_model_text.bind("<Control-KeyPress>", self.ru_paste)
        ttk.Button(self.quant_frame, text="Выбрать", command=lambda: self.select_file(self.quant_model_text)).pack(pady=5)
        
        ttk.Label(self.quant_frame, text="Метод:").pack(pady=(20,5))
        self.quant_method = ttk.Combobox(self.quant_frame, 
            values=["Q2_K", "Q3_K_S", "Q4_0", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"], 
            width=15, state="readonly")
        self.quant_method.set("Q4_K_M")
        self.quant_method.pack(pady=5)
        
        self.quant_status = ttk.Label(self.quant_frame, text="GPU: Проверка...")
        self.quant_status.pack(pady=20)
        
        self.quant_btn = ttk.Button(self.quant_frame, text="Квантовать", 
                                  command=lambda: self.safe_execute(self.quantize_model), state='disabled')
        self.quant_btn.pack(pady=20)
        
        btn_frame = ttk.Frame(self.quant_frame)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="Только слияние", command=lambda: self.safe_execute(self.merge_lora)).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="Только GGUF", command=lambda: self.safe_execute(self.convert_to_gguf)).pack(side='left', padx=10)
   
    def select_model_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с моделью", initialdir=str(SCRIPT_DIR))
        if folder:
            self.model_text.delete("1.0", 'end')
            self.model_text.insert("1.0", folder)

    def create_tooltip(self, widget, text):
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        label = tk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1,
                        font=('Arial', 9), wraplength=300)
        label.pack()
        tooltip.withdraw()
        
        def show(event):
            x, y = widget.winfo_rootx() + 25, widget.winfo_rooty() + 25
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def hide(event):
            tooltip.withdraw()
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
    
    def select_file(self, text_widget):
        filename = filedialog.askopenfilename(filetypes=[("Model files", "*.gguf *.bin *.safetensors *.pt")])
        if filename:
            text_widget.delete("1.0", 'end')
            text_widget.insert("1.0", filename)
            if text_widget == self.quant_model_text:
                self.quant_btn.config(state='normal')
        
    def download_model(self):
        from huggingface_hub import snapshot_download
        import shutil
        model_id = self.model_text.get("1.0", 'end-1c').strip()
        if not model_id or "/" not in model_id:
            messagebox.showerror("Ошибка", "HF ID: пользователь/модель")
            return
        
        self.download_btn.config(state='disabled', text="Скачиваю...")
        self.model_status.config(text="Скачиваю...")
        self.root.update()  # Обновляем UI
        
        def download_thread():
            try:
                local_dir = PROCESS_DIR / "downloaded_model"
                local_dir.mkdir(exist_ok=True)
                
                self.root.after(0, lambda: self.log(f"Скачиваю {model_id}"))
                
                # ПРЯМАЯ загрузка в GUI потоке!
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=3
                )
                
                # АВТОЗАМЕНА пути в поле!
                self.root.after(0, lambda: self.model_text.delete("1.0", 'end'))
                self.root.after(0, lambda: self.model_text.insert("1.0", str(local_dir)))
                
                self.root.after(0, lambda: self.model_status.config(text="Скачано!"))
                self.root.after(0, lambda: self.log(f"Модель готова: {local_dir}"))
                self.root.after(0, messagebox.showinfo, "Готово", f"Модель в:\n{local_dir}")
                
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Ошибка скачивания: {e}", is_error=True))
                self.root.after(0, messagebox.showerror, "Ошибка", str(e))
            finally:
                self.root.after(0, lambda: self.download_btn.config(state='normal', text="Скачать"))

        threading.Thread(target=download_thread, daemon=True).start()


    def select_dataset(self):
        filename = filedialog.askopenfilename(filetypes=[("JSONL", "*.jsonl"), ("All", "*.*")])
        if filename:
            self.dataset_text.delete("1.0", 'end')
            self.dataset_text.insert("1.0", filename)
    
    def toggle_params_tab(self, enable=False):
        """Блокирует/разблокирует вкладку Параметры"""
        if enable:
            self.notebook.tab(1, state='normal') 
        else:
            self.notebook.tab(1, state='disabled')

    
    def check_resources(self):
        model_path = self.model_text.get("1.0", 'end-1c').strip()
        dataset_path = self.dataset_text.get("1.0", 'end-1c').strip()

        self.model_status.config(text="Модель: проверка...")

        if not model_path or not dataset_path:
            self.model_status.config(text="Модель: данные не указаны")
            self.next_btn.config(state='disabled')
            self.toggle_params_tab(False)
            return
        
        if not (os.path.isabs(model_path) or "://" in model_path or "/" in model_path):
            self.model_status.config(text="Модель: НЕВЕРНЫЙ ПУТЬ")
            self.next_btn.config(state='disabled')
            self.toggle_params_tab(False)
            messagebox.showwarning("Внимание", "Путь модели должен быть абсолютным или HF‑идентификатором!")
            return
        
        if os.path.isdir(model_path):
            ok, msg = self.is_model_dir(model_path)
            if not ok:
                self.log(f"Модель: {msg}")
                self.model_status.config(text=f"Модель: {msg[:40]}...")
                self.next_btn.config(state='disabled')
                self.toggle_params_tab(False)
                messagebox.showwarning("Модель", f"Папка не похожа на модель:\n{msg}")
                return
            self.model_status.config(text="Модель: ПАПКА OK")
        else:
            self.model_status.config(text="Модель: HF / не папка")
        
        if not os.path.exists(dataset_path):
            self.log("Датасет не найден")
            self.next_btn.config(state='disabled')
            self.toggle_params_tab(False)
            messagebox.showwarning("Датасет", "Файл датасета не найден!")
            return
        
        self.current_model = model_path
        self.current_dataset = dataset_path
        self.log("Ресурсы проверены: модель + датасет")
        self.next_btn.config(state='normal')
        self.toggle_params_tab(True)
    
    def to_params(self):
        model_text = self.model_text.get("1.0", 'end-1c').strip()
        dataset_text = self.dataset_text.get("1.0", 'end-1c').strip()
        if model_text and dataset_text:
            self.notebook.select(1)
        else:
            messagebox.showwarning("Внимание", "Укажите модель и датасет!")
    
    def to_main(self):
        self.notebook.select(0)
    
    # VENV ПАЙПЛАЙН (использует run_in_venv)
    def start_training(self):
        if not all([self.current_model, self.current_dataset]):
            messagebox.showerror("Ошибка", "Укажите модель и датасет!")
            return
        confirm = messagebox.askyesno("Старт", f"ПОЛНЫЙ ПАЙПЛАЙН?\n{self.current_model}\n{self.current_dataset}")
        if confirm:
            self.log("VENV ПАЙПЛАЙН!")
            threading.Thread(target=self._full_pipeline, daemon=True).start()
    
    def _full_pipeline(self):
        try:
            self.log("[01] ОБУЧЕНИЕ LoRA")
            self.run_training_script()
            
            if self.merge_var.get():
                self.log("[02] СЛИЯНИЕ")
                self.merge_lora()
            
            if self.gguf_var.get():
                self.log("[03] GGUF")
                self.convert_to_gguf()
            
            if self.quant_var.get():
                self.log("[04] КВАНТ")
                self.quantize_model_final()
            
            self.log("VENV ПАЙПЛАЙН OK!")
        except Exception as e:
            self.log(f"{e}")
    
    def run_training_script(self):
        """LoRA + ЖДЕТ завершения"""
        if not all([self.current_model, self.current_dataset]):
            raise ValueError("Нет модели/датасета")       
        
        self.training_active = True  # ← БЛОКИРОВКА пайплайна!
        
        params = {
            'model_path': self.current_model,
            'dataset_path': self.current_dataset,
            'lora_path': str(self.lora_output),
            'epochs': int(float(self.epochs.get() or 1)),
            'lr': float(self.lr_combo.get() or "2e-4"),
            'batch_size': int(self.batch_size.get() or 1),
            'lora_r': int(self.lora_r.get() or 8),
            'lora_alpha': int(self.lora_alpha.get() or 16),
            'max_len': int(self.max_len.get() or 256),
            'lora_targets': [t.strip() for t in self.lora_targets.get().split(",")]
        }
        
        self.log(f"[01] ОБУЧЕНИЕ LoRA (PID в новом окне)")
        
        import json
        params_json = json.dumps(params)
        script_path = Path(__file__).parent / "train.py"
        
        import subprocess
        cmd = [str(VENV_PYTHON), str(script_path), params_json]
        
        CREATE_NEW_CONSOLE = 0x10
        process = subprocess.Popen(cmd, creationflags=CREATE_NEW_CONSOLE,
                                  cwd=str(Path(__file__).parent))
        
        self.log(f"✅ ОБУЧЕНИЕ ЗАПУЩЕНО! PID: {process.pid}")
        self.log(f"max_len: {params['max_len']}")
        
        process.wait()
        self.training_active = False
        
        if process.returncode != 0:
            raise RuntimeError("Обучение упало!")
        
        self.log(f"✅ LoRA готов: {self.lora_output}")

        
    def merge_lora(self):
        """Слияние через отдельный merge.py"""
        if not self.lora_output.exists():
            raise ValueError("LoRA не обучена!")
        if not self.current_model:
            raise ValueError("Нет базовой модели!")
        
        self.log("[02] СЛИЯНИЕ LoRA")
        
        params = {
            'model_path': self.current_model,
            'lora_path': str(self.lora_output),
            'merged_path': str(self.merged_output)
        }
        
        import json
        params_json = json.dumps(params)
        script_path = Path(__file__).parent / "merge.py"
        
        cmd = [str(VENV_PYTHON), str(script_path), params_json]
        CREATE_NEW_CONSOLE = 0x10
        process = subprocess.Popen(cmd, creationflags=CREATE_NEW_CONSOLE,
                                  cwd=str(Path(__file__).parent))
        
        self.log(f"СЛИЯНИЕ ЗАПУЩЕНО! PID: {process.pid}")
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("Слияние упало!")
        
        self.log("Слияние завершено: " + str(self.merged_output))

    
    def convert_to_gguf(self):
        """Конвертация HF -> GGUF через отдельное окно"""
        if not self.merged_output.exists():
            raise ValueError("Нет слитой модели!")
        if not self.current_model:
            raise ValueError("Нет базовой модели!")
        
        self.log("[03] КОНВЕРТАЦИЯ HF -> GGUF")
        
        params = {
            'hf_path': str(self.merged_output),
            'gguf_path': str(self.gguf_output)
        }
        
        import json
        params_json = json.dumps(params)
        script_path = Path(__file__).parent / "convert.py"
        
        cmd = [str(VENV_PYTHON), str(script_path), params_json]
        CREATE_NEW_CONSOLE = 0x10
        process = subprocess.Popen(cmd, creationflags=CREATE_NEW_CONSOLE,
                                  cwd=str(Path(__file__).parent))
        
        self.log(f"GGUF ЗАПУЩЕН PID: {process.pid}")
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("Конвертация упала!")
        
        self.log("GGUF готов: " + str(self.gguf_output))

    
    def quantize_model_final(self):
        """Квантование через отдельное quantize.py"""
        if not self.gguf_output.exists():
            raise ValueError("Нет f16.gguf!")
        
        quant_method = self.quant_method.get() if hasattr(self, 'quant_method') else "Q4_K_M"
        
        self.log(f"[04] КВАНТОВАНИЕ {quant_method}")
        
        params = {
            'f16_path': str(self.gguf_output),
            'q_path': str(self.quant_output),
            'quant_method': quant_method
        }
        
        import json
        params_json = json.dumps(params)
        script_path = Path(__file__).parent / "quantize.py"
        
        cmd = [str(VENV_PYTHON), str(script_path), params_json]
        CREATE_NEW_CONSOLE = 0x10
        process = subprocess.Popen(cmd, creationflags=CREATE_NEW_CONSOLE,
                                  cwd=str(Path(__file__).parent))
        
        self.log(f"КВАНТ ЗАПУЩЕН PID: {process.pid}")
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("Квантование упало!")
        
        self.log(f"{quant_method} готов: {self.quant_output}")

    def quantize_model(self):
        model_path = self.quant_model_text.get("1.0", 'end-1c').strip()
        if not model_path: 
            return
        self.safe_execute(self.quantize_model_final)
    
    def is_model_dir(self, path_str):
        """Проверяет, является ли каталог валидной моделью HF (f16, GGUF, safetensors и т.п.)"""
        path = Path(path_str)
        if not path.exists():
            return False, "Каталог не существует"

        # Проверяем наличие хотя бы минимального набора файлов
        required_files = ["config.json"]
        optional_files = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        ]

        # Проверяем, есть ли хоть один файл модели и конфиг
        found_model = any((path / f).exists() for f in required_files + optional_files)
        if not found_model:
            return False, "Нет ни одного файла модели (model.safetensors, pytorch_model.bin и т.п.)"

        if not (path / "config.json").exists():
            return False, "Нет config.json"

        return True, "Папка выглядит как валидная модель"

    
# Запуск
if __name__ == "__main__":
    app = TrainerGUI()
