# download.py - НОВЫЙ с huggingface_hub + прогресс
import sys
import json
import time
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def download_model(model_id, local_dir):
    """Скачивает модель с прогрессом"""
    local_path = Path(local_dir)
    local_path.mkdir(exist_ok=True)
    
    print(f"Скачиваю {model_id} → {local_dir}")
    
    try:
        # Полная модель с кэшем HF + прогресс
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,  # Копировать файлы
            resume_download=True,         # Продолжить при обрыве
            max_workers=4,               # Параллельная загрузка
            cache_dir=str(SCRIPT_DIR / "hf_cache")  # Собственный кэш
        )
        print("СКАЧАНО!")
        return True
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return False

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent.absolute()
    
    params = json.loads(sys.argv[1])
    model_id = params['model_id']
    local_dir = params['local_dir']
    
    success = download_model(model_id, local_dir)
    sys.exit(0 if success else 1)
