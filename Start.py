import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import webbrowser
from pathlib import Path
import glob

SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
print(f"Рабочая папка: {os.getcwd()}")

class SlidesApp:
    def __init__(self):
        self.root = tk.Tk()        
        self.root.resizable(False, False)
        self.root.title("LLM Тренер")
        self.root.geometry("900x600")
        self.root.minsize(880, 600)        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)        
        self.show_slide1()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def show_slide1(self):
        self.clear_window()
        
        # Заголовок
        title = tk.Label(self.root, text="🎉 Добро пожаловать!", 
                        font=('Arial', 28, 'bold'), bg='white')
        title.grid(row=0, column=0, pady=20, sticky='n')
        
        self.load_image("slide1.jpg")
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=1, column=0, pady=20, sticky='sew')
        ttk.Button(btn_frame, text="💝 Поддержать", 
                  command=self.boosty_click, width=20).pack(side='left', padx=15)
        ttk.Button(btn_frame, text="➡️ Пропуск", 
                  command=self.skip_click, width=20).pack(side='left')
    
    def boosty_click(self):
        print("Донат...")
        webbrowser.open("https://www.donationalerts.com/r/whiterabbit_")
        self.show_slide2a()
    
    def skip_click(self):
        self.show_slide2b()
    
    def show_slide2a(self):
        self.clear_window()
        
        title = tk.Label(self.root, text="❤️ Спасибо за поддержку!", 
                        font=('Arial', 26, 'bold'), fg='darkred', bg='white')
        title.grid(row=0, column=0, pady=20, sticky='n')
        
        self.load_image("slide2a.jpg")
        
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=1, column=0, pady=20, sticky='sew')
        ttk.Button(btn_frame, text="🚀 К программе", 
                  command=self.launch_trainer, width=20).pack(pady=10)
    
    def show_slide2b(self):
        self.clear_window()
        
        title = tk.Label(self.root, text="Начинаем!", 
                        font=('Arial', 28, 'bold'), fg='green', bg='white')
        title.grid(row=0, column=0, pady=20, sticky='n')
        
        self.load_image("slide2b.jpg")
        
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=1, column=0, pady=20, sticky='sew')
        
        ttk.Button(btn_frame, text="🚀 К программе", 
                  command=self.launch_trainer, width=20).pack(side='left', padx=15)
        ttk.Button(btn_frame, text="⬅️ На главную", 
                  command=self.show_slide1, width=20).pack(side='left', padx=15)
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def load_image(self, img_name):
        img_path = Path("images") / img_name
        print(f"🔍 {img_path.absolute()}: {img_path.exists()}")
        
        if img_path.exists():
            try:
                from PIL import Image, ImageTk
                img = Image.open(img_path)
                img.thumbnail((880, self.root.winfo_height()-20), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                label = tk.Label(self.root, image=photo)
                label.image = photo
                label.grid(row=0, column=0, pady=10, padx=20, sticky='nsew')                 
            except Exception as e:
                print(f"❌ PIL: {e}")
                self.show_placeholder(img_name)
        else:
            self.show_placeholder(img_name)
    
    def show_placeholder(self, img_name):
        canvas = tk.Canvas(self.root, bg='#e0f0ff', highlightthickness=0)
        canvas.grid(row=0, column=0, pady=10, padx=20, sticky='nsew')
        
        canvas.create_text(300, 100, text=f"images/{img_name}", 
                          font=('Arial', 16, 'bold'), fill='#0066cc', width=500)
        canvas.config(width=500, height=200)
    
    def launch_trainer(self):
        if Path("trainer.py").exists():
            subprocess.Popen([sys.executable, "trainer.py"])
            self.root.quit()
        else:
            messagebox.showerror("Ошибка", "trainer.py не найден!")
 
    def on_closing(self):
        print("👋 Закрытие")
        self.root.quit()

if __name__ == "__main__":    
    # Автоустановка PIL
    try:
        from PIL import Image
        print("PIL установлен")
    except ImportError:
        print("Устанавливаю Pillow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    
    try:
        app = SlidesApp()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        input("Нажми Enter...")