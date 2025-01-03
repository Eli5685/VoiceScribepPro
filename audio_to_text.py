import os
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
import threading
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import customtkinter as ctk
from tkinter import filedialog
from datetime import datetime
import time
from faster_whisper import WhisperModel
from tqdm import tqdm
import json
import wave
import mutagen
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
import torch

class AudioTranscriber(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Создание структуры папок в документах
        self.user_data_dir = os.path.join(os.path.expanduser("~"), "Documents", "VoiceScribePro")
        self.models_dir = os.path.join(self.user_data_dir, "models")
        self.recordings_dir = os.path.join(self.user_data_dir, "recordings")
        self.settings_dir = os.path.join(self.user_data_dir, "settings")
        
        # Создаем папки если их нет
        for directory in [self.user_data_dir, self.models_dir, self.recordings_dir, self.settings_dir]:
            os.makedirs(directory, exist_ok=True)

        # Настройка основного окна
        self.title("VoiceScribe Pro")
        self.geometry("1000x800")
        self.minsize(800, 600)
        
        # Установка темы и цветов
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Определение цветовой схемы
        self.colors = {
            "primary": "#1E1E1E",
            "primary_hover": "#2A2A2A",
            "secondary": "#2D2D2D",
            "secondary_hover": "#353535",
            "accent": "#6A5ACD",
            "accent_hover": "#7B6DDE",
            "bg_dark": "#171717",
            "header_bg": "#1A1A1A",  # Заменили rgba на hex
            "text_primary": "#FFFFFF",
            "text_secondary": "#B0B0B0",
            "border": "#3A3A3A",
            "progress": "#6A5ACD",
            "error": "#FF5252",
            "success": "#4CAF50"
        }

        # Загрузка настроек
        self.settings = self.load_settings()
        
        # Проверка и установка PyTorch с CUDA
        if self.settings.get("show_pytorch_dialog", True):
            self.check_pytorch_cuda()
        
        # Настройка сетки
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Параметры записи
        self.fs = 44100
        self.recording = False
        self.audio_data = []
        self.record_time = 0
        self.timer_running = False
        self.is_recorded_file = False
        
        # Создание элементов интерфейса
        self.create_widgets()
        
        # Инициализация модели
        self.model = None
        self.is_transcribing = False
        
        # Модели Whisper с информацией
        self.available_models = {
            "Tiny (быстрая, менее точная)": {
                "name": "tiny",
                "params": "78M",
                "vram": "~1 GB",
                "speed": "~10x быстрее large",
                "installed": False
            },
            "Base (средняя)": {
                "name": "base",
                "params": "148M",
                "vram": "~1 GB",
                "speed": "~7x быстрее large",
                "installed": False
            },
            "Small (точная)": {
                "name": "small",
                "params": "488M",
                "vram": "~2 GB",
                "speed": "~4x быстрее large",
                "installed": False
            },
            "Medium (очень точная)": {
                "name": "medium",
                "params": "1538M",
                "vram": "~5 GB",
                "speed": "~2x быстрее large",
                "installed": False
            },
            "Large (самая точная)": {
                "name": "large-v3",
                "params": "3158M",
                "vram": "~10 GB",
                "speed": "базовая скорость",
                "installed": False
            },
            "Turbo (быстрая и точная)": {
                "name": "turbo",
                "params": "1618M",
                "vram": "~6 GB",
                "speed": "~8x быстрее large",
                "installed": False
            }
        }

    def load_settings(self):
        try:
            settings_file = os.path.join(self.settings_dir, "settings.json")
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            default_settings = {
                "model": "base",
                "compute_type": "int8",
                "device": "cpu",
                "use_gpu": False,
                "show_pytorch_dialog": True,
                "save_path": self.user_data_dir,  # Путь сохранения по умолчанию
                "recording": {
                    "sample_rate": 44100,
                    "channels": 1,
                    "bit_depth": 16
                }
            }
            self.save_settings(default_settings)
            return default_settings

    def save_settings(self, settings):
        settings_file = os.path.join(self.settings_dir, "settings.json")
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

    def create_widgets(self):
        # Верхняя панель с полупрозрачным фоном
        self.top_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color="#2D2D2D",  # Полупрозрачный серый
            height=80
        )
        self.top_frame.grid(row=0, column=0, sticky="ew")
        self.top_frame.grid_columnconfigure((0, 1), weight=1)
        self.top_frame.grid_propagate(False)

        # Заголовок с полупрозрачным фоном
        header_frame = ctk.CTkFrame(
            self.top_frame,
            fg_color="transparent"
        )
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)

        self.header = ctk.CTkLabel(
            header_frame,
            text="VoiceScribe Pro",
            font=ctk.CTkFont(family="Roboto", size=32, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.header.grid(row=0, column=0, sticky="w", pady=15, padx=30)

        # Кнопка настроек
        self.settings_button = ctk.CTkButton(
            header_frame,
            text="НАСТРОЙКИ",
            command=self.show_settings,
            width=140,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            corner_radius=20
        )
        self.settings_button.grid(row=0, column=1, sticky="e", pady=15, padx=30)

        # Основной контейнер со скроллом
        self.main_scroll = ctk.CTkScrollableFrame(
            self,
            corner_radius=0,
            fg_color="transparent"
        )
        self.main_scroll.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.main_scroll.grid_columnconfigure(0, weight=1)

        # Секция источника аудио
        self.create_source_section()
        
        # Секция записи
        self.create_recording_section()
        
        # Секция результатов
        self.create_results_section()

    def create_source_section(self):
        # Фрейм выбора источника
        self.source_frame = ctk.CTkFrame(
            self.main_scroll,
            corner_radius=20,
            fg_color=self.colors["primary"],
            border_color=self.colors["border"],
            border_width=2
        )
        self.source_frame.grid(row=0, column=0, sticky="ew", padx=30, pady=20)
        self.source_frame.grid_columnconfigure((0, 1), weight=1)

        # Заголовок секции с иконкой
        header_frame = ctk.CTkFrame(
            self.source_frame,
            fg_color="transparent"
        )
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 15))
        
        self.source_label = ctk.CTkLabel(
            header_frame,
            text="🎵 ИСТОЧНИК АУДИО",
            font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.source_label.pack(side="left")

        # Кнопки выбора источника
        self.select_button = ctk.CTkButton(
            self.source_frame,
            text="ВЫБРАТЬ ФАЙЛ",
            command=self.select_file,
            height=50,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            corner_radius=25,
            image=None  # Здесь можно добавить иконку
        )
        self.select_button.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.record_button = ctk.CTkButton(
            self.source_frame,
            text="НАЧАТЬ ЗАПИСЬ",
            command=self.toggle_recording,
            height=50,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors["secondary"],
            hover_color=self.colors["secondary_hover"],
            corner_radius=25,
            image=None  # Здесь можно добавить иконку
        )
        self.record_button.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")

    def create_recording_section(self):
        # Фрейм записи
        self.record_frame = ctk.CTkFrame(
            self.main_scroll,
            corner_radius=20,
            fg_color=self.colors["primary"],
            border_color=self.colors["border"],
            border_width=2
        )
        self.record_frame.grid(row=1, column=0, sticky="ew", padx=30, pady=20)
        self.record_frame.grid_columnconfigure(0, weight=1)

        # Заголовок секции с иконкой
        header_frame = ctk.CTkFrame(
            self.record_frame,
            fg_color="transparent"
        )
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 0))
        
        self.record_label = ctk.CTkLabel(
            header_frame,
            text="🎙️ ЗАПИСЬ",
            font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.record_label.pack(side="left")

        # Таймер записи
        self.timer_label = ctk.CTkLabel(
            self.record_frame,
            text="00:00",
            font=ctk.CTkFont(family="Roboto", size=72, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.timer_label.grid(row=1, column=0, pady=(20, 20))

        # Индикатор уровня звука
        self.level_frame = ctk.CTkFrame(
            self.record_frame,
            fg_color="transparent"
        )
        self.level_frame.grid(row=2, column=0, sticky="ew", padx=40, pady=(0, 30))
        self.level_frame.grid_columnconfigure(0, weight=1)

        self.level_bar = ctk.CTkProgressBar(
            self.level_frame,
            height=20,
            corner_radius=10,
            progress_color=self.colors["accent"],
            border_color=self.colors["border"],
            border_width=1
        )
        self.level_bar.grid(row=0, column=0, sticky="ew")
        self.level_bar.set(0)

        # Метка статуса записи
        self.record_status = ctk.CTkLabel(
            self.record_frame,
            text="Готов к записи",
            font=ctk.CTkFont(size=13),
            text_color=self.colors["text_secondary"]
        )
        self.record_status.grid(row=3, column=0, pady=(0, 20))

    def create_results_section(self):
        # Фрейм результатов
        self.results_frame = ctk.CTkFrame(
            self.main_scroll,
            corner_radius=20,
            fg_color=self.colors["primary"],
            border_color=self.colors["border"],
            border_width=2
        )
        self.results_frame.grid(row=2, column=0, sticky="ew", padx=30, pady=20)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Заголовок секции с иконкой
        header_frame = ctk.CTkFrame(
            self.results_frame,
            fg_color="transparent"
        )
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        
        self.results_label = ctk.CTkLabel(
            header_frame,
            text="📝 ИНФОРМАЦИЯ",
            font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.results_label.pack(side="left")

        # Скроллируемый фрейм для информации
        self.info_scroll = ctk.CTkScrollableFrame(
            self.results_frame,
            height=150,
            fg_color=self.colors["secondary"],
            border_color=self.colors["border"],
            border_width=1,
            corner_radius=15
        )
        self.info_scroll.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.info_scroll.grid_columnconfigure(0, weight=1)

        # Информация о файле
        self.file_label = ctk.CTkLabel(
            self.info_scroll,
            text="Файл не выбран",
            wraplength=900,
            font=ctk.CTkFont(size=13),
            justify="left",
            text_color=self.colors["text_primary"]
        )
        self.file_label.grid(row=0, column=0, sticky="w", pady=10)

        # Прогресс конвертации
        self.progress_frame = ctk.CTkFrame(
            self.results_frame,
            fg_color="transparent"
        )
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=25,
            corner_radius=12,
            progress_color=self.colors["accent"],
            border_color=self.colors["border"],
            border_width=1
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.set(0)

        # Статус
        self.status_label = ctk.CTkLabel(
            self.results_frame,
            text="Готов к работе",
            font=ctk.CTkFont(size=13),
            text_color=self.colors["text_secondary"]
        )
        self.status_label.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 10))

        # Кнопка конвертации
        self.start_button = ctk.CTkButton(
            self.results_frame,
            text="НАЧАТЬ КОНВЕРТАЦИЮ",
            command=self.start_transcription,
            state="disabled",
            height=55,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            corner_radius=27
        )
        self.start_button.grid(row=4, column=0, pady=20, padx=20, sticky="ew")

    def update_timer(self):
        if self.timer_running:
            self.record_time += 1
            minutes = self.record_time // 60
            seconds = self.record_time % 60
            self.timer_label.configure(text=f"{minutes:02d}:{seconds:02d}")
            self.after(1000, self.update_timer)

    def update_level_indicator(self, indata):
        if len(indata) > 0:
            level = np.abs(indata).mean()
            normalized_level = min(1.0, level * 10)
            self.level_bar.set(normalized_level)
            self.update()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.record_time = 0
        self.timer_running = True
        self.update_timer()
        
        self.record_button.configure(
            text="ОСТАНОВИТЬ ЗАПИСЬ",
            fg_color=self.colors["error"],
            hover_color="#FF6B6B"
        )
        self.select_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.record_status.configure(
            text="Идёт запись...",
            text_color=self.colors["error"]
        )
        
        def audio_callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
                self.after(10, lambda: self.update_level_indicator(indata))
        
        # Используем настройки записи из settings
        self.stream = sd.InputStream(
            channels=self.settings["recording"]["channels"],
            samplerate=self.settings["recording"]["sample_rate"],
            callback=audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.timer_running = False
        self.stream.stop()
        self.stream.close()
        self.level_bar.set(0)
        
        self.record_button.configure(
            text="НАЧАТЬ ЗАПИСЬ",
            fg_color=self.colors["secondary"],
            hover_color=self.colors["secondary_hover"]
        )
        
        if len(self.audio_data) > 0:
            audio = np.concatenate(self.audio_data, axis=0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Используем os.path.join для корректного формирования пути
            filename = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # Создаем директорию если её нет
            
            # Сохраняем с настройками из settings
            write(filename, self.settings["recording"]["sample_rate"], audio)
            
            self.selected_file = filename
            self.is_recorded_file = True
            
            # Получение информации о файле
            info = self.get_audio_info(filename)
            info_text = f"📝 Записанный файл: {os.path.basename(filename)}\n\n"
            
            if "error" not in info:
                info_text += f"⏱️ Длительность: {info.get('duration', 'Н/Д')}\n"
                info_text += f"📦 Размер: {info.get('size', 'Н/Д')}\n"
                info_text += f"🎵 Частота дискретизации: {info.get('sample_rate', 'Н/Д')}\n"
                if "bit_depth" in info:
                    info_text += f"🎚️ Глубина звука: {info['bit_depth']}\n"
                if "channels" in info:
                    info_text += f"🔊 Каналы: {info['channels']}\n"
            
            self.file_label.configure(
                text=info_text,
                text_color=self.colors["text_primary"]
            )
            self.start_button.configure(state="normal")
            self.record_status.configure(
                text=f"✅ Запись сохранена (длительность: {self.record_time} сек)",
                text_color=self.colors["success"]
            )
        
        self.select_button.configure(state="normal")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Аудиофайлы", "*.mp3 *.wav *.m4a *.ogg")
            ]
        )
        if file_path:
            # Очищаем информацию о предыдущей записи
            self.record_status.configure(
                text="Готов к записи",
                text_color=self.colors["text_secondary"]
            )
            self.timer_label.configure(text="00:00")
            self.level_bar.set(0)
            
            self.selected_file = file_path
            self.is_recorded_file = False
            
            # Получение информации о файле
            info = self.get_audio_info(file_path)
            info_text = f"📁 Выбранный файл: {os.path.basename(file_path)}\n\n"
            
            if "error" not in info:
                info_text += f"⏱️ Длительность: {info.get('duration', 'Н/Д')}\n"
                info_text += f"📦 Размер: {info.get('size', 'Н/Д')}\n"
                if "bitrate" in info:
                    info_text += f"📊 Битрейт: {info['bitrate']}\n"
                info_text += f"🎵 Частота: {info.get('sample_rate', 'Н/Д')}\n"
                if "bit_depth" in info:
                    info_text += f"🎚️ Глубина: {info['bit_depth']}\n"
                if "channels" in info:
                    info_text += f"🔊 Каналы: {info['channels']}\n"
            
            self.file_label.configure(
                text=info_text,
                text_color=self.colors["text_primary"]
            )
            self.start_button.configure(state="normal")
            
            # Исправляем отображение модели
            model_name = self.settings.get("model", "base")
            model_display = next((k for k, v in self.available_models.items() 
                                if v["name"] == model_name), "Модель не выбрана")
            
            self.status_label.configure(
                text=f"🤖 Модель распознавания: {model_display}",
                text_color=self.colors["text_primary"]
            )

    def update_progress(self, progress):
        self.progress_bar.set(progress)
        self.update()

    def disable_interface(self):
        """Блокировка всех элементов управления"""
        self.select_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        
    def enable_interface(self):
        """Разблокировка всех элементов управления"""
        self.select_button.configure(state="normal")
        self.record_button.configure(state="normal")
        self.start_button.configure(state="normal")

    def start_transcription(self):
        if self.is_transcribing or not hasattr(self, 'selected_file'):
            return
            
        self.is_transcribing = True
        self.disable_interface()
        
        def transcribe():
            try:
                start_time = time.time()
                
                # Начало процесса
                self.progress_bar.set(0)
                self.status_label.configure(
                    text="⚙️ Загрузка модели...",
                    text_color=self.colors["text_primary"]
                )
                self.update()

                # Проверка CUDA для настроек
                cuda_available = False
                gpu_name = "Неизвестно"
                try:
                    import torch
                    if torch.cuda.is_available():
                        cuda_available = True
                        gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    pass

                device_var = ctk.StringVar(value="gpu" if self.settings.get("use_gpu", False) else "cpu")
                
                if self.settings.get("use_gpu", False) and not cuda_available:
                    self.settings["use_gpu"] = False
                    self.settings["device"] = "cpu"
                    self.save_settings(self.settings)
                    self.status_label.configure(
                        text="⚠️ GPU недоступен. Используется CPU.",
                        text_color=self.colors["error"]
                    )
                    self.update()
                    time.sleep(2)
                
                # Загрузка модели
                if not self.model:
                    device = "cuda" if self.settings.get("use_gpu", False) and cuda_available else "cpu"
                    model_name = self.settings["model"]
                    
                    # Проверяем наличие модели
                    if not self.check_model_installed(model_name):
                        self.status_label.configure(
                            text="⬇️ Скачивание модели...",
                            text_color=self.colors["text_primary"]
                        )
                        self.progress_bar.set(0.1)
                        self.update()
                    
                    try:
                        self.model = WhisperModel(
                            model_name,
                            device=device,
                            compute_type="float16" if device == "cuda" else "int8",
                            download_root=self.models_dir,
                            local_files_only=False
                        )
                        self.progress_bar.set(0.3)
                        self.update()
                    except Exception as e:
                        if "not found" in str(e).lower():
                            # Модель не найдена, пробуем скачать
                            self.status_label.configure(
                                text="⬇️ Скачивание модели...",
                                text_color=self.colors["text_primary"]
                            )
                            self.progress_bar.set(0.1)
                            self.update()
                            
                            self.model = WhisperModel(
                                model_name,
                                device=device,
                                compute_type="float16" if device == "cuda" else "int8",
                                download_root=self.models_dir,
                                local_files_only=False
                            )
                            self.progress_bar.set(0.3)
                            self.update()
                        else:
                            raise e
                
                # Сброс времени распознавания
                self.transcription_start_time = time.time()
                
                self.status_label.configure(
                    text="🎯 Идёт распознавание...",
                    text_color=self.colors["text_primary"]
                )
                self.update()

                # Распознавание
                segments, info = self.model.transcribe(
                    self.selected_file,
                    beam_size=5
                )
                
                # Сохранение результата
                self.progress_bar.set(0.9)
                self.update()
                
                base_name = os.path.splitext(os.path.basename(self.selected_file))[0]
                output_file = os.path.join(self.settings["save_path"], f"{base_name}_trsc.txt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    for segment in segments:
                        f.write(segment.text + "\n")

                # Удаление записанного файла, если это была запись
                if self.is_recorded_file and os.path.exists(self.selected_file):
                    try:
                        os.remove(self.selected_file)
                    except Exception:
                        pass  # Игнорируем ошибки при удалении

                # Вычисление времени обработки
                end_time = time.time()
                process_time = end_time - start_time
                hours = int(process_time // 3600)
                minutes = int((process_time % 3600) // 60)
                seconds = int(process_time % 60)
                
                time_str = ""
                if hours > 0:
                    time_str += f"{hours} ч "
                if minutes > 0:
                    time_str += f"{minutes} мин "
                time_str += f"{seconds} сек"

                # Завершение
                self.progress_bar.set(1.0)
                self.status_label.configure(
                    text=f"✅ Готово за {time_str}! Результат сохранён в {output_file}",
                    text_color=self.colors["success"]
                )

            except Exception as e:
                self.status_label.configure(
                    text=f"❌ Ошибка: {str(e)}",
                    text_color=self.colors["error"]
                )
                self.progress_bar.set(0)
                
            finally:
                self.is_transcribing = False
                self.enable_interface()

        # Запускаем в отдельном потоке
        thread = threading.Thread(target=transcribe)
        thread.start()

    def get_audio_info(self, file_path):
        """Получение информации об аудиофайле"""
        try:
            info = {}
            ext = os.path.splitext(file_path)[1].lower()
            
            # Получение размера файла
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            info["size"] = f"{size_mb:.2f} МБ"
            
            # Получение длительности и других параметров
            if ext == ".mp3":
                audio = MP3(file_path)
                info["duration"] = f"{int(audio.info.length // 60)}:{int(audio.info.length % 60):02d}"
                info["bitrate"] = f"{audio.info.bitrate // 1000} кбит/с"
                info["sample_rate"] = f"{audio.info.sample_rate // 1000} кГц"
            elif ext == ".wav":
                with wave.open(file_path, "rb") as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration = frames / float(rate)
                    info["duration"] = f"{int(duration // 60)}:{int(duration % 60):02d}"
                    info["channels"] = wav.getnchannels()
                    info["sample_rate"] = f"{rate // 1000} кГц"
                    info["bit_depth"] = f"{wav.getsampwidth() * 8} бит"
            
            return info
        except:
            return {"error": "Не удалось получить информацию о файле"}

    def show_settings(self):
        """Показать окно настроек"""
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("Настройки")
        settings_window.geometry("600x800")
        settings_window.grab_set()
        settings_window.resizable(False, False)

        # Контейнер со скроллом
        settings_scroll = ctk.CTkScrollableFrame(
            settings_window,
            fg_color=self.colors["primary"]
        )
        settings_scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Заголовок
        header = ctk.CTkLabel(
            settings_scroll,
            text="⚙️ Настройки",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        header.pack(pady=(0, 20))
        
        # 1. Выбор модели
        model_frame = ctk.CTkFrame(
            settings_scroll,
            fg_color=self.colors["secondary"],
            border_color=self.colors["border"],
            border_width=2,
            corner_radius=15
        )
        model_frame.pack(fill="x", pady=10)
        
        model_header = ctk.CTkFrame(
            model_frame,
            fg_color="transparent"
        )
        model_header.pack(fill="x", padx=15, pady=10)

        model_label = ctk.CTkLabel(
            model_header,
            text="🤖 Модель распознавания",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        model_label.pack(side="left")

        def show_models_info():
            self.show_models_info_window()

        info_button = ctk.CTkButton(
            model_header,
            text="ℹ️ ПОДРОБНЕЕ О МОДЕЛЯХ",
            command=show_models_info,
            width=180,
            height=25,
            font=ctk.CTkFont(size=12),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"]
        )
        info_button.pack(side="right")

        # Проверяем установленные модели
        for model_info in self.available_models.values():
            model_info["installed"] = self.check_model_installed(model_info["name"])
        
        selected_model = ctk.StringVar(value=[k for k, v in self.available_models.items() 
                                            if v["name"] == self.settings["model"]][0])
        
        for name, info in self.available_models.items():
            model_row = ctk.CTkFrame(
                model_frame,
                fg_color="transparent"
            )
            model_row.pack(fill="x", padx=15, pady=5)

            radio = ctk.CTkRadioButton(
                model_row,
                text=name,
                variable=selected_model,
                value=name,
                font=ctk.CTkFont(size=14),
                fg_color=self.colors["accent"],
                hover_color=self.colors["accent_hover"]
            )
            radio.pack(side="left")

            status = "✅" if info["installed"] else "⚠️"
            status_label = ctk.CTkLabel(
                model_row,
                text=f"{status} {'Установлена' if info['installed'] else 'Не установлена'}",
                font=ctk.CTkFont(size=12),
                text_color=self.colors["success"] if info["installed"] else self.colors["error"]
            )
            status_label.pack(side="right")

        # 2. Выбор устройства
        device_frame = ctk.CTkFrame(
            settings_scroll,
            fg_color=self.colors["secondary"],
            border_color=self.colors["border"],
            border_width=2,
            corner_radius=15
        )
        device_frame.pack(fill="x", pady=10)
        
        device_label = ctk.CTkLabel(
            device_frame,
            text="💻 Устройство обработки",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        device_label.pack(anchor="w", padx=15, pady=10)

        # Проверка CUDA для настроек
        cuda_available = False
        gpu_name = "Неизвестно"
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        device_var = ctk.StringVar(value="gpu" if self.settings.get("use_gpu", False) else "cpu")
        
        cpu_radio = ctk.CTkRadioButton(
            device_frame,
            text="💪 CPU (процессор)",
            variable=device_var,
            value="cpu",
            font=ctk.CTkFont(size=14),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"]
        )
        cpu_radio.pack(anchor="w", padx=25, pady=5)

        gpu_radio = ctk.CTkRadioButton(
            device_frame,
            text=f"🚀 GPU ({gpu_name if cuda_available else 'Недоступно'})",
            variable=device_var,
            value="gpu",
            font=ctk.CTkFont(size=14),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            state="normal" if cuda_available else "disabled"
        )
        gpu_radio.pack(anchor="w", padx=25, pady=5)
        
        # Кнопка настройки PyTorch/CUDA
        def configure_pytorch():
            self.check_pytorch_cuda()

        pytorch_button = ctk.CTkButton(
                device_frame,
            text="⚙️ НАСТРОИТЬ PYTORCH/CUDA",
            command=configure_pytorch,
            width=200,
            height=30,
                font=ctk.CTkFont(size=12),
            fg_color=self.colors["secondary_hover"],
            hover_color=self.colors["accent"]
        )
        pytorch_button.pack(anchor="w", padx=25, pady=(5, 15))

        # 3. Путь сохранения
        save_path_frame = ctk.CTkFrame(
            settings_scroll,
            fg_color=self.colors["secondary"],
            border_color=self.colors["border"],
            border_width=2,
            corner_radius=15
        )
        save_path_frame.pack(fill="x", pady=10)

        save_path_label = ctk.CTkLabel(
            save_path_frame,
            text="💾 Путь сохранения",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        save_path_label.pack(anchor="w", padx=15, pady=10)

        path_frame = ctk.CTkFrame(
            save_path_frame,
            fg_color="transparent"
        )
        path_frame.pack(fill="x", padx=15, pady=(0, 15))

        current_path = self.settings.get("save_path", self.user_data_dir)
        path_label = ctk.CTkLabel(
            path_frame,
            text=current_path,
                font=ctk.CTkFont(size=12),
                text_color=self.colors["text_secondary"]
            )
        path_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        def choose_path():
            new_path = filedialog.askdirectory(initialdir=current_path)
            if new_path:
                path_label.configure(text=new_path)
                self.settings["save_path"] = new_path
                self.save_settings(self.settings)

        change_path_button = ctk.CTkButton(
            path_frame,
            text="ИЗМЕНИТЬ",
            command=choose_path,
            width=100,
            height=30,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"]
        )
        change_path_button.pack(side="right")

        # Кнопка сохранения
        def save_settings():
            self.settings["model"] = self.available_models[selected_model.get()]["name"]
            self.settings["use_gpu"] = device_var.get() == "gpu"
            self.settings["device"] = "cuda" if self.settings["use_gpu"] else "cpu"
            self.save_settings(self.settings)
            self.model = None  # Сброс текущей модели
            settings_window.destroy()
        
        save_button = ctk.CTkButton(
            settings_scroll,
            text="СОХРАНИТЬ",
            command=save_settings,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"]
        )
        save_button.pack(pady=20)

    def show_models_info_window(self):
        """Показать окно с информацией о моделях"""
        info_window = ctk.CTkToplevel(self)
        info_window.title("Информация о моделях")
        info_window.geometry("1000x600")
        info_window.grab_set()

        # Контейнер со скроллом
        info_scroll = ctk.CTkScrollableFrame(
            info_window,
            fg_color=self.colors["primary"]
        )
        info_scroll.pack(fill="both", expand=True, padx=20, pady=20)

        # Заголовок
        header = ctk.CTkLabel(
            info_scroll,
            text="🤖 Модели Whisper",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        header.pack(pady=(0, 20))

        # Таблица моделей
        table_frame = ctk.CTkFrame(
            info_scroll,
            fg_color=self.colors["secondary"],
            border_color=self.colors["border"],
            border_width=2,
            corner_radius=15
        )
        table_frame.pack(fill="x", pady=5)
        table_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # Заголовки таблицы
        headers = ["Модель", "Размер", "VRAM", "Скорость", "Статус"]
        header_frame = ctk.CTkFrame(
            table_frame,
            fg_color=self.colors["accent"],
            corner_radius=15
        )
        header_frame.pack(fill="x", padx=2, pady=2)
        header_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # Фиксированная ширина для каждого столбца
        column_widths = [300, 100, 100, 150, 120]  # Ширина в пикселях
        
        for i, header_text in enumerate(headers):
            header_label = ctk.CTkLabel(
                header_frame,
                text=header_text,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.colors["text_primary"],
                width=column_widths[i]
            )
            header_label.grid(row=0, column=i, padx=5, pady=10)

        # Данные таблицы
        content_frame = ctk.CTkFrame(
            table_frame,
            fg_color="transparent"
        )
        content_frame.pack(fill="x", padx=2, pady=1)
        content_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        for i, (name, info) in enumerate(self.available_models.items()):
            row_frame = ctk.CTkFrame(
                content_frame,
                fg_color="transparent",
                height=40
            )
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_propagate(False)
            row_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

            # Название модели
            model_label = ctk.CTkLabel(
                row_frame,
                text=name,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=self.colors["text_primary"],
                width=column_widths[0]
            )
            model_label.grid(row=0, column=0, padx=5, pady=5)

            # Размер модели
            size_label = ctk.CTkLabel(
                row_frame,
                text=info["params"],
                font=ctk.CTkFont(size=13),
                text_color=self.colors["text_secondary"],
                width=column_widths[1]
            )
            size_label.grid(row=0, column=1, padx=5, pady=5)

            # VRAM
            vram_label = ctk.CTkLabel(
                row_frame,
                text=info["vram"],
                font=ctk.CTkFont(size=13),
                text_color=self.colors["text_secondary"],
                width=column_widths[2]
            )
            vram_label.grid(row=0, column=2, padx=5, pady=5)

            # Скорость
            speed_label = ctk.CTkLabel(
                row_frame,
                text=info["speed"],
                font=ctk.CTkFont(size=13),
                text_color=self.colors["text_secondary"],
                width=column_widths[3]
            )
            speed_label.grid(row=0, column=3, padx=5, pady=5)

            # Статус установки
            is_installed = self.check_model_installed(info["name"])
            status_label = ctk.CTkLabel(
                row_frame,
                text="✅ Установлена" if is_installed else "⚠️ Не установлена",
                font=ctk.CTkFont(size=13),
                text_color=self.colors["success"] if is_installed else self.colors["error"],
                width=column_widths[4]
            )
            status_label.grid(row=0, column=4, padx=5, pady=5)

            # Разделительная линия
            if i < len(self.available_models) - 1:
                separator = ctk.CTkFrame(
                    content_frame,
                    height=1,
                    fg_color=self.colors["border"]
                )
                separator.pack(fill="x", pady=1)

        # Кнопка закрытия
        close_button = ctk.CTkButton(
            info_scroll,
            text="ЗАКРЫТЬ",
            command=info_window.destroy,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors["secondary"],
            hover_color=self.colors["secondary_hover"]
        )
        close_button.pack(pady=20)

    def check_pytorch_cuda(self):
        """Проверка и установка PyTorch с CUDA"""
        try:
            import torch
            pytorch_installed = True
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pytorch_installed = False
            cuda_available = False
            
        # Проверяем наличие NVIDIA GPU через subprocess
        import subprocess
        has_gpu = False
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi'])
            has_gpu = True
        except:
            pass

        if not pytorch_installed:
            # Если PyTorch не установлен
            self.show_pytorch_install_dialog(
                "PyTorch не установлен",
                "PyTorch не установлен.\nВыберите версию для установки:"
            )
        elif has_gpu and not cuda_available:
            # Если есть GPU, но CUDA недоступна
            self.show_pytorch_install_dialog(
                "PyTorch установлен без поддержки CUDA",
                "У вас установлен PyTorch, но без поддержки CUDA.\n" +
                "Вы можете продолжить работу с текущей версией\n" +
                "или переустановить с поддержкой CUDA для ускорения работы на GPU."
            )
        elif not has_gpu and not cuda_available:
            # Если нет GPU, показываем информацию
            self.show_pytorch_install_dialog(
                "PyTorch установлен для CPU",
                "У вас установлен PyTorch для работы на CPU.\n" +
                "Видеокарта NVIDIA не обнаружена.\n" +
                "Можно продолжить работу с текущей версией.",
                show_install_buttons=False
            )
        elif cuda_available:
            # Если CUDA доступна, показываем информацию
            self.show_pytorch_install_dialog(
                "PyTorch установлен с CUDA",
                f"У вас установлен PyTorch с поддержкой CUDA.\n" +
                f"Доступная видеокарта: {torch.cuda.get_device_name(0)}\n" +
                "Можно продолжить работу с текущей версией.",
                show_install_buttons=False
            )

    def show_pytorch_install_dialog(self, title, message, show_install_buttons=True):
        """Показать диалог установки PyTorch"""
        try:
            choice_window = ctk.CTkToplevel(self)
            choice_window.title("Установка PyTorch")
            choice_window.geometry("500x450")
            choice_window.grab_set()
            
            # Центрируем окно на экране
            window_width = 500
            window_height = 450
            screen_width = choice_window.winfo_screenwidth()
            screen_height = choice_window.winfo_screenheight()
            
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 3
            
            choice_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
            # Заголовок
            header = ctk.CTkLabel(
                choice_window,
                text=title,
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color=self.colors["text_primary"]
            )
            header.pack(pady=20)
            
            # Описание
            description = ctk.CTkLabel(
                choice_window,
                text=message,
                font=ctk.CTkFont(size=14),
                wraplength=400
            )
            description.pack(pady=10)
            
            if show_install_buttons:
                def install_with_cuda():
                    choice_window.destroy()
                    self._perform_pytorch_install(with_cuda=True)
                    
                def install_without_cuda():
                    choice_window.destroy()
                    self._perform_pytorch_install(with_cuda=False)
                
                # Кнопки выбора
                cuda_button = ctk.CTkButton(
                    choice_window,
                    text="ПЕРЕУСТАНОВИТЬ С CUDA (GPU)",
                    command=install_with_cuda,
                    height=45,
                    font=ctk.CTkFont(size=15, weight="bold"),
                    fg_color=self.colors["accent"],
                    hover_color=self.colors["accent_hover"]
                )
                cuda_button.pack(pady=10, padx=20, fill="x")
                
                cpu_button = ctk.CTkButton(
                    choice_window,
                    text="ПЕРЕУСТАНОВИТЬ БЕЗ CUDA (CPU)",
                    command=install_without_cuda,
                    height=45,
                    font=ctk.CTkFont(size=15, weight="bold"),
                    fg_color=self.colors["secondary"],
                    hover_color=self.colors["secondary_hover"]
                )
                cpu_button.pack(pady=10, padx=20, fill="x")

                # Кнопка продолжения с текущей версией
                continue_button = ctk.CTkButton(
                    choice_window,
                    text="ПРОДОЛЖИТЬ С ТЕКУЩЕЙ ВЕРСИЕЙ",
                    command=choice_window.destroy,
                    height=45,
                    font=ctk.CTkFont(size=15, weight="bold"),
                    fg_color=self.colors["success"],
                    hover_color="#5BBF60"
                )
                continue_button.pack(pady=10, padx=20, fill="x")
            else:
                # Кнопка продолжения
                continue_button = ctk.CTkButton(
                    choice_window,
                    text="ПРОДОЛЖИТЬ",
                    command=choice_window.destroy,
                    height=45,
                    font=ctk.CTkFont(size=15, weight="bold"),
                    fg_color=self.colors["success"],
                    hover_color="#5BBF60"
                )
                continue_button.pack(pady=10, padx=20, fill="x")
            
            # Чекбокс "Больше не показывать"
            show_var = ctk.BooleanVar(value=self.settings.get("show_pytorch_dialog", True))
            
            def on_checkbox_change():
                self.settings["show_pytorch_dialog"] = show_var.get()
                self.save_settings(self.settings)
            
            checkbox = ctk.CTkCheckBox(
                choice_window,
                text="Показывать это окно при запуске",
                variable=show_var,
                command=on_checkbox_change,
                font=ctk.CTkFont(size=13),
                fg_color=self.colors["accent"],
                hover_color=self.colors["accent_hover"]
            )
            checkbox.pack(pady=20)
            
        except Exception as e:
            print(f"Ошибка создания окна выбора: {str(e)}")

    def _perform_pytorch_install(self, with_cuda=True):
        """Выполнение установки PyTorch"""
        try:
            import subprocess
            import sys
            
            # Создаем окно с информацией
            info_window = ctk.CTkToplevel(self)
            info_window.title("Установка PyTorch")
            info_window.geometry("400x200")
            info_window.grab_set()
            
            info_label = ctk.CTkLabel(
                info_window,
                text=f"Установка PyTorch {'с поддержкой CUDA' if with_cuda else 'без CUDA'}...\n" +
                     "Это может занять несколько минут.",
                font=ctk.CTkFont(size=14),
                wraplength=350
            )
            info_label.pack(pady=20)
            
            progress = ctk.CTkProgressBar(
                info_window,
                width=300,
                height=20,
                corner_radius=10,
                mode="indeterminate"
            )
            progress.pack(pady=20)
            progress.start()
            
            def install():
                try:
                    # Устанавливаем в пользовательскую директорию
                    if with_cuda:
                        # Устанавливаем PyTorch с CUDA
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", "--user", "--upgrade", "--no-cache-dir",
                            "torch", "torchvision", "torchaudio", 
                            "--index-url", "https://download.pytorch.org/whl/cu118"
                        ])
                    else:
                        # Устанавливаем PyTorch без CUDA
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", "--user", "--upgrade", "--no-cache-dir",
                            "torch", "torchvision", "torchaudio"
                        ])
                    
                    info_label.configure(
                        text=f"✅ PyTorch успешно установлен {'с CUDA' if with_cuda else 'без CUDA'}!\n" +
                             "Перезапустите приложение."
                    )

                    # Добавляем кнопку для закрытия окна
                    close_button = ctk.CTkButton(
                        info_window,
                        text="ЗАКРЫТЬ",
                        command=info_window.destroy,
                        height=35,
                        font=ctk.CTkFont(size=13, weight="bold"),
                        fg_color=self.colors["accent"],
                        hover_color=self.colors["accent_hover"]
                    )
                    close_button.pack(pady=10)

                except Exception as e:
                    error_msg = str(e)
                    if "PermissionError" in error_msg:
                        error_msg = "Ошибка доступа. Попробуйте запустить приложение от имени администратора."
                    
                    info_label.configure(
                        text=f"❌ Ошибка установки: {error_msg}\n" +
                             "Попробуйте установить PyTorch вручную или запустите приложение от имени администратора."
                    )
                    
                    # Добавляем кнопку для закрытия окна при ошибке
                    close_button = ctk.CTkButton(
                        info_window,
                        text="ЗАКРЫТЬ",
                        command=info_window.destroy,
                        height=35,
                        font=ctk.CTkFont(size=13, weight="bold"),
                        fg_color=self.colors["error"],
                        hover_color="#FF6B6B"
                    )
                    close_button.pack(pady=10)
                
                finally:
                    progress.stop()
                    progress.pack_forget()
            
            # Запускаем установку в отдельном потоке
            import threading
            thread = threading.Thread(target=install)
            thread.start()
            
        except Exception as e:
            print(f"Ошибка установки PyTorch: {str(e)}")

    def check_model_installed(self, model_name):
        """Проверка установки модели"""
        if model_name == "turbo":
            model_path = os.path.join(self.models_dir, "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
        elif model_name == "large-v3":
            model_path = os.path.join(self.models_dir, "models--Systran--faster-whisper-large-v3")
        else:
            model_path = os.path.join(self.models_dir, f"models--Systran--faster-whisper-{model_name}")
        return os.path.exists(model_path)

    def check_cuda_libraries(self):
        """Проверка и установка библиотек CUDA"""
        cuda_libs = {
            "cublas64_12.dll": "CUDA Runtime",
            "cudart64_12.dll": "CUDA Runtime",
            "cublasLt64_12.dll": "CUDA Runtime",
            "cufft64_11.dll": "CUDA Runtime",
            "curand64_10.dll": "CUDA Runtime",
            "cusolver64_11.dll": "CUDA Runtime",
            "cusparse64_12.dll": "CUDA Runtime",
            "cudnn64_8.dll": "cuDNN",
            "cudnn_ops_infer64_8.dll": "cuDNN",
            "cudnn_ops_train64_8.dll": "cuDNN",
            "cudnn_adv_infer64_8.dll": "cuDNN",
            "cudnn_adv_train64_8.dll": "cuDNN",
            "cudnn_cnn_infer64_8.dll": "cuDNN",
            "cudnn_cnn_train64_8.dll": "cuDNN"
        }
        
        missing_libs = {}
        for lib, package in cuda_libs.items():
            try:
                import ctypes
                ctypes.CDLL(lib)
            except OSError:
                if package not in missing_libs:
                    missing_libs[package] = []
                missing_libs[package].append(lib)
        
        if missing_libs:
            self.show_cuda_install_dialog(missing_libs)
            return False
        return True

    def show_cuda_install_dialog(self, missing_libs):
        """Показать диалог установки CUDA библиотек"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Установка CUDA")
        dialog.geometry("600x500")
        dialog.grab_set()
        
        # Центрируем окно
        window_width = 600
        window_height = 800
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 3
        dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Основной контейнер
        main_frame = ctk.CTkFrame(
            dialog,
            fg_color=self.colors["primary"],
            corner_radius=15
        )
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Заголовок
        header = ctk.CTkLabel(
            main_frame,
            text="⚠️ Отсутствуют необходимые библиотеки CUDA",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        header.pack(pady=20)

        # Описание
        description = ctk.CTkLabel(
            main_frame,
            text="Для работы с GPU требуется установить следующие компоненты:",
            font=ctk.CTkFont(size=14),
            wraplength=500
        )
        description.pack(pady=10)

        # Список отсутствующих компонентов
        info_frame = ctk.CTkScrollableFrame(
            main_frame,
            width=500,
            height=180,
            fg_color=self.colors["secondary"],
            corner_radius=10
        )
        info_frame.pack(pady=10, padx=20, fill="x")

        for package, libs in missing_libs.items():
            package_frame = ctk.CTkFrame(
                info_frame,
                fg_color="transparent"
            )
            package_frame.pack(fill="x", pady=5, padx=10)

            package_label = ctk.CTkLabel(
                package_frame,
                text=f"📦 {package}:",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.colors["text_primary"]
            )
            package_label.pack(anchor="w")

            for lib in libs:
                lib_frame = ctk.CTkFrame(
                    info_frame,
                    fg_color="transparent"
                )
                lib_frame.pack(fill="x", pady=2, padx=30)

                lib_label = ctk.CTkLabel(
                    lib_frame,
                    text=f"• {lib}",
                    font=ctk.CTkFont(size=13),
                    text_color=self.colors["text_secondary"]
                )
                lib_label.pack(anchor="w")

        # Инструкции по установке
        instructions_frame = ctk.CTkFrame(
            main_frame,
            fg_color=self.colors["secondary"],
            corner_radius=10
        )
        instructions_frame.pack(fill="x", padx=20, pady=15)

        instructions_label = ctk.CTkLabel(
            instructions_frame,
            text="Инструкции по установке:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        instructions_label.pack(anchor="w", padx=15, pady=(15, 10))

        def open_url(url):
            import webbrowser
            webbrowser.open(url)

        if "CUDA Runtime" in missing_libs:
            cuda_frame = ctk.CTkFrame(
                instructions_frame,
                fg_color="transparent"
            )
            cuda_frame.pack(fill="x", padx=15, pady=5)

            cuda_label = ctk.CTkLabel(
                cuda_frame,
                text="1. Установите CUDA Toolkit:",
                font=ctk.CTkFont(size=13),
                text_color=self.colors["text_primary"]
            )
            cuda_label.pack(anchor="w")

            cuda_link = "https://developer.nvidia.com/cuda-downloads"
            cuda_button = ctk.CTkButton(
                cuda_frame,
                text="Скачать CUDA Toolkit",
                command=lambda: open_url(cuda_link),
                height=30,
                font=ctk.CTkFont(size=12),
                fg_color=self.colors["accent"],
                hover_color=self.colors["accent_hover"]
            )
            cuda_button.pack(anchor="w", pady=5)

        if "cuDNN" in missing_libs:
            cudnn_frame = ctk.CTkFrame(
                instructions_frame,
                fg_color="transparent"
            )
            cudnn_frame.pack(fill="x", padx=15, pady=5)

            cudnn_label = ctk.CTkLabel(
                cudnn_frame,
                text="2. Установите cuDNN:",
                font=ctk.CTkFont(size=13),
                text_color=self.colors["text_primary"]
            )
            cudnn_label.pack(anchor="w")

            cudnn_link = "https://developer.nvidia.com/cudnn"
            cudnn_button = ctk.CTkButton(
                cudnn_frame,
                text="Скачать cuDNN",
                command=lambda: open_url(cudnn_link),
                height=30,
                font=ctk.CTkFont(size=12),
                fg_color=self.colors["accent"],
                hover_color=self.colors["accent_hover"]
            )
            cudnn_button.pack(anchor="w", pady=5)

        restart_label = ctk.CTkLabel(
            instructions_frame,
            text="После установки перезапустите приложение",
            font=ctk.CTkFont(size=13),
            text_color=self.colors["text_secondary"]
        )
        restart_label.pack(pady=15)

        # Кнопка закрытия
        close_button = ctk.CTkButton(
            main_frame,
            text="ПОНЯТНО",
            command=dialog.destroy,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"]
        )
        close_button.pack(pady=20)

if __name__ == "__main__":
    app = AudioTranscriber()
    app.mainloop()
