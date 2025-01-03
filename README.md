# 🎙️ VoiceScribePro

VoiceScribePro - это мощное приложение для преобразования речи в текст, использующее современные технологии искусственного интеллекта для точной транскрипции аудио.

![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

## ✨ Возможности

- 🎤 Запись аудио в реальном времени
- 📁 Поддержка аудиофайлов (MP3, WAV)
- 🤖 Несколько моделей распознавания разного размера и точности
- 🚀 Поддержка GPU для ускорения обработки
- 📝 Сохранение результатов в текстовый файл
- 🎯 Высокая точность распознавания
- 🌍 Поддержка множества языков

## 🛠️ Технологии

- **Python 3.11**
- **Faster Whisper** - оптимизированная версия Whisper от OpenAI
- **CustomTkinter** - современный пользовательский интерфейс
- **PyAudio** - запись аудио
- **CUDA** - ускорение на GPU (опционально)
- **PyTorch** - машинное обучение

## 💻 Системные требования

- Windows 10/11
- Python 3.11
- 4 ГБ RAM (минимум)
- 2 ГБ свободного места
- NVIDIA GPU с CUDA (опционально, для ускорения)

## 📥 Установка

### Способ 1: Установщик (рекомендуется)

1. Скачайте последнюю версию `VoiceScribePro_Setup.exe` из [релизов](https://github.com/your-repo/releases)
2. Запустите установщик от имени администратора
3. Следуйте инструкциям установщика
4. Запустите `VoiceScribePro.bat` с рабочего стола

### Способ 2: Ручная установка

1. Установите Python 3.11 с [официального сайта](https://www.python.org/downloads/)
2. Клонируйте репозиторий:

bash
git clone https://github.com/your-repo/VoiceScribePro.git

3. Установите зависимости:

bash
cd VoiceScribePro
pip install -r requirements.txt

4. Запустите приложение:

bash
python audio_to_text.py


## 🚀 Использование

1. Запустите программу через ярлык на рабочем столе
2. Выберите режим работы:
   - Запись аудио в реальном времени
   - Загрузка аудиофайла
3. Выберите модель распознавания:
   - Tiny (быстрая, менее точная)
   - Base (средняя)
   - Small (точная)
   - Medium (очень точная)
   - Large (самая точная)
   - Turbo (быстрая и точная)
4. Нажмите кнопку "Начать конвертацию"
5. Результат сохранится в текстовый файл

## ⚡ Ускорение на GPU

Для использования GPU:
1. Установите [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Установите [cuDNN](https://developer.nvidia.com/cudnn)
3. В настройках программы включите использование GPU

## 📝 Лицензия

Распространяется под лицензией MIT. Смотрите файл `LICENSE` для деталей.

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста:
1. Создайте форк проекта
2. Создайте ветку для вашей функции
3. Отправьте пулл-реквест

## 🙏 Благодарности

- [OpenAI](https://openai.com/) за модель Whisper
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) за оптимизацию
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) за UI компоненты
