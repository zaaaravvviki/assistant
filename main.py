import asyncio
import pyaudio
from vosk import Model, KaldiRecognizer
import g4f
from pydub import AudioSegment
from pydub.playback import play
import time
from rhvoice_wrapper import TTS
import json

# === Настройки ===
model = Model("vosk")
rec = KaldiRecognizer(model, 16000)


# === Озвучка ===
tts = TTS(threads=1)

# === Микрофон ===
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

# === Глобальная переменная ===
listening = True  # контролирует, слушаем ли мы микрофон

chat_history = []

def load_device_config():
    # Считываем ID устройства
    with open('device_id.txt', 'r', encoding='utf-8') as f:
        device_id = f.read().strip()

    # Загружаем все конфигурации
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    # Ищем подходящий конфиг по ID
    config = next((item for item in settings if item['id'] == device_id), None)

    if config is None:
        raise ValueError(f"Конфигурация для ID '{device_id}' не найдена.")

    return config

def speak(text):
    tts.to_file(filename='response.wav', text=text, voice='anna', format_='wav', sets=None)
    song = AudioSegment.from_wav("response.wav")
    play(song)

# === Логика обработки команды ===
async def process_voice_command(text, config):
    global chat_history
    system_message = config['system_message']
    bot_name = config['bot_name']
    # Удаляем имя бота из текста
    user_input = text.lower().replace(bot_name, "", 1).strip()

    chat_history.append({"role": "user", "content": user_input})
    chat_history.insert(0, {"role": "system", "content": system_message})

    try:
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.gpt_4o,
            messages=chat_history,
            # provider=g4f.Provider.GeekGpt,
        )
        chat_gpt_response = response
    except Exception as e:
        print(f"{g4f.Provider.GeekGpt.__name__}:", e)
        chat_gpt_response = "Извините, произошла ошибка."

    chat_history.append({"role": "assistant", "content": chat_gpt_response})
    print(f"{bot_name.capitalize()}: {chat_gpt_response}")

    # Озвучиваем ответ
    speak(chat_gpt_response)

# === Основная функция для прослушивания ===
async def main():
    global listening
    config = load_device_config()
    bot_name = config['bot_name']

    print(f"🎤 Говори что-нибудь (обращайся к боту по имени «{bot_name}»)")
    last_spoken_time = time.time()
    buffer_text = ""

    while True:
        if not listening:
            await asyncio.sleep(0.1)  # ничего не делаем пока не разрешено слушать
            continue

        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                buffer_text += " " + text
                last_spoken_time = time.time()
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                last_spoken_time = time.time()

        if buffer_text and time.time() - last_spoken_time > 2.0:
            final_text = buffer_text.strip()
            buffer_text = ""
            print(f"Вы сказали: {final_text}")
            if bot_name in final_text.lower():
                listening = False  # Отключаем микрофон
                await process_voice_command(final_text, config)
                listening = True  # Включаем обратно
            else:
                print("⏭️ Без обращения к боту — игнорируем.")

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("🛑 Завершено пользователем")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
