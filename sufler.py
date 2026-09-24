#!/usr/bin/env python3
"""Суфлёр: слушает системный звук, распознаёт речь, отвечает через Claude Code (подписка Pro).

Реплики с '?' отправляются в Claude автоматически, Enter отправляет последние реплики вручную.
"""
import argparse, ctypes, glob, json, os, queue, subprocess, sys, threading
from collections import deque

import numpy as np

# CUDA-библиотеки из pip-пакетов nvidia-* (системной CUDA нет)
try:
    import nvidia
    libs = sorted(glob.glob(os.path.join(nvidia.__path__[0], "*/lib/lib*.so.*")))
except ImportError:
    libs = []
for f in libs:
    try:
        ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
from faster_whisper import WhisperModel

LOCAL_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "large-v3-turbo")

RATE = 16000
JUNK = ("продолжение следует", "субтитр", "спасибо за просмотр")  # галлюцинации Whisper на тишине/музыке
SYSTEM = ("Ты суфлёр. Тебе дают расшифровку речи (с ошибками распознавания). "
          "Ответь на последний вопрос из блока \"Новое\": 3-5 коротких тезисов, которые можно сразу произнести вслух, "
          "на языке вопроса. Без вступлений. Блок \"Контекст\" нужен только для понимания. "
          "Если в блоке \"Новое\" нет содержательного вопроса (риторический, \"да?\", \"нет?\", \"понятно?\"), "
          "ответь одним словом SKIP. Никогда не отвечай на вопросы из блока \"Контекст\": на них уже ответили.")

p = argparse.ArgumentParser()
p.add_argument("--model", default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "large-v3-turbo",
               help="whisper: путь к модели или имя (small/medium/large-v3-turbo)")
p.add_argument("--lang", default="ru", help="ru/en/...; auto: автоопределение (на коротких фразах ошибается)")
p.add_argument("--claude", default="sonnet", help="модель Claude: haiku/sonnet/opus")
p.add_argument("--source", default=None, help="источник PulseAudio (по умолчанию монитор дефолтного выхода)")
p.add_argument("--silence", type=float, default=0.01, help="порог RMS тишины")
p.add_argument("--no-auto", action="store_true", help="не спрашивать автоматически на '?'")
a = p.parse_args()
lang = None if a.lang == "auto" else a.lang

try:
    whisper = WhisperModel(a.model, device="cuda", compute_type="float16")
    list(whisper.transcribe(np.zeros(RATE, np.float32), language="en")[0])  # проверка CUDA + прогрев
except Exception as e:
    print(f"[CUDA недоступна ({e}), работаю на CPU]", file=sys.stderr)
    whisper = WhisperModel(a.model, device="cpu", compute_type="int8")

history = deque(maxlen=20)
heard = asked = 0  # счётчики реплик: всего услышано / на момент последнего вопроса
lock = threading.Lock()  # один ответ за раз
out = threading.Lock()  # реплики не вклиниваются в печатающийся ответ


def ask():
    global asked
    if not lock.acquire(blocking=False):
        return
    try:
        lines = list(history)
        k = max(1, min(heard - asked, len(lines)))
        asked = heard
        cmd = ["claude", "-p", "--model", a.claude, "--setting-sources", "", "--strict-mcp-config",
               "--tools", "", "--no-session-persistence", "--system-prompt", SYSTEM,
               "--output-format", "stream-json", "--include-partial-messages", "--verbose"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.stdin.write("Контекст:\n" + "\n".join(lines[:-k]) + "\n\nНовое:\n" + "\n".join(lines[-k:]))
        proc.stdin.close()
        with out:
            head = ""  # копим начало ответа, чтобы молча пропустить SKIP
            for line in proc.stdout:
                ev = json.loads(line)
                d = ev.get("event", {}).get("delta", {}) if ev.get("type") == "stream_event" else {}
                t = d.get("text", "") if d.get("type") == "text_delta" else ""
                if ev.get("type") == "result" and ev.get("is_error"):
                    t = ev.get("result", "")
                if head is not None:
                    head += t
                    if len(head.strip()) > 4:
                        print("\n\033[1;32m" + head, end="", flush=True)
                        head = None
                else:
                    print(t, end="", flush=True)
            if head is None:
                print("\033[0m\n", flush=True)
    finally:
        lock.release()


def keyboard():
    for _ in sys.stdin:
        threading.Thread(target=ask, daemon=True).start()


def transcriber():
    global heard
    while True:
        segs, _ = whisper.transcribe(q.get(), language=lang, vad_filter=True, beam_size=5,
                                     initial_prompt=" ".join(list(history)[-3:]) or None)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not text or any(j in text.lower() for j in JUNK):
            continue
        history.append(text)
        heard += 1
        with out:
            print(f"\033[2m{text}\033[0m", flush=True)
        if not a.no_auto and "?" in text:
            threading.Thread(target=ask, daemon=True).start()


q = queue.Queue()
threading.Thread(target=transcriber, daemon=True).start()
src = a.source or subprocess.check_output(["pactl", "get-default-sink"], text=True).strip() + ".monitor"
rec = subprocess.Popen(["parec", "-d", src, "--format=s16le", f"--rate={RATE}", "--channels=1"],
                       stdout=subprocess.PIPE)
threading.Thread(target=keyboard, daemon=True).start()
print(f"Источник: {src}. Enter: спросить Claude, Ctrl+C: выход.", file=sys.stderr)

CHUNK = RATE // 10  # 100 мс
buf, quiet, pre = [], 0, deque(maxlen=3)  # pre: 300 мс до начала речи, чтобы не съедать первый слог
try:
    while data := rec.stdout.read(CHUNK * 2):
        x = np.frombuffer(data, np.int16).astype(np.float32) / 32768
        loud = np.sqrt(np.mean(x * x)) > a.silence
        if loud and not buf:
            buf.extend(pre)
        if loud or buf:
            buf.append(x)
        pre.append(x)
        quiet = 0 if loud else quiet + 1
        # Ограничение: энергетический VAD плохо режет речь на фоне музыки и шума, при необходимости заменить на silero-vad
        if buf and (quiet >= 8 or len(buf) >= 250):  # 0.8 с тишины или 25 с речи
            audio, buf = np.concatenate(buf), []
            if len(audio) >= RATE // 2:
                q.put(audio)
except KeyboardInterrupt:
    pass
finally:
    rec.terminate()
