"""
Telegram Ovoz Bot
1) Matn -> Ovoz (TTS: fish-audio/s2.1-pro-free via OpenRouter)
2) Ovoz -> Matn (STT: lokal Whisper / faster-whisper)
Thinking: Llama 3.3 70B (OpenRouter)
"""

import os
import asyncio
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv
import imageio_ffmpeg

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    BufferedInputFile,
)
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import aiohttp

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_THINKING_KEY = os.getenv("OPENROUTER_THINKING_KEY")
GEMINI_API_KEY = os.getenv("MIYYA_API_KEY")

# --- Konfiguratsiya ---
TTS_API_URL = "https://openrouter.ai/api/v1/audio/speech"
TTS_MODEL = "fish-audio/s2.1-pro-free:free"
THINKING_API_URL = "https://openrouter.ai/api/v1/chat/completions"
THINKING_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# --- Gemini AI (Google AI Studio) ---
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-3.5-flash-lite"  # eng barqaror bepul model

# --- STT (lokal Whisper) ---
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium
MAX_VOICE_SECONDS = 300  # 5 daqiqa chek


def _patch_av_for_whisper():
    """PyAV 19 da metadata_errors parametri olib tashlangan —
    faster-whisper mosligi uchun av.open ni yumshatamiz."""
    try:
        import faster_whisper.audio as fwa
        import av as _av

        class _AvProxy:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def open(self, file, mode="r", **kwargs):
                kwargs.pop("metadata_errors", None)
                return self._real.open(file, mode=mode, **kwargs)

        fwa.av = _AvProxy(_av)
        logger.info("PyAV patch qo'llandi (faster-whisper mosligi)")
    except Exception as e:
        logger.warning(f"PyAV patch bo'lmadi: {e}")


# --- FSM holatlar ---
class Flow(StatesGroup):
    choosing_mode = State()       # Matn->Ovoz yoki Ovoz->Matn
    waiting_tts_text = State()    # TTS: matn kutish
    waiting_grandpa_text = State() # Grandpa effekti: matn kutish
    waiting_stt = State()         # STT: ovoz xabari kutish
    waiting_script = State()      # Senariy tahlili: matn kutish
    ai_chat = State()             # AI Yordamchi: suhbat


# =============================================
# OVOZLAR KATALOGI
# =============================================

UZBEK_VOICES = {
    "🧔 Erkak (Rasmiy)": "44ad146afad04e4e99a7b98b8f97106f",
    "👨 Erkak (Tabiiy)": "fef226d2978a42d5b542e7577134f7b2",
}

FEMALE_VOICES = {
    "👩 Sarah (Yosh, jonli)": "933563129e564b19a115bedd57b7406a",
    "👩‍🦰 Laura (Ishonchli)": "e3cd384158934cc9a01029cd7d278634",
    "🧘 Selene (Sokin, meditativ)": "b347db033a6549378b48d00acb0d06cd",
    "👩 Hannah (Suhbatdosh)": "412a7bebb63345b6becf50f9da63ed5b",
}

MALE_VOICES = {
    "🧑 Ethan (Tushuntiruvchi)": "2ce3de29febb43d7a4a85a63085e1da1",
    "🧔 Adrian (Barqaror)": "7d44c98f619c4e98823a84ed60122f9f",
}

CHILD_VOICES = {
    "🧒 Bolacha (Neutal)": "4e407b98cb034f74b3863ec33c567c83",
    "👶 Kichkina bola": "051eccd6d8894155a0c544b8e5c0fd72",
    "👦 O'g'il bola": "501f126749464eef92d30fa913da1e8e",
    "🧒 Yosh bola (Playful)": "c0d6e010f2bb4198b98b78fb796e1b36",
}

# --- Trend ovozlari (Instagram/TikTok grandpa effekti) ---
TREND_VOICES = {
    "👴 Grandpa Spencer (Trend)": "245aac5cae7640d88a0e81da89c1e0f3",
    "👴 Grandpa (Hikoyachi)": "b0f79e4bac084142a2ca8266bd959ae5",
    "👴 Grandpa (Faktlar, PSA)": "697a707402bf47e9940b95a5f85fbe94",
}

ALL_VOICES = {}
ALL_VOICES.update(UZBEK_VOICES)
ALL_VOICES.update(FEMALE_VOICES)
ALL_VOICES.update(MALE_VOICES)
ALL_VOICES.update(CHILD_VOICES)
ALL_VOICES.update(TREND_VOICES)

VOICE_CATEGORIES = {
    "🇺🇿 O'zbek ovozlari": UZBEK_VOICES,
    "👴 Trend ovozlari": TREND_VOICES,
    "👩 Ayol ovozlari": FEMALE_VOICES,
    "🧔 Erkak ovozlari": MALE_VOICES,
    "🧒 Bola ovozlari": CHILD_VOICES,
}

# --- Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Foydalanuvchi ma'lumotlari ---
user_data = {}

# --- Standart ovoz: o'zbekcha ---
DEFAULT_VOICE = UZBEK_VOICES["🧔 Erkak (Rasmiy)"]

# --- Whisper (birinchi so'rovda yuklanadi) ---
_whisper = None
_whisper_lock = asyncio.Lock()


# =============================================
# KEYBOARDLAR
# =============================================

def mode_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="1️⃣ Matn → Ovoz")],
            [KeyboardButton(text="2️⃣ Ovoz → Matn")],
            [KeyboardButton(text="👴 Grandpa effekti")],
            [KeyboardButton(text="🤖 AI Yordamchi")],
            [KeyboardButton(text="📜 Senariy tahlili")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Tanlang...",
    )


def voice_category_keyboard() -> InlineKeyboardMarkup:
    buttons = []
    for cat_name in VOICE_CATEGORIES:
        buttons.append([InlineKeyboardButton(
            text=cat_name,
            callback_data=f"cat:{cat_name}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def voice_list_keyboard(category: str) -> InlineKeyboardMarkup:
    voices = VOICE_CATEGORIES.get(category, {})
    buttons = []
    for name, voice_id in voices.items():
        buttons.append([InlineKeyboardButton(
            text=name,
            callback_data=f"vsel:{voice_id}"
        )])
    buttons.append([InlineKeyboardButton(
        text="🔙 Orqaga",
        callback_data="back_to_cats"
    )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def restart_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🔄 Boshidan boshlash")]],
        resize_keyboard=True,
    )


# =============================================
# TTS FUNKSIYASI
# =============================================

async def text_to_speech(text: str, voice_id: str | None = None) -> bytes | None:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TTS_MODEL,
        "input": text,
        "response_format": "mp3",
    }
    if voice_id and voice_id != "default":
        payload["voice"] = voice_id

    payload["provider"] = {
        "options": {
            "fish-audio": {
                "normalize": True,
                "latency": "normal",
                "prosody": {"speed": 1.0, "normalize_loudness": True},
            }
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            TTS_API_URL, headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"TTS API xatosi {resp.status}: {error_text}")
                # Provider options xato bersa, oddiy so'rov
                if "provider" in error_text or "voice" in error_text:
                    payload.pop("provider", None)
                    payload.pop("voice", None)
                    async with session.post(
                        TTS_API_URL, headers=headers, json=payload,
                        timeout=aiohttp.ClientTimeout(total=180),
                    ) as retry_resp:
                        if retry_resp.status != 200:
                            return None
                        return await retry_resp.read()
                return None
            return await resp.read()


# =============================================
# GRANDPA EFFEKTI (FFmpeg)
# =============================================

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
TMP_DIR = Path(__file__).parent / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# Grandpa effekti sozlamalari (tuning uchun o'zgartirish mumkin)
GRANDPA_SETTINGS = {
    "pitch": 0.82,      # pastroq ovoz (0.75-0.90)
    "tempo": 0.97,      # biroz sekinroq (0.90-1.0)
    "low_boost_db": 3,  # ko'krak rezonansi
    "high_cut_db": -3,  # yuqori chastotani kesish (keksa ovoz)
    "clip_threshold": 0.8,  # asoftclip chegarasi
    "denoise_strength": 10, # afftdn NR darajasi
    "loudnorm_I": -16,  # loudnorm maqsadli loudness
}


async def apply_grandpa_effect(audio_bytes: bytes) -> bytes | None:
    """
    TTS audio fayliga 'Grandpa' (keksa odam) effektini qo'llaydi.
    FFmpeg orqali: pitch/formant pastga, EQ bilan yumshoqlik,
    shovqin pasaytirish, loudness normallashtirish.
    """
    s = GRANDPA_SETTINGS
    filter_str = (
        # 1) Pitch va formantni pastga (rubberband, formant=shifted default)
        f"rubberband=pitch={s['pitch']}:tempo={s['tempo']},"
        # 2) Ko'krak rezonansi (past chastota kuchaytirish)
        f"equalizer=f=180:t=q:w=1.5:g={s['low_boost_db']},"
        # 3) Yuqori chastotani kesish (keksa ovoz uchun)
        f"equalizer=f=3500:t=q:w=2:g={s['high_cut_db']},"
        # 4) Engil issiqlik (asoftclip tanh)
        f"asoftclip=type=tanh:threshold={s['clip_threshold']},"
        # 5) Shovqin pasaytirish
        f"afftdn=nr={s['denoise_strength']}:nf=-30,"
        # 6) Loudness normallashtirish
        f"loudnorm=I={s['loudnorm_I']}:TP=-1.5:LRA=11"
    )

    # Unikallik: uuid4 bilan bir vaqtda ko'p foydalanuvchi arashmaydi
    uid = uuid.uuid4().hex[:12]
    in_path = TMP_DIR / f"gp_in_{uid}.mp3"
    out_path = TMP_DIR / f"gp_out_{uid}.mp3"

    try:
        in_path.write_bytes(audio_bytes)

        cmd = [
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(in_path),
            "-af", filter_str,
            "-b:a", "128k",
            str(out_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error(f"Grandpa FFmpeg xatosi: {stderr.decode(errors='replace')[:300]}")
            return None

        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.error("Grandpa effekti bo'sh fayl qaytardi")
            return None

        return out_path.read_bytes()

    except Exception as e:
        logger.error(f"Grandpa effekti xatosi: {e}")
        return None
    finally:
        # Xavfsiz tozalash
        for p in (in_path, out_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# =============================================
# STT FUNKSIYASI (Lokal Whisper)
# =============================================

async def get_whisper():
    """Whisper modelini birinchi marta yuklaydi (thread-safe)."""
    global _whisper
    async with _whisper_lock:
        if _whisper is None:
            def _load():
                _patch_av_for_whisper()
                from faster_whisper import WhisperModel
                logger.info(f"Whisper modeli yuklanmoqda: {WHISPER_MODEL_SIZE}")
                return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            _whisper = await asyncio.to_thread(_load)
            logger.info("Whisper modeli tayyor!")
    return _whisper


async def speech_to_text(audio_path: str) -> tuple[str, str]:
    """Ovoz faylini matnga aylantiradi. (matn, til) qaytaradi."""
    model = await get_whisper()

    def _transcribe():
        segments, info = model.transcribe(
            audio_path,
            language="uz",           # O'zbekcha uchun majburlanadi (asosiy til)
            vad_filter=True,         # jimlikni kesib tashlash
            beam_size=5,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language

    return await asyncio.to_thread(_transcribe)


# =============================================
# THINKING MODEL
# =============================================

async def improve_prompt(text: str, mode: str = "tts") -> str:
    """Thinking model matnni ovozli nutqqa tayyorlaydi (o'zbekcha ustuvor)."""
    if mode == "stt":
        system_msg = (
            "Siz ovozdan matnga aylantirilgan matnni tozalaysiz.\n"
            "Vazifangiz:\n"
            "1. Punktuatsiyani qo'shing (nuqta, vergul, savol belgisi).\n"
            "2. O'zbekcha matn bo'lsa, imlosini tuzating: o', g', so'ngra, "
            "nima uchun, ndefa, xuddi, albatta.\n"
            "3. Transkripsiya xato so'zlarini kontekstdan kelib chiqib tuzating.\n"
            "4. Ma'noni hech qanday o'zgartirmang, qo'shimcha so'z qo'shmang.\n"
            "5. Matn boshqa tilda bo'lsa, o'sha tilda tozalang.\n\n"
            "Faqat tozalangan matnni qaytaring, boshqa hech narsa yozmang."
        )
    else:
        system_msg = (
            "Siz matnni ovozli nutqqa tayyorlash yordamchisisiz.\n"
            "Vazifangiz:\n"
            "1. Punktuatsiyani to'g'rilang — nutqda pauzalar shu belgilarga "
            "bog'lanadi (nuqta, vergul, savol/tanqis belgisi).\n"
            "2. O'zbekcha matn bo'lsa, imlosini tekshiring: o', g', so'ngra, "
            "nima uchun, ndefa, xuddi, albatta.\n"
            "3. Raqamlarni ovozda o'qiladigan so'zlarga aylantiring, chunki "
            "model raqamlarni noto'g'ri o'qiydi:\n"
            "   - 2025 -> 'ikki ming yigirma besh'\n"
            "   - 5 -> 'besh', 100 -> 'yuz', 3 -> 'uch'\n"
            "   - 12:30 -> 'o'n ikki yarim'\n"
            "4. Qisqartmalarni oching: 'Dr.' -> 'doktor', 'km' -> 'kilometr'.\n"
            "5. Agar o'qilishi qiyin bo'lsa, tabiiy ravishda ozgina qayta "
            "ifodalang, lekin ma'no saqlansin.\n"
            "6. Boshqa tildagi matn uchun — o'sha tilda tabiiy qilib tayyorlang "
            "va o'sha tilning imlosiga amal qiling.\n\n"
            "Ma'noni o'zgartirmang. Faqat tayyor matnni qaytaring."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_THINKING_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": THINKING_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ],
        "max_tokens": 1000,
        "reasoning": {"enabled": False},  # reasoning model — tezroq va ishonchliroq javob
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                THINKING_API_URL, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    return text
                data = await resp.json()
                content = data["choices"][0]["message"].get("content")
                if not content:
                    logger.warning("Thinking model bo'sh javob qaytardi")
                    return text
                # Thinking/markdown belgilarini tozalash (ovozga o'qilmasligi uchun)
                for junk in ("```", "`", "**", "__", "#"):
                    content = content.replace(junk, "")
                content = content.strip()
                return content if content else text
    except Exception as e:
        logger.error(f"Thinking xatosi: {e}")
        return text


# =============================================
# GEMINI AI (Google AI Studio)
# =============================================

async def gemini_chat(user_message: str, system_prompt: str | None = None) -> str | None:
    """Gemini API orqali matn javobini qaytaradi."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY topilmadi")
        return None

    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    contents = [{"parts": [{"text": user_message}]}]
    payload = {"contents": contents}

    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Gemini API xatosi {resp.status}: {error_text[:200]}")
                    return None
                data = await resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return None
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        return p["text"].strip()
                return None
    except Exception as e:
        logger.error(f"Gemini chat xatosi: {e}")
        return None


async def gemini_analyze_script(script_text: str) -> str | None:
    """Senariyni tahlil qiladi — tuzilish, qahramonlar, sahnalar."""
    system_prompt = (
        "Sen professional senariy tahlilchisisiz.\n"
        "Foydalanuvchi senariy yuboradi. Sen quyidagilarni qil:\n"
        "1. Senariy tuzilishini ajrat (boshlash, rivojlanish, yakun).\n"
        "2. Qahramonlarni sanab chiq (ism, xarakter).\n"
        "3. Asosiy g'oyani bir jumla bilan ayting.\n"
        "4. Ovoz yozish uchun tavsiyalar bering (ohang, temp).\n\n"
        "Juda qisqa va aniq javob bering (500 belgidan kam)."
    )
    return await gemini_chat(script_text, system_prompt)


async def gemini_bot_help(user_message: str) -> str | None:
    """Bot haqida savollarga javob beradi."""
    system_prompt = (
        "Sen Telegram ovoz boti yordamchisisiz.\n"
        "Bot imkoniyatlari:\n"
        "1. Matn → Ovoz: matnni ovozga aylantiradi (Fish Audio TTS)\n"
        "2. Ovoz → Matn: ovozli xabarni matnga aylantiradi (Whisper STT)\n"
        "3. Grandpa effekti: matnni keksa odam ovozida aylantiradi\n"
        "4. Turli ovozlar: o'zbek, ayol, erkak, bola, trend ovozlari\n"
        "5. AI Yordamchi: sizga savollarga javob beradi\n\n"
        "O'zbek tilida qisqa va tushunarli javob bering.\n"
        "Agar foydalanuvchi ovoz yaratish haqida so'rasa, /start buyrug'ini maslahat bering."
    )
    return await gemini_chat(user_message, system_prompt)


# =============================================
# HANDLERLAR
# =============================================

dp = Dispatcher(storage=MemoryStorage())


# --- /start: ovoz tanlashni so'raymiz ---
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    uid = message.from_user.id
    user_data[uid] = {"voice": DEFAULT_VOICE}
    await state.clear()

    await message.answer(
        "🎙️ <b>Ovoz Bot'ga xush kelibsiz!</b>\n\n"
        "Men ikkala yo'nalishda ishlayman:\n"
        "1️⃣ <b>Matn → Ovoz</b> — yozgan matningizni ovozli xabar qilaman\n"
        "2️⃣ <b>Ovoz → Matn</b> — ovozli xabaringizni matnga aylantiraman\n\n"
        "Avval <b>ovozni tanlang</b> (matn → ovoz rejimi uchun):\n",
        parse_mode=ParseMode.HTML,
        reply_markup=voice_category_keyboard(),
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "ℹ️ <b>Yo'riqnoma</b>\n\n"
        "1️⃣ /start — boshlash, ovoz tanlash\n"
        "2️⃣ Rejimni tanlang:\n"
        "   • 1️⃣ Matn → Ovoz (matn yozasiz, ovoz olasiz)\n"
        "   • 2️⃣ Ovoz → Matn (ovoz yuborasiz, matn olasiz)\n\n"
        "⚠️ Maksimal matn: ~3000 belgi\n"
        "⚠️ Maksimal ovoz: 5 daqiqa",
        parse_mode=ParseMode.HTML,
    )


# --- Ovoz kategoriyasi tanlandi ---
@dp.callback_query(F.data.startswith("cat:"))
async def category_selected(callback: CallbackQuery):
    category = callback.data[4:]
    await callback.message.edit_text(
        f"🎙️ <b>{category}</b> — ovozni tanlang:",
        parse_mode=ParseMode.HTML,
        reply_markup=voice_list_keyboard(category),
    )
    await callback.answer()


# --- Orqaga tugmasi ---
@dp.callback_query(F.data == "back_to_cats")
async def back_to_cats(callback: CallbackQuery):
    await callback.message.edit_text(
        "🎙️ <b>Ovoz kategoriyasini tanlang:</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=voice_category_keyboard(),
    )
    await callback.answer()


# --- OVOZ TANLANDI → rejani so'raymiz ---
@dp.callback_query(F.data.startswith("vsel:"))
async def voice_selected(callback: CallbackQuery, state: FSMContext):
    uid = callback.from_user.id
    voice_id = callback.data.split(":", 1)[1]

    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["voice"] = voice_id

    voice_name = "Default"
    for name, vid in ALL_VOICES.items():
        if vid == voice_id:
            voice_name = name
            break

    await callback.message.edit_text(
        f"✅ Ovoz tanlandi: <b>{voice_name}</b>\n\n"
        "Endi rejani tanlang:",
        parse_mode=ParseMode.HTML,
    )
    await callback.answer()

    await callback.message.answer(
        "📋 <b>Qaysi rejimda ishlaysiz?</b>\n\n"
        "1️⃣ <b>Matn → Ovoz</b>\n"
        "    Matn yozaman, ovozli xabar olaman\n\n"
        "2️⃣ <b>Ovoz → Matn</b>\n"
        "    Ovoz yuboraman, matn olaman",
        parse_mode=ParseMode.HTML,
        reply_markup=mode_keyboard(),
    )
    await state.set_state(Flow.choosing_mode)


# --- REJAMI TANLANDI ---
@dp.message(F.text == "🔄 Boshidan boshlash")
async def restart(message: Message, state: FSMContext):
    uid = message.from_user.id
    user_data[uid] = {"voice": DEFAULT_VOICE}
    await state.clear()

    await message.answer(
        "🔄 <b>Qaytadan boshlaymiz!</b>\n\n"
        "Avval <b>ovozni tanlang:</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=voice_category_keyboard(),
    )


@dp.message(Flow.choosing_mode)
async def mode_chosen(message: Message, state: FSMContext):
    text = (message.text or "").strip()

    if "Matn → Ovoz" in text or "1️⃣" in text:
        await message.answer(
            "📝 <b>Matn → Ovoz rejimi</b>\n\n"
            "Matn yozing — men uni ovozga aylantirib beraman:",
            parse_mode=ParseMode.HTML,
        )
        await state.set_state(Flow.waiting_tts_text)

    elif "Ovoz → Matn" in text or "2️⃣" in text:
        await message.answer(
            "🎙️ <b>Ovoz → Matn rejimi</b>\n\n"
            "Endi <b>ovozli xabar</b> yuboring (mikrofon tugmasi) —\n"
            "men uni matnga aylantirib beraman.\n\n"
            "Tilni avtomatik aniqlayman 🌍",
            parse_mode=ParseMode.HTML,
        )
        await state.set_state(Flow.waiting_stt)

    elif "Grandpa" in text or "👴" in text:
        await message.answer(
            "👴 <b>Grandpa Effekti</b>\n\n"
            "Matn yozing — men uni <b>keksa odam ovozida</b> aylantirib beraman:\n"
            "• Pastroq va tabiiy tembr\n"
            "• Biroz dag'al, xirillagan\n"
            "• Nutq tushunarli va ravon\n\n"
            "Boshqa ovoz rejimi uchun /start bosing.",
            parse_mode=ParseMode.HTML,
        )
        await state.set_state(Flow.waiting_grandpa_text)

    elif "AI Yordamchi" in text or "🤖" in text:
        await message.answer(
            "🤖 <b>AI Yordamchi</b>\n\n"
            "Men sizga yordam bera olaman! Savolingizni yozing:\n"
            "• Bot haqida savollar\n"
            "• Ovoz yaratish bo'yicha maslahat\n"
            "• Har qanday umumiy savol\n\n"
            "Chiqish uchun /start bosing.",
            parse_mode=ParseMode.HTML,
        )
        await state.set_state(Flow.ai_chat)

    elif "Senariy" in text or "📜" in text:
        await message.answer(
            "📜 <b>Senariy Tahlili</b>\n\n"
            "Senariy matnini yuboring — men uni tahlil qilaman:\n"
            "• Tuzilish (boshlash, rivojlanish, yakun)\n"
            "• Qahramonlar\n"
            "• Asosiy g'oya\n"
            "• Ovoz yozish tavsiyalari\n\n"
            "Chiqish uchun /start bosing.",
            parse_mode=ParseMode.HTML,
        )
        await state.set_state(Flow.waiting_script)

    else:
        await message.answer(
            "⚠️ Iltimos, tugmalardan birini tanlang:",
            reply_markup=mode_keyboard(),
        )


# =============================================
# 1) MATN → OVOZ (TTS)
# =============================================

@dp.message(Flow.waiting_tts_text)
async def tts_text_handler(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    uid = message.from_user.id

    if not text:
        await message.answer("⚠️ Iltimos, matn yozing.")
        return

    if len(text) > 3000:
        await message.answer("⚠️ Matn juda uzun! 3000 belgidan qisqaroq yozing.")
        return

    voice_id = user_data.get(uid, {}).get("voice", DEFAULT_VOICE)
    processing_msg = await message.answer("🧠 Matn tahlil qilinmoqda...")

    try:
        improved_text = await improve_prompt(text, mode="tts")
        logger.info(f"TTS matn: {improved_text[:80]}...")

        await processing_msg.edit_text("⏳ Ovoz yaratilmoqda...")

        audio_bytes = await text_to_speech(improved_text, voice_id)

        if audio_bytes is None:
            await processing_msg.edit_text(
                "❌ Xatolik yuz berdi. Qayta urinib ko'ring."
            )
            return

        audio_file = BufferedInputFile(audio_bytes, filename="voice.mp3")
        await message.answer_voice(
            voice=audio_file,
            caption=f"🔊 {len(text)} belgi",
        )
        await processing_msg.delete()

        await message.answer(
            "🔄 Yana ishlash uchun yoki boshqa rejimga o'tish:",
            reply_markup=restart_keyboard(),
        )

    except Exception as e:
        logger.error(f"TTS xatosi: {e}")
        await processing_msg.edit_text(f"❌ Xatolik: {str(e)[:200]}")


# =============================================
# 1b) GRANDPA EFFEKTI (Matn → Keksa Ovoz)
# =============================================

@dp.message(Flow.waiting_grandpa_text)
async def grandpa_text_handler(message: Message, state: FSMContext):
    text = (message.text or "").strip()
    uid = message.from_user.id

    if not text:
        await message.answer("⚠️ Iltimos, matn yozing.")
        return

    if len(text) > 2000:
        await message.answer("⚠️ Matn juda uzun! 2000 belgidan qisqaroq yozing.")
        return

    voice_id = user_data.get(uid, {}).get("voice", DEFAULT_VOICE)
    processing_msg = await message.answer("🧠 Matn tahlil qilinmoqda...")

    try:
        improved_text = await improve_prompt(text, mode="tts")
        logger.info(f"Grandpa matn: {improved_text[:80]}...")

        await processing_msg.edit_text("⏳ Ovoz yaratilmoqda...")

        audio_bytes = await text_to_speech(improved_text, voice_id)

        if audio_bytes is None:
            await processing_msg.edit_text(
                "❌ Ovoz yaratishda xatolik. Qayta urinib ko'ring."
            )
            return

        await processing_msg.edit_text("👴 Grandpa effekti qo'llanmoqda...")

        processed_bytes = await apply_grandpa_effect(audio_bytes)

        if processed_bytes is None:
            # Effekt muvaffaqiyatsiz bo'lsa — oddiy ovozni yuboramiz
            logger.warning("Grandpa effekti ishlamadi, oddiy ovoz yuboriladi")
            processed_bytes = audio_bytes

        audio_file = BufferedInputFile(processed_bytes, filename="grandpa_voice.mp3")
        await message.answer_voice(
            voice=audio_file,
            caption=f"👴 Grandpa effekti | {len(text)} belgi",
        )
        await processing_msg.delete()

        await message.answer(
            "🔄 Yana ishlash uchun yoki boshqa rejimga o'tish:",
            reply_markup=restart_keyboard(),
        )

    except Exception as e:
        logger.error(f"Grandpa xatosi: {e}")
        await processing_msg.edit_text(f"❌ Xatolik: {str(e)[:200]}")


# =============================================
# 3) AI YORDAMCHI (Gemini)
# =============================================

@dp.message(Flow.ai_chat)
async def ai_chat_handler(message: Message, state: FSMContext):
    """AI yordamchi bilan suhbat."""
    text = (message.text or "").strip()

    if not text:
        await message.answer("⚠️ Iltimos, savolingizni yozing.")
        return

    processing_msg = await message.answer("🤖 Javob tayyorlanmoqda...")

    try:
        response = await gemini_bot_help(text)

        if response:
            # Juda uzun javoblarni kesish (Telegram cheklovi)
            if len(response) > 4000:
                response = response[:4000] + "..."
            await processing_msg.edit_text(response)
        else:
            await processing_msg.edit_text(
                "❌ Javob olishda xatolik. Qayta urinib ko'ring."
            )

    except Exception as e:
        logger.error(f"AI chat xatosi: {e}")
        await processing_msg.edit_text(f"❌ Xatolik: {str(e)[:200]}")


# =============================================
# 4) SENARIY TAHLILI (Gemini)
# =============================================

@dp.message(Flow.waiting_script)
async def script_analysis_handler(message: Message, state: FSMContext):
    """Senariyni tahlil qiladi."""
    text = (message.text or "").strip()

    if not text:
        await message.answer("⚠️ Iltimos, senariy matnini yuboring.")
        return

    if len(text) < 50:
        await message.answer("⚠️ Senariy juda qisqa. Kamida 50 belgi yuboring.")
        return

    if len(text) > 10000:
        await message.answer("⚠️ Senariy juda uzun! 10000 belgidan qisqaroq yuboring.")
        return

    processing_msg = await message.answer("📜 Senariy tahlil qilinmoqda...")

    try:
        analysis = await gemini_analyze_script(text)

        if analysis:
            if len(analysis) > 4000:
                analysis = analysis[:4000] + "..."
            await processing_msg.edit_text(
                f"📜 <b>Senariy Tahlili:</b>\n\n{analysis}",
                parse_mode=ParseMode.HTML,
            )
        else:
            await processing_msg.edit_text(
                "❌ Tahlilda xatolik. Qayta urinib ko'ring."
            )

    except Exception as e:
        logger.error(f"Senariy tahlili xatosi: {e}")
        await processing_msg.edit_text(f"❌ Xatolik: {str(e)[:200]}")


# =============================================
# 2) OVOZ → MATN (STT)
# =============================================

@dp.message(Flow.waiting_stt, F.voice | F.audio)
async def stt_voice_handler(message: Message, state: FSMContext):
    """Ovozli xabarni matnga aylantiradi."""
    tg_file = message.voice or message.audio

    # Davomiylik tekshiruvi
    duration = tg_file.duration or 0
    if duration > MAX_VOICE_SECONDS:
        await message.answer(
            f"⚠️ Ovoz juda uzun ({duration}s). Maksimal {MAX_VOICE_SECONDS} soniya."
        )
        return

    processing_msg = await message.answer(
        "🧠 Ovoz tahlil qilinmoqda...\n"
        "⏳ Bir necha soniya kuting."
    )

    tmp_path = None
    try:
        # Faylni yuklab olish
        file = await message.bot.get_file(tg_file.file_id)
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        ext = "ogg" if message.voice else "mp3"
        tmp_path = os.path.join(tmp_dir, f"{message.from_user.id}_{tg_file.file_id}.{ext}")

        await message.bot.download_file(file.file_path, destination=tmp_path)
        logger.info(f"STT: fayl yuklandi {tmp_path} ({duration}s)")

        await processing_msg.edit_text("⏳ Matnga aylantirilmoqda...")

        # Transkripsiyalash
        raw_text, lang = await speech_to_text(tmp_path)

        if not raw_text:
            await processing_msg.edit_text(
                "❌ Ovozdan matn topilmadi.\n"
                "Iltimos, aniqroq va yaqinroq gapiring."
            )
            return

        logger.info(f"STT natija (til={lang}): {raw_text[:100]}")

        # Thinking model bilan tozalash
        await processing_msg.edit_text("🧹 Matn tozalanmoqda...")
        cleaned = await improve_prompt(raw_text, mode="stt")

        await processing_msg.edit_text(
            f"✅ <b>Tayyor!</b> (til: <code>{lang}</code>)\n\n"
            f"📝 {cleaned}",
            parse_mode=ParseMode.HTML,
        )

        await message.answer(
            "🔄 Yana ishlash uchun yoki boshqa rejimga o'tish:",
            reply_markup=restart_keyboard(),
        )

    except Exception as e:
        logger.error(f"STT xatosi: {e}")
        await processing_msg.edit_text(f"❌ Xatolik: {str(e)[:200]}")
    finally:
        # Vaqtinchalik faylni o'chirish
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@dp.message(Flow.waiting_stt)
async def stt_wrong_type(message: Message):
    await message.answer(
        "⚠️ Iltimos, <b>ovozli xabar</b> yuboring (mikrofon tugmasi bosib ushlab turing).",
        parse_mode=ParseMode.HTML,
    )



# =============================================
# ISHGA TUSHIRISH
# =============================================

async def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN topilmadi! .env faylni tekshiring.")
        return
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY topilmadi! .env faylni tekshiring.")
        return

    bot = Bot(token=BOT_TOKEN)
    logger.info("Bot ishga tushdi!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
