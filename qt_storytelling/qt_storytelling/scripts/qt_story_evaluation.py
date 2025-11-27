#!/usr/bin/env python3
"""
qt_story_eval.py

Simplified evaluation version of the QT interactive storyteller.

Differences from the full research version:
- NO mid-story interruption: once a story begins, QT finishes all 4 beats
  without stopping for "stop / switch / continue".
- Story choice is LIMITED to three categories:
    * funny
    * magic
    * adventure
- Still supports:
    * Camera-based simple face detection + greeting
    * Kid-safe small talk
    * Comfort replies when the child sounds sad/upset
    * Simple gestures and emotions

ROS I/O:
- Subscribes:
    /qt_asr/text      (std_msgs/String)  – recognised text
    /qt_vad/active   (std_msgs/Bool)    – VAD (not heavily used, but kept)
    <camera_topic>   (sensor_msgs/Image) – for face detection

- Publishes:
    /qt_robot/behavior/talkText (std_msgs/String)
    /qt_robot/gesture/play      (std_msgs/String)
    /qt_robot/emotion/show      (std_msgs/String)
    /qt_asr/control             (std_msgs/String)   "pause"/"resume"

Parameters:
    ~ollama_url   (str) default "http://localhost:11434/api/generate"
    ~ollama_model (str) default "qwen2.5:0.5b-instruct"
    ~camera_topic (str) default "/camera/color/image_raw"
"""

import json
import os
import re
import threading
import time
from typing import Dict, Optional

import cv2
import requests
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

# ---------------------------------------------------------------------
# Emoji / weird char cleaner
# ---------------------------------------------------------------------

EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)


def clean_for_speech(text: str) -> str:
    """Strip emoji, weird chars, and unwanted filler so QT TTS stays clean."""
    if not text:
        return ""
    # remove emoji / non-BMP
    text = EMOJI_RE.sub("", text)
    # keep only basic printable chars + newlines
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) <= 126)

    # remove annoying filler like "hey there" anywhere in the text
    # (case-insensitive, plus optional punctuation after it)
    text = re.sub(r"\b[Hh]ey there[!,. ]*", " ", text)

    # collapse extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# Gesture service (optional)
try:
    from qt_gesture_controller.srv import gesture_play, gesture_playRequest
    HAVE_GESTURE_SRV = True
except Exception:
    HAVE_GESTURE_SRV = False

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_TIMEOUT = 20.0

SAFE_KIDS_PREFIX = (
    "You are a friendly kids robot talking to a child aged 5–10. "
    "Reply in English. "
    "Keep replies very short (1–2 sentences), simple, cheerful, and age-appropriate. "
    "Avoid sensitive, scary, violent, political, adult, or personal topics. "
    "No brand names, no medical or legal advice. Be supportive and kind.\n\n"
)

COMFORT_PREFIX = (
    "You are a gentle, comforting robot friend talking to a child aged 5–10. "
    "Reply in 1–2 very short sentences using simple words. "
    "Your goals:\n"
    "- Show that you heard the feeling (name it if possible).\n"
    "- Say that the feeling is okay and normal.\n"
    "- Be warm, kind, and calm and also check on the child how the child is doing.\n"
    "- Do NOT give medical, diagnostic, or treatment advice.\n"
    "- Do NOT say you can fix anything.\n"
    "- If the child sounds very upset or unsafe, gently suggest they talk to a trusted grown-up.\n"
)

COMFORT_WORDS = [
    "sad", "upset", "lonely", "alone", "scared", "worried", "anxious",
    "afraid", "tired", "depressed", "angry", "mad", "nervous", "bored",
    "stressed", "overwhelmed", "crying", "cry"
]

# ---------------------------------------------------------------------
# Globals / state
# ---------------------------------------------------------------------

say_pub: Optional[rospy.Publisher] = None
gesture_pub: Optional[rospy.Publisher] = None
emotion_pub: Optional[rospy.Publisher] = None
asr_ctrl_pub: Optional[rospy.Publisher] = None

current_story_id: Optional[str] = None   # None when no story running
last_story_topic: Optional[str] = None

ask_another_pending: bool = False        # after a story ends
awaiting_story_choice: bool = False      # waiting for "funny / magic / adventure"

is_speaking: bool = False
post_speech_guard_until: float = 0.0

cancel_ev = threading.Event()
say_lock = threading.Lock()

# VAD (kept but only lightly used)
vad_active: bool = False

# Camera / face detection
bridge: Optional[CvBridge] = None
face_cascade: Optional[cv2.CascadeClassifier] = None

last_face_ts: float = 0.0
greeted_recently: bool = False
last_greet_spoken: float = 0.0
face_seen_this_story: bool = False
story_left_announced: bool = False

# Face / greeting params
DETECT_MIN_FACE_AREA: int = 40
GREET_RISING_EDGE_SEC: float = 2.0
GREET_RESET_SEC: float = 6.0
GREET_COOLDOWN: float = 20.0
FACE_TIMEOUT_SEC: float = 45.0

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase and remove non-alphabetic characters for simple comparisons."""
    return re.sub(r"[^a-z ]", "", (text or "").lower()).strip()

def say(text: str, manage_asr: bool = True) -> None:
    """Speak through QT. Optionally pause/resume ASR to avoid echo."""
    global is_speaking, post_speech_guard_until
    text = clean_for_speech(text)
    if not text:
        return
    with say_lock:
        is_speaking = True

    # Pause ASR (only if requested)
    if manage_asr and asr_ctrl_pub:
        try:
            asr_ctrl_pub.publish("pause")
        except Exception:
            pass

    # Publish speech
    try:
        say_pub.publish(text)
    except Exception:
        pass
    rospy.loginfo("[QT] say: %s", text)

    # Rough speech duration
    words = len(text.split())
    duration = min(3.0, max(0.9, 0.27 * words))
    end_time = time.time() + duration
    while time.time() < end_time and not rospy.is_shutdown():
        time.sleep(0.03)

    post_speech_guard_until = time.time() + 0.25
    with say_lock:
        is_speaking = False

    # Resume ASR (only if we paused it here)
    if manage_asr and asr_ctrl_pub:
        try:
            asr_ctrl_pub.publish("resume")
        except Exception:
            pass


def say_story(text: str) -> None:
    """
    Speak story text as a single line.
    In this evaluation version we do NOT allow interruption mid-story,
    and we do NOT pause/resume ASR per line (handled in run_story).
    """
    # Just one call; no extra splitting into smaller chunks
    if cancel_ev.is_set() or rospy.is_shutdown():
        return
    say(text, manage_asr=False)  # ASR is handled by run_story()
    # still respect post_speech_guard if we want tiny gap before next beat
def play_gesture(name: str, speed: float = 1.0) -> None:
    """Play a body gesture. Uses service if available, else just publishes name."""
    if not name:
        return
    # Gesture names from /qt_robot/gesture/list, e.g. "hi", "clapping", "shy"
    if HAVE_GESTURE_SRV:
        try:
            rospy.wait_for_service('/qt_robot/gesture/play', timeout=0.2)
            client = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
            client(gesture_playRequest(name=name, speed=speed))
            return
        except Exception:
            pass
    if gesture_pub:
        try:
            gesture_pub.publish(name)
        except Exception:
            pass


def show_emotion(name: str) -> None:
    """Show a face emotion, e.g. 'QT/happy', 'QT/sad'."""
    if not name or not emotion_pub:
        return
    try:
        emotion_pub.publish(name)
    except Exception:
        pass


def kid_safe_prompt(user_text: str) -> str:
    return (
        SAFE_KIDS_PREFIX +
        f'Child said: "{user_text}".\n'
        "Reply in 1–2 very short, cheerful sentences.\n"
    )


def comfort_prompt(user_text: str) -> str:
    return (
        COMFORT_PREFIX +
        f'\nThe child said: "{user_text}".\n'
        "Write your answer now.\n"
    )


def call_ollama(prompt: str, temperature: float = 0.6, top_p: float = 0.9) -> str:
    """Send a prompt to Ollama and return response text."""
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "top_p": top_p},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict):
                return (data.get("response") or "").strip()
            return str(data)
        rospy.logwarn("Ollama HTTP %d: %s", res.status_code, res.text[:150])
    except Exception as e:
        rospy.logwarn("Ollama error: %s", str(e))
    return "I'm thinking about that."


def kid_safe_reply(user_text: str) -> str:
    prompt = kid_safe_prompt(user_text)
    raw = call_ollama(prompt)
    return clean_for_speech(raw)


def comfort_reply(user_text: str) -> str:
    prompt = comfort_prompt(user_text)
    raw = call_ollama(prompt)
    return clean_for_speech(raw)


# ---------------------------------------------------------------------
# Pre-scripted stories (Ollama no longer used for stories)
# ---------------------------------------------------------------------

PRESET_STORIES = {
    "funny": [
        {"say": "Once there was a tiny robot who kept putting its shoes on the wrong feet."},
        {"say": "One day, the robot walked into class with one red shoe, one blue shoe, and socks on its hands."},
        {"say": "All the children laughed, and the robot laughed too, because it finally understood why walking felt so wobbly."},
        {"say": "From then on, the robot checked its feet every morning and sometimes wore silly socks just for fun."},
    ],
    "magic": [
        {"say": "In a quiet town, a little child found a soft glowing pebble under a tree."},
        {"say": "When they held it, tiny sparkles danced around and a gentle voice whispered, “You are braver than you know.”"},
        {"say": "The child used the magic pebble only to share kindness, like cheering up friends and calming worries."},
        {"say": "Even when the glow faded, the child remembered that the real magic was the kindness in their own heart."},
    ],
    "adventure": [
        {"say": "A curious child and a friendly robot set off to explore a bright forest after breakfast."},
        {"say": "They followed a winding path, crossed a small wooden bridge, and greeted a squirrel who seemed to show them the way."},
        {"say": "At the top of a sunny hill, they found a smooth stone with a painted heart, like a secret friendly treasure."},
        {"say": "They walked home tired but happy, knowing that gentle adventures can hide in everyday places."},
    ],
}


def get_preset_story(cat: str) -> list:
    """Return a pre-scripted 4-beat story for the given category."""
    story = PRESET_STORIES.get(cat)
    if story:
        return story
    # fallback if something weird happens
    return PRESET_STORIES["adventure"]


def parse_story_response(raw: str) -> Optional[list]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            beats = obj.get("beats") or []
            return beats[:4]
        except Exception:
            pass
    return None


def generate_story_beats(topic_hint: str) -> list:
    prompt = build_story_prompt(topic_hint)
    raw = call_ollama(prompt)
    beats = parse_story_response(raw) or []

    fallback = [
        {"say": "Once there was a tiny robot who loved to help children."},
        {"say": "One day, the robot found a shining path and followed it with a smile."},
        {"say": "At the end of the path, it discovered a cozy secret garden full of laughter."},
        {"say": "From then on, the robot told warm stories to every child who visited."},
    ]

    out = []
    for b in beats:
        say_txt = clean_for_speech((b.get("say") or "").strip())
        if not say_txt:
            continue
        out.append({"say": say_txt[:220]})
        if len(out) == 4:
            break

    while len(out) < 4:
        out.append(fallback[len(out)])

    return out[:4]


def run_story(beats: list) -> None:
    """
    Narrate a list of beats.
    ASR is paused for the entire story in this evaluation version.
    """
    global current_story_id, ask_another_pending
    current_story_id = "STORY"
    cancel_ev.clear()

    # Pause ASR for the entire story
    if asr_ctrl_pub:
        try:
            asr_ctrl_pub.publish("pause")
            rospy.loginfo("ASR mic: PAUSED for story")
        except Exception:
            pass

    try:
        for idx, beat in enumerate(beats):
            if cancel_ev.is_set() or rospy.is_shutdown():
                break

            text = beat.get("say", "")

            # Simple expressive behaviour
            emo, gest = choose_default_emotion_and_gesture(text, idx)
            if emo:
                show_emotion(emo)
            if gest:
                play_gesture(gest, speed=1.0)

            if text:
                say_story(text)  # no ASR control inside
    finally:
        # End of story: clear state and resume ASR
        current_story_id = None
        ask_another_pending = True
        if asr_ctrl_pub:
            try:
                asr_ctrl_pub.publish("resume")
                rospy.loginfo("ASR mic: RESUMED after story")
            except Exception:
                pass

    show_emotion("QT/happy")
    play_gesture("bye-bye")
    # Final prompt is a normal utterance, so manage_asr=True
    say("Would you like another story?")



# ---------------------------------------------------------------------
# Expressive mapping (emotions and gestures)
# ---------------------------------------------------------------------

AVAILABLE_EMOTIONS = ["QT/happy", "QT/sad", "QT/angry"]

STORY_GESTURES = [
    "QT/hi",
    "QT/clapping",
    "QT/hands-on-hip",
    "QT/shy",
    "QT/hands-side-back",
    "QT/happy",
    "QT/sad",
    "QT/bye-bye",
    "QT/show_left",
    "QT/sneezing",
    "QT/peekaboo-back",
    "QT/touch-head",
    "QT/touch-head-back",
    "QT/stretching",
    "QT/up_left",
    "QT/swipe_left",
    "QT/bored",
]


def choose_default_emotion_and_gesture(text: str, step_idx: int):
    text_l = (text or "").lower()

    if any(w in text_l for w in ["sad", "cry", "lonely", "upset", "afraid", "scared"]):
        emotion = "QT/sad"
    elif any(w in text_l for w in ["angry", "mad"]):
        emotion = "QT/angry"
    else:
        emotion = "QT/happy"

    if any(w in text_l for w in ["hello", "hi", "meet", "start", "begin", "once"]):
        gesture = "QT/hi"
    elif any(w in text_l for w in ["clap", "yay", "hurray", "hooray", "great", "well done", "happy"]):
        gesture = "QT/clapping"
    elif any(w in text_l for w in ["shy", "quiet", "soft"]):
        gesture = "QT/shy"
    elif any(w in text_l for w in ["surprise", "suddenly", "wow", "magic"]):
        gesture = "QT/hands-side-back"
    elif any(w in text_l for w in ["end", "home", "goodnight", "goodbye"]):
        gesture = "QT/bye-bye"
    else:
        gesture = STORY_GESTURES[step_idx % len(STORY_GESTURES)]

    return emotion, gesture


# ---------------------------------------------------------------------
# Question-less helpers
# ---------------------------------------------------------------------


def _sleep_int(seconds: float, cancel_event: threading.Event) -> None:
    end_time = time.time() + seconds
    while time.time() < end_time and not cancel_event.is_set() and not rospy.is_shutdown():
        rospy.sleep(0.1)


# ---------------------------------------------------------------------
# Camera / face detection
# ---------------------------------------------------------------------


def _find_haar_path() -> Optional[str]:
    candidates = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def on_image(msg: Image) -> None:
    global last_face_ts, greeted_recently, last_greet_spoken, face_seen_this_story, story_left_announced
    global current_story_id

    if bridge is None or face_cascade is None:
        return
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception:
        return

    small = cv_img
    if cv_img.shape[1] > 800:
        scale = 800.0 / cv_img.shape[1]
        small = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(DETECT_MIN_FACE_AREA, DETECT_MIN_FACE_AREA),
        )
    except Exception:
        faces = []
    now = rospy.Time.now().to_sec()

    if len(faces) > 0:
        prev = last_face_ts
        last_face_ts = now

        if (prev == 0.0 or (now - prev) > GREET_RISING_EDGE_SEC) and not greeted_recently and current_story_id is None:
            if (time.time() - last_greet_spoken) > GREET_COOLDOWN:
                show_emotion("QT/happy")
                play_gesture("QT/hi")
                say("Hi there! You can ask 'tell me a story' and I will tell one for you.")
                greeted_recently = True
                last_greet_spoken = time.time()

        if current_story_id:
            face_seen_this_story = True
        story_left_announced = False
    else:
        if (now - last_face_ts) > GREET_RESET_SEC:
            greeted_recently = False
        if current_story_id and face_seen_this_story:
            no_face_for = now - last_face_ts
            if no_face_for > FACE_TIMEOUT_SEC and not story_left_announced:
                cancel_ev.set()
                current_story_id = None
                face_seen_this_story = False
                story_left_announced = True
                say("Looks like you left. I will stop for now.")


# ---------------------------------------------------------------------
# VAD & ASR callbacks
# ---------------------------------------------------------------------


def on_vad(msg: Bool) -> None:
    global vad_active
    try:
        vad_active = bool(msg.data)
    except Exception:
        vad_active = False


def _choose_category_from_text(text_norm: str) -> str:
    """
    Decide between funny / magic / adventure, defaulting to 'adventure'.
    """
    if "funny" in text_norm or "silly" in text_norm:
        return "funny"
    if "magic" in text_norm or "magical" in text_norm or "wizard" in text_norm:
        return "magic"
    if "adventure" in text_norm or "explore" in text_norm or "journey" in text_norm:
        return "adventure"
    # default
    return "adventure"


def _topic_for_category(cat: str) -> str:
    if cat == "funny":
        return "a short, funny and gentle story for children"
    if cat == "magic":
        return "a short, magical story with kind magic for children"
    if cat == "adventure":
        return "a short, safe adventure story for children"
    return "a cheerful imaginative kids story"


def start_story_from_category(cat: str) -> None:
    global last_story_topic
    topic = _topic_for_category(cat)
    last_story_topic = topic  # still store a human-readable description if you want

    say(f"Okay! I’ll think of a {cat} story.")
    beats = get_preset_story(cat)  # use pre-scripted story instead of Ollama
    run_story(beats)


def on_asr(msg: String) -> None:
    """
    Main ASR callback.

    Evaluation behaviour:
    - If a story is running (current_story_id is not None), we IGNORE all ASR
      (no mid-story interruptions).
    - After a story ends, if QT asks “Would you like another story?” it waits
      for a yes/no, then asks which type: funny, magic, or adventure.
    - Otherwise:
        * detect comfort phrases
        * start story request
        * or small talk
    """
    global ask_another_pending, awaiting_story_choice, current_story_id

    if is_speaking:
        return

    text_raw = msg.data or ""
    text_norm = normalize(text_raw)
    if not text_norm:
        return

    rospy.loginfo("ASR: %s", text_norm)

    # basic noise + profanity filter
    if len(text_norm.split()) <= 1 and len(text_norm) <= 2:
        return
    if any(bad in text_norm for bad in ["fuck", "shit", "bitch"]):
        return

    # Guard right after our own speech
    if time.time() < post_speech_guard_until:
        return

    # 0) If a story is currently running, IGNORE input (no mid-story interruption)
    if current_story_id:
        return

    YES_WORDS = ["yes", "yeah", "yep", "sure", "please"]
    NO_WORDS = ["no", "nope", "nah", "not now", "later"]
    BYE_WORDS = ["bye", "goodbye", "see you", "see ya"]

    # 1) Handle "another story?" follow-up
    if ask_another_pending:
        # If child says bye, end gently
        if any(b in text_norm for b in BYE_WORDS):
            ask_another_pending = False
            say("Okay, bye for now!")
            return

        # If child clearly says no, stop
        if any(w in text_norm for w in NO_WORDS):
            ask_another_pending = False
            say("Okay, no problem.")
            return

        # If child clearly says yes AND mentions story, continue
        if any(w in text_norm for w in YES_WORDS) and "story" in text_norm:
            ask_another_pending = False
            awaiting_story_choice = True
            say("Great! Do you want a funny story, a magic story, or an adventure story?")
            return

        # Otherwise, treat it as normal chat after a story
        ask_another_pending = False
        # fall through to comfort/small talk handling below

    # 2) If we are already waiting for category choice
    if awaiting_story_choice:
        # Child changed their mind or said bye
        if any(b in text_norm for b in BYE_WORDS) or any(w in text_norm for w in NO_WORDS):
            awaiting_story_choice = False
            say("Okay, maybe another time.")
            return

        awaiting_story_choice = False
        category = _choose_category_from_text(text_norm)
        start_story_from_category(category)
        return

    # 3) Comfort detection (when child sounds upset)
    if "i feel" in text_norm or "i am " in text_norm or any(w in text_norm for w in COMFORT_WORDS):
        show_emotion("QT/sad")
        play_gesture("touch-head")
        reply = comfort_reply(text_raw)
        say(reply)
        return

    # 4) New story request
    if "story" in text_norm or "tell me a story" in text_norm:
        awaiting_story_choice = True
        say("Sure! Would you like a funny story, a magic story, or an adventure story?")
        return

    # 5) Otherwise: small talk
    answer = kid_safe_reply(text_raw)
    say(answer)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    global say_pub, gesture_pub, emotion_pub, asr_ctrl_pub
    global OLLAMA_MODEL, OLLAMA_URL
    global bridge, face_cascade

    rospy.init_node("qt_story_eval", anonymous=False)

    OLLAMA_MODEL = rospy.get_param("~ollama_model", OLLAMA_MODEL)
    OLLAMA_URL = rospy.get_param("~ollama_url", OLLAMA_URL)
    cam_topic = rospy.get_param("~camera_topic", "/camera/color/image_raw")

    rospy.loginfo("Ollama model: %s", OLLAMA_MODEL)
    rospy.loginfo("Ollama URL: %s", OLLAMA_URL)

    say_pub = rospy.Publisher('/qt_robot/behavior/talkText', String, queue_size=10)
    gesture_pub = rospy.Publisher('/qt_robot/gesture/play', String, queue_size=10)
    emotion_pub = rospy.Publisher('/qt_robot/emotion/show', String, queue_size=10)
    asr_ctrl_pub = rospy.Publisher('/qt_asr/control', String, queue_size=5)

    rospy.Subscriber('/qt_asr/text', String, on_asr)
    rospy.Subscriber('/qt_vad/active', Bool, on_vad)

    bridge = CvBridge()
    haar_path = _find_haar_path()
    if haar_path:
        fc = cv2.CascadeClassifier(haar_path)
        if not fc.empty():
            face_cascade = fc
            rospy.loginfo("Using Haar cascade: %s", haar_path)
        else:
            rospy.logwarn("Failed to load Haar cascade; face detection disabled.")
    else:
        rospy.logwarn("No Haar cascade file found; face detection disabled.")

    if face_cascade:
        rospy.Subscriber(cam_topic, Image, on_image, queue_size=1)
        rospy.loginfo("Listening to camera: %s", cam_topic)

    rospy.loginfo("Evaluation storyteller node started (non-interruptible, funny/magic/adventure).")
    rospy.spin()


if __name__ == "__main__":
    main()

