#!/usr/bin/env python3
"""
qt_interactive_storyteller.py

This ROS node turns the QT robot into a simple, interactive storyteller and
comfort companion.  It listens to spoken text from an offline ASR node,
generates short story segments using an Ollama model, and speaks them back
to the user.  The system supports the following high-level behaviours:

* **Story telling** – When a child says "story" or "tell me a story", the
  robot asks for a topic.  After the topic is provided, it generates a
  multi-step story using Ollama.  Each step consists of one or two
  sentences, delivered in short chunks so the child can interrupt.  The
  child can say "stop", "switch", or "continue" to control the story flow.

* **Comfort mode** – If the child expresses an emotion (for example,
  "I feel sad" or uses words like "lonely", "scared", etc.), the robot
  responds with a short, validating message.  These replies are generated
  via a separate comfort prompt to ensure a gentle tone.

* **Small talk** – In all other cases, the robot answers with a brief,
  kid-safe reply using a general chat prompt.  The ASR can run on a
  headset microphone; the node takes care to pause the ASR while the
  robot is speaking to avoid feeding its own voice back into the recogniser.

Compared to the original `qt_voice_story_orchestrator.py`, this script
omitted face detection and camera handling in the first revision to focus
on storytelling.  At your request the current version **reintroduces**
camera support for simple face recognition and greeting.  QT will watch
its camera feed, greet a child when a face appears after a period of
absence, and reset the greeting after no face is seen for a while.  The
script still leaves out FER emotion recognition to avoid extra model
dependencies.  Gestures and emotions are optional; if your QT SDK
supports them the node will publish accordingly.

To use this node you need:

* A running Ollama instance reachable at the configured URL (defaults
  to http://localhost:11434) with a suitable model.  The model name can
  be overridden via the `~ollama_model` ROS parameter.
* An ASR node that publishes recognised text to `/qt_asr/text` and a VAD
  node that publishes voice activity to `/qt_vad/active`.
* The QT SDK topics `/qt_robot/behavior/talkText`, `/qt_robot/gesture/play`
  and `/qt_robot/emotion/show` available.  Gestures are optional; if the
  gesture service is unavailable the node will fall back to publishing
  gesture names on the topic.

Launch this node after starting your ASR and TTS nodes.  When running
with headphones, the ASR will be paused while QT speaks and resumed
afterwards, minimising echo.
"""

import json
import os
import re
import threading
import time
import random
import requests
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

# Remove emojis and non-basic characters from LLM text
EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

def clean_for_speech(text):
    if not text:
        return ""
    # strip high-plane emoji characters
    text = EMOJI_RE.sub("", text)
    # optionally also drop other weird control chars
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) <= 126)
    return text.strip()

try:
    # Gesture service may not be available on all QT setups
    from qt_gesture_controller.srv import gesture_play, gesture_playRequest
    HAVE_GESTURE_SRV = True
except Exception:
    HAVE_GESTURE_SRV = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default endpoint for Ollama.  Override via ROS param `~ollama_url` if needed.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Default model for Ollama.  Override via ROS param `~ollama_model`.
OLLAMA_MODEL = "qwen3:1.7b"

# Timeout in seconds for generating a story or chat reply.  Increase if
# running heavy models on slow hardware.
OLLAMA_TIMEOUT = 20.0

# Prefix for general, kid-safe chat.  Ollama will prepend this when
# generating small-talk responses.
SAFE_KIDS_PREFIX = (
    "You are a friendly kids robot talking to a child aged 5-10. "
    "The story should be written in English for children aged 5-10"
    "Keep replies very short (1-2 sentences), simple, cheerful, and age-appropriate. "
    "Avoid sensitive, scary, violent, political, adult, or personal topics. "
    "No brand names, no medical or legal advice. Be supportive and kind.\n\n"
)

# Prefix for comfort messages.  Used when the child expresses sadness or
# similar feelings.  This encourages the model to return a gentle,
# validating response.
COMFORT_PREFIX = (
    "You are a gentle, comforting robot friend talking to a child aged 5-10. "
    "Reply in 1-2 very short sentences using simple words. "
    "Your goals:\n"
    "- Show that you heard the feeling (name it if possible).\n"
    "- Say that the feeling is okay and normal.\n"
    "- Be warm, kind, and calm and also check on the child how is the child doing too.\n"
    "- Do NOT give medical, diagnostic, or treatment advice.\n"
    "- Do NOT say you can fix anything.\n"
    "- If the child sounds very upset or unsafe, gently suggest they talk to a trusted grown-up.\n"
    "- You may optionally ask one small, gentle follow-up like "
    "\"Do you want to tell me more?\" or "
    "\"Would you like a story to help you feel better?\"\n"
)

COMFORT_WORDS = [
    "sad", "upset", "lonely", "alone", "scared", "worried", "anxious",
    "afraid", "tired", "depressed", "angry", "mad", "nervous", "bored",
    "stressed", "overwhelmed", "crying", "cry"
]

def kid_safe_prompt(user_text):
    """
    Build a safe prompt for general small-talk with a child (age 5-10),
    using the SAFE_KIDS_PREFIX defined above.
    """
    return (
        SAFE_KIDS_PREFIX +
        'Child said: "' + user_text + '".\n'
        "Reply in 1-2 very short, cheerful sentences.\n"
    )


def comfort_prompt(user_text):
    return (
        COMFORT_PREFIX +
        "\nThe child said: \"" + user_text + "\".\n"
        "Write your answer now."
    )

def comfort_reply(user_text):
    """Generate a warm, validating reply for emotional utterances."""
    prompt = COMFORT_PREFIX + (
        'The child just said: "' + user_text + '".\n'
        "Respond in a warm, understanding way that shows you listened, "
        "and optionally ask a tiny follow-up question like "
        "\"Do you want to tell me more?\" or \"Should we create a story together?\""
    )
    raw = call_ollama(prompt)
    return clean_for_speech(raw)


# -----------------------------------------------------------------------------
# Globals and state variables
# -----------------------------------------------------------------------------

# Publishers
say_pub = None
gesture_pub = None
emotion_pub = None
asr_ctrl_pub = None
audio_pub = None

# Story state
current_story_id = None
last_story_topic = None
pending_question = None
ask_another_pending = False
awaiting_story_choice = False
interrupt_choice_pending = False
is_speaking = False
post_speech_guard_until = 0.0

cancel_ev = threading.Event()
say_lock = threading.Lock()

# VAD state
vad_active = False

# -----------------------------------------------------------------------------
# Camera and face detection
# -----------------------------------------------------------------------------

# cv_bridge converter
bridge = None

# Haar cascade for face detection.  Will be loaded in main().
face_cascade = None

# Timing variables for greetings and presence
last_face_ts = 0.0
greeted_recently = False
last_greet_spoken = 0.0
face_seen_this_story = False
story_left_announced = False

# Face and greeting parameters
DETECT_MIN_FACE_AREA = 40  # minimum size of face to count as a detection (in pixels)
GREET_RISING_EDGE_SEC = 2.0  # greet after this many seconds of presence if not greeted recently
GREET_RESET_SEC = 6.0  # reset greeting state if no face seen for this many seconds
GREET_COOLDOWN = 20.0  # minimum seconds between greetings
FACE_TIMEOUT_SEC = 45.0  # stop story if child leaves for this long
QUESTION_WAIT_SEC = 4.0  # number of seconds to wait for the child's answer

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def normalize(text):
    """Lowercase and remove non-alphabetic characters for simple comparisons."""
    return re.sub(r"[^a-z ]", "", (text or "").lower()).strip()


def say(text):
    """Publish speech while pausing and resuming the ASR to avoid echo."""
    global is_speaking, post_speech_guard_until
    if not text:
        return
    with say_lock:
        is_speaking = True
    # Pause ASR mic
    if asr_ctrl_pub:
        try:
            asr_ctrl_pub.publish("pause")
        except Exception:
            pass
    # Speak via QT
    try:
        say_pub.publish(text)
    except Exception:
        pass
    rospy.loginfo("[QT] say: %s", text)
    # Estimate duration of speaking
    words = len(text.split())
    duration = min(2.2, max(0.8, 0.25 * words))
    end_time = time.time() + duration
    while time.time() < end_time and not rospy.is_shutdown():
        time.sleep(0.03)
    # Guard to avoid echo
    post_speech_guard_until = time.time() + 0.25
    with say_lock:
        is_speaking = False
    # Resume ASR
    if asr_ctrl_pub:
        try:
            asr_ctrl_pub.publish("resume")
        except Exception:
            pass


def say_story(text, listen_window=2.2):
    """Speak long text in small chunks, allowing the user to interrupt."""
    # Split into short phrases by punctuation or every 10 words
    parts = re.split(r"([.!?;:])", text)
    chunks = []
    current = ""
    for part in parts:
        if part in ".!?;:":
            current += part
            if current.strip():
                chunks.append(current.strip())
            current = ""
        else:
            token = part.strip()
            if not token:
                continue
            if current:
                current += " " + token
            else:
                current = token
    if current.strip():
        chunks.append(current.strip())
    # Further split long chunks by word count
    final_chunks = []
    for chunk in chunks:
        words = chunk.split()
        while len(words) > 10:
            final_chunks.append(" ".join(words[:6]))
            words = words[10:]
        if words:
            final_chunks.append(" ".join(words))
    # Speak each chunk
    for ch in final_chunks:
        if cancel_ev.is_set() or rospy.is_shutdown():
            break
        say(ch)
        # Wait until the post-speech guard expires
        while time.time() < post_speech_guard_until and not cancel_ev.is_set() and not rospy.is_shutdown():
            rospy.sleep(0.02)
        # Allow a brief window for the child to interrupt
        end = time.time() + listen_window
        while time.time() < end and not cancel_ev.is_set() and not rospy.is_shutdown():
            if interrupt_choice_pending or cancel_ev.is_set():
                return
            rospy.sleep(0.03)


def play_gesture(name, speed=1.0):
    """Play a gesture via service or topic."""
    if not name:
        return
    if HAVE_GESTURE_SRV:
        try:
            rospy.wait_for_service('/qt_robot/gesture/play', timeout=0.3)
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
            
def play_audio(name: str) -> None:
    """Play a named audio from /home/qtrobot/data/audios (without extension)."""
    if not name or audio_pub is None:
        return
    try:
        audio_pub.publish(name)
    except Exception:
        pass

def show_emotion(name: str) -> None:
    if not name or not emotion_pub:
        return
    try:
        emotion_pub.publish(name)
    except Exception:
        pass

def show_emotion(name):
    """Publish an emotion on the QT emotion topic."""
    if not name or not emotion_pub:
        return
    try:
        emotion_pub.publish(name)
    except Exception:
        pass


def kid_safe_reply(user_text):
    prompt = kid_safe_prompt(user_text)
    raw = call_ollama(prompt)
    return clean_for_speech(raw)
    
def choose_default_emotion_and_gesture(text, step_idx):
    """
    Choose an emotion (face) and gesture (body) using only what this QT supports.
    This is used when the LLM does not specify emotion/gesture.
    """
    text_l = (text or "").lower()

    # ---- Emotion (face) ----
    if any(w in text_l for w in ["sad", "cry", "lonely", "upset", "afraid", "scared"]):
        emotion = "QT/sad"
    elif any(w in text_l for w in ["angry", "mad"]):
        emotion = "QT/angry"
    else:
        # default to happy for kids' stories and chat
        emotion = "QT/happy"

    # ---- Gesture (body) ----
    if any(w in text_l for w in ["hello", "hi", "meet", "start", "begin"]):
        gesture = "QT/hi"
    elif any(w in text_l for w in ["clap", "yay", "hurray", "hooray", "great", "well done"]):
        gesture = "QT/clapping"
    elif any(w in text_l for w in ["shy", "quiet", "soft"]):
        gesture = "QT/shy"
    elif any(w in text_l for w in ["surprise", "suddenly", "wow"]):
        gesture = "QT/hands-side-back"
    elif "goodbye" in text_l or "home" in text_l or "end" in text_l:
        gesture = "QT/bye-bye"
    else:
        # cycle through all the known gestures so it doesn't repeat the same one
        gesture = STORY_GESTURES[step_idx % len(STORY_GESTURES)]

    return emotion, gesture


def call_ollama(prompt, temperature=0.6, top_p=0.9):
    """Send a prompt to Ollama and return the response text."""
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
        rospy.logwarn("Ollama HTTP %d: %s", res.status_code, res.text[:120])
    except Exception as e:
        rospy.logwarn("Ollama error: %s", str(e))
    return "I'm thinking about that."


def build_story_prompt(topic):
    return (
        "You are a friendly storytelling robot. Age 5-10. "
        "Short, cheerful, safe sentences. No brands, no violence or scary content. "
        "The story should be written in English for children aged 5-10.\n"
        "Do not use emojis or special characters—use only plain words.\n"
        "Return ONLY a JSON object like this:\n"
        "{\"beats\": [\n"
        "  {\"say\": \"...\", \"gesture\": \"...\", \"emotion\": \"...\"},\n"
        "  {\"say\": \"...\", \"gesture\": \"...\", \"emotion\": \"...\"},\n"
        "  {\"say\": \"...\", \"gesture\": \"...\", \"emotion\": \"...\", \"ask\": {\n"
        "      \"question\": \"...\",\n"
        "      \"expect_any\": [\"keyword1\", \"keyword2\"],\n"
        "      \"correct_say\": \"...\",\n"
        "      \"wrong_say\": \"...\",\n"
        "      \"timeout\": 4\n"
        "  }}\n"
        "]}\n"
        "Rules:\n"
        "- Topic: " + (topic or 'cheerful imaginative kids story') + ".\n"
        "- EXACTLY 4 beats in the 'beats' list.\n"
        "- Each 'say' is 1-3 short sentences, friendly and simple.\n"
        "- Allowed gestures: "
        "\"QT/hi\",\"QT/clapping\",\"QT/hands-on-hip\",\"QT/shy\",\"QT/hands-side-back\",\"QT/happy\",\"QT/sad\",\"QT/bye-bye\".\n"
        "- Allowed emotions: \"QT/happy\",\"QT/calm\",\"QT/thinking\".\n"
        "- 1 or 2 of the beats (near the end) may include an 'ask' object as shown above.\n"
        "- No text before or after the JSON.\n"
    )

def parse_story_response(raw):
    """Parse the JSON returned by Ollama into a list of beats."""
    raw = (raw or "").strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    # Find outermost JSON object
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


def generate_story_beats(topic_hint):
    """Generate a list of beats for a new story using Ollama."""
    prompt = build_story_prompt(topic_hint)
    raw = call_ollama(prompt)
    beats = parse_story_response(raw) or []
    # Fallback if parsing fails or too few beats
    fallback = [
        {"say": "Let's go on a tiny adventure through bright trees and chirpy birds.", "gesture": "QT/hi", "emotion": "QT/happy", "dur": 2.0},
        {"say": "We follow a shiny path and hear a gentle river singing nearby.", "gesture": "QT/look-around", "emotion": "QT/calm", "dur": 2.0},
        {"say": "Across the water, a cozy cave glows like a warm lantern.", "gesture": "QT/point-down", "emotion": "QT/thinking", "dur": 2.0},
        {"say": "Inside, a mirror shows a brave helper—you—and we cheer together!", "gesture": "QT/wow", "emotion": "QT/happy", "dur": 2.0},
    ]
    out = []
    for beat in beats:
        say_txt = (beat.get("say") or "").strip()
        if not say_txt:
            continue
        out.append({
            "say": say_txt[:220],
            "gesture": (beat.get("gesture") or "QT/hi").strip(),
            "emotion": (beat.get("emotion") or "QT/happy").strip(),
            "dur": float(beat.get("dur", 2.0))
        })
        if "ask" in beat and isinstance(beat["ask"], dict):
            # Copy ask details verbatim for our Q&A logic
            out[-1]["ask"] = beat["ask"]
        # Stop after 4 beats
        if len(out) == 4:
            break
    # If too few beats, append fallback
    while len(out) < 4:
        out.append(fallback[len(out)])
    return out[:4]

def run_story(beats):
    """Narrate a list of beats.  Each beat is a dict with keys 'say', 'gesture', 'emotion', and optional 'ask'."""
    global current_story_id, pending_question, ask_another_pending, interrupt_choice_pending
    current_story_id = "STORY"
    cancel_ev.clear()
    pending_question = None
    interrupt_choice_pending = False

    # Tell each beat
    for step_idx, beat in enumerate(beats):
        if cancel_ev.is_set() or rospy.is_shutdown():
            break

        # -------- expressive emotion + gesture --------
        emo = beat.get("emotion")
        gest = beat.get("gesture")

        # If model didn't give one, choose a default based on text + step index
        if not emo or not gest:
            default_emo, default_gest = choose_default_emotion_and_gesture(
                beat.get("say", ""), step_idx
            )
            if not emo:
                emo = default_emo
            if not gest:
                gest = default_gest

        # Play emotion (face) and gesture (body)
        if emo:
            show_emotion(emo)
        if gest:
            play_gesture(gest, speed=1.0)

        # -------- speech --------
        if beat.get("say"):
            # If there's an 'ask' we don't want to split the sentence mid-question
            if "ask" in beat:
                say(beat["say"])
            else:
                say_story(beat["say"])

        # -------- optional Q&A --------
        if beat.get("ask") and not cancel_ev.is_set():
            ask = beat["ask"]
            # Prepare question parameters
            q_text = ask.get("question", "What happens next?")
            expect = [w.lower() for w in ask.get("expect_any", [])]
            correct_say = ask.get("correct_say", "Yes, that's right!")
            wrong_say = ask.get("wrong_say", "Nice guess, let's see.")
            timeout = float(ask.get("timeout", 3.0))
            start_question(q_text, expect, correct_say, wrong_say, timeout)
            wait_for_question_or_timeout(cancel_ev)

        # Brief gap before next beat
        _sleep_int(beat.get("dur", 2.0), cancel_ev)

    # Story finished
    current_story_id = None
    ask_another_pending = True
    show_emotion("QT/happy")
    play_gesture("QT/bye-bye")
    say("Would you like another story?")


def start_question(question_text, expect_any, correct_say, wrong_say, timeout):
    """Set up a pending question for the child.  See wait_for_question_or_timeout()."""
    global pending_question
    pending_question = {
        "question": question_text,
        "expect": [w.lower() for w in expect_any],
        "correct_say": correct_say,
        "wrong_say": wrong_say,
        "deadline": time.time() + timeout,
        "answered": False,
        "outcome": None,
    }
    show_emotion("QT/thinking")
    play_gesture("QT/listen")
    say(question_text)


def wait_for_question_or_timeout(cancel_event):
    """
    Listen for an answer until either the child speaks (VAD goes active and then
    quiet) or a timeout elapses.  If no answer is provided, we will deliver
    the default wrong/correct reply depending on whether the child spoke.
    """
    global pending_question
    if pending_question is None:
        return
    deadline = pending_question.get("deadline", time.time())
    # Wait until we detect speech or timeout
    while not cancel_event.is_set() and time.time() < deadline:
        if vad_active:
            break
        rospy.sleep(0.07)
    # If speech started, wait until VAD goes inactive (child finished)
    if vad_active:
        gap_deadline = time.time() + 1.0
        while not cancel_event.is_set() and time.time() < gap_deadline:
            if not vad_active:
                break
            rospy.sleep(0.05)
    # If no answer flagged as answered yet, deliver generic reply
    if pending_question and not pending_question.get("answered"):
        show_emotion("QT/happy")
        say("Good guess! Let's see what happens next.")
    pending_question = None

def _sleep_int(seconds, cancel_event):
    """Sleep in small increments, checking for cancellation."""
    end_time = time.time() + seconds
    while time.time() < end_time and not cancel_event.is_set() and not rospy.is_shutdown():
        rospy.sleep(0.1)


def handle_pending_question(answer):
    """Check if a pending question exists and grade the answer.  Returns True if a question was handled."""
    global pending_question
    if pending_question is None:
        return False
    text_norm = normalize(answer)
    if any(k in text_norm for k in pending_question.get("expect", [])):
        show_emotion("QT/happy")
        say(pending_question.get("correct_say", "That's right!"))
        pending_question["answered"] = True
        pending_question["outcome"] = "correct"
    elif text_norm:
        show_emotion("QT/calm")
        say(pending_question.get("wrong_say", "Interesting, let's find out."))
        pending_question["answered"] = True
        pending_question["outcome"] = "wrong"
    else:
        # Silence or unknown answer will be handled in wait_for_question_or_timeout()
        pass
    return True

# -----------------------------------------------------------------------------
# Camera / face detection helpers
# -----------------------------------------------------------------------------

def _find_haar_path():
    """
    Try to locate a Haar cascade file for frontal face detection.  This helper
    searches common install locations for OpenCV.  Returns a path if found.
    """
    candidates = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def on_image(msg):
    """
    Process incoming camera images, perform simple face detection, and greet
    the child when they appear.  Also tracks when the child leaves to
    optionally stop the story.
    """
    global last_face_ts, greeted_recently, last_greet_spoken, face_seen_this_story, story_left_announced
    global current_story_id
    # Convert ROS image to OpenCV BGR
    if bridge is None or face_cascade is None:
        return
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception:
        return
    # Resize for speed if very large
    small = cv_img
    if cv_img.shape[1] > 800:
        scale = 800.0 / cv_img.shape[1]
        small = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = []
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(DETECT_MIN_FACE_AREA, DETECT_MIN_FACE_AREA),
        )
    except Exception:
        pass
    now = rospy.Time.now().to_sec()
    if len(faces) > 0:
        # Use the largest detected face
        x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
        # Convert coordinates back to original scale
        sx = float(cv_img.shape[1]) / float(small.shape[1])
        sy = float(cv_img.shape[0]) / float(small.shape[0])
        cx = int((x + w / 2) * sx)
        cy = int((y + h / 2) * sy)
        # Update last face timestamp
        prev = last_face_ts
        last_face_ts = now
        # Greeting logic: greet if rising edge (face appears after absence)
        if (prev == 0.0 or (now - prev) > GREET_RISING_EDGE_SEC) and not greeted_recently and current_story_id is None:
            # Avoid greeting too frequently
            if (time.time() - last_greet_spoken) > GREET_COOLDOWN:
                show_emotion("QT/happy")
                play_gesture("QT/hi")
                play_audio("QT/Komiku_Glouglou")
                say("Hi there! You can ask me for a story or just talk with me.")
                greeted_recently = True
                last_greet_spoken = time.time()
        # Mark that we saw a face during this story
        if current_story_id:
            face_seen_this_story = True
        # If the child has returned after we announced they left, reset the flag
        story_left_announced = False
    else:
        # No faces detected
        # Reset greeting if no face for a while
        if (now - last_face_ts) > GREET_RESET_SEC:
            greeted_recently = False
        # Auto-stop story if child leaves
        if current_story_id and face_seen_this_story:
            no_face_for = now - last_face_ts
            if no_face_for > FACE_TIMEOUT_SEC and not story_left_announced:
                cancel_ev.set()
                current_story_id = None
                face_seen_this_story = False
                story_left_announced = True
                say("Looks like you left. I will stop for now.")


# ---- Expressive behaviour config (use only gestures/emotions that exist on this QT) ----
AVAILABLE_EMOTIONS = ["QT/happy", "QT/sad", "QT/angry"]

STORY_GESTURES = [
    "QT/hi",               # greeting / start
    "QT/clapping",         # excited / happy
    "QT/hands-on-hip",     # confident / explaining
    "QT/shy",              # quieter moments
    "QT/hands-side-back",  # plot twists
    "QT/happy",            # playful motions
    "QT/sad",              # slower / sad moments
    "QT/bye-bye",          # story ending
    "QT/show_left",        # showing something to the side
    "QT/sneezing",         # funny moment
    "QT/send_kiss",        # affectionate / comforting
    "QT/peekaboo-back",    # playful
    "QT/touch-head",       # thinking / worried
    "QT/touch-head-back",  # variation
    "QT/stretching",       # relaxed / waking up
    "QT/up_left",          # reaching up
    "QT/swipe_left",       # moving attention / scene change
    "QT/bored",            # low energy part of story
]


def on_vad(msg):
    """Callback for voice activity detection."""
    global vad_active
    try:
        vad_active = bool(msg.data)
    except Exception:
        vad_active = False


def on_asr(msg):
    """
    Callback for each recognised speech segment.  Handles commands, story
    requests, comfort triggers and general chat.
    """
    global awaiting_story_choice, last_story_topic, ask_another_pending
    global interrupt_choice_pending, current_story_id
    global pending_question

    # Ignore if robot is speaking
    if is_speaking:
        return

    # Raw + normalised text
    text_raw = msg.data or ""
    text_norm = normalize(text_raw)
    tokens = text_norm.split()
    if not text_norm:
        return

    rospy.loginfo("ASR: %s", text_norm)

    # Command keywords
    CMD_STOP   = ["stop", "pause", "halt", "enough", "quit", "cancel", "end", "finish"]
    CMD_SWITCH = ["next", "change", "switch", "another", "different", "skip"]
    CMD_CONT   = ["continue", "keep going", "go on", "carry on", "resume"]
    YES_WORDS  = ["yes", "yeah", "yep", "sure", "okay", "ok", "please"]
    NO_WORDS   = ["no", "nope", "nah", "not now", "later"]

    def has_any(tokens, words):
        return any(t in words for t in tokens)
    
    is_command = any(w in text_norm for w in (CMD_STOP + CMD_SWITCH + CMD_CONT))

    # Respect a short guard to avoid echoing our own speech, unless it's a command
    if time.time() < post_speech_guard_until and not is_command:
        return

    # Simple noise filtering: skip very short or profane utterances
    if not is_command and (len(text_norm.split()) <= 1 and len(text_norm) <= 2):
        return
    if any(bad in text_norm for bad in ["fuck", "shit", "bitch"]):
        return

    # 1) Handle story follow-up choice (after asking "another story?")
    if ask_another_pending and not current_story_id:
        if any(w in text_norm for w in NO_WORDS):
            ask_another_pending = False
            say("Okay, no problem.")
            return
        if any(w in text_norm for w in YES_WORDS) or "story" in text_norm:
            ask_another_pending = False
            awaiting_story_choice = True
            say("Great! What kind of story would you like? You can say any topic.")
            return
        # Otherwise, treat as general chat below
        ask_another_pending = False

    # 2) If currently awaiting a story topic, take this utterance as the topic
    if awaiting_story_choice:
        awaiting_story_choice = False
        say("Okay! I'll think for a moment.")
        play_audio("QT/Komiku_Glouglou")
        last_story_topic = text_raw.strip()
        beats = generate_story_beats(last_story_topic)
        run_story(beats)
        return

    # 3) If we are handling an interruption (continue/stop/switch) during a story
    if interrupt_choice_pending:
        if any(w in text_norm for w in CMD_STOP):
            interrupt_choice_pending = False
            cancel_ev.set()
            pending_question = None
            say("Okay, I'll stop.")
            current_story_id = None
            ask_another_pending = False
            return

        if any(w in text_norm for w in CMD_SWITCH):
            interrupt_choice_pending = False
            cancel_ev.set()
            pending_question = None
            say("Okay, let's switch to a different story. What should the new story be about?")
            awaiting_story_choice = True
            return

        if any(w in text_norm for w in CMD_CONT):
            interrupt_choice_pending = False
            say("Okay, I'll continue.")
            if last_story_topic:
                beats = generate_story_beats(last_story_topic)
                run_story(beats)
            return

        # Unrecognised answer: ask again
        say("Should I continue, stop, or switch?")
        return

    # 4) Explicit commands at top level
    if any(w in text_norm for w in CMD_STOP):
        cancel_ev.set()
        pending_question = None
        say("Okay, I'll stop.")
        current_story_id = None
        ask_another_pending = False
        return

    if any(w in text_norm for w in CMD_SWITCH):
        cancel_ev.set()
        pending_question = None
        say("Okay, let's switch to a different story. What should the new story be about?")
        awaiting_story_choice = True
        return

    if any(w in text_norm for w in CMD_CONT):
        if not current_story_id and last_story_topic:
            say("Okay, I'll continue with a new part.")
            beats = generate_story_beats(last_story_topic)
            run_story(beats)
        return

    # 5) If a story is running and we get other speech, pause and ask for directive
    if current_story_id:
        cancel_ev.set()
        interrupt_choice_pending = True
        pending_question = None
        say("Oh! Do you want me to continue, stop, or switch to a different story?")
        return

    # 6) Comfort / emotional support
    if "i feel" in text_norm or "i am " in text_norm or any(w in text_norm for w in COMFORT_WORDS):
        show_emotion("QT/sad")
        play_gesture("QT/touch-head")
        reply = comfort_reply(text_raw)
        say(reply)
        return

    # 7) Story request outside any prompt
    if "story" in text_norm or "tell me a story" in text_norm:
        say("Sure! What kind of story would you like? You can say any topic.")
        awaiting_story_choice = True
        return

    # 8) If there is a pending Q&A from a story, grade the answer
    if handle_pending_question(text_raw):
        return

    # 9) General small talk
    answer = kid_safe_reply(text_raw)
    say(answer)

def main():
    """Initialise ROS publishers/subscribers and start the node."""
    global say_pub, gesture_pub, emotion_pub, asr_ctrl_pub, OLLAMA_MODEL, OLLAMA_URL
    rospy.init_node("qt_voice_story_orchestrator", anonymous=False)
    # Read parameters
    OLLAMA_MODEL = rospy.get_param("~ollama_model", OLLAMA_MODEL)
    OLLAMA_URL = rospy.get_param("~ollama_url", OLLAMA_URL)
    rospy.loginfo("Ollama model: %s", OLLAMA_MODEL)
    rospy.loginfo("Ollama URL: %s", OLLAMA_URL)
    # Setup publishers
    say_pub = rospy.Publisher('/qt_robot/behavior/talkText', String, queue_size=10)
    gesture_pub = rospy.Publisher('/qt_robot/gesture/play', String, queue_size=10)
    emotion_pub = rospy.Publisher('/qt_robot/emotion/show', String, queue_size=10)
    asr_ctrl_pub = rospy.Publisher('/qt_asr/control', String, queue_size=5)
    # Subscribe to ASR and VAD
    rospy.Subscriber('/qt_asr/text', String, on_asr)
    rospy.Subscriber('/qt_vad/active', Bool, on_vad)
    # Camera setup
    # Determine camera topic (default to color image)
    cam_topic = rospy.get_param("~camera_topic", "/camera/color/image_raw")
    # Initialise cv_bridge and Haar cascade
    global bridge, face_cascade
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
    # Only subscribe to camera if we have a cascade
    if face_cascade:
        rospy.Subscriber(cam_topic, Image, on_image, queue_size=1)
        rospy.loginfo("Listening to camera: %s", cam_topic)
    rospy.loginfo("Interactive storyteller node started.")
    rospy.spin()


if __name__ == "__main__":
    main()
