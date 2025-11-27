#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qt_clicker_story_orchestrator.py - Refined

QT robot interactive storyteller controlled ONLY by a presentation clicker.
Refactored for non-blocking speech to prevent ROS callback queue starvation/crash.
"""

import os
import re
import time
import threading
from typing import Optional, Dict

import requests
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Try gesture service; if missing, fall back to topic
try:
    from qt_gesture_controller.srv import gesture_play, gesture_playRequest
    HAVE_GESTURE_SRV = True
except Exception:
    HAVE_GESTURE_SRV = False

# ---------------------- Ollama config ----------------------

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"    # override with ~ollama_model param if needed
OLLAMA_TIMEOUT = 25.0

# ---------------------- Text cleaning ----------------------

EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

def clean_for_speech(text: str) -> str:
    if not text:
        return ""
    text = EMOJI_RE.sub("", text)
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) <= 126)
    return text.strip()

# ---------------------- Globals: ROS pubs/subs ----------------------

say_pub: Optional[rospy.Publisher] = None
gesture_pub: Optional[rospy.Publisher] = None
emotion_pub: Optional[rospy.Publisher] = None

bridge: Optional[CvBridge] = None
face_cascade: Optional[cv2.CascadeClassifier] = None

# Threading state for non-blocking speech
say_lock = threading.Lock()
is_speaking = False
say_thread: Optional[threading.Thread] = None # NEW: Track the speaking thread

# ---------------------- Face / presence state ----------------------

last_face_ts: float = 0.0
greeted_recently: bool = False
last_greet_spoken: float = 0.0
face_seen_this_story: bool = False

DETECT_MIN_FACE_AREA   = 40
GREET_RISING_EDGE_SEC  = 2.0
GREET_RESET_SEC        = 6.0
GREET_COOLDOWN         = 20.0
FACE_TIMEOUT_SEC       = 45.0

# ---------------------- Clicker & flow state ----------------------

# High-level mode flags
awaiting_main_choice  = False  # story vs chat
awaiting_story_type   = False  # funny / magic / adventure
awaiting_another      = False  # another story? yes/no

current_story_style: Optional[str] = None  # "funny" | "magic" | "adventure" | None

# Question after story
pending_question: Optional[Dict[str, any]] = None

# For adventure (double-next) vs magic (single-next)
_first_next_ts: float = 0.0
_next_single_pending: bool = False
_next_timer: Optional[threading.Timer] = None

# Triple-next to stop
last_next_ts: float = 0.0
next_press_count: int = 0

# Cancel current story run
cancel_ev = threading.Event()

# ---------------------- Basic robot helpers ----------------------

def _speak_and_wait(text: str):
    """
    NEW: This runs in a separate thread and contains the blocking wait.
    It simulates the speech duration for the QT robot.
    """
    global is_speaking
    
    # 1. Acquire lock and set speaking flag
    if not say_lock.acquire(blocking=False):
        rospy.logwarn("[QT] say skip: another speech is running.")
        return
    is_speaking = True

    try:
        # 2. Publish the text
        say_pub.publish(text)
        rospy.loginfo("[QT] say: %s", text)
        
        # 3. Blocking wait (safe inside this dedicated thread)
        words = max(1, len(text.split()))
        dur = max(0.8, min(3.0, 0.25 * words))
        end = time.time() + dur
        
        # Wait while checking for cancel or shutdown flags
        while time.time() < end and not rospy.is_shutdown() and not cancel_ev.is_set():
            time.sleep(0.03)

    except Exception as e:
        rospy.logerr("Speech thread error: %s", str(e))
        
    finally:
        # 4. Release lock and clear speaking flag
        is_speaking = False
        say_lock.release()

def say(text: str):
    """
    Speak text via QT. Starts a new thread and returns immediately,
    preventing the caller (like ROS callbacks) from blocking.
    """
    global say_thread
    
    text = clean_for_speech(text)
    if not text:
        return
    
    # Only allow starting a new speech thread if the current one is finished
    if say_thread is not None and say_thread.is_alive():
        # If we are interrupting, or if the system is overloaded, log it
        rospy.logwarn("[QT] Skipping speech: previous thread is still running.")
        return

    # Start the new thread
    say_thread = threading.Thread(target=_speak_and_wait, args=(text,))
    say_thread.start()

def wait_for_speech_to_finish(timeout: float = 60.0):
    """Wait for the currently active speech thread to finish."""
    global say_thread
    if say_thread is not None and say_thread.is_alive():
        say_thread.join(timeout=timeout)
    
def play_gesture(name: str, speed: float = 1.0):
    # ... (No change) ...
    if not name:
        return
    if HAVE_GESTURE_SRV:
        try:
            rospy.wait_for_service('/qt_robot/gesture/play', timeout=0.3)
            cli = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
            cli(gesture_playRequest(name=name, speed=speed))
            return
        except Exception:
            pass
    if gesture_pub is not None:
        try:
            gesture_pub.publish(name)
        except Exception:
            pass

def show_emotion(name: str):
    # ... (No change) ...
    if not name or emotion_pub is None:
        return
    try:
        emotion_pub.publish(name)
    except Exception:
        pass

# ---------------------- Ollama helpers ----------------------
# (No changes needed in Ollama functions)

def call_ollama(prompt: str, temperature: float = 0.6, top_p: float = 0.9) -> str:
    # ... (Function body remains the same) ...
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "top_p": top_p},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return (data.get("response") or "").strip()
            return str(data)
        rospy.logwarn("Ollama HTTP %d: %s", r.status_code, r.text[:120])
    except Exception as e:
        rospy.logwarn("Ollama error: %s", str(e))
    return "I'm thinking about that."

def build_story_prompt(style: str) -> str:
    # ... (Function body remains the same) ...
    style = style.lower()
    if style == "funny":
        mood = (
            "Tell a very short, FUNNY story for a child aged 5–10. "
            "Use silly things that are not scary. Maybe the robot or a cake does something silly. "
        )
    elif style == "magic":
        mood = (
            "Tell a short, MAGICAL story for a child aged 5–10. "
            "Use gentle, kind magic like glowing forests, friendly dragons, or stars. "
        )
    else:  # adventure
        mood = (
            "Tell a short, ADVENTURE story for a child aged 5–10. "
            "The hero is brave but safe. No real danger, just exciting exploring. "
        )
    return (
        "You are a friendly kids robot telling a story in English.\n"
        "Rules:\n"
        "- Audience is a child aged 5–10.\n"
        "- Use only simple, cheerful language.\n"
        "- No violence, no scary monsters, no injuries.\n"
        "- No brand names, no politics, no medical advice.\n"
        "- Write 6–8 short sentences, around 80–150 words total.\n"
        "- Do NOT ask the child questions and do NOT say 'The end'.\n"
        "- Do NOT use emojis.\n\n"
        f"{mood}"
        "Write the story now as plain text only."
    )

def generate_story_text(style: str) -> str:
    prompt = build_story_prompt(style)
    raw = call_ollama(prompt)
    return clean_for_speech(raw)

def split_story_into_chunks(text: str):
    # ... (Function body remains the same) ...
    if not text:
        return []
    parts = re.split(r"([.!?])", text)
    chunks = []
    current = ""
    for part in parts:
        if part in ".!?":
            current += part
            if current.strip():
                chunks.append(current.strip())
            current = ""
        else:
            t = part.strip()
            if not t:
                continue
            if current:
                current += " " + t
            else:
                current = t
    if current.strip():
        chunks.append(current.strip())

    final = []
    for ch in chunks:
        words = ch.split()
        while len(words) > 12:
            final.append(" ".join(words[:8]))
            words = words[8:]
        if words:
            final.append(" ".join(words))
    return final

# ---------------------- Question / evaluation ----------------------

def start_click_question(style: str):
    # ... (Function body remains the same) ...
    global pending_question

    style = (style or "").lower()
    if style == "funny":
        q_text = (
            "Which sentence is correct? "
            "Option one: 'The cake was good.' "
            "Option two: 'The cake were good.' "
            "Press Previous for option one, Next for option two."
        )
        correct_button = "prev"
        correct_say   = "Yes, we say 'The cake was good.' Great job!"
        wrong_say     = "Nice try. We say 'The cake was good.'"
    elif style == "magic":
        q_text = (
            "Which sentence sounds better? "
            "Option one: 'We used kind magic.' "
            "Option two: 'We use kind magic yesterday.' "
            "Press Previous for option one, Next for option two."
        )
        correct_button = "prev"
        correct_say   = "Right, 'We used kind magic.' sounds correct!"
        wrong_say     = "Good try. We say 'We used kind magic.'"
    else:  # adventure
        q_text = (
            "Which sentence is correct? "
            "Option one: 'The hero was brave.' "
            "Option two: 'The hero were brave.' "
            "Press Previous for option one, Next for option two."
        )
        correct_button = "prev"
        correct_say   = "Yes! 'The hero was brave.' is correct."
        wrong_say     = "Nice try. We say 'The hero was brave.'"

    pending_question = {
        "correct_button": correct_button,
        "correct_say": correct_say,
        "wrong_say": wrong_say,
        "answered": False,
    }

    show_emotion("QT/thinking")
    play_gesture("QT/listen")
    say(q_text)
    wait_for_speech_to_finish() # NEW: Wait for the question to be fully spoken

def wait_for_question_answer():
    """Block until the current pending_question is answered or cancelled."""
    global pending_question
    while not rospy.is_shutdown() and not cancel_ev.is_set():
        if pending_question is None:
            return
        if pending_question.get("answered", False):
            pending_question = None
            return
        rospy.sleep(0.05)

def ask_another_story():
    # ... (Function body remains the same) ...
    global awaiting_another
    awaiting_another = True
    show_emotion("QT/happy")
    play_gesture("QT/hi")
    say("Would you like another story? Press Previous for yes, Next for no.")
    wait_for_speech_to_finish() # NEW: Wait for the question to be fully spoken

# ---------------------- Story running ----------------------

# ---------------------- Story running ----------------------

def run_story(style: str):
    """Generate a story of given style and narrate it."""
    global current_story_style, face_seen_this_story
    current_story_style = style
    face_seen_this_story = False
    cancel_ev.clear()
    
    last_face_ts = rospy.Time.now().to_sec()

    # Generate
    show_emotion("QT/thinking")
    play_gesture("QT/touch-head")
    say(f"I will think of a short {style} story.")
    wait_for_speech_to_finish() # NEW: Wait for generation intro

    story_text = generate_story_text(style)
    chunks = split_story_into_chunks(story_text)

    show_emotion("QT/happy")
    play_gesture("QT/hi")

    for ch in chunks:
        if cancel_ev.is_set() or rospy.is_shutdown():
            break
        
        # We must wait for it to finish before sending the next chunk.
        say(ch)
        wait_for_speech_to_finish() # NEW: Waits for the *speech thread* to finish.
        
        # tiny gap (still okay to sleep a little here, as we are in the run_story thread)
        end = time.time() + 0.4
        while time.time() < end and not cancel_ev.is_set() and not rospy.is_shutdown():
            time.sleep(0.03)

    if cancel_ev.is_set():
        current_story_style = None
        return

    # 🎯 FIX: Store the current face-seen status before the Q&A section
    original_face_seen = face_seen_this_story
    
    # 🎯 FIX: Temporarily suppress auto-stop during the question phase.
    # We set it to False so the face-timeout in on_image() doesn't fire 
    # (it only fires if face_seen_this_story is True).
    face_seen_this_story = False 
    
    # Ask one question at the end
    # start_click_question handles the saying and waiting itself.
    start_click_question(style)
    
    # wait_for_question_answer will block until the clicker handles the answer.
    wait_for_question_answer()

    if cancel_ev.is_set():
        current_story_style = None
        # Restore the flag if the story was cancelled during the question phase
        face_seen_this_story = original_face_seen 
        return

    # 🎯 FIX: Restore the face-seen status after the question is complete
    face_seen_this_story = original_face_seen 

    # Ask another story is called from the clicker handler after the question is answered
    return

# ---------------------- Clicker handling ----------------------

def stop_all():
    """Stop any current story/question and reset state."""
    global awaiting_main_choice, awaiting_story_type, awaiting_another
    global pending_question, current_story_style, say_thread
    global _next_single_pending, _first_next_ts, _next_timer
    
    cancel_ev.set()
    
    # If a speech thread is active, try to wait briefly for it to stop
    if say_thread is not None and say_thread.is_alive():
         say_thread.join(timeout=0.5)

    awaiting_main_choice = False
    awaiting_story_type   = False
    awaiting_another     = False
    current_story_style  = None
    pending_question     = None
    _next_single_pending = False
    _first_next_ts = 0.0
    if _next_timer is not None:
        try:
            _next_timer.cancel()
        except Exception:
            pass
            
def _commit_magic_if_pending():
    """Timer callback: if single-next still pending, choose MAGIC story."""
    global _next_single_pending, awaiting_story_type
    if rospy.is_shutdown() or cancel_ev.is_set():
        return
    if _next_single_pending and awaiting_story_type and current_story_style is None:
        _next_single_pending = False
        awaiting_story_type = False
        show_emotion("QT/happy")
        # Since this is a thread, we must wrap run_story in another thread 
        # to prevent blocking this Timer's execution thread.
        threading.Thread(target=run_story, args=("magic",)).start()

def on_clicker(msg: String):
    """
    Main clicker handler.
    """
    global awaiting_main_choice, awaiting_story_type, awaiting_another
    global pending_question, _first_next_ts, _next_single_pending, _next_timer
    global last_next_ts, next_press_count
    global current_story_style
    # All 'global' declarations MUST be at the top of the function. 
    # The global declaration for pending_question was missing from the top.

    btn = (msg.data or "").strip().lower()
    if btn not in ("prev", "next"):
        return
        
    now = time.time()   
    # ... rest of the function ...
    
    # All long-running logic (like run_story) MUST be moved to a new thread 
    # when called from this callback to avoid crashing the node.

    # ----- Triple-NEXT to stop everything -----
    if btn == "next":
        if now - last_next_ts < 1.2:
            next_press_count += 1
        else:
            next_press_count = 1
        last_next_ts = now

        if next_press_count >= 3:
            stop_all()
            show_emotion("QT/calm")
            play_gesture("QT/bye-bye")
            # NOTE: We can't wait for speech here, or the callback blocks.
            say("Okay, I'll stop. Thank you for playing with me.")
            next_press_count = 0
            return
    else:
        next_press_count = 0

    # ----- 1) If a question is pending, treat PREV/NEXT as answers -----
    if pending_question is not None:
         # ... (Answer logic remains the same) ...
        correct_button = pending_question.get("correct_button", "prev")
        if btn == correct_button:
            show_emotion("QT/happy")
            say(pending_question.get("correct_say", "That's right!"))
        else:
            show_emotion("QT/sad")
            say(pending_question.get("wrong_say", "Nice try."))

        pending_question["answered"] = True
        pending_question = None
        current_story_style = None

        # Move to "another story?" menu
        ask_another_story()
        return

    # ----- 2) 'Another story?' yes/no -----
    if awaiting_another:
        awaiting_another = False
        if btn == "prev":
            # Start the setup for story choice, non-blocking
            show_emotion("QT/happy")
            say("Yay! Let's choose another story.")
            # We must wait for speech before setting the next state, otherwise 
            # we might process another click while robot is talking.
            # To avoid blocking the callback, we run the setup in a thread.
            threading.Thread(target=_story_type_setup).start()
        else:  # next = no
            show_emotion("QT/calm")
            play_gesture("QT/bye-bye")
            say("Okay, maybe later!")
        return

    # ----- 3) Main menu: story vs chat -----
    if awaiting_main_choice:
        awaiting_main_choice = False
        if btn == "prev":
            # Start the setup for story choice, non-blocking
            show_emotion("QT/happy")
            say("Great! Let's choose a story type.")
            threading.Thread(target=_story_type_setup).start()
        else:
            # simple chat placeholder
            show_emotion("QT/happy")
            play_gesture("QT/hi")
            say(
                "We can have a little robot chat later. "
                "For now, you can press Previous if you want a story."
            )
        return

    # ----- 4) Choosing story type -----
    if awaiting_story_type and current_story_style is None:
        if btn == "prev":
            # FUNNY story (start story generation in a NEW thread)
            awaiting_story_type = False
            _next_single_pending = False
            if _next_timer is not None:
                try:
                    _next_timer.cancel()
                except Exception:
                    pass
            show_emotion("QT/happy")
            say("A funny story is coming up!")
            threading.Thread(target=run_story, args=("funny",)).start()
            return

        if btn == "next":
            # Check for second 'next' (adventure)
            if _next_single_pending and (now - _first_next_ts) < 1.0:
                # double NEXT -> adventure (start story generation in a NEW thread)
                _next_single_pending = False
                if _next_timer is not None:
                    try:
                        _next_timer.cancel()
                    except Exception:
                        pass
                awaiting_story_type = False
                show_emotion("QT/happy")
                say("An adventure story is coming up!")
                threading.Thread(target=run_story, args=("adventure",)).start()
                return
            else:
                # first NEXT: might become magic if no second click arrives
                _next_single_pending = True
                _first_next_ts = now
                if _next_timer is not None:
                    try:
                        _next_timer.cancel()
                    except Exception:
                        pass
                # The timer callback (_commit_magic_if_pending) handles the thread start.
                _next_timer = threading.Timer(1.0, _commit_magic_if_pending)
                _next_timer.start()
                return

    # ----- 5) Clicks outside any waiting state: ignore -----
    return

def _story_type_setup():
    """Helper to set up story choice menu (must be called in a thread)."""
    global awaiting_story_type, _first_next_ts, _next_single_pending
    
    # Wait for the introductory speech to finish before asking for choice
    wait_for_speech_to_finish() 
    
    awaiting_story_type = True
    _first_next_ts = 0.0
    _next_single_pending = False
    say(
        "For a funny story press Previous. "
        "For a magic story press Next. "
        "For an adventure story press Next two times quickly."
    )

# ---------------------- Camera / face detection ----------------------

def _find_haar_path() -> Optional[str]:
    # ... (Function body remains the same) ...
    candidates = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def on_image(msg: Image):
    """
    Detect face presence and handle greetings / auto-stop when child leaves.
    **Runs in a ROS callback thread, MUST RETURN QUICKLY.**
    """
    global last_face_ts, greeted_recently, last_greet_spoken, face_seen_this_story
    global awaiting_main_choice

    if bridge is None or face_cascade is None:
        return
    # ... (Image processing and face detection logic remains the same) ...
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
        faces = []

    now = rospy.Time.now().to_sec()

    if len(faces) > 0:
        prev_ts = last_face_ts
        last_face_ts = now

        # greeting when face appears after a while
        if (prev_ts == 0.0 or (now - prev_ts) > GREET_RISING_EDGE_SEC) \
           and not greeted_recently \
           and current_story_style is None \
           and (time.time() - last_greet_spoken) > GREET_COOLDOWN:
            show_emotion("QT/happy")
            play_gesture("QT/hi")
            # Start greeting in a new thread to avoid blocking the image callback
            threading.Thread(target=_handle_greeting_and_menu).start()
            
        if current_story_style is not None:
            face_seen_this_story = True
    else:
        # no faces
        if (now - last_face_ts) > GREET_RESET_SEC:
            greeted_recently = False

        # auto-stop if child leaves in middle of story
        if current_story_style is not None and face_seen_this_story:
            no_face_for = now - last_face_ts
            if no_face_for > FACE_TIMEOUT_SEC and not cancel_ev.is_set():
                # Start stop logic in a new thread
                threading.Thread(target=_handle_autostop).start()

def _handle_greeting_and_menu():
    """Helper to handle greeting (must be called in a thread)."""
    global greeted_recently, last_greet_spoken, awaiting_main_choice
    
    # This logic can be long, so it's safely in a new thread
    say(
        "Hi there! Do you want me to tell a story or have a chat with me? "
        "Press Previous for a story, Next for a chat."
    )
    # Wait for the speech to finish before proceeding
    wait_for_speech_to_finish()
    
    greeted_recently = True
    last_greet_spoken = time.time()
    awaiting_main_choice = True

def _handle_autostop():
    """Helper to handle autostop (must be called in a thread)."""
    # This logic can be long, so it's safely in a new thread
    stop_all()
    show_emotion("QT/calm")
    say("Looks like you left. I will stop for now.")


# ---------------------- main() ----------------------
# (No changes needed in main setup)
def main():
    global say_pub, gesture_pub, emotion_pub, bridge, face_cascade
    global OLLAMA_MODEL, OLLAMA_URL

    rospy.init_node("qt_clicker_story_orchestrator", anonymous=False)

    # read params
    OLLAMA_MODEL = rospy.get_param("~ollama_model", OLLAMA_MODEL)
    OLLAMA_URL   = rospy.get_param("~ollama_url", OLLAMA_URL)
    cam_topic    = rospy.get_param("~camera_topic", "/camera/color/image_raw")

    rospy.loginfo("qt_clicker_story_orchestrator: Ollama model: %s", OLLAMA_MODEL)
    rospy.loginfo("qt_clicker_story_orchestrator: Ollama URL:   %s", OLLAMA_URL)

    say_pub     = rospy.Publisher('/qt_robot/behavior/talkText', String, queue_size=10)
    gesture_pub = rospy.Publisher('/qt_robot/gesture/play',      String, queue_size=10)
    emotion_pub = rospy.Publisher('/qt_robot/emotion/show',      String, queue_size=10)

    # camera / face
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

    if face_cascade is not None:
        rospy.Subscriber(cam_topic, Image, on_image, queue_size=1)
        rospy.loginfo("Listening to camera: %s", cam_topic)

    # clicker
    rospy.Subscriber('/qt_clicker/button', String, on_clicker)
    rospy.loginfo("Listening for clicker events on /qt_clicker/button")

    rospy.loginfo("Clicker-based storyteller started.")
    rospy.spin()

if __name__ == "__main__":
    main()
