#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qt_asr_vosk_node.py

Offline ASR for QTrobot using Vosk.

 - Uses Jabra Evolve2 30 SE (sounddevice index 25, ALSA hw:3,0).
 - Loads model from /home/qtrobot/models/vosk-en-small.
 - Publishes recognized text (final results only) to /qt_asr/text.
"""

import os
import json
import queue
import time
import rospy
import webrtcvad
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from std_msgs.msg import String, Bool

# --- VAD / control globals ---
PAUSED = False            # set by /qt_asr/control (pause|resume)
VAD_PUB = None            # /qt_vad/active publisher

# VAD tuning
VAD_AGGRESSIVENESS = 3    # 0..3 (3=most aggressive)
VAD_FRAME_MS = 20         # 10, 20, or 30 ms
VAD_HANG_MS  = 200        # keep VAD active this long after last voiced frame


def _on_asr_ctrl(msg):
    global PAUSED
    cmd = (msg.data or "").strip().lower()
    if cmd == "pause":
        PAUSED = True
        rospy.loginfo("ASR mic: PAUSED")
    elif cmd == "resume":
        PAUSED = False
        rospy.loginfo("ASR mic: RESUMED")


def main():
    rospy.init_node("qt_asr_vosk", anonymous=False)

    device    = int(rospy.get_param("~input_device", 25))  # from `python3 -m sounddevice`
    rate      = int(rospy.get_param("~rate", 16000))
    model_dir = rospy.get_param("~model_dir", "/home/qtrobot/models/vosk-en-small")

    if not os.path.isdir(model_dir):
        rospy.logerr("Vosk model dir not found: %s", model_dir)
        return

    rospy.loginfo("Loading Vosk model from %s", model_dir)
    model = Model(model_dir)
    rec = KaldiRecognizer(model, rate)
    rec.SetWords(True)

    pub = rospy.Publisher("/qt_asr/text", String, queue_size=10)

    global VAD_PUB
    VAD_PUB = rospy.Publisher('/qt_vad/active', Bool, queue_size=5)
    rospy.Subscriber('/qt_asr/control', String, _on_asr_ctrl)

    # --- Init WebRTC VAD (needs 'rate' so it must be inside main) ---
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_bytes = int(rate * (VAD_FRAME_MS / 1000.0)) * 2  # 16-bit mono
    vad_hang_until = 0.0

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            rospy.logwarn("ASR audio status: %s", status)
        audio_q.put(bytes(indata))

    try:
        stream = sd.RawInputStream(
            samplerate=rate,
            blocksize=8000,
            device=device,
            dtype='int16',
            channels=1,          # Jabra input = 1 channel
            callback=audio_callback
        )
    except Exception as e:
        rospy.logerr("Failed to open audio device %s: %s", str(device), e)
        return

    rospy.loginfo("Vosk ASR using device %s at %d Hz", str(device), rate)
    rospy.loginfo("Publishing recognized text to /qt_asr/text")

    with stream:
        while not rospy.is_shutdown():
            try:
                data = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # --- PAUSE: drain audio but don't feed recognizer ---
            if PAUSED:
                if VAD_PUB:
                    VAD_PUB.publish(False)
                rec.Reset()
                continue

            # --- WebRTC VAD over 20 ms sub-frames ---
            is_voiced = False
            if len(data) >= frame_bytes:
                # iterate over 20 ms chunks; if any chunk is speech -> voiced
                for i in range(0, len(data) - frame_bytes + 1, frame_bytes):
                    frame = data[i:i+frame_bytes]
                    try:
                        if vad.is_speech(frame, rate):
                            is_voiced = True
                            break
                    except Exception:
                        # if VAD throws on a weird frame, just treat as unvoiced
                        pass

            now = time.time()
            if is_voiced:
                vad_hang_until = now + (VAD_HANG_MS / 1000.0)

            vad_active = (now < vad_hang_until)
            if VAD_PUB:
                VAD_PUB.publish(vad_active)
             # Optional debug:
             # rospy.logdebug("VAD active: %s", vad_active)

            # --- Feed recognizer (final-only) ---
            if rec.AcceptWaveform(data):
                res = rec.Result()
                try:
                    j = json.loads(res)
                    txt = (j.get("text") or "").strip()
                    if txt:
                        rospy.loginfo("ASR: %s", txt)
                        pub.publish(txt)
                except Exception as e:
                    rospy.logwarn("ASR parse error: %s", e)
            else:
                # ignoring partials keeps it stable
                pass


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

