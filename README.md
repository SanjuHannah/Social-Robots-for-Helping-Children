This repository contains an interactive storytelling system developed for the QTrobot to support visually impaired and low-vision children through multi-sensory, expressive, and accessible story experiences.

The system provides two interaction modes:

1. Voice-Based Storytelling Mode

Children talk naturally to the robot.

QTrobot listens using offline VOSK ASR, responds with kid-safe dialogue, and generates stories via Ollama LLM.

Includes comfort responses (“I feel sad…”) and small-talk.

Facial detection triggers greetings.

Gestures + emotions synchronized with narration.

2. Clicker-Based Story Mode

For children who cannot rely on speech.

A tactile button (clicker) allows navigating story beats.

QTrobot reads pre-scripted stories with expressive gestures.

System Components:

Python 3 implementation using the QTrobot SDK

ROS Noetic integration

VOSK for offline speech recognition

Ollama (phi3 / qwen models) for story generation

RealSense camera for face detection

Gesture & emotion controllers for expressive output

Clicker listener node for accessibility mode

Purpose:

Designed to make storytelling inclusive for visually impaired children, while also engaging for low-vision or sighted children through gestures, sound, and social interaction.

Evaluation:

User testing conducted with college peers and sighted children using:

System Usability Scale (SUS)

Intrinsic Motivation Inventory (IMI)

Robotic Social Attributes Scale (RoSAS)

Results showed:

Strong usability

High engagement

Positive social perception of QTrobot

Repository Structure:
qt_storytelling/
 launch/
  ── voice_story.launch
  ── clicker_story.launch
 scripts/
   ── qt_voice_story_orchestrator.py
   ── qt_story_evaluation.py
   ── qt_clicker_listener.py

How to Run:
Voice Mode
roslaunch qt_storytelling voice_story.launch

Clicker Mode
roslaunch qt_storytelling clicker_story.launch

Future Work:

Pilot studies with visually impaired children

Improved FER (emotion recognition)

Multi-language storytelling

Haptic or audio-rich story augmentation
