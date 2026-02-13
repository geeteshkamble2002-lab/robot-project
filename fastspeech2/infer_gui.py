"""Launch FastSpeech2 TTS Gradio UI. Run from project root: python infer_gui.py"""
import sys
import os
# Ensure we can import infer (project root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import launch_gui
launch_gui()
