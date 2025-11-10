"""
BeatBuff - Local Enhanced UI Edition 🎧
Gesture-controlled music player with polished HUD overlay and local playlist support.
"""

import os
import cv2
import math
import time
import random
import numpy as np
import vlc
import mediapipe as mp

# --- CONFIG ---
VLC_PATH = r"C:\Program Files\VideoLAN\VLC"
MUSIC_DIR = os.path.join(os.getcwd(), "songs")
GESTURE_HOLD = 0.5
FRAME_W, FRAME_H = 960, 720  # higher res for better UI look

if os.path.isdir(VLC_PATH):
    os.add_dll_directory(VLC_PATH)
import vlc

# --- Globals ---
player = vlc.MediaPlayer()
volume = 70
player.audio_set_volume(volume)
status = "Idle"
current_song = None
current_genre = None
current_index = 0
playlist = []
genre_map = {}

banner_text = ""
banner_time = 0
banner_duration = 1.5

# --- Build playlist ---
def build_playlist(root):
    pl, gm = [], {}
    idx = 0
    for rootdir, _, files in os.walk(root):
        rel = os.path.relpath(rootdir, root)
        genre = rel if rel != "." else "General"
        for f in files:
            if f.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg")):
                path = os.path.join(rootdir, f)
                title = os.path.splitext(f)[0]
                pl.append((path, genre, title))
                gm.setdefault(genre, []).append(idx)
                idx += 1
    return pl, gm

playlist, genre_map = build_playlist(MUSIC_DIR)
if not playlist:
    raise SystemExit("❌ No music found in /songs/. Please add mp3 files.")

# --- Core Player ---
def play_index(i):
    global player, current_index, current_song, current_genre, status, banner_text, banner_time
    filepath, genre, title = playlist[i]
    if player.is_playing():
        player.stop()
    media = vlc.Media(filepath)
    player.set_media(media)
    player.play()
    player.audio_set_volume(volume)
    current_index, current_song, current_genre = i, title, genre
    status = f"▶ {title}"
    banner_text, banner_time = f"▶ Now Playing: {title}", time.time()

def play_next():
    global current_index
    current_index = (current_index + 1) % len(playlist)
    play_index(current_index)

def play_prev():
    global current_index
    current_index = (current_index - 1) % len(playlist)
    play_index(current_index)

def play_recommendation():
    global current_index
    if current_genre and current_genre in genre_map and len(genre_map[current_genre]) > 1:
        other = random.choice([i for i in genre_map[current_genre] if i != current_index])
        play_index(other)
    else:
        play_next()

# --- Gesture Detection ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def finger_pattern(hand, w, h):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x else 0)
    for i in range(1, 5):
        fingers.append(1 if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y else 0)
    return fingers

def detect_gesture(hand, w, h):
    fingers = finger_pattern(hand, w, h)
    total = sum(fingers)
    thumb, index, middle, ring, pinky = fingers
    dist = math.hypot(int(hand.landmark[4].x*w)-int(hand.landmark[8].x*w),
                      int(hand.landmark[4].y*h)-int(hand.landmark[8].y*h))
    if total == 5: return "open"
    if total == 0: return "fist"
    if fingers == [0,1,0,0,0]: return "next"
    if fingers == [0,1,1,0,0]: return "prev"
    if fingers == [1,1,0,0,1]: return "vol_up"
    if fingers == [0,1,0,0,1]: return "vol_down"
    if thumb and index and dist < 40: return "recommend"
    return "none"

# --- Enhanced HUD ---
def draw_card(img, song, genre):
    """Draw 'Now Playing' card with cover image if available"""
    x, y, w, h = 20, 20, 300, 100
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)
    cover_path = os.path.join(MUSIC_DIR, genre, "cover.jpg")
    if os.path.exists(cover_path):
        cover = cv2.imread(cover_path)
        cover = cv2.resize(cover, (h-10, h-10))
        img[y+5:y+h-5, x+5:x+h-5] = cover
    cv2.putText(img, f"{song}", (x+h+10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(img, f"Genre: {genre}", (x+h+10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

def draw_volume_bar(img, vol):
    h, w, _ = img.shape
    bar_x, bar_y, bar_w, bar_h = 20, h-60, 400, 22
    fill = int((vol/100)*bar_w)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), (0,255,0), -1)
    cv2.putText(img, f"Volume: {vol}%", (bar_x, bar_y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

def draw_banner(img, text):
    """Top floating banner"""
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (img.shape[1],80), (0,200,255), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.putText(img, text, (40,50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 3)

# --- Camera loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

play_index(0)
last_gesture = "none"
gesture_start = 0
last_action = 0

while True:
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture = "none"
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand, w, h)

    now = time.time()
    if gesture != last_gesture:
        gesture_start = now
    elif gesture != "none" and (now - gesture_start) > GESTURE_HOLD and (now - last_action) > 1:
        if gesture == "open":
            player.play()
            status, banner_text = "▶ Play", "▶ Playing"
        elif gesture == "fist":
            player.pause()
            status, banner_text = "⏸ Pause", "⏸ Paused"
        elif gesture == "next":
            play_next()
            banner_text = "⏭ Next Track"
        elif gesture == "prev":
            play_prev()
            banner_text = "⏮ Previous Track"
        elif gesture == "vol_up":
            volume = min(100, volume + 10)
            player.audio_set_volume(volume)
            banner_text = f"🔊 Volume {volume}%"
        elif gesture == "vol_down":
            volume = max(0, volume - 10)
            player.audio_set_volume(volume)
            banner_text = f"🔉 Volume {volume}%"
        elif gesture == "recommend":
            play_recommendation()
            banner_text = "🎵 Recommended Song"
        last_action = now
        banner_time = now

    last_gesture = gesture

    # --- Draw UI ---
    draw_card(img, current_song or "No Track", current_genre or "General")
    draw_volume_bar(img, volume)

    # Temporary banner for actions
    if time.time() - banner_time < banner_duration and banner_text:
        draw_banner(img, banner_text)

    cv2.imshow("BeatBuff 🎧 Enhanced UI", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
player.stop()
