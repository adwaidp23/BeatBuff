"""
BeatBuff - Local Playlist Edition
Plays local songs from songs/<genre>/* using VLC.
Refined gesture controls + HUD overlay.
"""

import os
import time
import math
import random
import cv2
import numpy as np
import vlc
import mediapipe as mp

# --- CONFIG ---
# Path to VLC install (change if needed)
VLC_PATH = r"C:\Users\Public\Desktop\VLC"  # update if necessary

# Where your local music is stored (organized by genre folders)
MUSIC_DIR = os.path.join(os.getcwd(), "songs")

# UI & behavior settings
FRAME_W, FRAME_H = 640, 480
GESTURE_HOLD = 0.5   # seconds to hold gesture to confirm
RECOMMEND_COUNT = 4

# --- Setup VLC (ensure lib path is added before importing vlc) ---
if os.path.isdir(VLC_PATH):
    os.add_dll_directory(VLC_PATH)
import vlc

# --- Globals ---
player = vlc.MediaPlayer()
volume = 70
player.audio_set_volume(volume)
status = "Idle"
current_index = 0
playlist = []          # list of (filepath, genre, title)
genre_map = {}         # genre -> list of indices in playlist
current_genre = None
current_song = None

# --- Build local playlist by scanning MUSIC_DIR ---
def build_playlist(root_dir):
    pl = []
    gmap = {}
    idx = 0
    for root, dirs, files in os.walk(root_dir):
        # skip root itself if it's songs/ and contains subfolders; if there are mp3 directly in songs/ they will be included with genre = "":
        rel = os.path.relpath(root, root_dir)
        genre = rel.replace("\\", "/") if rel != "." else ""
        for f in sorted(files):
            if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")):
                path = os.path.join(root, f)
                title = os.path.splitext(f)[0]
                pl.append((path, genre, title))
                gmap.setdefault(genre, []).append(idx)
                idx += 1
    return pl, gmap

playlist, genre_map = build_playlist(MUSIC_DIR)
if not playlist:
    print("No tracks found in", MUSIC_DIR)
    print("Place music files under songs/<genre>/yourfiles.mp3 and re-run.")
    raise SystemExit

# start at first track
current_index = 0
current_song = playlist[current_index][2]
current_genre = playlist[current_index][1] or ""

# --- utility functions ---
def play_index(i):
    global player, current_index, current_song, current_genre, status
    if i < 0 or i >= len(playlist):
        return
    filepath, genre, title = playlist[i]
    if player.is_playing():
        player.stop()
        time.sleep(0.1)
    media = vlc.Media(filepath)
    player.set_media(media)
    player.audio_set_volume(volume)
    # try play and allow small buffering
    player.play()
    current_index = i
    current_song = title
    current_genre = genre
    status = f"Playing: {title}"
    print(status)

def play_next():
    global current_index
    next_idx = (current_index + 1) % len(playlist)
    play_index(next_idx)

def play_prev():
    global current_index
    prev_idx = (current_index - 1) % len(playlist)
    play_index(prev_idx)

def play_recommendation():
    # pick songs from same genre if available, else choose random from playlist
    if current_genre and current_genre in genre_map and len(genre_map[current_genre]) > 1:
        candidates = [i for i in genre_map[current_genre] if i != current_index]
        if candidates:
            chosen = random.choice(candidates)
            play_index(chosen)
            return
    # fallback: random different song
    candidates = list(range(len(playlist)))
    candidates.remove(current_index)
    if candidates:
        play_index(random.choice(candidates))

# --- Mediapipe / gesture detection ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65, min_tracking_confidence=0.6)

def landmark_distance(a, b, w, h):
    x1, y1 = int(a.x * w), int(a.y * h)
    x2, y2 = int(b.x * w), int(b.y * h)
    return math.hypot(x2-x1, y2-y1)

def detect_simple_gesture(hand, w, h):
    # returns one of: open, fist, next, prev, vol_up, vol_down, recommend, none
    tips = [4,8,12,16,20]
    fingers = []
    # thumb (use x compare)
    fingers.append(1 if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x else 0)
    # other fingers (tip.y < pip.y)
    for i in range(1,5):
        fingers.append(1 if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y else 0)
    # commonly used patterns:
    total = sum(fingers)
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    pinky_tip = hand.landmark[20]

    # Open palm
    if total == 5:
        return "open"
    # Fist
    if total == 0:
        return "fist"
    # Index only -> Next
    if fingers == [0,1,0,0,0]:
        return "next"
    # Index+middle -> Prev
    if fingers == [0,1,1,0,0]:
        return "prev"
    # Rock-on -> volume up (thumb,index,pinky)
    if fingers == [1,1,0,0,1]:
        return "vol_up"
    # Horns (index+pinky) -> volume down
    if fingers == [0,1,0,0,1]:
        return "vol_down"
    # Pinch (thumb close to index) -> recommend
    if fingers[0] == 1 and fingers[1] == 1:
        if landmark_distance(thumb_tip, index_tip, w, h) < 40:
            return "recommend"
    return "none"

# --- HUD drawing functions ---
def draw_hud(img, status_text, vol, song_title, genre, recs):
    h, w, _ = img.shape
    # translucent top bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (w, 110), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    # title / status
    cv2.putText(img, "BeatBuff (Local)", (16,32), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,200,20), 2)
    cv2.putText(img, status_text, (16,68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, f"{song_title or 'No track'}", (16,96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
    # bottom HUD
    cv2.rectangle(overlay, (0, h-90), (w, h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    # volume bar
    bar_x, bar_y, bar_w, bar_h = 20, h-60, 300, 18
    fill = int((vol/100.0) * bar_w)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (70,70,70), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), (0,200,0), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 1)
    cv2.putText(img, f"Vol: {vol}%", (bar_x+bar_w+15, bar_y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    # recommendations right side
    rec_x = w - 360
    cv2.putText(img, "Recommended:", (rec_x, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    y = h-60
    for r in recs[:RECOMMEND_COUNT]:
        cv2.putText(img, f"• {r}", (rec_x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,0), 1)
        y += 22
    # footer instruction
    cv2.putText(img, "Hold gesture ~0.6s to confirm. Press ESC to exit.", (16, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return img

# Build initial recommendations (first RECOMMEND_COUNT titles)
def get_recommendation_titles(genre):
    recs = []
    if genre and genre in genre_map:
        for i in genre_map[genre][:RECOMMEND_COUNT]:
            recs.append(playlist[i][2])
    else:
        for p in playlist[:RECOMMEND_COUNT]:
            recs.append(p[2])
    return recs

# --- Main loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# start first track automatically (optional)
play_index(current_index)

last_gesture = "none"
gesture_start_time = 0
last_action_time = 0
recommendations = get_recommendation_titles(current_genre)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # reduce processing by skipping frames occasionally if needed (uncomment if slow)
    # Use full speed first; if lag observed, skip processing on odd frames:
    # frame_id += 1
    # if frame_id % 2 == 1: show frame and continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture = "none"
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = detect_simple_gesture(handLms, w, h)

    now = time.time()
    # detect stable hold
    if gesture != last_gesture:
        gesture_start_time = now
    elif gesture != "none" and (now - gesture_start_time) >= GESTURE_HOLD and (now - last_action_time) > 0.8:
        # confirm gesture
        if gesture == "open":
            # play/resume
            try:
                player.play()
                status = f"Playing: {playlist[current_index][2]}"
            except:
                pass
        elif gesture == "fist":
            player.pause()
            status = "Paused"
        elif gesture == "next":
            play_next()
            status = f"Next: {playlist[current_index][2]}"
            recommendations = get_recommendation_titles(current_genre)
        elif gesture == "prev":
            play_prev()
            status = f"Prev: {playlist[current_index][2]}"
            recommendations = get_recommendation_titles(current_genre)
        elif gesture == "vol_up":
            volume = min(100, volume + 10)
            player.audio_set_volume(volume)
            status = f"Volume: {volume}%"
        elif gesture == "vol_down":
            volume = max(0, volume - 10)
            player.audio_set_volume(volume)
            status = f"Volume: {volume}%"
        elif gesture == "recommend":
            play_recommendation()
            status = f"Recommended: {playlist[current_index][2]}"
            recommendations = get_recommendation_titles(current_genre)
        last_action_time = now

    last_gesture = gesture

    # auto-refresh recommendations every 30s
    if time.time() - last_action_time > 30:
        recommendations = get_recommendation_titles(current_genre)

    # draw HUD
    frame = draw_hud(frame, status, volume, current_song, current_genre, recommendations)
    cv2.imshow("BeatBuff - Local Playlist", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
player.stop()
