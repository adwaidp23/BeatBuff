"""
BeatBuff - Lag-Free Gesture Controlled Music Player (Modern UI + Fixed Camera)
Author: ChatGPT (Polished UI Version)
"""

import os, cv2, math, time, random, numpy as np, threading
import mediapipe as mp
from pygame import mixer

# --- CONFIG ---
MUSIC_DIR = os.path.join(os.getcwd(), "songs")
GESTURE_HOLD = 0.5
FRAME_W, FRAME_H = 640, 360
MIN_CONFIDENCE = 0.8

# --- AUDIO ---
mixer.init()
volume = 70
current_song, current_genre, current_index = None, None, 0
playlist, genre_map = [], {}
banner_text, banner_time = "", 0

# --- COLORS ---
COLORS = {
    "primary": (0, 190, 255),
    "accent": (255, 140, 0),
    "card": (25, 25, 35),
    "card_alpha": 0.7,
    "text": (255, 255, 255),
    "muted": (180, 180, 180),
    "success": (0, 230, 0),
}

# --- BUILD PLAYLIST ---
def build_playlist(root):
    pl, gm = [], {}
    i = 0
    for r, _, files in os.walk(root):
        genre = os.path.basename(r) if os.path.basename(r) else "General"
        for f in files:
            if f.lower().endswith((".mp3", ".wav", ".ogg", ".flac", ".m4a")):
                path = os.path.join(r, f)
                pl.append((path, genre, os.path.splitext(f)[0]))
                gm.setdefault(genre, []).append(i)
                i += 1
    return pl, gm

def play_index(i):
    global current_song, current_genre, current_index, banner_text, banner_time
    if not playlist:
        return
    i %= len(playlist)
    path, genre, title = playlist[i]
    try:
        mixer.music.load(path)
        mixer.music.play()
        mixer.music.set_volume(volume / 100)
        current_song, current_genre, current_index = title, genre, i
        banner_text, banner_time = f"Now Playing: {title}", time.time()
    except Exception as e:
        print("Error playing:", e)

def play_next(): play_index(current_index + 1)
def play_prev(): play_index(current_index - 1)
def toggle_play_pause():
    global banner_text, banner_time
    if mixer.music.get_busy():
        mixer.music.pause()
        banner_text, banner_time = "Paused", time.time()
    else:
        mixer.music.unpause()
        banner_text, banner_time = "Playing", time.time()
def set_volume(v):
    global volume, banner_text, banner_time
    volume = max(0, min(100, v))
    mixer.music.set_volume(volume / 100)
    banner_text, banner_time = f"Volume: {volume}%", time.time()

# --- CAMERA THREAD ---
class CameraThread:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("🎥 Initializing camera...")
        for _ in range(20):
            ret, _ = self.cap.read()
            if ret: break
            time.sleep(0.1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        print("✅ Camera ready.")

    def _update(self):
        while self.running:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            f = cv2.flip(f, 1)
            with self.lock:
                self.frame = f
    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()

# --- HAND GESTURES ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=MIN_CONFIDENCE,
                       min_tracking_confidence=MIN_CONFIDENCE)
def detect_gesture(hand, w, h):
    tips = [4,8,12,16,20]
    f = []
    f.append(1 if hand.landmark[4].x > hand.landmark[3].x else 0)
    for i in range(1,5):
        f.append(1 if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y else 0)
    thumb,index,middle,ring,pinky = f
    total = sum(f)
    if total==5: return "open"
    if total==0: return "fist"
    if f==[0,1,0,0,0]: return "next"
    if f==[0,1,1,0,0]: return "prev"
    if f==[0,0,0,0,1]: return "vol_up"
    if f==[1,0,0,0,0]: return "vol_down"
    if f==[0,1,0,0,1]: return "play_pause"
    if total==2 and index and middle: return "random"
    return "none"

# --- UI ---
def draw_rounded_rect(img, pt1, pt2, color, radius=10, alpha=1.0):
    overlay = img.copy()
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def text(img, msg, pos, size=0.6, color=(255,255,255), thick=1):
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

def draw_now_playing(img):
    x,y,w,h = 25,25,350,100
    draw_rounded_rect(img,(x,y),(x+w,y+h),COLORS["card"],radius=15,alpha=COLORS["card_alpha"])
    text(img,"NOW PLAYING",(x+15,y+25),0.55,COLORS["accent"],2)
    song = current_song if current_song else "No track"
    genre = current_genre if current_genre else "General"
    text(img,song,(x+15,y+55),0.8,COLORS["text"],2)
    text(img,f"Genre: {genre}",(x+15,y+85),0.6,(200,160,255),2)

def draw_volume(img):
    h = img.shape[0]
    bar_x, bar_y, bar_w, bar_h = 25, h-80, 300, 20
    draw_rounded_rect(img,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),(60,60,60),radius=8,alpha=0.7)
    fill = int((volume/100)*bar_w)
    draw_rounded_rect(img,(bar_x,bar_y),(bar_x+fill,bar_y+bar_h),COLORS["success"],radius=8,alpha=1)
    text(img,f"Volume: {volume}%",(bar_x,bar_y-10),0.6,COLORS["text"])

def draw_controls(img):
    h,w = img.shape[:2]
    x,y,wid = w-300,25,260
    draw_rounded_rect(img,(x,y),(x+wid,y+230),COLORS["card"],radius=15,alpha=COLORS["card_alpha"])
    controls=[
        ("Playback Controls:",COLORS["accent"]),
        ("🖐 Open - Play",(255,255,255)),
        ("✊ Fist - Pause",(255,255,255)),
        ("☝ Index - Next",(255,255,255)),
        ("✌ Two - Prev",(255,255,255)),
        ("🤙 Pinky - Vol Up",(255,255,255)),
        ("👍 Thumb - Vol Down",(255,255,255)),
        ("🤟 Index+Pinky - Toggle",(255,255,255)),
        ("✌✌ Two Mid - Random",(255,255,255)),
    ]
    for i,(t,col) in enumerate(controls):
        text(img,t,(x+15,y+25+i*23),0.5,col)

def draw_status(img,gesture,fps):
    h,w = img.shape[:2]
    cv2.rectangle(img,(0,h-28),(w,h),(0,0,0),-1)
    text(img,f"Gesture: {gesture.upper()}",(10,h-8),0.5,
         COLORS["success"] if gesture!="none" else COLORS["muted"])
    text(img,f"FPS: {fps:.1f}",(w-100,h-8),0.5,COLORS["muted"])

# --- MAIN LOOP ---
def main():
    global playlist, genre_map, banner_text, banner_time
    playlist, genre_map = build_playlist(MUSIC_DIR)
    if not playlist:
        print("No songs found in /songs directory.")
        return

    cam = CameraThread()
    play_index(0)
    print("BeatBuff started — use hand gestures to control playback.")
    fps_time, frames, fps = time.time(), 0, 0
    last_gesture, gesture_start, last_action = "none", 0, 0
    gesture_cooldown = 0.7

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frames += 1
        if time.time()-fps_time >= 1:
            fps, frames, fps_time = frames, 0, time.time()

        # Gesture detection
        if frames % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            gesture = "none"
            if res.multi_hand_landmarks:
                for hand in res.multi_hand_landmarks:
                    gesture = detect_gesture(hand, frame.shape[1], frame.shape[0])
                    break
            now = time.time()
            if gesture != last_gesture:
                gesture_start = now
            elif gesture != "none" and now-gesture_start>GESTURE_HOLD and now-last_action>gesture_cooldown:
                if gesture=="open": mixer.music.unpause(); banner_text,banner_time="▶ Playing",time.time()
                elif gesture=="fist": mixer.music.pause(); banner_text,banner_time="⏸ Paused",time.time()
                elif gesture=="next": play_next()
                elif gesture=="prev": play_prev()
                elif gesture=="vol_up": set_volume(volume+10)
                elif gesture=="vol_down": set_volume(volume-10)
                elif gesture=="play_pause": toggle_play_pause()
                elif gesture=="random": play_index(random.randint(0,len(playlist)-1))
                last_action = now
            last_gesture = gesture
        else:
            gesture = last_gesture

        # Draw UI
        draw_now_playing(frame)
        draw_volume(frame)
        draw_controls(frame)
        draw_status(frame,gesture,fps)
        if time.time()-banner_time < 2 and banner_text:
            cv2.rectangle(frame,(0,0),(frame.shape[1],40),COLORS["primary"],-1)
            text(frame,banner_text,(20,25),0.8,(255,255,255),2)

        cv2.imshow("BeatBuff - Modern UI (Fixed Camera)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k==27: break
        elif k==ord('+'): set_volume(volume+10)
        elif k==ord('-'): set_volume(volume-10)
        elif k==ord(' '): toggle_play_pause()
        elif k==ord('n'): play_next()
        elif k==ord('p'): play_prev()

    cam.stop()
    mixer.music.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
