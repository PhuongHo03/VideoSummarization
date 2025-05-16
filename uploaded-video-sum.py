import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import tempfile
import time

st.title('Tóm tắt video bằng phương pháp khử nền')

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "summary_available" not in st.session_state:
    st.session_state.summary_available = False
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None

summary_video_path = "summary_video.mp4"
temp_video_path = "temp_video.mp4"

uploaded_file = st.file_uploader("Tải lên video", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file.close()
    st.session_state.uploaded_video_path = temp_file.name
    st.success("Tải lên thành công!")

def toggle_stream():
    if st.session_state.streaming:
        st.session_state.streaming = False
        process_summary(st.session_state.video_fps)
        st.session_state.summary_available = True
    else:
        st.session_state.streaming = True
        st.session_state.summary_available = False

if st.session_state.uploaded_video_path:
    button_label = "Dừng" if st.session_state.streaming else "Bắt đầu"
    st.button(button_label, on_click=toggle_stream)

video_placeholder1 = st.empty()
video_placeholder2 = st.empty()
progress_bar_placeholder = st.empty()  
time_placeholder = st.empty()  

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

summary_frames = []

def process_summary(fps):
    if len(summary_frames) > 0:
        height, width, _ = summary_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        for frame in summary_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        out.release()
        ffmpeg_cmd = f'ffmpeg -i {temp_video_path} -vcodec libx264 -crf 23 {summary_video_path} -y'
        process = subprocess.run(ffmpeg_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            st.error("FFmpeg Error!")
            st.text(process.stderr.decode("utf-8"))
        else:
            st.success("Tóm tắt thành công!")

        os.remove(temp_video_path)
    else:
        st.warning("Không có khung hình nào để tóm tắt!")

if st.session_state.streaming and st.session_state.uploaded_video_path:
    cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.session_state.video_fps = fps
    duration = total_frames / fps  

    start_time = time.time()
    
    while st.session_state.streaming:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elapsed_time = (time.time() - start_time)  
        expected_frame = int(elapsed_time * fps)

        if expected_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fgMask = bg_subtractor.apply(frame)
        fgMask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
        
        kernel = np.ones((5,5), np.uint8)
        fgMask = cv2.erode(fgMask, kernel, iterations=1) 
        fgMask = cv2.dilate(fgMask, kernel, iterations=1)
        fgMask = cv2.GaussianBlur(fgMask, (3,3), 0)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        _,fgMask = cv2.threshold(fgMask,130,255,cv2.THRESH_BINARY)

        fgMask = cv2.Canny(fgMask,20,200)
        contours,_ = cv2.findContours(fgMask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for i in range(len(contours)):
            (x, y, w, h) = cv2.boundingRect(contours[i])
            area = cv2.contourArea(contours[i])
            if area > 300:
                cv2.drawContours(fgMask, contours[i], 0, (0, 0, 255), 6)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if np.sum(fgMask) > 50000:
            if current_frame % (fps/10) == 0:
                summary_frames.append(frame)

        progress_bar_placeholder.empty()
        progress_bar_placeholder.slider("Tiến trình", 0.0, duration, elapsed_time, disabled=True)

        time_placeholder.text(f"Thời gian: {elapsed_time:.2f} / {duration:.2f} giây\nFPS: {fps}")

        video_placeholder1.image(frame, channels="RGB", caption="Video Gốc")
        video_placeholder2.image(fgMask, channels="RGB", caption="Video Xử Lý")

    cap.release()
    video_placeholder1.empty()
    video_placeholder2.empty()
    progress_bar_placeholder.empty()
    time_placeholder.empty()

elif st.session_state.summary_available and os.path.exists(summary_video_path):
    if st.button("Tóm tắt"):
        cap = cv2.VideoCapture(summary_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        start_time = time.time()

        while True:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            elapsed_time = time.time() - start_time
            expected_frame = int(elapsed_time * fps)

            if expected_frame >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder1.image(frame_rgb, channels="RGB", caption="Video Tóm Tắt")
            progress_bar_placeholder.slider("Tiến trình", 0.0, duration, elapsed_time, disabled=True)
            time_placeholder.text(f"Thời gian: {elapsed_time:.2f} / {duration:.2f} giây\nFPS: {fps}")

        cap.release()
        video_placeholder1.empty()
        progress_bar_placeholder.empty()
        time_placeholder.empty()
