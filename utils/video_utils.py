import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fourcc=None):
    if fourcc is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
