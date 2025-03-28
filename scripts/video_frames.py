import os
import cv2
import glob

# Output directory (now includes 'video' subfolder)
file_path = "../sample-results/video/"
os.makedirs(file_path, exist_ok=True)  # Creates directory if it doesn't exist

for filename in glob.iglob("../data/videos//*.mp4", recursive=True):
    vidcap = cv2.VideoCapture(filename)
    print("Processing:", filename)  # Debug: Confirm which file is being processed
    
    # Extract just the video name (without path or extension)
    video_name = os.path.splitext(os.path.basename(filename))[0]  # e.g., "my_video"
    
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1, length, 60):
        vidcap.set(1, i)
        ret, still = vidcap.read()
        if ret:  # Only save if the frame was read correctly
            # New output format: "../sample-results/video/my_video_frame_60.png"
            output_name = f"{video_name}_frame_{i}.png"
            output_path = os.path.join(file_path, output_name)
            cv2.imwrite(output_path, still)
            print(f"Saved: {output_path}")  # Optional: Log saved frames