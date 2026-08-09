import pathlib
import  cv2
def read_video(video_path):
    path = pathlib.Path(video_path)
    c = cv2.VideoCapture(path)
    frames = []
    while c.isOpened():
        frames.append(c.read())
        print(frames[-1])
    c.release(s)
    print(frames)
    return frames

    # return video
    pass
def save_video(frames , name = 'output_video.avi' , width = 1920 , height = 1080):
    # print("number of")
    # output_filename = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # H.264 codec
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec for MP4
    fps = 30                                 # Frames per second
    # width, height = 1, 480
    o = cv2.VideoWriter(name, fourcc, fps, (width, height))
    for f in frames:
        o.write(f)
    o.release()

    # return
    return
# def blend_frames(frame_a, frame_b, alpha=0.5):
#     pass
# read_video("./data/blended.avi")


