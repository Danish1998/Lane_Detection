import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import image_processor


def process_video(in_clip, output='../tests/test_results.mp4', time_interval=None):
    """Processes the input video one frame at a time
    Args:
        in_clip: relative path to the input video file
        output: name and path to save the resulting video file
        time_interval: tuple containing start and end time in seconds.
                       If None then the full video file is processed
    Returns:
        output: name and path to the processed video file
    """
    if not time_interval:
        clip = VideoFileClip(in_clip)
    else:
        clip = VideoFileClip(in_clip).subclip(time_interval)

    out_clip = clip.fl_image(image_processor.process_img)
    out_clip.write_videofile(output, audio=False)

    return output


def show_video(clip_name):
    """Displays the video clip in IPython kernel
    Args:
        clip_name: relative path to the video file
    """
    HTML("""
    <video width="640" height="360" controls>
      <source src="{0}">
    </video>
    """.format(clip_name))


def get_frames(in_clip, save_dir='../data/'):
    """Extracts frames at regular intervals from input video file
    Args:
        in_clip: relative path to the input video file
        save_dir: directory where the extracted frames are saved
    Returns:
        frames: list of path to extracted frames
    """
    save_dir = os.path.join(save_dir, in_clip.split('.')[:-1])
    clip = VideoFileClip(in_clip)
    frames = []
    for i in range(20):
        frame_name = 'frame_' + str(i) + '.jpeg'
        out_file = os.path.join(save_dir, frame_name)
        time = i * 60 + 30
        clip.save_frame(out_file, t=time)
        frames.append(out_file)
    return frames
