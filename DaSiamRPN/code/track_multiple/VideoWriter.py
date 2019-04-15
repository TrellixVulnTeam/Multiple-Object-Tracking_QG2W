import os
from moviepy.editor import ImageSequenceClip

images_folder = "./tracks/traffic_footage_tracks/final_tracks"

clip = ImageSequenceClip(images_folder, fps=29.97)

clip.write_videofile("./tracks/traffic_footage_tracks/final_tracks.mp4")

