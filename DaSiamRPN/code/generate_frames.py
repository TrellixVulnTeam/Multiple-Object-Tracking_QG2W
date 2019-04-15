import cv2
import argparse

"Parse command line arguments"
parser = argparse.ArgumentParser(description='Generate frames from video')
parser.add_argument('--video_path', required=True, help='input video')
parser.add_argument('--output_folder', required=True, help='output folder path')

args = parser.parse_args()

s = cv2.VideoCapture(args.video_path)

success, img = s.read()
count = 0
if (args.output_folder[-1] != '/'):
    args.output_folder = args.output_folder + "/" 
while success:
    img_name = str(count).zfill(6)
    img_path = args.output_folder + ("%s.jpg" % img_name)
    cv2.imwrite(img_path, img)
    success, img = s.read()
    count += 1
duration = count/25.0
print("Count: %d" % count)
print("Duration: %d" % duration)
