@echo off
call conda activate esim

python tools/prepare_video.py -i video.mp4 -o video_input
python -m esim.cli --input video_input --output video_out --contrast-threshold 0.2
python -m esim.event_frames video_out/events.npz --output video_out/event_frames --fps 25
