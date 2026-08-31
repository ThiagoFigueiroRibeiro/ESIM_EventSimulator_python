@echo off
call conda activate esim

python tools/calculate_centers.py --npz video_out/events.npz --fps 240 --first_events 200 --baricenter_frames ./baricenter_frames --plot_events