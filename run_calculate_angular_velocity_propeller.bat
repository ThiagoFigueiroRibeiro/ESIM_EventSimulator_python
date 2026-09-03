@echo off
call conda activate esim

python tools/calculate_angular_velocity_propeller.py --npz video_out/events.npz --fps 240 --first_events 200 --tracking_method pca