@echo off
call conda activate esim

python tools/calculate_angular_velocity.py --npz video_out/events.npz --origin_x 320 --origin_y 320 --analysis_events_per_bin 2000