@echo off
call conda activate esim

python tools/calculate_angular_velocity_2_blade_propeller.py --npz video_out/events.npz --origin_x 320 --origin_y 320 --events_per_window 20000
