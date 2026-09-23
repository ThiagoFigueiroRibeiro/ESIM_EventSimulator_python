@echo off
call conda activate esim

python tools/calculate_angular_velocity_barycenter.py --npz video_out/events.npz --origin_x 320 --origin_y 320 --num_bins 240 --analysis_events_per_bin 2000 --angular_fit 