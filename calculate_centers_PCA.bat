@echo off
call conda activate esim

python tools/calculate_centers_PCA.py --npz video_out/events.npz --fps 240 --first_events 100 --PCA_frames ./PCA_frames --plot_events --pca_len_mode "range" --pc1_boost 0.5 --pc2_boost 0.25