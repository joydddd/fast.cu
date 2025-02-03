sudo systemctl isolate multi-user
sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
ncu -o profile -f python benchmark.py