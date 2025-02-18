sudo systemctl isolate multi-user
sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo cp MemoryWorkloadAnalysis_Chart1.section /usr/lib/x86_64-linux-gnu/nsight-compute/sections/MemoryWorkloadAnalysis_Chart.section
echo "start running benchmark.py" 
ncu -o profile -f -k regex:matmulKernel --set detailed python benchmark.py 