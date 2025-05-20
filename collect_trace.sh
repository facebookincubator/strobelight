#!/bin/bash

# Set variables
CUDA_PROGRAM="test_cuda_api_multi_gpu.cu"   # src code of the cuda program that will be traced
EXECUTABLE="test_cuda_api_multi_gpu"        # Executable of the cuda program that will be traced 
REPO="/efs/NFLX-GENAI-PROJECTS/GPUSNOOP"    # full path of the directory where repository is cloned
TRACE_DURATION=30  			    # Run gpuevent_snoop for 30 seconds
GPU_EVENTSNOOP="$REPO/strobelight/strobelight/src/_build/profilers/gpuevent_snoop"     # BPF user program

# Step 1: Compile the CUDA Program
echo "Compiling $CUDA_PROGRAM..."
/usr/local/cuda/bin/nvcc $CUDA_PROGRAM -o $EXECUTABLE 
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful"

# Step 2: Run the Program in the Background
echo "Starting $EXECUTABLE..."
./$EXECUTABLE &
CUDA_PID=$!

# Give some time for the process to start
sleep 3

# Step 3: Verify the Process is Running
if ! ps -p $CUDA_PID > /dev/null; then
    echo "Error: CUDA process ($CUDA_PID) is not running!"
    exit 1
fi
echo "CUDA process running with PID: $CUDA_PID"

# Step 4: Run gpuevent_snoop for 30 Seconds
echo "Running gpuevent_snoop for $TRACE_DURATION seconds..."
sudo $GPU_EVENTSNOOP -p $CUDA_PID -a -s -v --args --duration=$TRACE_DURATION

# Step 5: Kill the CUDA Program After Tracing (Optional)
echo "Stopping CUDA program..."
kill $CUDA_PID

echo "Tracing completed."

