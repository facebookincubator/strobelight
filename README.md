# BPF GPUEventSnoop with LLM-based CUDA Trace Analysis

Traces CUDA GPU kernel functions via BPF and provides in-depth analysis through visualizations and optional LLM (Large Language Model)-powered summaries.

---

## 🚀 Prerequisites

- NVIDIA GPU instance  
- Ubuntu with kernel headers  
- CUDA Toolkit installed (`nvcc` should be at `/usr/local/cuda/bin/nvcc`)

---

## 🔧 Install Required Packages

```bash
sudo apt update
sudo apt install -y clang llvm libbpf-dev
sudo apt install -y linux-headers-$(uname -r)
sudo apt install -y build-essential git cmake libelf-dev libfl-dev pkg-config

🛠️ Build Strobelight and GPUEventSnoop
cd strobelight
./scripts/build.sh

- BPF source for user and kernel prog is located in: strobelight/strobelight/src/profilers/gpuevent_snoop
- After a successful build, binaries will be dumped in: strobelight/strobelight/src/_build/profilers

🧪 Run the Profiler
cd strobelight/strobelight/src/_build/profilers
./gpuevent_snoop -p <PID>
./gpuevent_snoop --help  # For all options

Supported CUDA routines traced:
- cudaMalloc, cudaFree
- cudaMemcpy, cudaMemcpyAsync
- cudaLaunchKernel
- cudaStreamCreate, cudaStreamDestroy
- cudaStreamSynchronize, cudaDeviceSynchronize

🧫 Test Programs and Tracing

Build a Sample Program
/usr/local/cuda/bin/nvcc test_cuda_api_multi_gpu.cu -o test_cuda_api_multi_gpu

Run and Collect Trace

Update demo collect_trace.sh with the correct $REPO path. Script will run the above cuda program and trace it

$ ./collect_trace.sh > sample-trace.out
A sample trace file (sample-trace.out) is provided for testing.


📊 CUDA Trace Analysis (LLM-Enhanced)

This toolset parses and analyzes CUDA traces using visualizations and LLMs like OpenAI GPT.

⸻

🔧 Setup Environment

cd llm-analysis
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

📈 Run Analysis

With OpenAI API key::
python ./enhanced_cuda_trace_analysis.py trace.out --llm_mode openai --api_key YOUR_API_KEY

Without API key (mock mode):
python ./enhanced_cuda_trace_analysis.py trace.out --llm_mode mock

Results will be stored in the cuda_analysis_results folder.


🧰 Command Line Options
- --trace_file: Path to the CUDA trace file (required)
- --output_dir: Output folder (default: ./cuda_analysis_results)
- --llm_mode: LLM mode (mock, openai, local; default: mock)
- --api_key: OpenAI API key (required for openai mode)
- --model_endpoint: Local LLM API endpoint (default: http://localhost:8000)
- --skip_parsing: Skip trace parsing
- --skip_analysis: Skip trace data analysis
- --skip_visualization: Skip visualization generation
- --test_llm: Run LLM test suite

📂 Output Artifacts
- Parsed trace data (JSON)
- Analysis results (JSON)
- Visualizations (PNG)
- Enhanced dashboards
- LLM analysis (Markdown, HTML)
- Final summary reports

Sample outputs:
- llm-sample-results/
- sample_llm_analysis_report.html


.
├── strobelight/                   # Strobelight GPU profiler
├── llm-analysis/                 # LLM analysis tools
├── collect_trace.sh              # CUDA trace demo script
├── test_cuda_api_multi_gpu.cu    # demo CUDA program for tracing
├── sample_llm_analysis_report.html # Example output
├── llm-sample-results/           # Example LLM results
└── README.md


🧠 Components in llm-analysis/
- cuda_trace_parser.py – Parses trace data
- cuda_trace_analyzer.py – Analyzes kernel launches
- cuda_visualization_organizer.py – Generates visualizations
- enhanced_cuda_llm_analyzer.py – Performs LLM analysis
- enhanced_cuda_trace_analysis.py – CLI wrapper
- cuda_prompt_templates.py – Prompt templates for LLMs
- cuda_llm_analysis_tester.py – LLM test suite

📬 Feedback & Contributions

Feedback and contributions are welcome!
Please open an issue or pull request to help improve this GPU tracing and analysis framework.


