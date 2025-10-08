# PixelHolo Performance Timing Guide

This guide explains how to use the comprehensive timing system added to PixelHolo to measure and optimize performance.

## 🚀 Quick Start

### 1. Run the Application with Timing
```bash
# Run the main application (timing is automatically enabled)
python -m pixelholo.app

# Or run the test script to see timing in action
python test_timing.py
```

### 2. View Performance Summary
The application will automatically print a performance summary at the end:
```
📊 PERFORMANCE SUMMARY
============================================================
Session Duration: 45.23s
Total Operations: 12
Total Measured Time: 38.45s
Memory Efficiency: 85.0%

📈 OPERATION BREAKDOWN:
  tts_model_loading:
    Count: 1
    Total Time: 8.234s
    Average Time: 8.234s
    Min Time: 8.234s
    Max Time: 8.234s
    Avg Memory: 1024.5MB
```

## 📊 What Gets Measured

### Model Operations
- **TTS Model Loading**: Time to load Chatterbox-TTS model
- **LipSync Model Loading**: Time to load Wav2Lip model
- **Background Removal**: Time to process video background removal

### Processing Operations
- **TTS Generation**: Time to generate speech from text
- **Lip-sync Generation**: Time to create lip-synced video
- **Voice Isolation**: Time to extract voice from audio
- **Audio Extraction**: Time to extract audio from video

### AI Operations
- **Ollama Response**: Time for AI to generate responses
- **Web Search**: Time for internet search operations
- **Internet Detection**: Time for ML model to determine if search is needed

## 🔍 Understanding the Output

### Timing Information
```
⏱️  TIMING: tts_generation
   Duration: 2.456s
   Memory: 1024.5MB → 1089.2MB
   GPU Memory: 2048.1MB → 2156.3MB
   GPU Peak: 2200.5MB
   Metadata: {'text_length': 45}
```

### Performance Metrics
- **Duration**: How long the operation took
- **Memory**: RAM usage before and after
- **GPU Memory**: GPU memory usage (if CUDA available)
- **Metadata**: Additional context (text length, file sizes, etc.)

## 📈 Performance Analysis

### Run the Analysis Script
```bash
python analyze_performance.py
```

This will provide:
- **Operation Ranking**: Operations sorted by total time
- **Detailed Statistics**: Min, max, average, standard deviation
- **Memory Analysis**: RAM and GPU memory usage patterns
- **Bottleneck Identification**: Operations that are slow or variable
- **Optimization Recommendations**: Specific suggestions for improvement

### Example Analysis Output
```
🔍 OPERATION RANKING (by total time):
 1. lipsync_generation    | Total: 15.23s | Avg: 3.81s | Count: 4
 2. background_removal     | Total: 12.45s | Avg: 12.45s | Count: 1
 3. tts_generation        | Total: 8.67s | Avg: 2.17s | Count: 4
 4. tts_model_loading     | Total: 8.23s | Avg: 8.23s | Count: 1

⚠️  PERFORMANCE BOTTLENECKS:
🐌 High Average Time Operations:
   - background_removal: 12.45s average
   - lipsync_generation: 3.81s average

💡 OPTIMIZATION RECOMMENDATIONS:
🎬 Video Processing Optimization:
   - Use frame sampling instead of processing every frame
   - Implement GPU parallel processing
   - Use video streaming for real-time processing
```

## 🛠️ Customizing Timing

### Add Custom Timing
```python
from pixelholo.timing import time_operation

# Time any operation
with time_operation("my_custom_operation", param1="value1"):
    # Your code here
    do_something()
```

### Access Profiler Directly
```python
from pixelholo.timing import get_profiler

profiler = get_profiler()
# Get current timings
timings = profiler.timings
# Get summary
summary = profiler.get_summary()
```

## 📁 Log Files

### Performance Log
- **File**: `performance_log.json`
- **Contains**: All timing data, memory usage, metadata
- **Format**: JSON with detailed operation breakdown

### Log Structure
```json
{
  "session_start": 1703123456.789,
  "session_duration": 45.23,
  "timings": [
    {
      "operation": "tts_generation",
      "duration": 2.456,
      "memory_before": 1024.5,
      "memory_after": 1089.2,
      "gpu_memory_before": 2048.1,
      "gpu_memory_after": 2156.3,
      "metadata": {"text_length": 45}
    }
  ],
  "summary": { ... }
}
```

## 🎯 Optimization Tips

### Based on Common Bottlenecks

1. **Model Loading (8-15s)**
   - Cache models after first load
   - Use quantized models
   - Pre-load in background

2. **Background Removal (10-20s)**
   - Process every 5th frame instead of all frames
   - Use GPU acceleration
   - Cache processed segments

3. **Lip-sync Generation (3-8s)**
   - Use smaller model variants
   - Process in chunks
   - Implement streaming

4. **TTS Generation (1-3s)**
   - Cache common phrases
   - Use faster TTS models
   - Implement text chunking

## 🔧 Troubleshooting

### No Timing Data
- Ensure you're running the updated code
- Check that `pixelholo.timing` module is imported
- Verify the application completed successfully

### Missing GPU Memory Data
- CUDA must be available
- PyTorch with CUDA support required
- GPU memory tracking only works with CUDA

### Performance Log Not Found
- Run the application first
- Check file permissions
- Ensure the application completed without errors

## 📊 Example Workflow

1. **Run Application**: `python -m pixelholo.app`
2. **Interact**: Have several conversations with the avatar
3. **View Summary**: Check the performance summary at the end
4. **Analyze**: Run `python analyze_performance.py`
5. **Optimize**: Implement recommendations based on bottlenecks
6. **Re-test**: Run again to measure improvements

This timing system will help you identify exactly where time is being spent and guide optimization efforts for maximum performance gains.
