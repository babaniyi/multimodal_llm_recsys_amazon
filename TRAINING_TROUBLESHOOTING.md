# Training Troubleshooting Guide

## 🚨 Common Issues and Solutions

### **Issue 1: Training Hangs/Stuck**

**Symptoms:**
- Logging stops after "Logging system initialized"
- Multiple multiprocessing workers spawned
- High CPU usage but no progress

**Root Cause:**
- Multiprocessing issues with `num_workers > 0`
- Memory issues with large batch sizes
- Dataset initialization problems

**Solution:**
```json
{
    "data_config": {
        "num_workers": 0,        // Disable multiprocessing
        "batch_size": 16,        // Reduce batch size
        "pin_memory": false      // Disable pin_memory
    }
}
```

### **Issue 2: Out of Memory (OOM)**

**Symptoms:**
- CUDA out of memory errors
- Process killed by system

**Solution:**
```json
{
    "data_config": {
        "batch_size": 8,         // Further reduce batch size
        "gradient_accumulation_steps": 2  // Compensate with accumulation
    }
}
```

### **Issue 3: Slow Training**

**Symptoms:**
- Each step takes 40+ seconds
- Low GPU utilization

**Solution:**
```json
{
    "data_config": {
        "fast_training_mode": true,  // Skip image processing
        "batch_size": 16,           // Optimal batch size
        "num_workers": 0            // Avoid multiprocessing issues
    }
}
```

### **Issue 4: Dataset Loading Errors**

**Symptoms:**
- KeyError: 'instruction'
- Missing image files
- Data format errors

**Solution:**
1. **Check data format:**
```bash
python -c "
import json
with open('data/processed/train.json', 'r') as f:
    data = json.load(f)
    print(f'Loaded {len(data)} samples')
    print('Sample keys:', list(data[0].keys()))
"
```

2. **Re-run data processing:**
```bash
python run_multimodal_workflow.py --steps data
```

## 🔧 Quick Fixes

### **Reset Configuration**
```json
{
    "data_config": {
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "pin_memory": false,
        "fast_training_mode": true
    }
}
```

### **Kill Stuck Processes**
```bash
# Kill all training processes
pkill -f "src.training.train"
pkill -f "run_multimodal_workflow"

# Check if processes are gone
ps aux | grep python | grep -v grep
```

### **Clear GPU Memory**
```bash
# If using CUDA
python -c "import torch; torch.cuda.empty_cache()"
```

## 🚀 Safe Training Configuration

**Recommended settings for stable training:**

```json
{
    "data_config": {
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "pin_memory": false,
        "fast_training_mode": true,
        "enable_image_processing": false
    },
    "training_config": {
        "save_every": 100,
        "eval_freq": 100,
        "fp16": true
    }
}
```

## 📊 Monitoring Training

### **Check Training Progress**
```bash
# Monitor logs
tail -f logs/train.log

# Check GPU usage
nvidia-smi

# Check process status
ps aux | grep python
```

### **Expected Log Pattern**
```
2025-09-24 15:07:55,389 - __main__ - INFO -    Training loss: 75.418
2025-09-24 15:08:02,974 - __main__ - INFO - Ep 1 (Step 000000): Train loss 70.816
2025-09-24 15:08:15,076 - __main__ - INFO - Ep 1 (Step 000025): Train loss 9.141  # ~13s per step
```

## 🎯 Next Steps

1. **Use the fixed configuration** (num_workers: 0, batch_size: 16)
2. **Start training again:**
   ```bash
   python run_multimodal_workflow.py --steps train
   ```
3. **Monitor for 5-10 minutes** to ensure it's working
4. **If still stuck, try fast mode:**
   ```json
   "fast_training_mode": true
   ```

The key fix is disabling multiprocessing (`num_workers: 0`) which was causing the hanging issue.

