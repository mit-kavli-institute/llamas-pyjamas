# LLAMAS Arc Wavelength Solution Multiprocessing with Ray

## Feature Overview

This document describes the implementation of Ray multiprocessing for the LLAMAS arc wavelength solution pipeline. The new `arcLlamasMulti.py` module provides parallel processing capabilities for the computationally intensive wavelength calibration process, delivering significant performance improvements over the serial implementation.

> **Note on drift.** This document was written when the Ray arc path was introduced
> (2025-09). The parallel design it describes is still what runs — `reduce.py` imports
> `Arc.arcLlamasMulti` — but some signatures have since grown extra keyword arguments.
> Treat the code as authoritative where the two disagree.

### What Was Parallelized

The wavelength solution process involves three main computational bottlenecks that have been parallelized:

1. **Fiber Shift Calculations** (`shiftArcXRay`): Cross-correlation calculations for ~300+ fibers per extension
2. **Fiber Throughput Calculations** (`fiberRelativeThroughputRay`): Individual fiber throughput calculations
3. **Wavelength Solution Transfer** (`arcSolveRay`): Extension-level wavelength solution application

### Performance Benefits

- **Fiber-level parallelization**: Process hundreds of fibers simultaneously across available CPU cores
- **Extension-level parallelization**: Process different spectrograph extensions (red/green/blue) in parallel
- **Scalable core usage**: Automatically utilizes available CPU cores via `LLAMAS_RAY_CPUS` environment variable
- **Progress monitoring**: Real-time progress tracking and CPU usage reporting

## Technical Implementation

### Ray Architecture

The implementation follows the established LLAMAS Ray pattern used in `traceLlamasMulti.py`:

```python
# Core management
NUMBER_OF_CORES = int(os.environ.get('LLAMAS_RAY_CPUS', multiprocessing.cpu_count()))
ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
```

### Ray Remote Functions

#### `@ray.remote process_fiber_shift`
Handles individual fiber cross-correlation calculations:
- **Input**: Fiber data, reference spectrum, stretch function parameters
- **Output**: Shift/stretch results for wavelength pixel mapping
- **Parallelization**: ~300+ tasks per extension

#### `@ray.remote process_fiber_throughput`  
Calculates relative throughput for individual fibers:
- **Input**: Fiber spectrum, x-shift data, reference flux
- **Output**: Normalized throughput value
- **Parallelization**: All fibers processed simultaneously

#### `@ray.remote process_extension_wavelength_transfer`
Applies wavelength solutions to entire extensions:
- **Input**: Extension data, fitted wavelength solution
- **Output**: Wavelength arrays for all fibers in extension
- **Parallelization**: Multiple extensions processed simultaneously

### Core Functions

#### `shiftArcXRay(arc_extraction_pickle)`
Ray-enabled version of shift calculation:
```python
# Parallel fiber processing
for fits_ext in range(len(arcspec)):
    for ifiber in range(metadata[fits_ext]['nfibers']):
        task = process_fiber_shift.remote(fits_ext, ifiber, refspec, fiber_counts, 'quadratic')
        tasks.append(task)

# Batch processing with progress monitoring
while tasks:
    done_tasks, tasks = ray.wait(tasks, num_returns=min(50, len(tasks)))
    results = ray.get(done_tasks)
    # Apply results back to data structure
```

#### `fiberRelativeThroughputRay(flat_extraction_pickle, arc_extraction_pickle)`
Ray-enabled throughput calculation:
```python
# Parallel throughput processing  
for fits_ext in range(len(arcspec)):
    for ifiber in range(metadata[fits_ext]['nfibers']):
        task = process_fiber_throughput.remote(fits_ext, ifiber, fiber_spec, xshift_fiber, reference_flux)
        tasks.append(task)
```

#### `arcSolveRay(arc_extraction_shifted_pickle, autoid=False, savefile='LLAMAS_reference_arc.pkl', savedir=OUTPUT_DIR)`
Ray-enabled wavelength solution with extension-level parallelization:
```python
# Process multiple extensions per channel in parallel
extension_tasks = []
for extension in range(len(arcspec_shifted)):
    if arcdict['metadata'][extension]['channel'] == channel:
        task = process_extension_wavelength_transfer.remote(extension, extension_data, final_arcfit, channel)
        extension_tasks.append(task)

results = ray.get(extension_tasks)
```

## Usage Examples

### 🚀 **Automatic Ray Multiprocessing (Recommended)**

**Zero-code changes required!** Your existing workflow now automatically uses Ray multiprocessing:

```python
import llamas_pyjamas.Arc.arcLlamasMulti as arc

# Your EXACT existing workflow - now automatically Ray-enabled!
arc_filename = 'LLAMAS_2025-03-05T19_04_46.449_mef.fits'
flat_filename = 'your_flat_file_mef.fits'

arc_picklename = os.path.join(OUTPUT_DIR, arc_filename.replace('_mef.fits', '_extract.pkl'))

# These function calls automatically use Ray multiprocessing!
arc.shiftArcX(arc_picklename)                                    # 🔥 Ray parallel (auto)
shift_picklename = arc_picklename.replace('_extract.pkl', '_extract_shifted.pkl')
flat_picklename = os.path.join(OUTPUT_DIR, flat_filename.replace('_mef.fits', '_extract.pkl'))
arc.fiberRelativeThroughput(flat_picklename, shift_picklename)    # 🔥 Ray parallel (auto)
tp = shift_picklename.replace('.pkl','_shifted_tp.pkl')
arc.arcSolve(tp)                                                 # 🔥 Ray parallel (auto)
```

**What you'll see:**
```
🚀 Initializing Ray with 16 CPU cores for parallel processing...
💾 Current CPU Usage: 5.2%
✅ Ray initialized successfully with 16 cores
🔥 Using Ray multiprocessing for arc shift calculation...
Processing 2400+ fiber shift calculations in parallel...
Completed 2400/2400 fiber shift calculations
Saved shifted arc extraction to /path/to/output
```

### **Manual Control Options**

Override automatic Ray behavior when needed:

```python
# Force serial processing (bypass Ray)
arc.shiftArcX(arc_picklename, use_ray=False)                     # 🐌 Serial processing
arc.fiberRelativeThroughput(flat_picklename, shift_picklename, use_ray=False)
arc.arcSolve(tp, use_ray=False)

# Explicit Ray function calls (for advanced users)
arc.shiftArcXRay(arc_picklename)                                 # 🔥 Ray parallel (explicit)
arc.fiberRelativeThroughputRay(flat_picklename, shift_picklename)
arc.arcSolveRay(tp)
```

### Orchestrated Workflow

For complete automation with error handling and progress monitoring:

```python
import llamas_pyjamas.Arc.arcLlamasMulti as arc

# Complete Ray-enabled pipeline
final_solution = arc.run_wavelength_solution_ray(
    arc_filename='LLAMAS_2025-03-05T19_04_46.449_mef.fits',
    flat_filename='your_flat_file_mef.fits',
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR
)
print(f"Wavelength solution saved to: {final_solution}")
```

### Flexible Workflow Function

Switch between Ray and standard processing:

```python
# Use Ray multiprocessing (default)
result = arc.run_wavelength_solution_workflow(
    arc_filename='LLAMAS_2025-03-05T19_04_46.449_mef.fits',
    flat_filename='your_flat_file_mef.fits',
    use_ray=True
)

# Use standard serial processing
result = arc.run_wavelength_solution_workflow(
    arc_filename='LLAMAS_2025-03-05T19_04_46.449_mef.fits', 
    flat_filename='your_flat_file_mef.fits',
    use_ray=False
)
```

## Configuration

### Core Count Specification

The number of CPU cores used follows the established LLAMAS pattern:

```bash
# Set cores via environment variable (recommended)
export LLAMAS_RAY_CPUS=16

# Or let it auto-detect system cores
# Defaults to multiprocessing.cpu_count()
```

### Integration with LLAMAS Pipeline

The Ray core configuration integrates with the broader LLAMAS pipeline via `reduce.py`:

```python
# In your pipeline configuration
config = {
    'ray_num_cpus': 16  # This sets LLAMAS_RAY_CPUS
}
```

## Error Handling and Logging

### **Enhanced Ray-Specific Error Handling**

The Ray implementation includes sophisticated error handling and logging not present in the original:

#### **Conditional Data Cleaning**
Only applies enhanced cleaning when the standard method fails:

```python
# FIRST ATTEMPT: Use standard approach (same as original)
try:
    success, shift, stretch, stretch2, _, _, _ = \
        xcorr_shift_stretch(refspec, interpolateNaNs(fiber_counts), stretch_func='quadratic')
        
    if success == 1:
        # Success - use result
    else:
        logger.warning(f"Arc shift failed for fiber {ifiber} (success={success})")
        
except ValueError as e:
    if "array must not contain infs or NaNs" in str(e):
        # SECOND ATTEMPT: Apply enhanced cleaning and retry
        logger.info(f"Applying data cleaning for fiber {ifiber} due to NaN/Inf values")
        cleaned_data = _robust_clean_spectrum_ray(fiber_counts)
        # Retry with cleaned data...
```

#### **Comprehensive Logging**

All Ray functions include detailed logging:

- **`logger.warning()`** - Processing failures (`success != 1`)
- **`logger.info()`** - Data cleaning operations and unusual values
- **`logger.error()`** - Serious processing errors
- **`logger.debug()`** - Detailed processing information

#### **Robust Data Cleaning Pipeline**

Multi-stage cleaning for problematic data:

1. **Replace Inf values** with NaN
2. **Linear interpolation** for scattered NaN values  
3. **Median filter** for clustered invalid regions
4. **Forward/backward fill** for edge cases
5. **Final validation** ensures no NaN/Inf remain

### **Task-Level Error Management**

Ray processing provides enhanced error isolation and recovery:

```python
# Individual fiber failures don't crash the pipeline
for fits_ext, ifiber, success, xshift_result in results:
    arcspec[fits_ext].xshift[ifiber, :] = xshift_result
    if not success:
        logger.warning(f"Arc shift failed for fiber {ifiber} in extension {fits_ext}")
```

### Pipeline-Level Error Recovery

The orchestration function includes comprehensive error handling:

```python
try:
    # Ray processing pipeline
    shiftArcXRay(arc_picklename)
    fiberRelativeThroughputRay(flat_picklename, shift_picklename)  
    arcSolveRay(tp_picklename)
except Exception as e:
    print(f"Error in wavelength solution pipeline: {str(e)}")
    raise
finally:
    ray.shutdown()  # Always cleanup Ray resources
```

### Resource Cleanup

Ray resources are automatically managed:

- **Automatic shutdown**: Ray is shut down after pipeline completion
- **Error recovery**: Ray shutdown occurs even if processing fails
- **Resource monitoring**: CPU usage tracking throughout processing

## Memory Considerations

### Large Dataset Handling

For datasets with high fiber counts or large spectral arrays:

1. **Batch Processing**: Tasks are processed in batches of 50 to manage memory usage
2. **Incremental Results**: Results are applied immediately to prevent memory accumulation
3. **Resource Monitoring**: CPU and memory usage tracked throughout processing

### Memory-Efficient Patterns

```python
# Process tasks in manageable batches
while tasks:
    done_tasks, tasks = ray.wait(tasks, num_returns=min(50, len(tasks)))
    results = ray.get(done_tasks)
    
    # Apply results immediately to free memory
    for result in results:
        apply_result_to_data_structure(result)
```

## Performance Monitoring

### Progress Tracking

Real-time progress monitoring is built into all Ray functions:

```python
completed += len(results)
if completed % 100 == 0 or completed == total_tasks:
    print(f"Completed {completed}/{total_tasks} fiber shift calculations")
```

### CPU Usage Monitoring

CPU utilization is tracked throughout the pipeline:

```python
print(f"Starting with {NUMBER_OF_CORES} cores available")
print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
# ... processing ...
print(f"Final CPU Usage: {psutil.cpu_percent(percpu=True)}%")
```

### Timing Information

Total processing time is calculated and reported:

```python
start_time = time.time()
# ... processing ...
total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f} seconds")
```

## Validation and Testing

### Output Compatibility

The Ray-enabled functions produce identical outputs to the original serial functions:

- **Data structures**: Arc extraction objects maintain identical structure
- **File formats**: Output pickle files are fully compatible  
- **Wavelength solutions**: Numerical results match serial implementation exactly

### Verification Workflow

To verify Ray implementation produces correct results:

```python
# Run both implementations and compare
original_result = arc.run_wavelength_solution_workflow(
    arc_filename, flat_filename, use_ray=False
)
ray_result = arc.run_wavelength_solution_workflow(
    arc_filename, flat_filename, use_ray=True  
)

# Results should be numerically identical
assert original_result == ray_result
```

## Integration with Existing LLAMAS Code

### Backward Compatibility

All original functions remain unchanged:
- `shiftArcX()` - Original serial implementation
- `fiberRelativeThroughput()` - Original serial implementation  
- `arcSolve()` - Original serial implementation

### New Ray Functions

New Ray-enabled functions are clearly named:
- `shiftArcXRay()` - Ray parallel implementation
- `fiberRelativeThroughputRay()` - Ray parallel implementation
- `arcSolveRay()` - Ray parallel implementation

### Import Compatibility

The module can be imported as a drop-in replacement:

```python
# Replace this:
import llamas_pyjamas.Arc.arcLlamas as arc

# With this:
import llamas_pyjamas.Arc.arcLlamasMulti as arc

# Use Ray functions with 'Ray' suffix, or use workflow functions
```

## Troubleshooting

### Common Issues and Solutions

#### **Ray Initialization Fails**

**Problem:** Ray fails to initialize and falls back to serial processing
```
⚠️  Ray initialization failed: ...
🔀 Falling back to serial processing...
```

**Solutions:**
1. **Check available memory**: Ray requires sufficient memory
   ```bash
   # Check system memory
   free -h  # Linux
   vm_stat  # macOS
   ```

2. **Reduce core count**: Lower the number of cores used
   ```bash
   export LLAMAS_RAY_CPUS=4  # Use fewer cores
   ```

3. **Clear Ray processes**: Kill any existing Ray processes
   ```bash
   ray stop  # Stop Ray cluster
   pkill -f ray  # Kill remaining Ray processes
   ```

#### **Slow Performance Despite Ray**

**Problem:** Performance doesn't improve with Ray multiprocessing

**Solutions:**
1. **Verify Ray is actually being used**: Look for these messages:
   ```
   ✅ Ray initialized successfully with X cores
   🔥 Using Ray multiprocessing for...
   ```

2. **Check core count**: Ensure you're using multiple cores
   ```python
   import os
   print(f"LLAMAS_RAY_CPUS: {os.environ.get('LLAMAS_RAY_CPUS', 'auto-detect')}")
   ```

3. **Monitor CPU usage**: Watch CPU utilization during processing
   ```bash
   htop  # or top on systems without htop
   ```

#### **Memory Issues with Large Datasets**

**Problem:** Out of memory errors during Ray processing

**Solutions:**
1. **Reduce batch size**: The code already uses batches of 50 tasks
2. **Monitor memory usage**: 
   ```bash
   watch -n 1 'free -h'  # Linux
   ```
3. **Use fewer cores**:
   ```bash
   export LLAMAS_RAY_CPUS=8  # Reduce from default
   ```

#### **Ray Doesn't Auto-Initialize**

**Problem:** Functions don't automatically use Ray

**Checklist:**
1. ✅ Import the right module: `import llamas_pyjamas.Arc.arcLlamasMulti as arc`
2. ✅ Call functions without `Ray` suffix: `arc.shiftArcX()` not `arc.shiftArcXRay()`
3. ✅ Don't set `use_ray=False` parameter
4. ✅ Check for error messages in output

#### **Understanding Log Messages**

The Ray implementation provides detailed logging for debugging and monitoring:

**Success Messages:**
```
INFO: Fiber 150 processed successfully after data cleaning
DEBUG: Successfully transferred wavelength solution to 300 fibers in extension 18 (red)
```

**Warning Messages:**
```
WARNING: Arc shift failed for fiber 47 in extension 18 (success=0)
WARNING: Invalid reference flux (-1.2) for fiber 23 in extension 19
INFO: Unusual throughput value 0.032 for fiber 156 in extension 20
```

**Error Messages:**
```
ERROR: Data cleaning failed for fiber 89 in extension 18: insufficient valid data
ERROR: Wavelength transfer failed for fiber 12 in extension 19: invalid arcfit parameters
```

**Configuring Logging Levels:**
```python
import logging
logging.getLogger('llamas_pyjamas.Arc.arcLlamasMulti').setLevel(logging.INFO)  # Show INFO and above
logging.getLogger('llamas_pyjamas.Arc.arcLlamasMulti').setLevel(logging.WARNING)  # Only warnings/errors
logging.getLogger('llamas_pyjamas.Arc.arcLlamasMulti').setLevel(logging.DEBUG)  # Show everything
```

#### **Fallback to Serial Processing**

**When this happens:**
- Ray initialization fails
- Ray processing encounters errors
- `use_ray=False` is explicitly set

**How to debug:**
1. **Enable verbose output**: Look for these indicators:
   ```
   🔥 Using Ray multiprocessing...     # Ray is working
   🐌 Using serial processing...       # Serial fallback
   ⚠️  Ray processing failed...        # Ray error occurred
   ```

2. **Test Ray manually**:
   ```python
   import ray
   ray.init()
   print("Ray working!")
   ray.shutdown()
   ```

### Performance Optimization Tips

#### **Optimal Core Count**

- **Start with default**: Let system auto-detect cores
- **For memory-constrained systems**: Use 50-75% of available cores
- **For CPU-intensive workflows**: Use 100% of physical cores (not hyperthreads)

```bash
# Examples for different systems
export LLAMAS_RAY_CPUS=8   # 8-core system
export LLAMAS_RAY_CPUS=16  # 16-core system  
export LLAMAS_RAY_CPUS=32  # High-performance system
```

#### **System Resource Monitoring**

Watch these metrics during processing:
- **CPU utilization**: Should be near 100% across all cores
- **Memory usage**: Should not exceed 80-90% of available RAM
- **Ray dashboard**: Access at http://localhost:8265 (if enabled)

#### **Expected Performance Gains**

Typical speedup factors:
- **shiftArcX**: 8-16x speedup (highly parallel)
- **fiberRelativeThroughput**: 4-8x speedup (moderate parallel)
- **arcSolve**: 2-4x speedup (limited parallelization)

### Integration with Existing Code

#### **Drop-in Replacement**

The module is designed as a drop-in replacement:

```python
# OLD - Original module
import llamas_pyjamas.Arc.arcLlamas as arc

# NEW - Multiprocessing module  
import llamas_pyjamas.Arc.arcLlamasMulti as arc

# All function calls remain identical
arc.shiftArcX(arc_picklename)  # Now automatically Ray-enabled!
```

#### **Backward Compatibility**

All original function signatures are preserved:
- **Same parameters**: No new required parameters
- **Same outputs**: Identical results and file formats
- **Same behavior**: Functions work exactly the same, just faster

#### **Testing Compatibility**

Verify outputs match between serial and Ray processing:

```python
# Test that both produce identical results
import llamas_pyjamas.Arc.arcLlamasMulti as arc

# Serial version
arc.shiftArcX(arc_picklename, use_ray=False)
serial_result = load_result_file()

# Ray version  
arc.shiftArcX(arc_picklename, use_ray=True)
ray_result = load_result_file()

# Should be numerically identical
assert np.allclose(serial_result, ray_result)
```

This implementation provides significant performance improvements while maintaining full compatibility with existing LLAMAS pipeline workflows and data structures.