Examples
========

This section provides practical examples of using LLAMAS Pyjamas for common data reduction tasks.

Example 1: Basic Spectral Extraction
-------------------------------------

Extract spectra from a single observation:

.. code-block:: python

   from llamas_pyjamas.Extract import extractLlamas
   from llamas_pyjamas.Trace import traceLlamas
   from llamas_pyjamas.File import llamasIO
   
   # Load trace information
   trace = traceLlamas.loadTraces('master_trace.pkl')
   
   # Load science data
   data_loader = llamasIO.llamasOneCamera()
   hdu_data, header = data_loader.load_single_extension('science.fits', ext=0)
   
   # Perform extraction
   extractor = extractLlamas.ExtractLlamas(trace, hdu_data, header, optimal=True)
   
   # Save results
   extractor.saveExtraction('output/')

Example 2: Batch Processing Multiple Files
-------------------------------------------

Process multiple science files with the same calibrations:

.. code-block:: python

   import glob
   from llamas_pyjamas.Extract import extractLlamas
   
   # Get list of science files
   science_files = glob.glob('data/science/*.fits')
   
   # Load master calibrations
   trace = traceLlamas.loadTraces('calibrations/master_trace.pkl')
   arc_dict = extractLlamas.load_extractions('calibrations/arc_solution.pkl')
   
   # Process each file
   for science_file in science_files:
       print(f"Processing {science_file}")
       
       # Load data
       data_loader = llamasIO.llamasOneCamera()
       hdu_data, header = data_loader.load_single_extension(science_file)
       
       # Extract
       extractor = extractLlamas.ExtractLlamas(trace, hdu_data, header, optimal=True)
       
       # Apply wavelength calibration
       science_dict = {'extractions': [extractor], 'metadata': [{'channel': 'red'}]}
       calibrated = arcLlamas.arcTransfer(science_dict, arc_dict)
       
       # Save
       output_name = f"extracted_{os.path.basename(science_file).replace('.fits', '.pkl')}"
       extractLlamas.save_extractions(calibrated['extractions'], savefile=output_name)

Example 3: Custom Fiber Selection
----------------------------------

Extract only specific fibers of interest:

.. code-block:: python

   import numpy as np
   from llamas_pyjamas.Extract import extractLlamas
   
   # Load full extraction
   extractions, metadata = extractLlamas.load_extractions('full_extraction.pkl')
   
   # Define fibers of interest (e.g., central region)
   fibers_of_interest = np.arange(100, 200)  # Fibers 100-199
   
   # Extract subset
   for ext in extractions:
       # Create mask for selected fibers
       mask = np.zeros(ext.counts.shape[0], dtype=bool)
       mask[fibers_of_interest] = True
       
       # Apply selection
       ext.counts = ext.counts[mask]
       ext.wave = ext.wave[mask]
       ext.relative_throughput = ext.relative_throughput[mask]
   
   # Save subset
   extractLlamas.save_extractions(extractions, savefile='subset_extraction.pkl')

Example 4: White Light Image with Custom Colormap
--------------------------------------------------

Create a white light image with custom visualization:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from llamas_pyjamas.Image import WhiteLight
   from llamas_pyjamas.Extract import extractLlamas
   
   # Load extracted data
   extractions, metadata = extractLlamas.load_extractions('science_extraction.pkl')
   
   # Create white light FITS file
   whitelight_file = WhiteLight.WhiteLightFits(extractions, metadata, 'custom_whitelight.fits')
   
   # Load the created image for custom plotting
   from astropy.io import fits
   with fits.open('custom_whitelight.fits') as hdul:
       image_data = hdul[0].data
   
   # Create custom visualization
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   # Standard visualization
   im1 = ax1.imshow(image_data, cmap='viridis', origin='lower')
   ax1.set_title('Standard White Light Image')
   ax1.set_xlabel('X Position')
   ax1.set_ylabel('Y Position')
   plt.colorbar(im1, ax=ax1, label='Flux (ADU)')
   
   # Log scale visualization
   log_data = np.log10(image_data + 1)  # Add 1 to handle zeros
   im2 = ax2.imshow(log_data, cmap='plasma', origin='lower')
   ax2.set_title('Log Scale White Light Image')
   ax2.set_xlabel('X Position')
   ax2.set_ylabel('Y Position')
   plt.colorbar(im2, ax=ax2, label='Log10(Flux + 1)')
   
   plt.tight_layout()
   plt.savefig('custom_whitelight_comparison.png', dpi=300)
   plt.show()

Example 5: Quality Assessment Pipeline
--------------------------------------

Comprehensive quality assessment of extraction results:

.. code-block:: python

   import matplotlib.pyplot as plt
   from llamas_pyjamas.QA import llamasQA
   from llamas_pyjamas.Extract import extractLlamas
   
   # Load extraction results
   extractions, metadata = extractLlamas.load_extractions('science_extraction.pkl')
   
   # Create QA plots
   qa = llamasQA.LlamasQA()
   
   # Plot fiber throughput
   plt.figure(figsize=(12, 8))
   for i, ext in enumerate(extractions):
       plt.subplot(2, 2, i+1)
       fiber_numbers = np.arange(len(ext.relative_throughput))
       plt.plot(fiber_numbers, ext.relative_throughput, 'o-', alpha=0.7)
       plt.xlabel('Fiber Number')
       plt.ylabel('Relative Throughput')
       plt.title(f"{metadata[i]['channel'].title()} Channel")
       plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('fiber_throughput_qa.png', dpi=300)
   plt.show()
   
   # Generate extraction profile plots
   qa.plot_extraction_profiles(extractions, metadata, save_plots=True)
   
   # Create summary statistics
   qa.generate_qa_report(extractions, metadata, output_file='qa_report.txt')

Example 6: Wavelength Calibration Quality Check
------------------------------------------------

Assess the quality of wavelength calibrations:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from llamas_pyjamas.Arc import arcLlamas
   from llamas_pyjamas.Extract import extractLlamas
   
   # Load arc extraction with wavelength solution
   arc_extractions, arc_metadata = extractLlamas.load_extractions('arc_calibrated.pkl')
   
   # Check wavelength coverage and resolution
   for i, ext in enumerate(arc_extractions):
       channel = arc_metadata[i]['channel']
       
       plt.figure(figsize=(12, 6))
       
       # Plot example spectra from different fibers
       for fiber in [50, 150, 250]:  # Sample fibers
           if fiber < ext.wave.shape[0]:
               wavelength = ext.wave[fiber]
               flux = ext.counts[fiber]
               
               # Only plot where wavelength solution exists
               valid = wavelength > 0
               plt.plot(wavelength[valid], flux[valid], 
                       alpha=0.7, label=f'Fiber {fiber}')
       
       plt.xlabel('Wavelength (Å)')
       plt.ylabel('Flux (ADU)')
       plt.title(f'{channel.title()} Channel - Wavelength Calibration Check')
       plt.legend()
       plt.grid(True, alpha=0.3)
       
       # Calculate and display wavelength statistics
       all_waves = ext.wave[ext.wave > 0]
       wave_min, wave_max = np.min(all_waves), np.max(all_waves)
       wave_range = wave_max - wave_min
       
       plt.text(0.02, 0.98, f'Range: {wave_min:.1f} - {wave_max:.1f} Å\n'
                            f'Coverage: {wave_range:.1f} Å',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
       
       plt.tight_layout()
       plt.savefig(f'wavelength_calibration_{channel}.png', dpi=300)
       plt.show()

Example 7: Parallel Processing Large Dataset
---------------------------------------------

Use Ray for parallel processing of large datasets:

.. code-block:: python

   import ray
   from llamas_pyjamas.Extract import extractLlamas
   import glob
   
   # Initialize Ray
   ray.init()
   
   # Get list of all science files
   science_files = glob.glob('large_dataset/*.fits')
   
   # Create Ray remote extraction workers
   @ray.remote
   def process_file(filename, trace_file, output_dir):
       """Process a single file with Ray."""
       try:
           # Load trace
           trace = traceLlamas.loadTraces(trace_file)
           
           # Load science data
           data_loader = llamasIO.llamasOneCamera()
           hdu_data, header = data_loader.load_single_extension(filename)
           
           # Extract
           extractor = extractLlamas.ExtractLlamas(trace, hdu_data, header, optimal=True)
           
           # Save
           output_name = os.path.join(output_dir, f"extracted_{os.path.basename(filename)}.pkl")
           extractor.saveExtraction(output_name)
           
           return f"Successfully processed {filename}"
           
       except Exception as e:
           return f"Error processing {filename}: {str(e)}"
   
   # Submit all jobs
   futures = []
   for science_file in science_files:
       future = process_file.remote(science_file, 'master_trace.pkl', 'parallel_output/')
       futures.append(future)
   
   # Collect results
   results = ray.get(futures)
   
   # Print summary
   successful = sum(1 for r in results if "Successfully" in r)
   failed = len(results) - successful
   print(f"Processing complete: {successful} successful, {failed} failed")
   
   # Shutdown Ray
   ray.shutdown()

Example 8: Creating Master Calibrations
----------------------------------------

Create master calibration files for a observing run:

.. code-block:: python

   from llamas_pyjamas.Bias import llamasBias
   from llamas_pyjamas.Flat import flatLlamas
   from llamas_pyjamas.Trace import traceLlamas
   import glob
   
   def create_master_calibrations(bias_dir, flat_dir, output_dir):
       """Create master calibration files."""
       
       # 1. Master Bias
       print("Creating master bias...")
       bias_files = glob.glob(f"{bias_dir}/*.fits")
       bias_processor = llamasBias.BiasLlamas()
       master_bias = bias_processor.master_bias(bias_files)
       
       # Save master bias
       fits.writeto(f"{output_dir}/master_bias.fits", master_bias, overwrite=True)
       
       # 2. Master Flat and Trace
       print("Processing flat fields and creating traces...")
       flat_files = glob.glob(f"{flat_dir}/*.fits")
       
       all_traces = []
       for flat_file in flat_files:
           # Load and bias-subtract flat
           flat_data = fits.getdata(flat_file)
           flat_corrected = flat_data - master_bias
           
           # Trace fibers
           tracer = traceLlamas.TraceLlamas()
           tracer.traceSingleCamera(flat_corrected)
           all_traces.append(tracer)
       
       # Save master trace
       traceLlamas.saveTraces(all_traces, f"{output_dir}/master_traces.pkl")
       
       # 3. Create normalized flat field
       flat_processor = flatLlamas.LlamasFlatFielding()
       # Extract flat field spectra first, then create normalized flat
       # (This would require the full extraction pipeline)
       
       print(f"Master calibrations saved to {output_dir}/")
   
   # Usage
   create_master_calibrations('calibrations/bias/', 
                            'calibrations/flats/', 
                            'master_calibrations/')

Running the Examples
--------------------

To run these examples:

1. **Download sample data** from the LLAMAS data repository
2. **Modify file paths** to match your data location
3. **Install dependencies** as described in the installation guide
4. **Run examples in order** - some depend on outputs from previous examples

Each example includes error handling and can be adapted for your specific data and requirements.