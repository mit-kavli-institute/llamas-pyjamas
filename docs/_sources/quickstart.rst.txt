Quick Start Guide
=================

This guide will walk you through the basic steps of processing LLAMAS data using the pyjamas pipeline.

Basic Pipeline Overview
-----------------------

The LLAMAS data reduction pipeline consists of several key steps:

1. **Bias Subtraction** - Remove detector bias and dark current
2. **Flat Field Correction** - Correct for pixel-to-pixel variations
3. **Fiber Tracing** - Identify and trace fiber positions across the detector
4. **Spectral Extraction** - Extract 1D spectra from traced fibers
5. **Wavelength Calibration** - Apply wavelength solutions using arc lamps
6. **White Light Image Generation** - Create 2D reconstructed images

Step-by-Step Example
--------------------

1. Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from llamas_pyjamas.Bias import llamasBias
   from llamas_pyjamas.Flat import flatLlamas
   from llamas_pyjamas.Trace import traceLlamas
   from llamas_pyjamas.Extract import extractLlamas
   from llamas_pyjamas.Arc import arcLlamas
   from llamas_pyjamas.Image import WhiteLight
   from llamas_pyjamas.File import llamasIO

2. Load and Prepare Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load FITS data
   data_loader = llamasIO.llamasAllCameras()
   flat_data = data_loader.load_fits_file('flat_field.fits')
   science_data = data_loader.load_fits_file('science_target.fits')
   arc_data = data_loader.load_fits_file('thar_arc.fits')

3. Bias Subtraction
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create master bias
   bias_processor = llamasBias.BiasLlamas()
   master_bias = bias_processor.master_bias(bias_files)
   
   # Subtract bias from all frames
   flat_data_corrected = flat_data - master_bias
   science_data_corrected = science_data - master_bias
   arc_data_corrected = arc_data - master_bias

4. Flat Field Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create normalized flat field
   flat_processor = flatLlamas.LlamasFlatFielding()
   normalized_flat = flat_processor.flatcube(flat_extractions, 'normalized_flat.fits')

5. Fiber Tracing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Trace fiber positions using flat field
   tracer = traceLlamas.TraceLlamas()
   tracer.traceSingleCamera(flat_data_corrected)
   
   # Generate profile weights for optimal extraction
   fiberimg, profimg, bpmask = tracer.profileFit()

6. Spectral Extraction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Extract spectra using optimal extraction
   extractor = extractLlamas.ExtractLlamas(tracer, science_data_corrected, header, optimal=True)
   
   # Save extraction results
   extractor.saveExtraction('output')

7. Wavelength Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Extract arc spectra
   arc_extractor = extractLlamas.ExtractLlamas(tracer, arc_data_corrected, header, optimal=True)
   
   # Calculate fiber shifts relative to reference
   arcLlamas.shiftArcX('arc_extraction.pkl')
   
   # Solve wavelength calibration
   arcLlamas.arcSolve('arc_extraction_shifted.pkl')
   
   # Transfer calibration to science data
   science_dict = extractLlamas.load_extractions('science_extraction.pkl')
   arc_dict = extractLlamas.load_extractions('LLAMAS_reference_arc.pkl')
   calibrated_science = arcLlamas.arcTransfer(science_dict, arc_dict)

8. Generate White Light Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create white light FITS file
   extractions, metadata = extractLlamas.load_extractions('calibrated_science.pkl')
   whitelight_file = WhiteLight.WhiteLightFits(extractions, metadata, 'whitelight.fits')
   
   # Generate quick-look image
   WhiteLight.WhiteLight(extractions, metadata, ds9plot=True)

Complete Processing Script
--------------------------

Here's a complete script that processes a full dataset:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Complete LLAMAS data reduction script
   """
   
   import os
   from llamas_pyjamas.Bias import llamasBias
   from llamas_pyjamas.Flat import flatLlamas
   from llamas_pyjamas.Trace import traceLlamas
   from llamas_pyjamas.Extract import extractLlamas
   from llamas_pyjamas.Arc import arcLlamas
   from llamas_pyjamas.Image import WhiteLight
   from llamas_pyjamas.File import llamasIO
   
   def process_llamas_observation(bias_files, flat_files, science_files, arc_files, output_dir):
       """Process a complete LLAMAS observation."""
       
       # Create output directory
       os.makedirs(output_dir, exist_ok=True)
       
       # 1. Create master bias
       print("Creating master bias...")
       bias_processor = llamasBias.BiasLlamas()
       master_bias = bias_processor.master_bias(bias_files)
       
       # 2. Process flat fields
       print("Processing flat fields...")
       flat_data = llamasIO.llamasAllCameras().load_fits_file(flat_files[0])
       flat_corrected = flat_data - master_bias
       
       # 3. Trace fibers
       print("Tracing fibers...")
       tracer = traceLlamas.TraceLlamas()
       tracer.traceSingleCamera(flat_corrected)
       fiberimg, profimg, bpmask = tracer.profileFit()
       
       # 4. Extract flat field spectra
       print("Extracting flat field spectra...")
       flat_extractor = extractLlamas.ExtractLlamas(tracer, flat_corrected, flat_data.header)
       flat_extractor.saveExtraction(output_dir)
       
       # 5. Process science data
       print("Processing science data...")
       for science_file in science_files:
           science_data = llamasIO.llamasAllCameras().load_fits_file(science_file)
           science_corrected = science_data - master_bias
           
           # Extract science spectra
           science_extractor = extractLlamas.ExtractLlamas(
               tracer, science_corrected, science_data.header, optimal=True
           )
           science_extractor.saveExtraction(output_dir)
       
       # 6. Wavelength calibration
       print("Performing wavelength calibration...")
       arc_data = llamasIO.llamasAllCameras().load_fits_file(arc_files[0])
       arc_corrected = arc_data - master_bias
       
       arc_extractor = extractLlamas.ExtractLlamas(tracer, arc_corrected, arc_data.header)
       arc_extractor.saveExtraction(output_dir)
       
       # Calculate wavelength solution
       arcLlamas.shiftArcX(os.path.join(output_dir, 'arc_extraction.pkl'))
       arcLlamas.arcSolve(os.path.join(output_dir, 'arc_extraction_shifted.pkl'))
       
       # 7. Generate white light images
       print("Generating white light images...")
       extractions, metadata = extractLlamas.load_extractions(
           os.path.join(output_dir, 'science_extraction.pkl')
       )
       WhiteLight.WhiteLightFits(extractions, metadata, 
                                os.path.join(output_dir, 'whitelight.fits'))
       
       print(f"Processing complete! Results saved to {output_dir}")
   
   if __name__ == "__main__":
       # Example usage
       bias_files = ['bias_01.fits', 'bias_02.fits', 'bias_03.fits']
       flat_files = ['flat_01.fits']
       science_files = ['science_target.fits']
       arc_files = ['thar_arc.fits']
       
       process_llamas_observation(bias_files, flat_files, science_files, arc_files, 'output')

Next Steps
----------

* See the :doc:`tutorials` for more detailed examples
* Check the :doc:`examples` for specific use cases
* Read the API documentation for detailed function descriptions
* Visit the :doc:`contributing` guide if you'd like to contribute to the project