Examples
========

This section provides practical examples of using LLAMAS PyJamas for various data reduction tasks.

Basic Extraction Example
------------------------

Extract spectra from a single observation::

    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
    from llamas_pyjamas.Trace.traceLlamas import TraceLlamas
    
    # Load traces
    trace_data = TraceLlamas()
    trace_data.loadTraces('trace_file.pkl')
    
    # Extract spectra  
    extractor = ExtractLlamas()
    extractions = extractor.extractOptimal('science.fits', trace_data)
    
    # Save results
    extractor.saveExtraction(extractions, 'extracted_spectra.pkl')

Wavelength Calibration Example
------------------------------

Calculate wavelength solution from arc lamp::

    import llamas_pyjamas.Arc.arcLlamas as arc
    
    # Process arc lamp observation
    wavelength_soln = arc.processArc('arc_lamp.fits', 'trace_file.pkl')
    
    # Apply to science data
    calibrated_spectra = arc.applyWavelengthSolution(
        'extracted_spectra.pkl', 
        wavelength_soln
    )

Parallel Processing Example
---------------------------

Process multiple files using Ray::

    import ray
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamasRay
    
    # Initialize Ray
    ray.init()
    
    # Process files in parallel
    files = ['obs1.fits', 'obs2.fits', 'obs3.fits']
    actors = [ExtractLlamasRay.remote() for _ in range(len(files))]
    
    results = ray.get([
        actor.process.remote(f, 'traces.pkl') 
        for actor, f in zip(actors, files)
    ])
    
    ray.shutdown()

GUI Example
-----------

Launch the interactive extraction GUI::

    from llamas_pyjamas.GUI import guiExtract
    
    # Start GUI
    app = guiExtract.LlamasGUI()
    app.run()

Data Cube Construction
----------------------

Create 3D data cubes from RSS files::

    from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor
    
    # Initialize cube constructor
    constructor = CubeConstructor()
    
    # Build cube from RSS files
    rss_files = ['obs1_rss.fits', 'obs2_rss.fits']
    cube = constructor.constructCube(
        rss_files, 
        wavelength_range=(3700, 7000),
        spatial_sampling=0.75
    )
    
    # Save cube
    constructor.saveCube(cube, 'data_cube.fits')