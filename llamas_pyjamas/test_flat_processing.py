#!/usr/bin/env python3
"""
Test script for the new flat field processing workflow in flatLlamas.py

This script demonstrates how to use the process_flat_field_complete() function
to process LLAMAS flat field data with wavelength calibration.
"""

import os
import sys
from pathlib import Path

# Add the llamas_pyjamas package to the path if needed
sys.path.insert(0, str(Path(__file__).parent))

from Flat.flatLlamas import process_flat_field_complete
from config import OUTPUT_DIR, CALIB_DIR, DATA_DIR

def test_flat_processing():
    """
    Example test function showing how to run the complete flat field processing.
    
    You'll need to update the file paths below to point to your actual flat field files.
    """
    
    print("="*60)
    print("LLAMAS Flat Field Processing Test")
    print("="*60)
    
    # ==== UPDATE THESE PATHS TO YOUR ACTUAL FILES ====
    
    # Example flat field file paths - update these to your actual files
    red_flat_file = os.path.join(DATA_DIR, "your_red_flat.fits")
    green_flat_file = os.path.join(DATA_DIR, "your_green_flat.fits") 
    blue_flat_file = os.path.join(DATA_DIR, "your_blue_flat.fits")
    
    # Optional: specify custom arc calibration file
    # If None, will use LLAMAS_reference_arc.pkl in CALIB_DIR
    arc_calib_file = None  # or os.path.join(CALIB_DIR, "LLAMAS_reference_arc.pkl")
    
    # Output directory for results
    output_dir = OUTPUT_DIR  # or specify custom path like "/path/to/output"
    
    # Directory containing trace files
    trace_dir = CALIB_DIR   # or specify custom path like "/path/to/traces"
    
    # ================================================
    
    # Check if input files exist
    input_files = [red_flat_file, green_flat_file, blue_flat_file]
    missing_files = [f for f in input_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: The following input files are missing:")
        for f in missing_files:
            print(f"  {f}")
        print("\nPlease update the file paths in this script to point to your actual flat field files.")
        return False
    
    print("Input files:")
    print(f"  Red flat:   {red_flat_file}")
    print(f"  Green flat: {green_flat_file}")
    print(f"  Blue flat:  {blue_flat_file}")
    print(f"  Arc calib:  {arc_calib_file or 'Default (LLAMAS_reference_arc.pkl)'}")
    print(f"  Output dir: {output_dir}")
    print(f"  Trace dir:  {trace_dir}")
    print()
    
    try:
        print("Starting flat field processing workflow...")
        print("-" * 40)
        
        # Run the complete processing workflow
        results = process_flat_field_complete(
            red_flat_file=red_flat_file,
            green_flat_file=green_flat_file,
            blue_flat_file=blue_flat_file,
            arc_calib_file=arc_calib_file,
            output_dir=output_dir,
            trace_dir=trace_dir
        )
        
        print("-" * 40)
        print("✅ Processing completed successfully!")
        print()
        
        # Print results summary
        print("RESULTS SUMMARY:")
        print(f"Status: {results['processing_status']}")
        print(f"Combined flat file: {results['combined_flat_file']}")
        print(f"Calibrated flat file: {results['calibrated_flat_file']}")
        print()
        
        # List generated FITS files
        output_files = results['output_files']
        if output_files:
            print(f"Generated {len(output_files)} pixel map FITS files:")
            for i, output_file in enumerate(output_files, 1):
                filename = os.path.basename(output_file)
                print(f"  {i:2d}. {filename}")
        else:
            print("No pixel map FITS files were generated.")
        
        print()
        print("You can now use these pixel maps for flat field correction in your science data processing!")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Missing required file - {e}")
        print("Make sure all input files and calibration files exist.")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: Processing failed - {e}")
        print("Check the log files for more detailed error information.")
        import traceback
        traceback.print_exc()
        return False


def check_requirements():
    """Check if required directories and files exist."""
    
    print("Checking system requirements...")
    
    # Check if required directories exist
    dirs_to_check = [
        ("Data directory", DATA_DIR),
        ("Output directory", OUTPUT_DIR), 
        ("Calibration directory", CALIB_DIR)
    ]
    
    missing_dirs = []
    for name, path in dirs_to_check:
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (missing)")
            missing_dirs.append(path)
    
    # Check for arc calibration file
    arc_file = os.path.join(CALIB_DIR, "LLAMAS_reference_arc.pkl")
    if os.path.exists(arc_file):
        print(f"  ✅ Arc calibration: {arc_file}")
    else:
        print(f"  ⚠️  Arc calibration: {arc_file} (missing - you may need to specify custom path)")
    
    # Check for trace files
    import glob
    trace_files = glob.glob(os.path.join(CALIB_DIR, "LLAMAS_master*traces.pkl"))
    if trace_files:
        print(f"  ✅ Found {len(trace_files)} trace files in {CALIB_DIR}")
    else:
        print(f"  ❌ No trace files found in {CALIB_DIR}")
        missing_dirs.append("trace files")
    
    print()
    
    if missing_dirs:
        print("⚠️  Some required directories or files are missing.")
        print("You may need to:")
        print("  - Create missing directories")
        print("  - Run trace calibration first to generate trace files") 
        print("  - Run arc calibration to generate LLAMAS_reference_arc.pkl")
        print("  - Update paths in config.py if your setup is different")
        print()
    
    return len(missing_dirs) == 0


if __name__ == "__main__":
    print("LLAMAS Flat Field Processing Test Script")
    print("=" * 50)
    print()
    
    # Check system requirements first
    if not check_requirements():
        print("Please resolve the missing requirements before running the test.")
        sys.exit(1)
    
    print()
    
    # Run the test
    success = test_flat_processing()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("The new flat field processing workflow is working correctly.")
    else:
        print("\n💥 Test failed!")
        print("Please check the error messages above and resolve any issues.")
        sys.exit(1)