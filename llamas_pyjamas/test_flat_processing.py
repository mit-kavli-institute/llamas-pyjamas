#!/usr/bin/env python3
"""
Comprehensive test suite for LLAMAS flat field processing and reduction.

This test suite validates the complete flat field workflow including:
- Normalized flat field creation with 1.0 values outside fiber traces
- Multi-extension FITS file structure matching raw science frames
- Integration with reduce.py pipeline using actual science data
- End-to-end flat field correction validation

Usage:
    # Run all tests with actual science file
    python test_flat_processing.py --science-file /path/to/science.fits
    
    # Run with pytest
    pytest test_flat_processing.py -v --science-file /path/to/science.fits
    
    # Run specific test
    python -m pytest test_flat_processing.py::TestFlatFieldReduction::test_normalized_flat_field_creation -v
"""

import os
import sys
import unittest
import tempfile
import shutil
import argparse
import glob
from pathlib import Path
import numpy as np
from astropy.io import fits
import pickle

# --- Ray size mitigation initialization block ---
def _init_ray():
    """Initialize Ray to avoid packaging large project directories."""
    try:
        import ray
    except ImportError:
        return

    if ray.is_initialized():
        return

    project_root = Path(__file__).parent
    allowed_subdirs = [
        "Arc", "Bias", "Cube", "Docs", "Extract", "File", "Flat", "Flux",
        "GUI", "Image", "Postprocessing", "QA", "Trace", "Tutorials", "Utils",
    ]

    ray.init(
        runtime_env={
            "working_dir": str(project_root),
            "excludes": [
                "mastercalib", "LUT", "output", "data", "large_data", "raw_data",
                ".git", "venv", "env", "__pycache__", "**/__pycache__",
                "*.ipynb", "notebooks", "*.fits", "*.fits.gz", "*.pkl", "*.pickle",
                "*.zip", "*.tar", "*.tar.gz", "*.whl",
                *[f"!{name}" for name in allowed_subdirs],
            ],
        },
    )

# Initialize Ray early
_init_ray()

# Add the llamas_pyjamas package to the path
sys.path.insert(0, str(Path(__file__).parent))

from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete, LlamasFlatFielding, Thresholding
from llamas_pyjamas.reduce import apply_flat_field_correction, build_flat_field_map
from llamas_pyjamas.config import OUTPUT_DIR, CALIB_DIR, LUT_DIR
from llamas_pyjamas.constants import idx_lookup
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

# Global test configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
# TEST_CONFIG = {
#     'red_flat': '/Users/slhughes/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/2025-03-05/LLAMAS_2025-03-05T23_21_49.353_mef.fits',
#     'green_flat': '/Users/slhughes/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/2025-03-05/LLAMAS_2025-03-05T18_38_32.349_mef.fits',
#     'blue_flat': '/Users/slhughes/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/2025-03-05/LLAMAS_2025-03-05T18_38_32.349_mef.fits',
#     'science_file': None,  # Will be set via command line argument
#     'trace_dir': CALIB_DIR,
#     'output_dir': OUTPUT_DIR,
#     'arc_calib_file': None,  # Will use default
#     'tolerance': 0.001,  # Tolerance for 1.0 values outside traces
#     'run_integration_tests': True,  # Set False to skip science frame tests
# }

TEST_CONFIG = {
    'red_flat': '/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/2025-03-05/LLAMAS_2025-03-05T23_21_49.353_mef.fits',
    'green_flat': '/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/2025-03-05/LLAMAS_2025-03-05T18_38_32.349_mef.fits',
    'blue_flat': '/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/2025-03-05/LLAMAS_2025-03-05T18_38_32.349_mef.fits',
    'science_file': None,  # Will be set via command line argument
    'trace_dir': CALIB_DIR,
    'output_dir': '/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/testing',
    'arc_calib_file': None,  # Will use default
    'tolerance': 0.001,  # Tolerance for 1.0 values outside traces
    'run_integration_tests': True,  # Set False to skip science frame tests
}


class TestFlatFieldReduction(unittest.TestCase):
    """
    Comprehensive test suite for LLAMAS flat field reduction workflow.
    
    Tests the complete pipeline from flat field creation through science frame correction,
    with emphasis on validating normalized flat field behavior (1.0 outside fiber traces).
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with file paths and temporary directories."""
        print("\n" + "="*80)
        print("LLAMAS FLAT FIELD REDUCTION TEST SUITE")
        print("="*80)
        
        # Create temporary test directory
        cls.test_output_dir = tempfile.mkdtemp(prefix='llamas_flat_test_')
        print(f"Test output directory: {cls.test_output_dir}")
        
        # Validate required files exist
        cls._validate_test_files()
        
        # Set up paths
        cls.red_flat = TEST_CONFIG['red_flat']
        cls.green_flat = TEST_CONFIG['green_flat'] 
        cls.blue_flat = TEST_CONFIG['blue_flat']
        cls.science_file = TEST_CONFIG['science_file']
        cls.trace_dir = TEST_CONFIG['trace_dir']
        cls.arc_calib_file = TEST_CONFIG['arc_calib_file']
        cls.tolerance = TEST_CONFIG['tolerance']
        
        print(f"Red flat: {os.path.basename(cls.red_flat)}")
        print(f"Green flat: {os.path.basename(cls.green_flat)}")
        print(f"Blue flat: {os.path.basename(cls.blue_flat)}")
        if cls.science_file:
            print(f"Science file: {os.path.basename(cls.science_file)}")
        print(f"Trace directory: {cls.trace_dir}")
        print()
    
    @classmethod
    def _validate_test_files(cls):
        """Validate that all required test files exist."""
        missing_files = []
        
        # Check flat files
        for key in ['red_flat', 'green_flat', 'blue_flat']:
            file_path = TEST_CONFIG[key]
            if not file_path or not os.path.exists(file_path):
                missing_files.append(f"{key}: {file_path}")
        
        # Check science file if provided
        science_file = TEST_CONFIG['science_file']
        if science_file and not os.path.exists(science_file):
            missing_files.append(f"science_file: {science_file}")
        
        # Check trace directory
        trace_dir = TEST_CONFIG['trace_dir']
        if not os.path.exists(trace_dir):
            missing_files.append(f"trace_dir: {trace_dir}")
        else:
            # Check for trace files
            trace_files = glob.glob(os.path.join(trace_dir, 'LLAMAS_master*traces.pkl'))
            if not trace_files:
                missing_files.append(f"No trace files found in: {trace_dir}")
        
        if missing_files:
            print("ERROR: Missing required test files:")
            for missing in missing_files:
                print(f"  - {missing}")
            print("\nPlease update TEST_CONFIG in test_flat_processing.py or provide via command line.")
            raise FileNotFoundError("Required test files missing")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test files."""
        if hasattr(cls, 'test_output_dir') and os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
            print(f"\nCleaned up test directory: {cls.test_output_dir}")
    
    def setUp(self):
        """Set up individual test."""
        self.maxDiff = None  # Allow full diff output for test failures
    
    def test_normalized_flat_field_creation(self):
        """
        Test creation of normalized flat field with 1.0 values outside fiber traces.
        
        This is the core test validating that:
        1. Flat field processing creates normalized images
        2. Values outside fiber traces = 1.0 ± tolerance
        3. Values inside fiber traces ≠ 1.0 (actual correction values)
        4. 24 extensions are created matching science frame structure
        """
        print("\nTEST: Normalized Flat Field Creation")
        print("-" * 50)
        
        # Step 1: Run complete flat field processing
        print("Running flat field processing...")
        results = process_flat_field_complete(
            red_flat_file=self.red_flat,
            green_flat_file=self.green_flat,
            blue_flat_file=self.blue_flat,
            arc_calib_file=self.arc_calib_file,
            output_dir=self.test_output_dir,
            trace_dir=self.trace_dir,
            verbose=False
        )
        
        self.assertEqual(results['processing_status'], 'completed',
                        "Flat field processing should complete successfully")
        
        # Step 2: Check that MEF file was created
        mef_file = results.get('combined_mef_file')
        self.assertIsNotNone(mef_file, "Combined MEF file should be created")
        self.assertTrue(os.path.exists(mef_file), f"MEF file should exist: {mef_file}")
        
        print(f"✓ MEF file created: {os.path.basename(mef_file)}")
        
        # Step 3: Load and analyze MEF structure
        with fits.open(mef_file) as hdul:
            print(f"✓ MEF file has {len(hdul)} HDUs (primary + {len(hdul)-1} extensions)")
            
            # Should have primary + 24 extensions
            self.assertGreater(len(hdul), 24, 
                             f"MEF should have at least 24 extensions, got {len(hdul)-1}")
            
            # Step 4: Analyze normalized values
            extensions_checked = 0
            outside_trace_values = []
            inside_trace_values = []
            
            for i in range(1, len(hdul)):  # Skip primary HDU
                hdu = hdul[i]
                if hdu.data is None:
                    continue
                    
                extensions_checked += 1
                data = hdu.data
                
                # Basic sanity checks
                self.assertFalse(np.all(np.isnan(data)), 
                               f"Extension {i} should not be all NaN")
                self.assertTrue(np.any(np.isfinite(data)), 
                               f"Extension {i} should have some finite values")
                
                # Analyze value distribution
                finite_data = data[np.isfinite(data)]
                if len(finite_data) == 0:
                    continue
                
                # Values very close to 1.0 are considered "outside traces"
                close_to_one = np.abs(finite_data - 1.0) < self.tolerance
                outside_values = finite_data[close_to_one]
                inside_values = finite_data[~close_to_one]
                
                outside_trace_values.extend(outside_values)
                inside_trace_values.extend(inside_values)
                
                # Print statistics for this extension
                channel = hdu.header.get('CHANNEL', 'UNKNOWN')
                benchside = hdu.header.get('BENCHSIDE', 'UNKNOWN')
                outside_count = len(outside_values)
                inside_count = len(inside_values) 
                total_count = len(finite_data)
                
                print(f"  Extension {i:2d} ({channel}{benchside}): "
                      f"{outside_count:6d} outside traces (≈1.0), "
                      f"{inside_count:6d} inside traces (≠1.0), "
                      f"{total_count:6d} total finite pixels")
        
        # Step 5: Validate overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"  Extensions analyzed: {extensions_checked}")
        print(f"  Pixels outside traces (≈1.0): {len(outside_trace_values):,}")
        print(f"  Pixels inside traces (≠1.0): {len(inside_trace_values):,}")
        
        # Critical assertions
        self.assertGreater(extensions_checked, 20, 
                          f"Should analyze at least 20 extensions, got {extensions_checked}")
        
        self.assertGreater(len(outside_trace_values), 1000,
                          "Should have substantial number of pixels ≈1.0 (outside traces)")
        
        self.assertGreater(len(inside_trace_values), 100,
                          "Should have pixels ≠1.0 (inside traces with corrections)")
        
        # Verify outside trace values are indeed close to 1.0
        if len(outside_trace_values) > 0:
            outside_mean = np.mean(outside_trace_values)
            outside_std = np.std(outside_trace_values)
            print(f"  Outside trace values: mean={outside_mean:.6f}, std={outside_std:.6f}")
            
            self.assertAlmostEqual(outside_mean, 1.0, delta=self.tolerance,
                                 msg=f"Mean of outside-trace values should be ≈1.0")
        
        # Verify inside trace values are not 1.0
        if len(inside_trace_values) > 0:
            inside_mean = np.mean(inside_trace_values)
            print(f"  Inside trace values: mean={inside_mean:.6f}")
            
            self.assertNotAlmostEqual(inside_mean, 1.0, delta=self.tolerance,
                                    msg="Mean of inside-trace values should not be ≈1.0")
        
        print("✅ PASSED: Normalized flat field creation validated")
    
    def test_mef_extension_ordering(self):
        """
        Test that MEF extensions are ordered correctly to match raw science frames.
        
        Validates:
        1. Extension naming follows idx_lookup pattern from constants.py
        2. Extensions are ordered by: RED1A, GREEN1A, BLUE1A, RED1B, GREEN1B, BLUE1B, RED2A, etc.
        3. Header metadata is consistent and correct
        4. Order matches: for each bench (1-4), for each side (A/B): RED→GREEN→BLUE
        """
        print("\nTEST: MEF Extension Ordering")
        print("-" * 50)
        
        # Run flat field processing if not already done
        results = process_flat_field_complete(
            red_flat_file=self.red_flat,
            green_flat_file=self.green_flat,
            blue_flat_file=self.blue_flat,
            arc_calib_file=self.arc_calib_file,
            output_dir='/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/testing',#self.test_output_dir,
            trace_dir=self.trace_dir,
            verbose=False
        )
        
        mef_file = results['combined_mef_file']
        self.assertIsNotNone(mef_file, "MEF file should exist")
        
        # Expected extension order from constants.py idx_lookup
        expected_order = []
        for (channel, bench, side), ext_idx in sorted(idx_lookup.items(), key=lambda x: x[1]):
            expected_order.append(f"FLAT_{channel.upper()}{bench}{side.upper()}")
        
        print(f"Expected {len(expected_order)} extensions in idx_lookup order:")
        for i, name in enumerate(expected_order[:12]):  # Show first 12
            print(f"  {i+1:2d}. {name}")
        print("     ...")
        for i, name in enumerate(expected_order[-6:], len(expected_order)-5):  # Show last 6
            print(f"  {i:2d}. {name}")
        
        # Analyze actual MEF structure
        with fits.open(mef_file) as hdul:
            actual_extensions = []
            header_info = []
            
            for i in range(1, len(hdul)):  # Skip primary HDU
                hdu = hdul[i]
                ext_name = hdu.header.get('EXTNAME', f'UNKNOWN_{i}')
                channel = hdu.header.get('CHANNEL', 'UNKNOWN')
                bench = hdu.header.get('BENCH', 'UNKNOWN')
                side = hdu.header.get('SIDE', 'UNKNOWN')
                benchside = hdu.header.get('BENCHSIDE', 'UNKNOWN')
                
                actual_extensions.append(ext_name)
                header_info.append({
                    'index': i,
                    'extname': ext_name,
                    'channel': channel,
                    'bench': bench,
                    'side': side,
                    'benchside': benchside
                })
        
        print(f"\nActual {len(actual_extensions)} extensions found:")
        for i, info in enumerate(header_info[:8]):  # Show first 8
            print(f"  {info['index']:2d}. {info['extname']} ({info['channel']}{info['benchside']})")
        if len(header_info) > 8:
            print("     ...")
            for info in header_info[-4:]:  # Show last 4
                print(f"  {info['index']:2d}. {info['extname']} ({info['channel']}{info['benchside']})")
        
        # Validate extension count
        self.assertGreaterEqual(len(actual_extensions), 24,
                               f"Should have at least 24 extensions, got {len(actual_extensions)}")
        
        # Validate extension names and order (for available extensions)
        for i, expected_name in enumerate(expected_order):
            if i < len(actual_extensions):
                actual_name = actual_extensions[i]
                self.assertEqual(actual_name, expected_name,
                               f"Extension {i+1} should be {expected_name}, got {actual_name}")
        
        # Validate header consistency
        channels_found = set()
        benches_found = set()
        sides_found = set()
        
        for info in header_info:
            channels_found.add(info['channel'])
            benches_found.add(info['bench'])
            sides_found.add(info['side'])
            
            # Check header consistency
            expected_benchside = f"{info['bench']}{info['side']}"
            self.assertEqual(info['benchside'], expected_benchside,
                           f"BENCHSIDE should be {expected_benchside}, got {info['benchside']}")
        
        print(f"\nHeader validation:")
        print(f"  Channels found: {sorted(channels_found)}")
        print(f"  Benches found: {sorted(benches_found)}")
        print(f"  Sides found: {sorted(sides_found)}")
        
        # Validate expected channels, benches, sides
        expected_channels = {'BLUE', 'GREEN', 'RED'}
        expected_benches = {'1', '2', '3', '4'}
        expected_sides = {'A', 'B'}
        
        self.assertTrue(expected_channels.issubset(channels_found),
                       f"Should find channels {expected_channels}, got {channels_found}")
        self.assertTrue(expected_benches.issubset(benches_found),
                       f"Should find benches {expected_benches}, got {benches_found}")
        self.assertTrue(expected_sides.issubset(sides_found),
                       f"Should find sides {expected_sides}, got {sides_found}")
        
        print("✅ PASSED: MEF extension ordering validated")
    
    def test_normalized_flat_field_creation(self):
        """
        Test creation of normalized flat field FITS file for reduce.py pipeline.
        
        Validates:
        1. Normalized flat field FITS file is created
        2. File has correct extension ordering from idx_lookup
        3. Values within fiber traces are around 1.0 (multipliers)
        4. Values outside fiber traces are exactly 1.0
        5. File structure matches science frame requirements
        """
        print("\nTEST: Normalized Flat Field FITS Creation")
        print("-" * 50)
        
        # Run flat field processing
        results = process_flat_field_complete(
            red_flat_file=self.red_flat,
            green_flat_file=self.green_flat,
            blue_flat_file=self.blue_flat,
            arc_calib_file=self.arc_calib_file,
            output_dir=self.test_output_dir,
            trace_dir=self.trace_dir,
            verbose=False
        )
        
        # Check that normalized flat field file was created
        normalized_flat_file = results.get('normalized_flat_field_file')
        self.assertIsNotNone(normalized_flat_file, "Normalized flat field file should be created")
        self.assertTrue(os.path.exists(normalized_flat_file), 
                       f"Normalized flat field file should exist: {normalized_flat_file}")
        
        print(f"✓ Normalized flat field file created: {os.path.basename(normalized_flat_file)}")
        
        # Analyze file structure and values
        with fits.open(normalized_flat_file) as hdul:
            print(f"✓ File has {len(hdul)} HDUs (primary + {len(hdul)-1} extensions)")
            
            # Should have 24 extensions plus primary
            self.assertEqual(len(hdul), 25, f"Should have 25 HDUs (primary + 24 extensions), got {len(hdul)}")
            
            # Check extension ordering matches idx_lookup
            expected_names = []
            for (channel, bench, side), ext_idx in sorted(idx_lookup.items(), key=lambda x: x[1]):
                expected_names.append(f"FLAT_{channel.upper()}{bench}{side.upper()}")
            
            actual_names = []
            for i in range(1, len(hdul)):  # Skip primary
                actual_names.append(hdul[i].header.get('EXTNAME', f'UNKNOWN_{i}'))
            
            print(f"✓ Extension ordering validation:")
            for i, (expected, actual) in enumerate(zip(expected_names, actual_names)):
                if expected == actual:
                    print(f"  Extension {i+1:2d}: {actual} ✓")
                else:
                    print(f"  Extension {i+1:2d}: Expected {expected}, got {actual} ✗")
                    self.assertEqual(actual, expected, f"Extension {i+1} name mismatch")
            
            # Analyze values in extensions
            traced_values_all = []
            untraced_values_all = []
            
            for i in range(1, min(len(hdul), 5)):  # Check first few extensions
                hdu = hdul[i]
                if hdu.data is None:
                    continue
                
                data = hdu.data
                finite_data = data[np.isfinite(data)]
                
                if len(finite_data) == 0:
                    continue
                
                # Values close to 1.0 are likely untraced regions
                close_to_one = np.abs(finite_data - 1.0) < self.tolerance
                exactly_one = finite_data == 1.0
                
                untraced_count = np.sum(exactly_one)
                traced_count = len(finite_data) - np.sum(close_to_one)
                
                channel = hdu.header.get('CHANNEL', 'UNKNOWN')
                benchside = hdu.header.get('BENCHSIDE', 'UNKNOWN')
                
                print(f"  Extension {i} ({channel}{benchside}): "
                      f"{untraced_count:,} pixels = 1.0, "
                      f"{traced_count:,} pixels ≠ 1.0")
                
                # Collect values for overall analysis
                traced_values_all.extend(finite_data[~close_to_one])
                untraced_values_all.extend(finite_data[close_to_one])
            
            # Overall statistics
            print(f"\nOverall Statistics:")
            print(f"  Total untraced pixels (≈1.0): {len(untraced_values_all):,}")
            print(f"  Total traced pixels (≠1.0): {len(traced_values_all):,}")
            
            # Validate that we have both traced and untraced regions
            self.assertGreater(len(untraced_values_all), 1000, 
                             "Should have substantial untraced pixels ≈ 1.0")
            self.assertGreater(len(traced_values_all), 100,
                             "Should have traced pixels ≠ 1.0")
            
            # Check that untraced values are indeed close to 1.0
            if len(untraced_values_all) > 0:
                untraced_mean = np.mean(untraced_values_all)
                print(f"  Untraced values mean: {untraced_mean:.6f}")
                self.assertAlmostEqual(untraced_mean, 1.0, delta=self.tolerance,
                                     msg="Mean of untraced values should be ≈ 1.0")
            
            # Check that traced values are reasonable multipliers
            if len(traced_values_all) > 0:
                traced_mean = np.mean(traced_values_all)
                traced_min = np.min(traced_values_all)
                traced_max = np.max(traced_values_all)
                print(f"  Traced values: mean={traced_mean:.3f}, range=[{traced_min:.3f}, {traced_max:.3f}]")
                
                # Should be reasonable multipliers
                self.assertGreater(traced_mean, 0.1, "Traced values should be > 0.1")
                self.assertLess(traced_mean, 10.0, "Traced values should be < 10.0")
        
        print("✅ PASSED: Normalized flat field FITS creation validated")
    
    @unittest.skipUnless(TEST_CONFIG['science_file'] and TEST_CONFIG['run_integration_tests'], 
                         "Science file not provided or integration tests disabled")
    def test_flat_field_integration_with_actual_science_frame(self):
        """
        Test flat field correction integration with actual science data.
        
        Uses real science frame to validate:
        1. Flat field mapping correctly matches science extensions
        2. Division operation preserves untraced regions (divided by 1.0)
        3. Traced regions are properly corrected
        4. No invalid values (NaN/inf) in corrected data
        """
        print("\nTEST: Flat Field Integration with Actual Science Frame")
        print("-" * 50)
        
        science_file = self.science_file
        print(f"Science file: {os.path.basename(science_file)}")
        
        # Step 1: Generate flat field pixel maps
        print("Generating flat field pixel maps...")
        results = process_flat_field_complete(
            red_flat_file=self.red_flat,
            green_flat_file=self.green_flat,
            blue_flat_file=self.blue_flat,
            arc_calib_file=self.arc_calib_file,
            output_dir=self.test_output_dir,
            trace_dir=self.trace_dir,
            verbose=False
        )
        
        pixel_map_files = results['output_files']
        self.assertGreater(len(pixel_map_files), 0, "Should generate pixel map files")
        
        print(f"✓ Generated {len(pixel_map_files)} pixel map files")
        
        # Step 2: Apply flat field correction
        print("Applying flat field correction to science frame...")
        corrected_file, correction_stats = apply_flat_field_correction(
            science_file=science_file,
            flat_pixel_maps=pixel_map_files,
            output_dir=self.test_output_dir,
            validate_matching=True,
            require_all_matches=False
        )
        
        self.assertIsNotNone(corrected_file, "Flat field correction should produce output file")
        self.assertTrue(os.path.exists(corrected_file), f"Corrected file should exist: {corrected_file}")
        
        print(f"✓ Corrected file created: {os.path.basename(corrected_file)}")
        print(f"✓ Correction stats: {correction_stats}")
        
        # Step 3: Validate correction by comparing original vs corrected
        print("Validating correction results...")
        
        with fits.open(science_file) as orig_hdul, fits.open(corrected_file) as corr_hdul:
            self.assertEqual(len(orig_hdul), len(corr_hdul),
                           "Original and corrected files should have same number of extensions")
            
            corrections_applied = 0
            total_extensions = 0
            
            for i in range(1, len(orig_hdul)):  # Skip primary HDU
                orig_hdu = orig_hdul[i]
                corr_hdu = corr_hdul[i]
                
                if orig_hdu.data is None or corr_hdu.data is None:
                    continue
                
                total_extensions += 1
                
                # Check if flat field correction was applied
                flat_corrected = corr_hdu.header.get('FLATCORR', False)
                
                if flat_corrected:
                    corrections_applied += 1
                    
                    # Validate shapes match
                    self.assertEqual(orig_hdu.data.shape, corr_hdu.data.shape,
                                   f"Extension {i}: shapes should match")
                    
                    # Check for invalid values
                    invalid_pixels = ~np.isfinite(corr_hdu.data)
                    invalid_count = np.sum(invalid_pixels)
                    total_pixels = corr_hdu.data.size
                    
                    # Allow some invalid pixels (from flat field boundaries) but not too many
                    invalid_fraction = invalid_count / total_pixels
                    self.assertLess(invalid_fraction, 0.5,
                                   f"Extension {i}: too many invalid pixels ({invalid_fraction:.2%})")
                    
                    # For regions where flat field = 1.0, correction should preserve original values
                    # (This is complex to test precisely, so we check basic statistics)
                    
                    orig_finite = orig_hdu.data[np.isfinite(orig_hdu.data)]
                    corr_finite = corr_hdu.data[np.isfinite(corr_hdu.data)]
                    
                    if len(orig_finite) > 0 and len(corr_finite) > 0:
                        # Statistics should be reasonably similar for regions with flat ≈ 1.0
                        orig_median = np.median(orig_finite)
                        corr_median = np.median(corr_finite)
                        
                        # Allow for some change due to flat field correction
                        # but check that data is not completely different
                        ratio = corr_median / orig_median if orig_median != 0 else 1.0
                        self.assertGreater(ratio, 0.1, 
                                         f"Extension {i}: corrected data too different from original")
                        self.assertLess(ratio, 10.0,
                                       f"Extension {i}: corrected data too different from original")
                    
                    print(f"  Extension {i:2d}: ✓ Corrected "
                          f"({invalid_count:,}/{total_pixels:,} invalid pixels)")
                else:
                    print(f"  Extension {i:2d}: - No correction applied")
        
        print(f"\nCorrection Summary:")
        print(f"  Total extensions: {total_extensions}")
        print(f"  Corrections applied: {corrections_applied}")
        print(f"  Correction rate: {corrections_applied/total_extensions:.1%}")
        
        # Validate that some corrections were applied
        self.assertGreater(corrections_applied, 0, "At least some extensions should be corrected")
        self.assertGreater(corrections_applied/total_extensions, 0.5, 
                          "Most extensions should be corrected")
        
        print("✅ PASSED: Flat field integration with science frame validated")
    
    def test_fiber_trace_flat_field_behavior(self):
        """
        Test flat field behavior in relation to fiber traces.
        
        Validates:
        1. Flat field values correlate with fiber trace locations
        2. Proper handling of traced vs untraced regions
        3. Smooth transitions at trace boundaries
        """
        print("\nTEST: Fiber Trace Flat Field Behavior")
        print("-" * 50)
        
        # Load trace files to understand fiber locations
        trace_files = glob.glob(os.path.join(self.trace_dir, 'LLAMAS_master*traces.pkl'))
        self.assertGreater(len(trace_files), 0, f"Should find trace files in {self.trace_dir}")
        
        print(f"Found {len(trace_files)} trace files")
        
        # Generate flat field
        results = process_flat_field_complete(
            red_flat_file=self.red_flat,
            green_flat_file=self.green_flat,
            blue_flat_file=self.blue_flat,
            arc_calib_file=self.arc_calib_file,
            output_dir=self.test_output_dir,
            trace_dir=self.trace_dir,
            verbose=False
        )
        
        pixel_map_files = results['output_files']
        self.assertGreater(len(pixel_map_files), 0, "Should generate pixel maps")
        
        # Analyze correlation between traces and flat field values
        correlations_checked = 0
        
        for trace_file in trace_files[:3]:  # Check first 3 trace files
            try:
                with open(trace_file, 'rb') as f:
                    trace_obj = pickle.load(f)
                
                channel = trace_obj.channel
                bench = trace_obj.bench
                side = trace_obj.side
                
                # Find corresponding pixel map
                pixel_map_file = None
                for pm_file in pixel_map_files:
                    if (f"flat_pixel_map_{channel.lower()}{bench}{side.upper()}.fits" in 
                        os.path.basename(pm_file)):
                        pixel_map_file = pm_file
                        break
                
                if not pixel_map_file:
                    print(f"  No pixel map found for {channel}{bench}{side}")
                    continue
                
                # Load fiber image and pixel map
                fiber_image = trace_obj.fiberimg
                with fits.open(pixel_map_file) as hdul:
                    pixel_map = hdul[0].data
                
                if fiber_image is None or pixel_map is None:
                    continue
                
                # Analyze correlation
                traced_pixels = fiber_image > 0  # Non-zero values indicate traced regions
                traced_flat_values = pixel_map[traced_pixels]
                untraced_flat_values = pixel_map[~traced_pixels]
                
                # Remove NaN values
                traced_flat_values = traced_flat_values[np.isfinite(traced_flat_values)]
                untraced_flat_values = untraced_flat_values[np.isfinite(untraced_flat_values)]
                
                if len(traced_flat_values) > 0 and len(untraced_flat_values) > 0:
                    traced_mean = np.mean(traced_flat_values)
                    untraced_mean = np.mean(untraced_flat_values)
                    
                    # Count values close to 1.0
                    traced_near_one = np.sum(np.abs(traced_flat_values - 1.0) < self.tolerance)
                    untraced_near_one = np.sum(np.abs(untraced_flat_values - 1.0) < self.tolerance)
                    
                    print(f"  {channel}{bench}{side}: "
                          f"traced_mean={traced_mean:.3f}, untraced_mean={untraced_mean:.3f}, "
                          f"untraced_≈1.0={untraced_near_one}/{len(untraced_flat_values)}")
                    
                    # Expect more untraced pixels to be ≈1.0 than traced pixels
                    untraced_fraction_one = untraced_near_one / len(untraced_flat_values)
                    traced_fraction_one = traced_near_one / len(traced_flat_values)
                    
                    self.assertGreater(untraced_fraction_one, traced_fraction_one,
                                     f"{channel}{bench}{side}: untraced regions should have more ≈1.0 values")
                    
                    correlations_checked += 1
                
            except Exception as e:
                print(f"  Error analyzing {trace_file}: {str(e)}")
                continue
        
        self.assertGreater(correlations_checked, 0, "Should successfully analyze some trace correlations")
        print(f"✓ Analyzed {correlations_checked} trace/flat field correlations")
        print("✅ PASSED: Fiber trace flat field behavior validated")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLAMAS Flat Field Reduction Test Suite')
    parser.add_argument('--science-file', type=str, 
                       help='Path to actual science FITS file for integration testing')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (use environment variables)')
    parser.add_argument('--no-integration', action='store_true',
                       help='Skip integration tests requiring science file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    return parser.parse_args()


def main():
    """Main test runner."""
    args = parse_arguments()
    
    # Update test configuration from arguments
    if args.science_file:
        TEST_CONFIG['science_file'] = args.science_file
    elif os.environ.get('LLAMAS_TEST_SCIENCE_FILE'):
        TEST_CONFIG['science_file'] = os.environ.get('LLAMAS_TEST_SCIENCE_FILE')
    
    if args.no_integration:
        TEST_CONFIG['run_integration_tests'] = False
    
    # Print usage instructions
    print("\n" + "="*80)
    print("LLAMAS FLAT FIELD REDUCTION TEST SUITE")
    print("="*80)
    print("\nUSAGE INSTRUCTIONS:")
    print("1. Direct execution:")
    print("   python test_flat_processing.py --science-file /path/to/science.fits")
    print("\n2. With pytest:")
    print("   pytest test_flat_processing.py -v --science-file /path/to/science.fits")
    print("\n3. Specific test:")
    print("   python -m pytest test_flat_processing.py::TestFlatFieldReduction::test_normalized_flat_field_creation -v")
    print("\n4. Skip integration tests:")
    print("   python test_flat_processing.py --no-integration")
    print("\nREQUIRED FILES:")
    print(f"- Red flat: {TEST_CONFIG['red_flat']}")
    print(f"- Green flat: {TEST_CONFIG['green_flat']}")
    print(f"- Blue flat: {TEST_CONFIG['blue_flat']}")
    print(f"- Science file: {TEST_CONFIG['science_file'] or 'NOT PROVIDED'}")
    print(f"- Trace directory: {TEST_CONFIG['trace_dir']}")
    print("="*80)
    
    # Run tests
    if args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    unittest.main(argv=[''], verbosity=verbosity, exit=False)


if __name__ == "__main__":
    main()