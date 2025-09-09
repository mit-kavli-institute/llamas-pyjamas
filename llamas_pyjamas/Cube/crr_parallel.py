"""
CRR Ray Parallelization Module

Implementation of Ray-based parallelization for the Covariance-regularized 
Reconstruction (CRR) cube construction. This module provides distributed
processing capabilities for large-scale LLAMAS IFU data cubes using the
Ray framework for cluster compatibility.

The parallelization strategy distributes wavelength slices across multiple
workers, with each worker handling a batch of wavelengths to minimize
communication overhead and optimize memory usage.

Classes:
    CRRWorker: Ray remote worker for processing wavelength batches
    ParallelCRRManager: Manager for coordinating distributed reconstruction

Functions:
    parallel_cube_construction: Main entry point for parallel reconstruction
    setup_ray_cluster: Initialize Ray cluster with proper configuration
    estimate_memory_requirements: Estimate memory usage for planning

Author: Generated for LLAMAS Pipeline
Date: September 2025
"""

import numpy as np
import ray
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import psutil
import time
from dataclasses import asdict

from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.Cube.crr_cube_constructor import (
    CRRCubeConstructor, CRRCubeConfig, RSSData, CRRDataCube
)


def setup_ray_cluster(n_workers: Optional[int] = None,
                     memory_limit_gb: Optional[float] = None,
                     local_mode: bool = True) -> Dict[str, Any]:
    """Initialize Ray cluster with proper configuration for CRR processing.
    
    Args:
        n_workers: Number of worker processes (None = auto-detect)
        memory_limit_gb: Memory limit per worker in GB
        local_mode: Whether to run in local mode or connect to cluster
        
    Returns:
        Dictionary with cluster information
    """
    logger = setup_logger(__name__)
    
    if ray.is_initialized():
        logger.info("Ray already initialized")
        return ray.cluster_resources()
        
    # Auto-detect system resources
    if n_workers is None:
        n_workers = psutil.cpu_count(logical=False)  # Physical cores
        
    if memory_limit_gb is None:
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_limit_gb = max(2.0, total_memory_gb * 0.8 / n_workers)  # 80% of total, divided by workers
    
    # Ray initialization configuration
    ray_config = {
        'num_cpus': n_workers,
        'object_store_memory': int(memory_limit_gb * 1024**3 * 0.3),  # 30% for object store
        'logging_level': logging.INFO
    }
    
    if local_mode:
        ray.init(**ray_config)
        logger.info(f"Ray initialized locally: {n_workers} workers, "
                   f"{memory_limit_gb:.1f} GB memory limit per worker")
    else:
        # Connect to existing cluster
        ray.init(address='auto')
        logger.info("Connected to existing Ray cluster")
    
    cluster_info = ray.cluster_resources()
    cluster_info['memory_limit_gb'] = memory_limit_gb
    cluster_info['n_workers'] = n_workers
    
    return cluster_info


def estimate_memory_requirements(rss_data: RSSData, 
                                config: CRRCubeConfig) -> Dict[str, float]:
    """Estimate memory requirements for CRR reconstruction.
    
    Args:
        rss_data: RSS data structure
        config: CRR configuration
        
    Returns:
        Dictionary with memory estimates in GB
    """
    n_fibers, n_wavelengths = rss_data.flux.shape
    
    # Estimate output grid size
    if rss_data.fiber_positions.ndim == 3:
        positions = rss_data.fiber_positions[0]  # Use first wavelength
    else:
        positions = rss_data.fiber_positions
        
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    
    # Add padding
    padding = 2 * config.kernel_radius_limit
    x_range += 2 * padding
    y_range += 2 * padding
    
    # Grid dimensions
    n_x = int(np.ceil(x_range / config.pixel_scale))
    n_y = int(np.ceil(y_range / config.pixel_scale))
    n_pixels = n_x * n_y
    
    # Memory estimates (in GB)
    bytes_per_gb = 1024**3
    
    # Input data per wavelength slice
    input_slice_gb = (n_fibers * 3 * 4) / bytes_per_gb  # flux, ivar, mask (float32)
    
    # Kernel matrix A (most memory intensive)
    kernel_matrix_gb = (n_pixels * n_fibers * 4) / bytes_per_gb  # float32
    
    # Weight matrix W  
    weight_matrix_gb = (n_pixels * n_fibers * 4) / bytes_per_gb  # float32
    
    # Output arrays per wavelength
    output_slice_gb = (n_pixels * 3 * 4) / bytes_per_gb  # flux, ivar, covariance
    
    # Full cube (all wavelengths)
    full_cube_gb = output_slice_gb * n_wavelengths
    
    # Peak memory per worker (worst case - processing one wavelength)
    worker_peak_gb = input_slice_gb + kernel_matrix_gb + weight_matrix_gb + output_slice_gb
    
    memory_estimates = {
        'n_fibers': n_fibers,
        'n_wavelengths': n_wavelengths,
        'n_pixels': n_pixels,
        'input_slice_gb': input_slice_gb,
        'kernel_matrix_gb': kernel_matrix_gb,
        'weight_matrix_gb': weight_matrix_gb,
        'output_slice_gb': output_slice_gb,
        'worker_peak_gb': worker_peak_gb,
        'full_cube_gb': full_cube_gb
    }
    
    return memory_estimates


@ray.remote
class CRRWorker:
    """Ray remote worker for processing batches of wavelength slices.
    
    Each worker maintains its own CRRCubeConstructor instance and processes
    assigned wavelength batches independently to minimize data transfer.
    """
    
    def __init__(self, config: CRRCubeConfig):
        """Initialize CRR worker.
        
        Args:
            config: CRR configuration parameters
        """
        self.config = config
        self.constructor = CRRCubeConstructor(config)
        self.logger = setup_logger(f"{__name__}.Worker")
        
    def setup_worker(self, fiber_positions: np.ndarray) -> bool:
        """Setup worker with grid information.
        
        Args:
            fiber_positions: Fiber positions for grid setup
            
        Returns:
            Success flag
        """
        try:
            self.constructor.setup_output_grid(fiber_positions)
            self.logger.info("Worker grid setup complete")
            return True
        except Exception as e:
            self.logger.error(f"Worker setup failed: {e}")
            return False
    
    def process_wavelength_batch(self, 
                                wavelength_indices: List[int],
                                rss_data: RSSData) -> Dict[str, Any]:
        """Process batch of wavelength slices.
        
        Args:
            wavelength_indices: List of wavelength indices to process
            rss_data: RSS data structure
            
        Returns:
            Dictionary with reconstruction results for this batch
        """
        self.logger.info(f"Processing wavelength batch: {len(wavelength_indices)} slices")
        
        batch_results = {
            'wavelength_indices': wavelength_indices,
            'flux_slices': [],
            'ivar_slices': [],
            'covar_slices': [],
            'processing_times': [],
            'errors': []
        }
        
        for wave_idx in wavelength_indices:
            start_time = time.time()
            
            try:
                flux_slice, ivar_slice, covar_slice = self.constructor.process_wavelength_slice(
                    wave_idx, rss_data
                )
                
                batch_results['flux_slices'].append(flux_slice)
                batch_results['ivar_slices'].append(ivar_slice)
                batch_results['covar_slices'].append(covar_slice)
                batch_results['errors'].append(None)
                
                processing_time = time.time() - start_time
                batch_results['processing_times'].append(processing_time)
                
                self.logger.debug(f"Wavelength {wave_idx} processed in {processing_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing wavelength {wave_idx}: {e}")
                
                # Add empty arrays for failed wavelength
                n_x = len(self.constructor.output_grid_x)
                n_y = len(self.constructor.output_grid_y)
                
                batch_results['flux_slices'].append(np.zeros((n_x, n_y)))
                batch_results['ivar_slices'].append(np.zeros((n_x, n_y)))
                batch_results['covar_slices'].append(np.zeros((n_x, n_y)))
                batch_results['errors'].append(str(e))
                batch_results['processing_times'].append(0.0)
        
        total_time = sum(batch_results['processing_times'])
        self.logger.info(f"Batch processing complete: {len(wavelength_indices)} slices "
                        f"in {total_time:.1f}s")
        
        return batch_results


class ParallelCRRManager:
    """Manager for coordinating distributed CRR reconstruction."""
    
    def __init__(self, config: CRRCubeConfig):
        """Initialize parallel manager.
        
        Args:
            config: CRR configuration parameters
        """
        self.config = config
        self.logger = setup_logger(__name__)
        self.workers = []
        
    def create_workers(self, n_workers: int) -> List:
        """Create Ray workers for parallel processing.
        
        Args:
            n_workers: Number of workers to create
            
        Returns:
            List of worker references
        """
        self.logger.info(f"Creating {n_workers} CRR workers")
        
        self.workers = [CRRWorker.remote(self.config) for _ in range(n_workers)]
        
        return self.workers
    
    def setup_workers(self, fiber_positions: np.ndarray) -> bool:
        """Setup all workers with grid information.
        
        Args:
            fiber_positions: Fiber positions for grid setup
            
        Returns:
            Success flag
        """
        self.logger.info("Setting up workers with grid information")
        
        # Setup all workers in parallel
        setup_futures = [worker.setup_worker.remote(fiber_positions) for worker in self.workers]
        
        setup_results = ray.get(setup_futures)
        
        success = all(setup_results)
        if success:
            self.logger.info("All workers setup successfully")
        else:
            self.logger.error("Some workers failed to setup")
            
        return success
    
    def distribute_wavelengths(self, n_wavelengths: int, batch_size: int) -> List[List[int]]:
        """Distribute wavelength indices across workers in batches.
        
        Args:
            n_wavelengths: Total number of wavelengths
            batch_size: Wavelengths per batch
            
        Returns:
            List of wavelength index batches for each worker
        """
        # Create batches
        wavelength_batches = []
        for i in range(0, n_wavelengths, batch_size):
            batch = list(range(i, min(i + batch_size, n_wavelengths)))
            wavelength_batches.append(batch)
        
        # Distribute batches across workers
        n_workers = len(self.workers)
        worker_assignments = [[] for _ in range(n_workers)]
        
        for i, batch in enumerate(wavelength_batches):
            worker_idx = i % n_workers
            worker_assignments[worker_idx].extend(batch)
        
        # Convert back to batches for each worker
        worker_batches = []
        for assignment in worker_assignments:
            if assignment:  # Only include non-empty assignments
                # Split large assignments into smaller batches
                worker_batch_list = []
                for i in range(0, len(assignment), batch_size):
                    worker_batch_list.append(assignment[i:i + batch_size])
                worker_batches.extend(worker_batch_list)
        
        self.logger.info(f"Distributed {n_wavelengths} wavelengths into "
                        f"{len(worker_batches)} batches across {n_workers} workers")
        
        return worker_batches


def parallel_cube_construction(rss_data: RSSData,
                             config: CRRCubeConfig,
                             n_workers: Optional[int] = None,
                             wavelength_batch_size: int = 50,
                             memory_limit_gb: Optional[float] = None) -> CRRDataCube:
    """Main entry point for parallel CRR cube reconstruction.
    
    Args:
        rss_data: RSS data structure
        config: CRR configuration
        n_workers: Number of workers (None = auto-detect)
        wavelength_batch_size: Wavelengths per batch
        memory_limit_gb: Memory limit per worker in GB
        
    Returns:
        Reconstructed CRR data cube
    """
    logger = setup_logger(__name__)
    logger.info("Starting parallel CRR cube reconstruction")
    
    # Estimate memory requirements
    memory_est = estimate_memory_requirements(rss_data, config)
    logger.info(f"Memory estimate: {memory_est['worker_peak_gb']:.2f} GB peak per worker")
    
    # Initialize Ray cluster
    cluster_info = setup_ray_cluster(n_workers, memory_limit_gb)
    n_workers = cluster_info.get('n_workers', 1)
    
    try:
        # Create parallel manager
        manager = ParallelCRRManager(config)
        workers = manager.create_workers(n_workers)
        
        # Setup workers
        if not manager.setup_workers(rss_data.fiber_positions):
            raise RuntimeError("Worker setup failed")
        
        # Distribute wavelengths
        n_wavelengths = len(rss_data.wavelength)
        wavelength_batches = manager.distribute_wavelengths(n_wavelengths, wavelength_batch_size)
        
        # Process batches in parallel
        logger.info(f"Processing {n_wavelengths} wavelengths in {len(wavelength_batches)} batches")
        
        # Submit all batch jobs
        batch_futures = []
        for i, batch in enumerate(wavelength_batches):
            worker_idx = i % len(workers)
            future = workers[worker_idx].process_wavelength_batch.remote(batch, rss_data)
            batch_futures.append(future)
        
        # Collect results
        start_time = time.time()
        batch_results = ray.get(batch_futures)
        processing_time = time.time() - start_time
        
        logger.info(f"Parallel processing completed in {processing_time:.1f}s")
        
        # Assemble final cube
        output_cube = assemble_cube_from_batches(batch_results, rss_data, config)
        
        logger.info("CRR cube assembly completed")
        return output_cube
        
    finally:
        # Cleanup Ray resources
        if ray.is_initialized():
            ray.shutdown()


def assemble_cube_from_batches(batch_results: List[Dict[str, Any]],
                             rss_data: RSSData,
                             config: CRRCubeConfig) -> CRRDataCube:
    """Assemble final cube from parallel batch results.
    
    Args:
        batch_results: List of batch processing results
        rss_data: Original RSS data
        config: CRR configuration
        
    Returns:
        Assembled CRR data cube
    """
    logger = setup_logger(__name__)
    logger.info("Assembling cube from parallel results")
    
    # Determine cube dimensions from first result
    first_batch = batch_results[0]
    first_slice = first_batch['flux_slices'][0]
    n_x, n_y = first_slice.shape
    n_wavelengths = len(rss_data.wavelength)
    
    # Initialize output cube
    cube_shape = (n_x, n_y, n_wavelengths)
    output_cube = CRRDataCube(cube_shape)
    
    # Store metadata
    output_cube.metadata.update({
        'regularization_lambda': config.regularization_lambda,
        'pixel_scale': config.pixel_scale,
        'kernel_radius_limit': config.kernel_radius_limit,
        'reconstruction_radius_limit': config.reconstruction_radius_limit,
        'fiber_diameter': config.fiber_diameter,
        'seeing_fwhm_ref': rss_data.seeing_fwhm,
        'wavelength_range': (rss_data.wavelength.min(), rss_data.wavelength.max()),
        'n_fibers': rss_data.flux.shape[0],
        'reconstruction_method': 'CRR_Parallel',
        'reference': 'Liu et al. (2020)',
        'parallel_processing': True
    })
    
    # Collect all slices by wavelength index
    wavelength_data = {}
    processing_errors = []
    
    for batch_result in batch_results:
        wave_indices = batch_result['wavelength_indices']
        flux_slices = batch_result['flux_slices']
        ivar_slices = batch_result['ivar_slices']
        covar_slices = batch_result['covar_slices']
        errors = batch_result['errors']
        
        for i, wave_idx in enumerate(wave_indices):
            wavelength_data[wave_idx] = {
                'flux': flux_slices[i],
                'ivar': ivar_slices[i],
                'covar': covar_slices[i],
                'error': errors[i]
            }
            
            if errors[i] is not None:
                processing_errors.append(f"Wavelength {wave_idx}: {errors[i]}")
    
    # Fill cube arrays
    for wave_idx in range(n_wavelengths):
        if wave_idx in wavelength_data:
            data = wavelength_data[wave_idx]
            output_cube.flux[:, :, wave_idx] = data['flux']
            output_cube.ivar[:, :, wave_idx] = data['ivar']
            output_cube.covariance_diagonal[:, :, wave_idx] = data['covar']
            
            # Set quality flags
            no_coverage = data['ivar'] == 0
            low_coverage = (data['ivar'] > 0) & (data['ivar'] < 1e-6)
            
            output_cube.set_flag(output_cube.FLAG_NOCOV, no_coverage)
            output_cube.set_flag(output_cube.FLAG_LOWCOV, low_coverage)
            output_cube.mask[:, :, wave_idx] = ~(no_coverage | low_coverage)
            
        else:
            logger.warning(f"Missing data for wavelength index {wave_idx}")
            output_cube.set_flag(output_cube.FLAG_NOCOV, 
                               np.ones((n_x, n_y), dtype=bool))
    
    # Log any processing errors
    if processing_errors:
        logger.warning(f"Processing errors encountered: {len(processing_errors)}")
        for error in processing_errors[:10]:  # Log first 10 errors
            logger.warning(error)
    
    logger.info(f"Cube assembly completed: {cube_shape}")
    return output_cube