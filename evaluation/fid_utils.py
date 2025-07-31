import os
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.fid_score import compute_statistics_of_path


def calculate_fid_given_paths_and_save_stats(
    paths, 
    cache_path, 
    batch_size=50, 
    device='cuda', 
    dims=2048, 
    num_workers=8
):
    """
    Calculate FID and save reference dataset statistics to cache file.
    
    Args:
        paths: List containing two paths - [target_folder, reference_folder]
        cache_path: Path to save reference dataset statistics
        batch_size: Batch size for processing
        device: Device to use for computation
        dims: Feature dimension
        num_workers: Number of worker processes
    
    Returns:
        float: FID value
    """
    target_path, ref_path = paths
    
    # Initialize InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    # Compute statistics for target folder
    m1, s1 = compute_statistics_of_path(
        target_path, model, batch_size, dims, device, num_workers
    )
    
    # Compute statistics for reference folder
    m2, s2 = compute_statistics_of_path(
        ref_path, model, batch_size, dims, device, num_workers
    )
    
    # Save reference folder statistics to cache
    np.savez(cache_path, mu=m2, sigma=s2)
    
    # Calculate FID score
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


def calculate_fid_given_paths_with_cache(
    paths, 
    cache_path, 
    batch_size=50, 
    device='cuda', 
    dims=2048, 
    num_workers=8
):
    """
    Calculate FID using cached reference dataset statistics.
    
    Args:
        paths: List containing target folder path
        cache_path: Path to cached reference dataset statistics file
        batch_size: Batch size for processing
        device: Device to use for computation
        dims: Feature dimension
        num_workers: Number of worker processes
    
    Returns:
        float: FID value
    """
    target_path = paths[0]
    
    # Initialize InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    # Compute statistics for target folder
    m1, s1 = compute_statistics_of_path(
        target_path, model, batch_size, dims, device, num_workers
    )
    
    # Load cached reference folder statistics
    data = np.load(cache_path)
    m2, s2 = data['mu'], data['sigma']
    
    # Calculate FID score
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


def calculate_fid_batch(
    target_paths, 
    reference_path=None, 
    cache_path=None, 
    batch_size=50, 
    device='cuda', 
    dims=2048, 
    num_workers=8
):
    """
    Calculate FID scores for multiple target paths efficiently.
    
    Args:
        target_paths: List of target folder paths
        reference_path: Reference folder path (if not using cache)
        cache_path: Path to cached reference statistics (if not computing fresh)
        batch_size: Batch size for processing
        device: Device to use for computation
        dims: Feature dimension
        num_workers: Number of worker processes
    
    Returns:
        dict: Dictionary mapping target paths to FID values
    """
    if reference_path is None and cache_path is None:
        raise ValueError("Either reference_path or cache_path must be provided")
    
    # Initialize InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    # Get reference statistics
    if cache_path and os.path.exists(cache_path):
        # Load from cache
        data = np.load(cache_path)
        m_ref, s_ref = data['mu'], data['sigma']
    elif reference_path:
        # Compute reference statistics
        m_ref, s_ref = compute_statistics_of_path(
            reference_path, model, batch_size, dims, device, num_workers
        )
        # Save to cache if path provided
        if cache_path:
            np.savez(cache_path, mu=m_ref, sigma=s_ref)
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    # Calculate FID for each target path
    fid_scores = {}
    for target_path in target_paths:
        # Compute target statistics
        m_target, s_target = compute_statistics_of_path(
            target_path, model, batch_size, dims, device, num_workers
        )
        
        # Calculate FID
        fid_value = calculate_frechet_distance(m_target, s_target, m_ref, s_ref)
        fid_scores[target_path] = fid_value
    
    return fid_scores