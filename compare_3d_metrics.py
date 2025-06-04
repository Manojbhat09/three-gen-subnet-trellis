import os
import time

class Compare3DMetrics:
    def __init__(self, reference_mesh, generated_mesh, output_dir, resolution=512, num_views=8):
        """Initialize the comparison system."""
        self.reference_mesh = reference_mesh
        self.generated_mesh = generated_mesh
        self.output_dir = output_dir
        self.resolution = resolution
        self.num_views = num_views
        self.logger = None
        self._resources = []
        self._cache = {}
        
    def _register_resource(self, resource, cleanup_func):
        """Register a resource for cleanup."""
        self._resources.append((resource, cleanup_func))
        
    def _cleanup_resources(self):
        """Clean up all registered resources."""
        for resource, cleanup_func in reversed(self._resources):
            try:
                cleanup_func(resource)
            except Exception as e:
                self.logger.error(f"Error cleaning up resource: {str(e)}")
        self._resources.clear()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup_resources()
        if self.logger:
            self.logger.info("Resources cleaned up")
            
    def _validate_config(self):
        """Validate the configuration parameters."""
        required_params = {
            'resolution': (int, 'Resolution must be a positive integer'),
            'num_views': (int, 'Number of views must be a positive integer'),
            'output_dir': (str, 'Output directory must be specified'),
            'reference_mesh': (str, 'Reference mesh path must be specified'),
            'generated_mesh': (str, 'Generated mesh path must be specified')
        }
        
        for param, (expected_type, error_msg) in required_params.items():
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Missing required parameter: {error_msg}")
            if not isinstance(getattr(self, param), expected_type):
                raise TypeError(f"Invalid type for {param}: expected {expected_type.__name__}, got {type(getattr(self, param)).__name__}")
        
        # Validate paths exist
        if not os.path.exists(self.reference_mesh):
            raise FileNotFoundError(f"Reference mesh not found: {self.reference_mesh}")
        if not os.path.exists(self.generated_mesh):
            raise FileNotFoundError(f"Generated mesh not found: {self.generated_mesh}")
        
        # Validate output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate resolution and num_views
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if self.num_views <= 0:
            raise ValueError("Number of views must be positive")
        
        self.logger.info("Configuration validation passed")

    def _track_progress(self, operation_name, total_steps):
        """Create a progress tracking context manager."""
        class ProgressTracker:
            def __init__(self, logger, operation_name, total_steps):
                self.logger = logger
                self.operation_name = operation_name
                self.total_steps = total_steps
                self.current_step = 0
                self.start_time = time.time()
                
            def __enter__(self):
                self.logger.info(f"Starting {self.operation_name}")
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                if exc_type is None:
                    self.logger.info(f"Completed {self.operation_name} in {duration:.2f} seconds")
                else:
                    self.logger.error(f"Failed {self.operation_name} after {duration:.2f} seconds")
                    
            def update(self, step=1):
                self.current_step += step
                progress = (self.current_step / self.total_steps) * 100
                self.logger.info(f"{self.operation_name} progress: {progress:.1f}%")
                
        return ProgressTracker(self.logger, operation_name, total_steps)
        
    def _get_cache_key(self, operation, *args, **kwargs):
        """Generate a cache key for an operation."""
        key_parts = [operation]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(key_parts)
        
    def _cached_operation(self, operation_name, func, *args, **kwargs):
        """Execute an operation with caching."""
        cache_key = self._get_cache_key(operation_name, *args, **kwargs)
        
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit for {operation_name}")
            return self._cache[cache_key]
            
        self.logger.debug(f"Cache miss for {operation_name}")
        result = func(*args, **kwargs)
        self._cache[cache_key] = result
        return result
        
    def _compute_view_metrics(self, view_idx):
        """Compute metrics for a single view with caching."""
        def compute():
            # Existing computation code here
            pass
            
        return self._cached_operation("compute_view_metrics", compute, view_idx)
        
    def _generate_view_visualization(self, view_idx):
        """Generate visualization for a single view with caching."""
        def generate():
            # Existing visualization code here
            pass
            
        return self._cached_operation("generate_view_visualization", generate, view_idx)
        
    def _clear_cache(self):
        """Clear the operation cache."""
        self._cache.clear()
        self.logger.debug("Cache cleared")
        
    def _compute_metrics(self):
        """Compute all metrics between the meshes."""
        with self._track_progress("Computing metrics", self.num_views) as progress:
            for i in range(self.num_views):
                try:
                    self._compute_view_metrics(i)
                    progress.update()
                except Exception as e:
                    self.logger.error(f"Failed to compute metrics for view {i}: {str(e)}")
                    continue
                    
    def _generate_visualizations(self):
        """Generate visualizations for the comparison."""
        with self._track_progress("Generating visualizations", self.num_views) as progress:
            for i in range(self.num_views):
                try:
                    self._generate_view_visualization(i)
                    progress.update()
                except Exception as e:
                    self.logger.error(f"Failed to generate visualization for view {i}: {str(e)}")
                    continue

    def run_comparison(self):
        """Run the full comparison pipeline."""
        try:
            self._validate_config()
            self._setup_logging()
            self._load_meshes()
            self._compute_metrics()
            self._generate_visualizations()
            self._save_results()
            self.logger.info("Comparison completed successfully")
        except Exception as e:
            self.logger.error(f"Comparison failed: {str(e)}")
            raise
        finally:
            self._cleanup_resources()
            self._clear_cache() 