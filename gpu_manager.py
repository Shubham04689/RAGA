# import torch
# import logging

# logger = logging.getLogger(__name__)

# class GPUManager:
#     """Manages GPU resources and optimization with improved memory monitoring."""
    
#     def __init__(self, config):
#         self.config = config
#         self.device = self._setup_device()
#         self._init_memory_tracking()
        
#     def _init_memory_tracking(self):
#         self.initial_memory = 0
#         if self.device.type == 'cuda':
#             try:
#                 torch.cuda.reset_peak_memory_stats()
#                 self.initial_memory = torch.cuda.memory_allocated()
#             except Exception as e:
#                 logger.warning(f"Could not initialize memory tracking: {str(e)}")
    
#     def _setup_device(self):
#         if self.config.use_cuda and torch.cuda.is_available():
#             torch.cuda.set_device(self.config.cuda_device)
#             if torch.cuda.get_device_capability()[0] >= 8:
#                 torch.backends.cuda.matmul.allow_tf32 = True
#                 torch.backends.cudnn.allow_tf32 = True
#             device = torch.device(f'cuda:{self.config.cuda_device}')
#             logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
#             return device
#         else:
#             if self.config.use_cuda:
#                 logger.warning("CUDA requested but not available. Using CPU.")
#             return torch.device('cpu')
    
#     def get_memory_usage(self):
#         try:
#             if self.device.type != 'cuda':
#                 return None
#             current_memory = torch.cuda.memory_allocated()
#             max_memory = torch.cuda.max_memory_allocated()
#             if max_memory <= 0:
#                 return 0.0
#             usage = current_memory / max_memory
#             logger.debug(f"Current memory: {current_memory / 1024**2:.2f}MB")
#             logger.debug(f"Max memory: {max_memory / 1024**2:.2f}MB")
#             logger.debug(f"Memory usage ratio: {usage:.2%}")
#             return usage
#         except Exception as e:
#             logger.error(f"Error calculating memory usage: {str(e)}")
#             return None

#     def optimize_memory(self):
#         if self.device.type == 'cuda':
#             try:
#                 torch.cuda.empty_cache()
#                 if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
#                     torch.backends.cuda.enable_mem_efficient_sdp(True)
#                 memory_usage = self.get_memory_usage()
#                 if memory_usage is not None and memory_usage > self.config.memory_buffer:
#                     logger.warning(f"High GPU memory usage detected: {memory_usage:.2%}")
#                     self._emergency_cleanup()
#                 self._log_memory_stats()
#             except Exception as e:
#                 logger.error(f"Memory optimization failed: {str(e)}")
    
#     def _emergency_cleanup(self):
#         try:
#             torch.cuda.empty_cache()
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             import gc
#             gc.collect()
#             self._log_memory_stats()
#         except Exception as e:
#             logger.error(f"Emergency cleanup failed: {str(e)}")
    
#     def _log_memory_stats(self):
#         if self.device.type == 'cuda':
#             try:
#                 allocated = torch.cuda.memory_allocated() / 1024**2
#                 reserved = torch.cuda.memory_reserved() / 1024**2
#                 max_allocated = torch.cuda.max_memory_allocated() / 1024**2
#                 logger.info("Memory stats (MB):")
#                 logger.info(f"  Allocated: {allocated:.2f}")
#                 logger.info(f"  Reserved: {reserved:.2f}")
#                 logger.info(f"  Max Allocated: {max_allocated:.2f}")
#             except Exception as e:
#                 logger.error(f"Failed to log memory stats: {str(e)}")
import torch
import logging
import psutil

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and optimization with enhanced memory and system resource monitoring."""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._init_memory_tracking()
        self.check_system_memory()  # Log system RAM stats on initialization

    def _init_memory_tracking(self):
        """Initialize GPU memory tracking."""
        self.initial_memory = 0
        if self.device.type == 'cuda':
            try:
                torch.cuda.reset_peak_memory_stats()
                self.initial_memory = torch.cuda.memory_allocated()
            except Exception as e:
                logger.warning(f"Could not initialize GPU memory tracking: {str(e)}")

    def _setup_device(self):
        """Configure and return the appropriate torch device."""
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.config.cuda_device)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            device = torch.device(f'cuda:{self.config.cuda_device}')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return device
        else:
            if self.config.use_cuda:
                logger.warning("CUDA requested but not available. Using CPU.")
            return torch.device('cpu')

    def get_memory_usage(self):
        """Return the current GPU memory usage ratio."""
        try:
            if self.device.type != 'cuda':
                return None
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            if max_memory <= 0:
                return 0.0
            usage = current_memory / max_memory
            logger.debug(f"GPU Memory Usage: Current: {current_memory / 1024**2:.2f}MB, Max: {max_memory / 1024**2:.2f}MB, Ratio: {usage:.2%}")
            return usage
        except Exception as e:
            logger.error(f"Error calculating GPU memory usage: {str(e)}")
            return None

    def optimize_memory(self):
        """Optimize GPU memory usage and monitor system RAM."""
        if self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                memory_usage = self.get_memory_usage()
                if memory_usage is not None and memory_usage > self.config.memory_buffer:
                    logger.warning(f"High GPU memory usage detected: {memory_usage:.2%}")
                    self._emergency_cleanup()
                self._log_memory_stats()
            except Exception as e:
                logger.error(f"GPU memory optimization failed: {str(e)}")
        # Check system RAM as well
        self.check_system_memory()

    def _emergency_cleanup(self):
        """Force cleanup when GPU memory usage is too high."""
        try:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            import gc
            gc.collect()
            self._log_memory_stats()
        except Exception as e:
            logger.error(f"Emergency GPU cleanup failed: {str(e)}")

    def _log_memory_stats(self):
        """Log detailed GPU memory statistics."""
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                logger.info("GPU Memory stats (MB):")
                logger.info(f"  Allocated: {allocated:.2f}")
                logger.info(f"  Reserved: {reserved:.2f}")
                logger.info(f"  Max Allocated: {max_allocated:.2f}")
            except Exception as e:
                logger.error(f"Failed to log GPU memory stats: {str(e)}")

    def check_system_memory(self):
        """Log available system RAM using psutil."""
        try:
            mem = psutil.virtual_memory()
            available_mb = mem.available / 1024**2
            total_mb = mem.total / 1024**2
            logger.info(f"System RAM: {available_mb:.2f}MB available out of {total_mb:.2f}MB total ({mem.percent}% used)")
            return available_mb
        except Exception as e:
            logger.error(f"Error checking system memory: {str(e)}")
            return None
