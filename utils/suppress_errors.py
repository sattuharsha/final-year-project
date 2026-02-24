"""
Suppress known harmless errors on Windows + Python 3.12 + multiprocess.
This should be imported early in any script to prevent ResourceTracker noise.
"""
import sys


def setup_error_suppression():
    """Setup unraisable hook to suppress ResourceTracker shutdown errors."""
    def _quiet_unraisable(hook):
        def wrapper(unraisable):
            try:
                exc_value = getattr(unraisable, "exc_value", None)
                exc_type = getattr(unraisable, "exc_type", None)
                msg = str(exc_value or "")
                # Suppress ResourceTracker / RLock recursion_count errors
                if ("_recursion_count" in msg or 
                    "ResourceTracker" in msg or 
                    (exc_type and "ResourceTracker" in str(exc_type)) or
                    (exc_value and isinstance(exc_value, AttributeError) and "_recursion_count" in str(exc_value))):
                    return  # Silently ignore
            except Exception:
                pass  # If suppression itself fails, ignore
            if hook is not None:
                hook(unraisable)
        return wrapper
    
    _default_hook = getattr(sys, "unraisablehook", None)
    sys.unraisablehook = _quiet_unraisable(_default_hook)
