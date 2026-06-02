# Fix for Ray + PypeIt signal handler conflict
# This must be done before any imports that might trigger PypeIt imports
import signal
import sys

_original_signal = signal.signal

def _safe_signal_handler(signum, handler):
    """
    Wrapper for signal.signal that gracefully handles Ray's DeferSigint context.

    This prevents crashes when PypeIt tries to set signal handlers while Ray
    workers are deserializing objects in a DeferSigint context.
    """
    try:
        return _original_signal(signum, handler)
    except ValueError as e:
        if "DeferSigint" in str(e) or "signal handler" in str(e).lower():
            # Silently ignore signal handler setup in Ray workers
            # This is safe because Ray manages its own signal handling
            return None
        raise

# Apply the monkey-patch globally
signal.signal = _safe_signal_handler

# Now proceed with normal imports
from .File import *
from .Trace import *
from .Flat import *
from .Extract import extractLlamas, save_extractions, load_extractions
from .Image import WhiteLight
from .QA import *
from .Utils.utils import *
from .config import *
from .File import llamasOneCamera, llamasAllCameras, getBenchSideChannel
from .LUT import *
from .Cube import *
from .Bias import BiasNotFoundError, BiasReadModeError, generate_fallback_bias_hdu
# Import for Ray pickling compatibility
import llamas_pyjamas.Trace.traceLlamasMaster as traceLlamasMaster
import cloudpickle

sys.modules['traceLlamasMaster'] = traceLlamasMaster

# Register a custom reducer for TraceRay to ensure proper serialization
def TraceRay_reducer(obj):
    """Custom reducer for TraceRay objects to handle class reference issues."""
    correct_class = sys.modules['traceLlamasMaster'].TraceRay
    return (correct_class, (), obj.__dict__)

# Apply the custom reducer
cloudpickle.register_pickle_by_value(traceLlamasMaster)
try:
    import copyreg
    copyreg.pickle(traceLlamasMaster.TraceRay, TraceRay_reducer)
except:
    pass

__version__ = '0.1.0'