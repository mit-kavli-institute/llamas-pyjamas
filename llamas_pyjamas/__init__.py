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
# Import for Ray pickling compatibility
import llamas_pyjamas.Trace.traceLlamasMaster as traceLlamasMaster
import sys
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