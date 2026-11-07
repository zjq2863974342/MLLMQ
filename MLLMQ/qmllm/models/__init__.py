import importlib
import os
import sys

import hf_transfer
from loguru import logger

# Conditionally import model wrappers — many of them have heavy / optional
# dependencies (cv2, etc.) that may not be available in every environment.
# Only Janus (which only needs transformers + torch) is always imported.
try:
    from qmllm.models.internvl2 import InternVL2
except ImportError:
    pass

try:
    from qmllm.models.llava_onevision import LLaVA_onevision
except ImportError:
    pass

try:
    from qmllm.models.llava_v15 import LLaVA_v15
except ImportError:
    pass

try:
    from qmllm.models.qwen2_vl import Qwen2_VL
except ImportError:
    pass

try:
    from qmllm.models.qwen2_5_vl import Qwen2_5_VL
except ImportError:
    pass

try:
    from qmllm.models.vila import vila
except ImportError:
    pass

from qmllm.models.janus import JanusModel

from qmllm.utils.registry import MODEL_REGISTRY

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]