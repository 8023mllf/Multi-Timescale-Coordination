import os
import sys

# 让 diffusion_policy 作为顶层包可被 import
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from .deploy_policy import *
