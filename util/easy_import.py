# Import commonly used modules and functions
import io
import re
import mne
import sys
import json
import joblib
import argparse
import itertools
import omegaconf
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

from rich import print, inspect
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import redirect_stdout, redirect_stderr
from IPython.display import display

from .logging import logger
from .data import MyData

n_jobs = 32
