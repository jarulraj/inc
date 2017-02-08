#!/usr/bin/env python

###################################################################################
# INCREMENTAL EXPERIMENTS
###################################################################################

from __future__ import print_function

import time
import argparse
import csv
import logging
import math
import os
import subprocess
import pprint
import matplotlib
import numpy as np

matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LinearLocator

###################################################################################
# LOGGING CONFIGURATION
###################################################################################
LOG = logging.getLogger(__name__)
LOG_handler = logging.StreamHandler()
LOG_formatter = logging.Formatter(
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S'
)
LOG_handler.setFormatter(LOG_formatter)
LOG.addHandler(LOG_handler)
LOG.setLevel(logging.INFO)

MAJOR_STRING = "++++++++++++++++++++++++++++\n\n"
MINOR_STRING = "----------------\n"

###################################################################################
# OUTPUT CONFIGURATION
###################################################################################

BASE_DIR = os.path.dirname(__file__)
OPT_FONT_NAME = 'Helvetica'
OPT_GRAPH_HEIGHT = 150
OPT_GRAPH_WIDTH = 400

# Make a list by cycling through the colors you care about
# to match the length of your data.
NUM_COLORS = 5
COLOR_MAP = ( '#418259', '#bd5632', '#e1a94c', '#7d6c5b', '#364d38', '#c4e1c6')
COLOR_MAP_2 = ( '#F58A87', '#80CA86', '#9EC9E9', '#D89761', '#FED113' )
COLOR_MAP_3 = ( '#2b3742', '#c9b385', '#610606', '#1f1501' )

OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "o", "\\\\" , ".", "\\\\\\"])

OPT_STACK_COLORS = ('#2b3742', '#c9b385', '#610606', '#1f1501')
OPT_LINE_STYLES= ('-', ':', '--', '-.')

# SET FONT

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
TINY_FONT_SIZE = 10
LEGEND_FONT_SIZE = 18

SMALL_LABEL_FONT_SIZE = 10
SMALL_LEGEND_FONT_SIZE = 10

XAXIS_MIN = 0.25
XAXIS_MAX = 3.75

# SET TYPE1 FONTS
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{euler}']

LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE, weight='bold')
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)
TINY_FP = FontProperties(style='normal', size=TINY_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE, weight='bold')

SMALL_LABEL_FP = FontProperties(style='normal', size=SMALL_LABEL_FONT_SIZE, weight='bold')
SMALL_LEGEND_FP = FontProperties(style='normal', size=SMALL_LEGEND_FONT_SIZE, weight='bold')

YAXIS_TICKS = 3
YAXIS_ROUND = 1000.0

###################################################################################
# CONFIGURATION
###################################################################################

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

PELOTON_BUILD_DIR = BASE_DIR + "/../tuner/build"
SDBENCH = PELOTON_BUILD_DIR + "/bin/sdbench"

OUTPUT_FILE = "outputfile.summary"

## INDEX USAGE TYPES
INDEX_USAGE_TYPE_PARTIAL_FAST  = 1
INDEX_USAGE_TYPE_PARTIAL_MEDIUM  = 2
INDEX_USAGE_TYPE_PARTIAL_SLOW  = 3
INDEX_USAGE_TYPE_FULL = 4
INDEX_USAGE_TYPE_NEVER = 5

INDEX_USAGE_TYPES_STRINGS = {
    1 : "partial-fast",
    2 : "partial-medium",
    3 : "partial-slow",
    4 : "full",
    5 : "never"
}

INDEX_USAGE_TYPES_STRINGS_SUBSET = INDEX_USAGE_TYPES_STRINGS.copy()
INDEX_USAGE_TYPES_STRINGS_SUBSET.pop(4, None)

MOTIVATION_STRINGS_SUBSET = INDEX_USAGE_TYPES_STRINGS.copy()
MOTIVATION_STRINGS_SUBSET.pop(2, None)

INDEX_USAGE_TYPES_SCALE = INDEX_USAGE_TYPES_STRINGS.copy()
INDEX_USAGE_TYPES_SCALE.pop(2, None)
INDEX_USAGE_TYPES_SCALE.pop(3, None)
INDEX_USAGE_TYPES_SCALE.pop(4, None)

## LAYOUT TYPES
LAYOUT_MODE_ROW = 1
LAYOUT_MODE_COLUMN = 2
LAYOUT_MODE_HYBRID = 3

LAYOUT_MODE_STRINGS = {
    1 : 'row',
    2 : 'column',
    3 : 'hybrid'
}

## QUERY COMPLEXITY TYPES
QUERY_COMPLEXITY_SIMPLE   = 1
QUERY_COMPLEXITY_MODERATE = 2
QUERY_COMPLEXITY_COMPLEX  = 3

QUERY_COMPLEXITYS_ALL = [QUERY_COMPLEXITY_SIMPLE, QUERY_COMPLEXITY_MODERATE, QUERY_COMPLEXITY_COMPLEX]

QUERY_COMPLEXITY_STRINGS = {
    1 : "simple",
    2 : "moderate",
    3 : "complex"
}

## WRITE COMPLEXITY TYPES
WRITE_COMPLEXITY_SIMPLE   = 1
WRITE_COMPLEXITY_COMPLEX  = 2
WRITE_COMPLEXITY_INSERT   = 3

WRITE_COMPLEXITY_STRINGS = {
    1 : "simple",
    2 : "complex"
}

## WRITE RATIO TYPES
WRITE_RATIO_READ_ONLY   = 0.0
WRITE_RATIO_READ_HEAVY  = 0.1
WRITE_RATIO_BALANCED    = 0.5
WRITE_RATIO_WRITE_HEAVY = 0.9

WRITE_RATIOS_ALL = [WRITE_RATIO_READ_ONLY, WRITE_RATIO_READ_HEAVY,
                    WRITE_RATIO_BALANCED, WRITE_RATIO_WRITE_HEAVY]

WRITE_RATIO_STRINGS = {
    0.0 : "read-only",
    0.1 : "read-heavy",
    0.5 : "balanced",
    0.9 : "write-heavy"
}

## TUNER MODEL TYPES
TUNER_MODEL_TYPE_DEFAULT = 1
TUNER_MODEL_TYPE_BC  = 2
TUNER_MODEL_TYPE_COLT = 3
TUNER_MODEL_TYPE_RI  = 4

TUNER_MODEL_TYPES_STRINGS = {
    1 : "default",
    2 : "bc",
    3 : "colt",
    4 : "ri"
}

DEFAULT_INDEX_USAGE_TYPE = INDEX_USAGE_TYPE_PARTIAL_FAST
DEFAULT_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
DEFAULT_WRITE_COMPLEXITY = WRITE_COMPLEXITY_COMPLEX
DEFAULT_TUNER_MODEL_TYPE = TUNER_MODEL_TYPE_DEFAULT
DEFAULT_SCALE_FACTOR = 100
DEFAULT_COLUMN_COUNT = 100
DEFAULT_WRITE_RATIO = WRITE_RATIO_READ_ONLY
DEFAULT_TUPLES_PER_TG = 1000
DEFAULT_PHASE_LENGTH = 100
DEFAULT_QUERY_COUNT = 1000
DEFAULT_SELECTIVITY = 0.001
DEFAULT_PROJECTIVITY = 0.1
DEFAULT_VERBOSITY = 0
DEFAULT_CONVERGENCE_MODE = 0
DEFAULT_CONVERGENCE_QUERY_THRESHOLD = 100
DEFAULT_VARIABILITY_THRESHOLD = 50
DEFAULT_INDEX_COUNT_THRESHOLD = 30
DEFAULT_INDEX_UTILITY_THRESHOLD = 0.25
DEFAULT_WRITE_RATIO_THRESHOLD = 1.0
DEFAULT_LAYOUT_MODE = LAYOUT_MODE_ROW
DEFAULT_ANALYZE_SAMPLE_COUNT_THRESHOLD = 10
DEFAULT_DURATION_BETWEEN_PAUSES = 5000
DEFAULT_DURATION_OF_PAUSE = 2000
DEFAULT_TILE_GROUPS_INDEXED_PER_ITERATION = 10
DEFAULT_HOLISTIC_INDEX_ENABLED = 0
DEFAULT_MULTI_STAGE = 0

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5)

## EXPERIMENTS
QUERY_EXPERIMENT = 1
CONVERGENCE_EXPERIMENT = 2
TIME_SERIES_EXPERIMENT = 3
VARIABILITY_EXPERIMENT = 4
SELECTIVITY_EXPERIMENT = 5
SCALE_EXPERIMENT = 6
INDEX_COUNT_EXPERIMENT = 7
TREND_EXPERIMENT = 8
MOTIVATION_EXPERIMENT = 9
HYBRID_EXPERIMENT = 10
MODEL_EXPERIMENT = 11

## DIRS
QUERY_DIR = BASE_DIR + "/results/query"
CONVERGENCE_DIR = BASE_DIR + "/results/convergence"
TIME_SERIES_DIR = BASE_DIR + "/results/time_series"
VARIABILITY_DIR = BASE_DIR + "/results/variability"
SELECTIVITY_DIR = BASE_DIR + "/results/selectivity"
SCALE_DIR = BASE_DIR + "/results/scale"
INDEX_COUNT_DIR = BASE_DIR + "/results/index_count"
LAYOUT_DIR = BASE_DIR + "/results/layout"
TREND_DIR = BASE_DIR + "/results/trend"
MOTIVATION_DIR = BASE_DIR + "/results/motivation"
HOLISTIC_DIR = BASE_DIR + "/results/holistic"
HYBRID_DIR = BASE_DIR + "/results/hybrid"
MODEL_DIR = BASE_DIR + "/results/model"

## INDEX USAGE TYPES
INDEX_USAGE_TYPES_ALL = [INDEX_USAGE_TYPE_PARTIAL_FAST, INDEX_USAGE_TYPE_PARTIAL_MEDIUM, INDEX_USAGE_TYPE_PARTIAL_SLOW, INDEX_USAGE_TYPE_NEVER]
INDEX_USAGE_TYPES_PARTIAL = [INDEX_USAGE_TYPE_PARTIAL_FAST, INDEX_USAGE_TYPE_PARTIAL_MEDIUM, INDEX_USAGE_TYPE_PARTIAL_SLOW]
INDEX_USAGE_TYPES_MOTIVATION = [INDEX_USAGE_TYPE_PARTIAL_SLOW, INDEX_USAGE_TYPE_FULL, INDEX_USAGE_TYPE_PARTIAL_FAST]
INDEX_USAGE_TYPES_SCALE = [INDEX_USAGE_TYPE_PARTIAL_FAST, INDEX_USAGE_TYPE_NEVER]

## QUERY EXPERIMENT
REFLEX_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_ALL
REFLEX_EXP_WRITE_RATIOS = WRITE_RATIOS_ALL
REFLEX_EXP_QUERY_COMPLEXITYS = QUERY_COMPLEXITYS_ALL
REFLEX_EXP_QUERY_COUNT = 3000
REFLEX_EXP_PHASE_LENGTHS = [50, 100, 250, 500]

REFLEX_CSV = "reflex.csv"

##  CONVERGENCE EXPERIMENT
CONVERGENCE_EXP_CONVERGENCE_MODE = 1
CONVERGENCE_EXP_PHASE_LENGTH = 50
CONVERGENCE_EXP_VARIABILITY_THRESHOLD = 10
CONVERGENCE_EXP_QUERY_COUNT = 3000
CONVERGENCE_EXP_INDEX_COUNT = 30
CONVERGENCE_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_PARTIAL
CONVERGENCE_EXP_WRITE_RATIOS = WRITE_RATIOS_ALL
CONVERGENCE_EXP_QUERY_COMPLEXITYS = QUERY_COMPLEXITYS_ALL
CONVERGENCE_CSV = "convergence.csv"

##  TIME SERIES EXPERIMENT
TIME_SERIES_EXP_PHASE_LENGTHS = [100]
TIME_SERIES_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_ALL
TIME_SERIES_EXP_INDEX_COUNT_THRESHOLD = 20
TIME_SERIES_EXP_QUERY_COUNT = 3000
TIME_SERIES_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_HEAVY]
TIME_SERIES_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_MODERATE
TIME_SERIES_EXP_VARIABILITY_THRESHOLD = 10
TIME_SERIES_LATENCY_MODE = 1
TIME_SERIES_INDEX_MODE = 2
TIME_SERIES_PLOT_MODES = [TIME_SERIES_LATENCY_MODE, TIME_SERIES_INDEX_MODE]
TIME_SERIES_LATENCY_CSV = "time_series_latency.csv"
TIME_SERIES_INDEX_CSV = "time_series_index.csv"

##  VARIABILITY EXPERIMENT
VARIABILITY_EXP_PHASE_LENGTH = 50
VARIABILITY_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_ALL
VARIABILITY_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
VARIABILITY_EXP_QUERY_COMPLEXITYS = QUERY_COMPLEXITYS_ALL
VARIABILITY_EXP_VARIABILITY_THRESHOLDS = [3, 5, 10, 20]
VARIABILITY_CSV = "variability.csv"

## SELECTIVITY EXPERIMENT
SELECTIVITY_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_SCALE
SELECTIVITY_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
SELECTIVITY_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
SELECTIVITY_EXP_PHASE_LENGTH = 500
SELECTIVITY_EXP_SELECTIVITYS = [0.0005, 0.005, 0.05, 0.5]
SELECTIVITY_CSV = "selectivity.csv"

## SCALE EXPERIMENT
SCALE_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_SCALE
SCALE_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
SCALE_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
SCALE_EXP_PHASE_LENGTH = 500
SCALE_EXP_SCALES = [1, 10, 100, 1000]
SCALE_CSV = "scale.csv"

## INDEX_COUNT EXPERIMENT
INDEX_COUNT_EXP_INDEX_USAGE_TYPE = INDEX_USAGE_TYPE_PARTIAL_FAST
INDEX_COUNT_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
INDEX_COUNT_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_MODERATE
INDEX_COUNT_EXP_INDEX_COUNT_THRESHOLDS = [3, 5, 10]
INDEX_COUNT_EXP_PHASE_LENGTH = 250
INDEX_COUNT_EXP_VARIABILITY_THRESHOLD = 10
INDEX_COUNT_EXP_QUERY_COUNT = 3000
INDEX_COUNT_LATENCY_MODE = 1
INDEX_COUNT_INDEX_MODE = 2
INDEX_COUNT_PLOT_MODES = [INDEX_COUNT_LATENCY_MODE, INDEX_COUNT_INDEX_MODE]
INDEX_COUNT_LATENCY_CSV = "index_count_latency.csv"
INDEX_COUNT_INDEX_CSV = "index_count_index.csv"

## LAYOUT EXPERIMENT
LAYOUT_EXP_LAYOUT_MODES = [LAYOUT_MODE_ROW, LAYOUT_MODE_HYBRID]
LAYOUT_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
LAYOUT_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_MODERATE
LAYOUT_EXP_COLUMN_COUNT = 500
LAYOUT_EXP_SELECTIVITIES = [0.01, 0.1]
LAYOUT_EXP_PROJECTIVITIES = [0.01, 0.1]
LAYOUT_EXP_INDEX_USAGES_TYPES = [INDEX_USAGE_TYPE_NEVER, INDEX_USAGE_TYPE_PARTIAL_MEDIUM]
LAYOUT_EXP_PHASE_LENGTH = 100
LAYOUT_EXP_QUERY_COUNT = 1000
LAYOUT_EXP_SCALE_FACTOR = 100
LAYOUT_EXP_VARIABILITY_THRESHOLD = 3
LAYOUT_CSV = "layout.csv"

## TREND EXPERIMENT
TREND_EXP_TUNING_COUNT = 300
TREND_EXP_METHODS = ["Data", "Holt-Winters Forecast"]
TREND_CSV = "trend.csv"
TREND_LINE_COLORS = ( '#594F4F', '#45ADA8')

## MOTIVATION EXPERIMENT
MOTIVATION_EXP_INDEX_USAGE_TYPES = INDEX_USAGE_TYPES_MOTIVATION
MOTIVATION_EXP_INDEX_COUNT_THRESHOLD = 5
MOTIVATION_EXP_SCALE_FACTOR = 1000
MOTIVATION_EXP_QUERY_COUNT = 5000
MOTIVATION_EXP_PHASE_LENGTHS = [MOTIVATION_EXP_QUERY_COUNT * 10]
MOTIVATION_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY]
MOTIVATION_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
MOTIVATION_LATENCY_MODE = 1
MOTIVATION_PLOT_MODES = [MOTIVATION_LATENCY_MODE]
MOTIVATION_LATENCY_CSV = "motivation_latency.csv"
MOTIVATION_EXP_DURATION_OF_PAUSE = DEFAULT_DURATION_OF_PAUSE * 5
MOTIVATION_EXP_TILE_GROUPS_INDEXED_PER_ITERATION = 1

## HOLISTIC EXPERIMENT
HOLISTIC_EXPERIMENT_MULTI_STAGE = 1
HOLISTIC_EXPERIMENT_HOLISTIC_INDEXING = [0, 1, 2]
HOLISTIC_EXPERIMENT_PHASE_LENGTH = 200
HOLISTIC_EXPERIMENT_QUERY_COUNT = 1000
HOLISTIC_EXPERIMENT_SCALE_FACTOR = 1000
HOLISTIC_EXPERIMENT_COLUMN_COUNT = 60
HOLISTIC_EXPERIMENT_QUERY_COMPLEXITY = QUERY_COMPLEXITY_MODERATE
HOLISTIC_EXPERIMENT_WRITE_COMPLEXITY = WRITE_COMPLEXITY_INSERT
HOLISTIC_EXPERIMENT_WRITE_RATIO_THRESHOLD = 0.5
HOLISTIC_EXPERIMENT_LAYOUT_MODE = LAYOUT_MODE_COLUMN
HOLISTIC_EXPERIMENT_HOLISTIC_INDEXING_STRINGS = {
    0 : 'peloton',
    1 : 'adaptive',
    2 : 'holistic'
}
HOLISTIC_CSV = 'holistic.csv'

## HYBRID EXPERIMENT
HYBRID_EXP_INDEX_USAGE_TYPES = [INDEX_USAGE_TYPE_FULL, INDEX_USAGE_TYPE_PARTIAL_FAST]
HYBRID_EXP_SCALES = [1000, 10000]
HYBRID_EXP_FRACTIONS = ["0%", "20%", "40%", "60%", "80%", "100%"]
HYBRID_CSV = 'hybrid.csv'

##  MODEL EXPERIMENT
MODEL_EXP_PHASE_LENGTHS = [500]
MODEL_EXP_INDEX_USAGE_TYPE = INDEX_USAGE_TYPE_PARTIAL_FAST
MODEL_EXP_TUNER_MODEL_TYPES = [TUNER_MODEL_TYPE_BC, TUNER_MODEL_TYPE_COLT, TUNER_MODEL_TYPE_RI]
MODEL_EXP_INDEX_COUNT_THRESHOLD = 50
MODEL_EXP_QUERY_COUNT = 5000
MODEL_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY]
MODEL_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_MODERATE
MODEL_EXP_VARIABILITY_THRESHOLD = 30
MODEL_LATENCY_MODE = 1
MODEL_INDEX_MODE = 2
MODEL_PLOT_MODES = [MODEL_LATENCY_MODE, MODEL_INDEX_MODE]
MODEL_LATENCY_CSV = "model_latency.csv"
MODEL_INDEX_CSV = "model_index.csv"
MODEL_EXP_SCALE_FACTOR = 300

###################################################################################
# UTILS
###################################################################################

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def loadDataFile(path):
    file = open(path, "r")
    reader = csv.reader(file)

    data = []

    row_num = 0
    for row in reader:
        row_data = []
        column_num = 0
        for col in row:
            row_data.append(float(col))
            column_num += 1
        row_num += 1
        data.append(row_data)

    return data

def next_power_of_10(n):
    return (10 ** math.ceil(math.log(n, 10)))

def get_upper_bound(n):
    return (math.ceil(n / YAXIS_ROUND) * YAXIS_ROUND)

# # MAKE GRID
def makeGrid(ax):
    axes = ax.get_axes()
    axes.yaxis.grid(True, color=OPT_GRID_COLOR)
    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(AXIS_LINEWIDTH)
    ax.set_axisbelow(True)

# # SAVE GRAPH
def saveGraph(fig, output, width, height):
    size = fig.get_size_inches()
    dpi = fig.get_dpi()
    LOG.debug("Current Size Inches: %s, DPI: %d" % (str(size), dpi))

    new_size = (width / float(dpi), height / float(dpi))
    fig.set_size_inches(new_size)
    new_size = fig.get_size_inches()
    new_dpi = fig.get_dpi()
    LOG.debug("New Size Inches: %s, DPI: %d" % (str(new_size), new_dpi))

    pp = PdfPages(output)
    fig.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()
    LOG.info("OUTPUT: %s", output)

def get_result_file(base_result_dir, result_dir_list, result_file_name):

    # Start with result dir
    final_result_dir = base_result_dir + "/"

    # Add each entry in the list as a sub-dir
    for result_dir_entry in result_dir_list:
        final_result_dir += result_dir_entry + "/"

    # Create dir if needed
    if not os.path.exists(final_result_dir):
        os.makedirs(final_result_dir)

    # Add local file name
    result_file_name = final_result_dir + result_file_name

    #pprint.pprint(result_file_name)
    return result_file_name

###################################################################################
# LEGEND
###################################################################################

def create_legend_index_usage_type():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_VALUES = INDEX_USAGE_TYPES_STRINGS_SUBSET.values()

    figlegend = pylab.figure(figsize=(17, 0.5))
    idx = 0
    lines = [None] * (len(LEGEND_VALUES) + 1)
    data = [1]
    x_values = [1]

    TITLE = "INDEX USAGE MODE:"
    LABELS = [TITLE, "PARTIAL-FAST", "PARTIAL-MODERATE", "PARTIAL-SLOW", "DISABLED"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(len(LEGEND_VALUES)):
        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[idx - 1], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx - 1], markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=7,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_index_usage.pdf')

def create_legend_motivation():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_VALUES = MOTIVATION_STRINGS_SUBSET.values()

    figlegend = pylab.figure(figsize=(14, 0.5))
    idx = 0
    lines = [None] * (len(LEGEND_VALUES) + 1)
    data = [1]
    x_values = [1]

    TITLE = "INDEX USAGE MODE:"
    LABELS = [TITLE, "FULL", "VALUE-BASED PARTIAL", "VALUE-AGNOSTIC PARTIAL"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(len(LEGEND_VALUES)):
        if idx == 1:
            color_idx = 1
        elif idx == 2:
            color_idx = 0
        elif idx == 3:
            color_idx = 2
        
        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[color_idx], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[color_idx], markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_motivation.pdf')


def create_bar_legend_index_usage_type():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(15, 0.5))

    LEGEND_VALUES = INDEX_USAGE_TYPES_STRINGS.values()
    LEGEND_VALUES = LEGEND_VALUES[:-1]

    num_items = len(LEGEND_VALUES) + 1
    ind = np.arange(1)
    margin = 0.10
    width = (1.-2.*margin)/num_items
    data = [1]

    bars = [None] * num_items

    # TITLE
    idx = 0
    bars[idx] = ax1.bar(ind + margin + (idx * width), data, width,
                        color = 'w',
                        linewidth=0)

    idx = 1
    for group in xrange(len(LEGEND_VALUES)):
        bars[idx] = ax1.bar(ind + margin + (idx * width), data, width,
                              color=COLOR_MAP_2[idx - 1],
                              hatch=OPT_PATTERNS[idx - 1],
                              linewidth=BAR_LINEWIDTH)
        idx = idx + 1

    TITLE = "INDEX USAGE MODE:"
    LABELS = [TITLE, "PARTIAL-FAST", "PARTIAL-MODERATE", "PARTIAL-SLOW"]

    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_bar_index_usage_type.pdf')
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

def create_legend_trend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))
    idx = 0
    lines = [None] * (len(TREND_EXP_METHODS) + 1)
    data = [1]
    x_values = [1]

    TITLE = "TREND:"
    LABELS = [TITLE, "DATA", "HOLT-WINTERS FORECAST"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(len(TREND_EXP_METHODS)):
        lines[idx], = ax1.plot(x_values, data,
                               color=TREND_LINE_COLORS[idx - 1],
                               linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx - 1],
                               markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=5,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_trend.pdf')

def create_legend_index_usage_type_subset():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_SIZE = 3

    figlegend = pylab.figure(figsize=(9, 0.5))
    idx = 0
    lines = [None] * (LEGEND_SIZE + 1)
    data = [1]
    x_values = [1]

    TITLE = "INDEX USAGE MODE:"
    LABELS = [TITLE, "PARTIAL-FAST", "DISABLED"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(LEGEND_SIZE):
        color_idx = idx
        if idx == 2:
            color_idx = 4
        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[color_idx - 1], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[color_idx - 1], markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_index_usage_subset.pdf')

def create_legend_index_count():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_SIZE = 4

    figlegend = pylab.figure(figsize=(10, 0.5))
    idx = 0
    lines = [None] * (LEGEND_SIZE + 1)
    data = [1]
    x_values = [1]

    TITLE = "INDEX STORAGE BUDGET:"
    LABELS = [TITLE, "2 GB", "4 GB", "6 GB"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(LEGEND_SIZE):
        lines[idx], = ax1.plot(x_values, data, color=COLOR_MAP_2[idx - 1], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx - 1], markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_index_count.pdf')

def create_legend_layout():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(13, 0.5))

    TITLE = "TUNING MODES:"
    LABELS = [TITLE, "DISABLED", "INDEX", "LAYOUT", "BOTH"]

    num_items = len(LABELS) + 1
    ind = np.arange(1)
    margin = 0.10
    width = (1.-2.*margin)/num_items
    data = [1]

    bars = [None] * num_items

    # TITLE
    idx = 0
    bars[idx] = ax1.bar(ind + margin + (idx * width), data, width,
                        color = 'w',
                        linewidth=0)

    idx = 1
    for group in xrange(len(LABELS)):
        bars[idx] = ax1.bar(ind + margin + (idx * width), data, width,
                              color=COLOR_MAP_2[idx - 1],
                              linewidth=BAR_LINEWIDTH)
        idx = idx + 1

    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=5,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_layout.pdf')

def create_legend_holistic():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_SIZE = 3

    figlegend = pylab.figure(figsize=(12, 0.5))
    idx = 0
    lines = [None] * (LEGEND_SIZE + 1)
    data = [1]
    x_values = [1]

    TITLE = "INDEX TUNING MODES:"
    LABELS = [TITLE, "HOLISTIC", "SMIX", "INCREMENTAL"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(LEGEND_SIZE):
        lines[idx], = ax1.plot(x_values, data, color=COLOR_MAP_3[idx - 1], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx - 1], markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_holistic.pdf')

def create_legend_hybrid():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(8, 0.5))

    TITLE = "SCAN TYPES:"
    LABELS = [TITLE, "HYBRID", "INDEX"]

    num_items = len(LABELS) + 1
    ind = np.arange(1)
    margin = 0.10
    width = (1.-2.*margin)/num_items
    data = [1]

    bars = [None] * num_items

    # TITLE
    idx = 0
    bars[idx] = ax1.bar(ind + margin + (idx * width), data, width,
                        color = 'w',
                        linewidth=0)

    idx = 1
    for group in xrange(len(LABELS)):
        bars[idx] = ax1.bar(ind + margin + (idx * width), data, width,
                              color=COLOR_MAP_3[idx - 1],
                              linewidth=BAR_LINEWIDTH)
        idx = idx + 1

    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=5,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_hybrid.pdf')

def create_legend_model():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_SIZE = 3

    figlegend = pylab.figure(figsize=(13, 0.5))
    idx = 0
    lines = [None] * (LEGEND_SIZE + 1)
    data = [1]
    x_values = [1]

    TITLE = "DECISION LOGIC:"
    LABELS = [TITLE, "RETROSPECTIVE", "COLT", "PREDICTIVE"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 1

    for group in xrange(LEGEND_SIZE):
        lines[idx], = ax1.plot(x_values, data, color=COLOR_MAP_3[idx - 1], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx - 1], markersize=OPT_MARKER_SIZE)
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=3)

    figlegend.savefig('legend_model.pdf')

###################################################################################
# PLOT
###################################################################################

def create_query_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in REFLEX_EXP_PHASE_LENGTHS]
    N = len(x_values)
    ind = np.arange(N)

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=OPT_COLORS[idx],
                 linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=OPT_MARKER_SIZE,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    YAXIS_MIN = 0
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.set_ylim(bottom=YAXIS_MIN)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Phase Lengths", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_convergence_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = CONVERGENCE_EXP_WRITE_RATIOS
    N = len(x_values)
    M = len(INDEX_USAGE_TYPES_PARTIAL)
    ind = np.arange(N)
    margin = 0.1
    width = (1.-2.*margin)/M
    bars = [None] * N

    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        bars[group] =  ax1.bar(ind + margin + (group * width),
                               y_values, width,
                               color=COLOR_MAP_2[group],
                               hatch=OPT_PATTERNS[group],
                               linewidth=BAR_LINEWIDTH)

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Convergence time (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Fraction of updates", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    #ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig


def create_time_series_line_chart(datasets, plot_mode):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in range(1, TIME_SERIES_EXP_QUERY_COUNT + 1)]
    N = len(x_values)
    ind = np.arange(N)

    TIME_SERIES_OPT_LINE_WIDTH = 3.0
    TIME_SERIES_OPT_MARKER_SIZE = 5.0
    TIME_SERIES_OPT_MARKER_FREQUENCY = TIME_SERIES_EXP_QUERY_COUNT/10

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=OPT_COLORS[idx],
                 linewidth=TIME_SERIES_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=TIME_SERIES_OPT_MARKER_SIZE,
                 markevery=TIME_SERIES_OPT_MARKER_FREQUENCY,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    # LATENCY
    if plot_mode == TIME_SERIES_LATENCY_MODE:
        ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)
    # INDEX
    elif plot_mode == TIME_SERIES_INDEX_MODE:
        ax1.set_ylabel("Index count", fontproperties=LABEL_FP)
        YAXIS_MIN = 0
        YAXIS_MAX = 10
        ax1.set_ylim([YAXIS_MIN, YAXIS_MAX])


    # X-AXIS
    #ax1.set_xticks(ind + 0.5)
    major_ticks = np.arange(0, TIME_SERIES_EXP_QUERY_COUNT + 1,
                            TIME_SERIES_OPT_MARKER_FREQUENCY)
    ax1.set_xticks(major_ticks)
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    #ax1.set_xticklabels(x_values)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_holistic_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in range(1, len(datasets[0]) + 1)]
    N = len(x_values)
    ind = np.arange(N)

    TIME_SERIES_OPT_LINE_WIDTH = 3.0
    TIME_SERIES_OPT_MARKER_SIZE = 5.0
    TIME_SERIES_OPT_MARKER_FREQUENCY = TIME_SERIES_EXP_QUERY_COUNT/30

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        print(len(ind), len(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=COLOR_MAP_3[idx],
                 linewidth=TIME_SERIES_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=TIME_SERIES_OPT_MARKER_SIZE,
                 markevery=TIME_SERIES_OPT_MARKER_FREQUENCY,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_yscale('log', nonposy='clip')
    ax1.tick_params(axis='y', which='minor', left='off', right='off')
    ax1.set_yticklabels(["", "1", "10", "100", "1000", "10000"])

    # LATENCY
    ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    #ax1.set_xticks(ind + 0.5)
    major_ticks = np.arange(0, len(x_values) + 1,
                            300)
    ax1.set_xticks(major_ticks)
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    #ax1.set_xticklabels(x_values)

    # LABELS
    y_mark = 0.8
    x_mark_count = 1.0/3
    x_mark_offset = x_mark_count/2 - x_mark_count/4
    x_marks = np.arange(0, 1, x_mark_count)

    ADAPT_LABELS = (["Scan (Copying)", "Scan (Proactive)", "Insert (Dropping)"])

    for idx, x_mark in enumerate(x_marks):
            ax1.text(x_mark + x_mark_offset,
                     y_mark,
                     ADAPT_LABELS[idx],
                     transform=ax1.transAxes,
                     bbox=dict(facecolor='lightgrey', alpha=0.75))

    # ADD VLINES
    plot.axvline(x=1000, color='k', linestyle='--', linewidth=1.0)
    plot.axvline(x=2000, color='k', linestyle='--', linewidth=1.0)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_variability_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in VARIABILITY_EXP_VARIABILITY_THRESHOLDS]
    N = len(x_values)
    ind = np.arange(N)

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=OPT_COLORS[idx],
                 linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=OPT_MARKER_SIZE,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Variability Threshold", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_selectivity_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in SELECTIVITY_EXP_SELECTIVITYS]
    N = len(x_values)
    ind = np.arange(N)

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        color_idx = idx
        if idx == 1:
            color_idx = 3
        ax1.plot(ind + 0.5, y_values,
                 color=OPT_COLORS[color_idx],
                 linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[color_idx],
                 markersize=OPT_MARKER_SIZE,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    YAXIS_MIN = 0
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.set_ylim(bottom=YAXIS_MIN)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Selectivity", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_scale_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in SCALE_EXP_SCALES]
    N = len(x_values)
    ind = np.arange(N)

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        color_idx = idx
        if idx == 1:
            color_idx = 3
        ax1.plot(ind + 0.5, y_values,
                 color=OPT_COLORS[color_idx],
                 linewidth=OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[color_idx],
                 markersize=OPT_MARKER_SIZE,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (s)", fontproperties=LABEL_FP)
    ax1.set_yscale('log', nonposy='clip')
    ax1.tick_params(axis='y', which='minor', left='off', right='off')
    ax1.set_yticklabels(["", "", "1", "10", "100", "1000"])

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Scale factor", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_index_count_line_chart(datasets, plot_mode):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in range(1, INDEX_COUNT_EXP_QUERY_COUNT + 1)]
    N = len(x_values)
    ind = np.arange(N)

    INDEX_COUNT_OPT_LINE_WIDTH = 3.0
    INDEX_COUNT_OPT_MARKER_SIZE = 5.0
    INDEX_COUNT_OPT_MARKER_FREQUENCY = INDEX_COUNT_EXP_QUERY_COUNT/10

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=COLOR_MAP_2[idx],
                 linewidth=INDEX_COUNT_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=INDEX_COUNT_OPT_MARKER_SIZE,
                 markevery=INDEX_COUNT_OPT_MARKER_FREQUENCY,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    # LATENCY
    if plot_mode == INDEX_COUNT_LATENCY_MODE:
        ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)
    # INDEX
    elif plot_mode == INDEX_COUNT_INDEX_MODE:
        ax1.set_ylabel("Index count", fontproperties=LABEL_FP)
        YAXIS_MIN = 0
        YAXIS_MAX = 8
        ax1.set_ylim([YAXIS_MIN, YAXIS_MAX])

    # X-AXIS
    #ax1.set_xticks(ind + 0.5)
    major_ticks = np.arange(0, INDEX_COUNT_EXP_QUERY_COUNT + 1,
                            INDEX_COUNT_OPT_MARKER_FREQUENCY)
    ax1.set_xticks(major_ticks)
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    #ax1.set_xticklabels(x_values)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig


def create_layout_bar_chart(datasets, title=""):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = ['Disabled', 'Index', 'Layout', 'Both']
    N = len(x_values)
    ind = np.arange(N)
    M = 1
    margin = 0.05
    margin_left_right = 0.3
    width = (1.-2.*margin)/M
    bars = [None] * 1

    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        bars[group] =  ax1.bar(ind + margin_left_right + (group * width),
                               y_values, width,
                               color=COLOR_MAP_2,
                               linewidth=BAR_LINEWIDTH)

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Total time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(np.arange(N)+0.75)
    ax1.set_xticklabels(x_values)
    ax1.set_xlabel('Tuning Mode', fontproperties=LABEL_FP)
    #ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_trend_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in range(1, TREND_EXP_TUNING_COUNT + 1)]
    N = len(x_values)
    ind = np.arange(N)

    TREND_OPT_LINE_WIDTH = 3.0
    TREND_OPT_MARKER_SIZE = 5.0
    TREND_OPT_MARKER_FREQUENCY = TREND_EXP_TUNING_COUNT/10

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=TREND_LINE_COLORS[idx],
                 linewidth=TREND_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=TREND_OPT_MARKER_SIZE,
                 markevery=TREND_OPT_MARKER_FREQUENCY,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Index Utility", fontproperties=LABEL_FP)
    YAXIS_MIN = 0
    ax1.set_ylim(bottom=YAXIS_MIN)

    # X-AXIS
    major_ticks = np.arange(0, TREND_EXP_TUNING_COUNT + 1,
                            TREND_OPT_MARKER_FREQUENCY)
    ax1.set_xticks(major_ticks)
    ax1.set_xlabel("Tuning Period", fontproperties=LABEL_FP)
    #ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_motivation_line_chart(datasets, plot_mode):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    # x_values = [str(i) for i in range(1, MOTIVATION_EXP_QUERY_COUNT + 1)]
    # N = len(x_values)
    ind = np.arange(MOTIVATION_EXP_QUERY_COUNT)

    MOTIVATION_OPT_LINE_WIDTH = 0.0
    MOTIVATION_OPT_MARKER_SIZE = 5.0
    MOTIVATION_OPT_MARKER_FREQUENCY = MOTIVATION_EXP_QUERY_COUNT/100

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        x_values = []
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
                    if line == 0:
                        x_values.append(0.0)
                    else:
                        x_values.append(x_values[line-1] + datasets[group][line][col])
        # Convert to second
        x_values = [float(x_val / 1000.0) for x_val in x_values]
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=OPT_COLORS[idx],
                 linewidth=MOTIVATION_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=MOTIVATION_OPT_MARKER_SIZE,
                 markevery=MOTIVATION_OPT_MARKER_FREQUENCY,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_yscale('log', nonposy='clip')

    # LATENCY
    if plot_mode == MOTIVATION_LATENCY_MODE:
        ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    #ax1.set_xticks(ind + 0.5)
    #major_ticks = np.arange(0, 151, 150/10)
    #ax1.set_xticks(major_ticks)
    ax1.set_xlabel("Query sequence", fontproperties=LABEL_FP)
    #ax1.set_xticklabels(x_values)

    # ADD VLINES
    #plot.axvline(x=83.1, color='k', linestyle='--', linewidth=1.0)
    #plot.axvline(x=122.3, color='k', linestyle='--', linewidth=1.0)
    #plot.axvline(x=125.6, color='k', linestyle='--', linewidth=1.0)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_hybrid_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = HYBRID_EXP_FRACTIONS
    N = len(x_values)
    M = len(HYBRID_EXP_INDEX_USAGE_TYPES)
    ind = np.arange(N)
    margin = 0.1
    width = (1.-2.*margin)/M
    bars = [None] * N

    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        bars[group] =  ax1.bar(ind + margin + (group * width),
                               y_values, width,
                               color=COLOR_MAP_3[group],
                               hatch=OPT_PATTERNS[group],
                               linewidth=BAR_LINEWIDTH)

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Percentage of table indexed (%)", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    #ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_model_line_chart(datasets, plot_mode):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in range(1, MODEL_EXP_QUERY_COUNT + 1)]
    N = len(x_values)
    ind = np.arange(N)

    MODEL_OPT_LINE_WIDTH = 3.0
    MODEL_OPT_MARKER_SIZE = 5.0
    MODEL_OPT_MARKER_FREQUENCY = MODEL_EXP_QUERY_COUNT/10

    idx = 0
    for group in xrange(len(datasets)):
        # GROUP
        y_values = []
        for line in  xrange(len(datasets[group])):
            for col in  xrange(len(datasets[group][line])):
                if col == 1:
                    y_values.append(datasets[group][line][col])
        LOG.info("group_data = %s", str(y_values))
        ax1.plot(ind + 0.5, y_values,
                 color=COLOR_MAP_3[idx],
                 linewidth=MODEL_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx],
                 markersize=MODEL_OPT_MARKER_SIZE,
                 markevery=MODEL_OPT_MARKER_FREQUENCY,
                 label=str(group))
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()

    # LATENCY
    if plot_mode == MODEL_LATENCY_MODE:
        ax1.set_ylabel("Latency (ms)", fontproperties=LABEL_FP)
    # INDEX
    elif plot_mode == MODEL_INDEX_MODE:
        ax1.set_ylabel("Index count", fontproperties=LABEL_FP)
        YAXIS_MIN = 0
        YAXIS_MAX = 10
        ax1.set_ylim([YAXIS_MIN, YAXIS_MAX])


    # X-AXIS
    #ax1.set_xticks(ind + 0.5)
    major_ticks = np.arange(0, MODEL_EXP_QUERY_COUNT + 1,
                            MODEL_OPT_MARKER_FREQUENCY)
    ax1.set_xticks(major_ticks)
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    #ax1.set_xticklabels(x_values)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

###################################################################################
# PLOT HELPERS
###################################################################################

# QUERY -- PLOT
def reflex_plot():

    for query_complexity in REFLEX_EXP_QUERY_COMPLEXITYS:
        for write_ratio in REFLEX_EXP_WRITE_RATIOS:

            datasets = []
            for index_usage_type in REFLEX_EXP_INDEX_USAGE_TYPES:
                # Get result file
                result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                   WRITE_RATIO_STRINGS[write_ratio],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(QUERY_DIR, result_dir_list, REFLEX_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_query_line_chart(datasets)

            file_name = "reflex" + "-" + \
                        QUERY_COMPLEXITY_STRINGS[query_complexity] + "-" + \
                        WRITE_RATIO_STRINGS[write_ratio] + ".pdf"

            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# CONVERGENCE -- PLOT
def convergence_plot():

    for query_complexity in CONVERGENCE_EXP_QUERY_COMPLEXITYS:

            datasets = []
            for index_usage_type in CONVERGENCE_EXP_INDEX_USAGE_TYPES:
                # Get result file
                result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(CONVERGENCE_DIR, result_dir_list, CONVERGENCE_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_convergence_bar_chart(datasets)

            file_name = "convergence" + "-" + \
                        QUERY_COMPLEXITY_STRINGS[query_complexity] + ".pdf"

            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# TIME SERIES -- PLOT
def time_series_plot():

    num_graphs = len(TIME_SERIES_EXP_WRITE_RATIOS)

    for graph_itr in range(0, num_graphs):

        # Pick parameters for time series graph set
        query_complexity = TIME_SERIES_EXP_QUERY_COMPLEXITY
        write_ratio = TIME_SERIES_EXP_WRITE_RATIOS[graph_itr]

        for phase_length in TIME_SERIES_EXP_PHASE_LENGTHS:

            for plot_mode in TIME_SERIES_PLOT_MODES:

                # LATENCY
                if plot_mode == TIME_SERIES_LATENCY_MODE:
                    CSV_FILE = TIME_SERIES_LATENCY_CSV
                    OUTPUT_STRING = "latency"
                # INDEX COUNT
                elif plot_mode == TIME_SERIES_INDEX_MODE:
                    CSV_FILE = TIME_SERIES_INDEX_CSV
                    OUTPUT_STRING = "index"

                datasets = []
                for index_usage_type in TIME_SERIES_EXP_INDEX_USAGE_TYPES:

                        # Get result file
                        result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                           WRITE_RATIO_STRINGS[write_ratio],
                                           QUERY_COMPLEXITY_STRINGS[query_complexity],
                                           str(phase_length)]
                        result_file = get_result_file(TIME_SERIES_DIR,
                                                      result_dir_list,
                                                      CSV_FILE)

                        dataset = loadDataFile(result_file)
                        datasets.append(dataset)

                fig = create_time_series_line_chart(datasets, plot_mode)

                file_name = "time-series" + "-" + OUTPUT_STRING + "-" + \
                            QUERY_COMPLEXITY_STRINGS[query_complexity] + "-" + \
                            WRITE_RATIO_STRINGS[write_ratio] + "-" + \
                            str(phase_length) + ".pdf"

                saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 3.0, height=OPT_GRAPH_HEIGHT/1.5)

# VARIABILITY -- PLOT
def variability_plot():

    for query_complexity in VARIABILITY_EXP_QUERY_COMPLEXITYS:

            datasets = []
            for index_usage_type in VARIABILITY_EXP_INDEX_USAGE_TYPES:

                # Get result file
                result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(VARIABILITY_DIR, result_dir_list, VARIABILITY_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_variability_line_chart(datasets)

            file_name = "variability" + "-" + \
                        QUERY_COMPLEXITY_STRINGS[query_complexity] + ".pdf"

            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# SELECTIVITY -- PLOT
def selectivity_plot():

    datasets = []
    for index_usage_type in SELECTIVITY_EXP_INDEX_USAGE_TYPES:
        # Get result file
        result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type]]
        result_file = get_result_file(SELECTIVITY_DIR, result_dir_list, SELECTIVITY_CSV)

        dataset = loadDataFile(result_file)
        datasets.append(dataset)

    fig = create_selectivity_line_chart(datasets)

    file_name = "selectivity" + ".pdf"

    saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# SCALE -- PLOT
def scale_plot():

    datasets = []
    for index_usage_type in SELECTIVITY_EXP_INDEX_USAGE_TYPES:
        # Get result file
        result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type]]
        result_file = get_result_file(SCALE_DIR, result_dir_list, SCALE_CSV)

        dataset = loadDataFile(result_file)
        datasets.append(dataset)

    fig = create_scale_line_chart(datasets)

    file_name = "scale" + ".pdf"

    saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# INDEX COUNT -- PLOT
def index_count_plot():

    for plot_mode in INDEX_COUNT_PLOT_MODES:

        # LATENCY
        if plot_mode == INDEX_COUNT_LATENCY_MODE:
            CSV_FILE = INDEX_COUNT_LATENCY_CSV
            OUTPUT_STRING = "latency"
        # INDEX COUNT
        elif plot_mode == INDEX_COUNT_INDEX_MODE:
            CSV_FILE = INDEX_COUNT_INDEX_CSV
            OUTPUT_STRING = "index"

        datasets = []
        for index_count_threshold in INDEX_COUNT_EXP_INDEX_COUNT_THRESHOLDS:

            # Get result file
            result_dir_list = [str(index_count_threshold)]
            result_file = get_result_file(INDEX_COUNT_DIR,
                                          result_dir_list,
                                          CSV_FILE)

            dataset = loadDataFile(result_file)
            datasets.append(dataset)

        fig = create_index_count_line_chart(datasets, plot_mode)

        file_name = "index-count" + "-" + OUTPUT_STRING + ".pdf"

        saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 3.0, height=OPT_GRAPH_HEIGHT/1.5)


# TREND -- PLOT
def trend_plot():

    num_lines = len(TREND_EXP_METHODS)

    datasets = []
    for line_itr in range(0, num_lines):

        # Get result file
        result_dir_list = [str(line_itr)]
        result_file = get_result_file(TREND_DIR, result_dir_list, TREND_CSV)

        dataset = loadDataFile(result_file)
        datasets.append(dataset)

    fig = create_trend_line_chart(datasets)

    file_name = "trend" + ".pdf"

    saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 2.0, height=OPT_GRAPH_HEIGHT)

# LAYOUT -- PLOT
def layout_plot():

    for projectivity in LAYOUT_EXP_PROJECTIVITIES:
        for selectivity in LAYOUT_EXP_SELECTIVITIES:

            datasets = []

            # Get result file
            result_dir_list = [str(selectivity), str(projectivity)]
            result_file = get_result_file(LAYOUT_DIR, result_dir_list, LAYOUT_CSV)

            dataset = loadDataFile(result_file)
            datasets.append(dataset)

            fig = create_layout_bar_chart(datasets)

            file_name = "layout-" + str(projectivity) + "-" + str(selectivity) + ".pdf"
            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# HOLISTIC -- PLOT
def holistic_plot():
    datasets = []
    for holistic_index_enabled in HOLISTIC_EXPERIMENT_HOLISTIC_INDEXING:
        result_dir_list = [HOLISTIC_EXPERIMENT_HOLISTIC_INDEXING_STRINGS[holistic_index_enabled]]

        result_file = get_result_file(HOLISTIC_DIR, result_dir_list, HOLISTIC_CSV)

        dataset = loadDataFile(result_file)
        datasets.append(dataset)

    # Reverse the order of the lines for better visual
    datasets = datasets[::-1]
    fig = create_holistic_line_chart(datasets)

    file_name = "holistic.pdf";

    saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 3.0, height=OPT_GRAPH_HEIGHT/1.5)

# HYBRID -- PLOY
def hybrid_plot():

    for scale_factor in HYBRID_EXP_SCALES:

        datasets = []
        for index_usage_type in HYBRID_EXP_INDEX_USAGE_TYPES:

            # Get result file
            result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                               str(scale_factor)]
            result_file = get_result_file(HYBRID_DIR, result_dir_list, HYBRID_CSV)

            dataset = loadDataFile(result_file)
            datasets.append(dataset)

        fig = create_hybrid_bar_chart(datasets)

        file_name = "hybrid" + "-" + \
                    str(scale_factor) + ".pdf"

        saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# MODEL -- PLOT
def model_plot():

    num_graphs = len(MODEL_EXP_WRITE_RATIOS)

    for graph_itr in range(0, num_graphs):

        # Pick parameters for time series graph set
        query_complexity = MODEL_EXP_QUERY_COMPLEXITY
        write_ratio = MODEL_EXP_WRITE_RATIOS[graph_itr]

        for phase_length in MODEL_EXP_PHASE_LENGTHS:

            for plot_mode in MODEL_PLOT_MODES:

                # LATENCY
                if plot_mode == MODEL_LATENCY_MODE:
                    CSV_FILE = MODEL_LATENCY_CSV
                    OUTPUT_STRING = "latency"
                # INDEX COUNT
                elif plot_mode == MODEL_INDEX_MODE:
                    CSV_FILE = MODEL_INDEX_CSV
                    OUTPUT_STRING = "index"

                datasets = []
                for tuner_model_type in MODEL_EXP_TUNER_MODEL_TYPES:

                        # Get result file
                        result_dir_list = [TUNER_MODEL_TYPES_STRINGS[tuner_model_type],
                                           WRITE_RATIO_STRINGS[write_ratio],
                                           QUERY_COMPLEXITY_STRINGS[query_complexity],
                                           str(phase_length)]
                        result_file = get_result_file(MODEL_DIR,
                                                      result_dir_list,
                                                      CSV_FILE)

                        dataset = loadDataFile(result_file)
                        datasets.append(dataset)

                fig = create_model_line_chart(datasets, plot_mode)

                file_name = "model" + "-" + OUTPUT_STRING + "-" + \
                            QUERY_COMPLEXITY_STRINGS[query_complexity] + "-" + \
                            WRITE_RATIO_STRINGS[write_ratio] + "-" + \
                            str(phase_length) + ".pdf"

                saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 2.0, height=OPT_GRAPH_HEIGHT)


###################################################################################
# EVAL HELPERS
###################################################################################

# CLEAN UP RESULT DIR
def clean_up_dir(result_directory):

    subprocess.call(['rm', '-rf', result_directory])
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

# RUN EXPERIMENT
def run_experiment(
    program=SDBENCH,
    column_count=DEFAULT_COLUMN_COUNT,
    convergence_query_threshold=DEFAULT_CONVERGENCE_QUERY_THRESHOLD,
    query_complexity=DEFAULT_QUERY_COMPLEXITY,
    variability_threshold=DEFAULT_VARIABILITY_THRESHOLD,
    index_usage_type=DEFAULT_INDEX_USAGE_TYPE,
    tuner_model_type=DEFAULT_TUNER_MODEL_TYPE,
    analyze_sample_count_threshold=DEFAULT_ANALYZE_SAMPLE_COUNT_THRESHOLD,
    tuples_per_tg=DEFAULT_TUPLES_PER_TG,
    duration_between_pauses=DEFAULT_DURATION_BETWEEN_PAUSES,
    duration_of_pause=DEFAULT_DURATION_OF_PAUSE,
    scale_factor=DEFAULT_SCALE_FACTOR,
    tile_groups_indexed_per_iteration=DEFAULT_TILE_GROUPS_INDEXED_PER_ITERATION,
    convergence_mode=DEFAULT_CONVERGENCE_MODE,
    projectivity=DEFAULT_PROJECTIVITY,
    query_count=DEFAULT_QUERY_COUNT,
    selectivity=DEFAULT_SELECTIVITY,
    phase_length=DEFAULT_PHASE_LENGTH,
    verbosity=DEFAULT_VERBOSITY,
    write_complexity=DEFAULT_WRITE_COMPLEXITY,
    write_ratio=DEFAULT_WRITE_RATIO,
    index_count_threshold=DEFAULT_INDEX_COUNT_THRESHOLD,
    index_utility_threshold=DEFAULT_INDEX_UTILITY_THRESHOLD,
    write_ratio_threshold=DEFAULT_WRITE_RATIO_THRESHOLD,
    layout_mode=DEFAULT_LAYOUT_MODE,
    holistic_index_enabled=DEFAULT_HOLISTIC_INDEX_ENABLED,
    multi_stage=DEFAULT_MULTI_STAGE):

    # subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)
    PROGRAM_OUTPUT_FILE_NAME = "program.txt"
    PROGRAM_OUTPUT_FILE = open(PROGRAM_OUTPUT_FILE_NAME, "w")
    arg_list = [program,
                     "-a", str(column_count),
                     "-b", str(convergence_query_threshold),
                     "-c", str(query_complexity),
                     "-d", str(variability_threshold),
                     "-e", str(index_usage_type),
                     "-f", str(analyze_sample_count_threshold),
                     "-g", str(tuples_per_tg),
                     "-i", str(duration_between_pauses),
                     "-j", str(duration_of_pause),
                     "-k", str(scale_factor),
                     "-m", str(tile_groups_indexed_per_iteration),
                     "-o", str(convergence_mode),
                     "-p", str(projectivity),
                     "-q", str(query_count),
                     "-s", str(selectivity),
                     "-t", str(phase_length),
                     "-v", str(tuner_model_type),
                     "-u", str(write_complexity),
                     "-w", str(write_ratio),
                     "-x", str(index_count_threshold),
                     "-y", str(index_utility_threshold),
                     "-z", str(write_ratio_threshold),
                     "-l", str(layout_mode),
                     "-n", str(multi_stage),
                     "-r", str(holistic_index_enabled)
                     ]
    arg_string = ' '.join(arg_list[1:])
    pprint.pprint(arg_string)
    subprocess.call(arg_list,
                    stdout = PROGRAM_OUTPUT_FILE)
    subprocess.call(["rm -f " + PROGRAM_OUTPUT_FILE_NAME], shell=True)

# MOTIVATION -- PLOT
def motivation_plot():

    num_graphs = len(MOTIVATION_EXP_WRITE_RATIOS)

    for graph_itr in range(0, num_graphs):

        # Pick parameters for time series graph set
        query_complexity = MOTIVATION_EXP_QUERY_COMPLEXITY
        write_ratio = MOTIVATION_EXP_WRITE_RATIOS[graph_itr]

        for phase_length in MOTIVATION_EXP_PHASE_LENGTHS:

            for plot_mode in MOTIVATION_PLOT_MODES:

                # LATENCY
                if plot_mode == MOTIVATION_LATENCY_MODE:
                    CSV_FILE = MOTIVATION_LATENCY_CSV
                    OUTPUT_STRING = "latency"

                datasets = []
                for index_usage_type in MOTIVATION_EXP_INDEX_USAGE_TYPES:

                        # Get result file
                        result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                           WRITE_RATIO_STRINGS[write_ratio],
                                           QUERY_COMPLEXITY_STRINGS[query_complexity],
                                           str(phase_length)]
                        result_file = get_result_file(MOTIVATION_DIR,
                                                      result_dir_list,
                                                      CSV_FILE)

                        dataset = loadDataFile(result_file)
                        datasets.append(dataset)

                fig = create_motivation_line_chart(datasets, plot_mode)

                file_name = "motivation" + "-" + OUTPUT_STRING + "-" + \
                            QUERY_COMPLEXITY_STRINGS[query_complexity] + "-" + \
                            WRITE_RATIO_STRINGS[write_ratio] + "-" + \
                            str(phase_length) + ".pdf"

                saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 2, height=OPT_GRAPH_HEIGHT)


###################################################################################
# UTILITIES
###################################################################################

# COLLECT STATS

# Collect result to a given file that already exists
def collect_aggregate_stat(independent_variable,
                           result_file_name):

    # Open result file in append mode
    result_file = open(result_file_name, "a")

    # Sum up stat
    itr = 1
    stat = 0
    with open(OUTPUT_FILE) as fp:
        for line in fp:
            line = line.strip()
            data = line.split(" ")
            stat += float(data[-1])
            itr += 1

    #pprint.pprint(stat)
    result_file.write(str(independent_variable) + " , " + str(stat) + "\n")
    result_file.close()

# Collect result to a given file that already exists
def collect_stat(independent_variable,
                 result_file_name,
                 stat_offset):

    # Open result file in append mode
    result_file = open(result_file_name, "a")

    # Sum up stat
    itr = 1
    with open(OUTPUT_FILE) as fp:
        for line in fp:
            line = line.strip()
            data = line.split(" ")
            stat = float(data[stat_offset])

            result_file.write(str(itr) + " , " + str(stat) + "\n")
            itr += 1

    result_file.close()


###################################################################################
# EVAL
###################################################################################

# QUERY -- EVAL
def reflex_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(QUERY_DIR)
    print("QUERY EVAL")

    for query_complexity in REFLEX_EXP_QUERY_COMPLEXITYS:
        print(MAJOR_STRING)

        for write_ratio in REFLEX_EXP_WRITE_RATIOS:
            print(MINOR_STRING)

            for index_usage_type in REFLEX_EXP_INDEX_USAGE_TYPES:
                for phase_length in REFLEX_EXP_PHASE_LENGTHS:
                    print("> query_complexity: " + str(query_complexity) +
                            " write_ratio: " + str(write_ratio) +
                            " index_usage_type: " + str(index_usage_type) +
                            " phase_length: " + str(phase_length) )

                    # Get result file
                    result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(QUERY_DIR, result_dir_list, REFLEX_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   index_usage_type=index_usage_type,
                                   write_ratio=write_ratio,
                                   query_count=REFLEX_EXP_QUERY_COUNT,
                                   query_complexity=query_complexity)

                    # Collect stat
                    collect_aggregate_stat(phase_length, result_file)

# CONVERGENCE -- EVAL
def convergence_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(CONVERGENCE_DIR)
    print("CONVERGENCE EVAL")

    for query_complexity in CONVERGENCE_EXP_QUERY_COMPLEXITYS:
        print(MAJOR_STRING)

        for index_usage_type in CONVERGENCE_EXP_INDEX_USAGE_TYPES:
            print(MINOR_STRING)

            for write_ratio in CONVERGENCE_EXP_WRITE_RATIOS:
                    print("> query_complexity: " + str(query_complexity) +
                            " index_usage_type: " + str(index_usage_type) +
                            " write_ratio: " + str(write_ratio) )

                    # Get result file
                    result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(CONVERGENCE_DIR, result_dir_list, CONVERGENCE_CSV)

                    # Run experiment (only one phase)
                    run_experiment(phase_length=CONVERGENCE_EXP_PHASE_LENGTH,
                                   query_count=CONVERGENCE_EXP_QUERY_COUNT,
                                   index_count_threshold=CONVERGENCE_EXP_INDEX_COUNT,
                                   index_usage_type=index_usage_type,
                                   write_ratio=write_ratio,
                                   query_complexity=query_complexity,
                                   convergence_mode=CONVERGENCE_EXP_CONVERGENCE_MODE,
                                   variability_threshold=CONVERGENCE_EXP_VARIABILITY_THRESHOLD)

                    # Collect stat
                    collect_aggregate_stat(write_ratio, result_file)

# TIME SERIES -- EVAL
def time_series_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TIME_SERIES_DIR)
    print("TIME SERIES EVAL")

    num_graphs = len(TIME_SERIES_EXP_WRITE_RATIOS)

    for graph_itr in range(0, num_graphs):
        print(MAJOR_STRING)

        # Pick parameters for time series graph set
        query_complexity = TIME_SERIES_EXP_QUERY_COMPLEXITY
        write_ratio = TIME_SERIES_EXP_WRITE_RATIOS[graph_itr]

        for phase_length in TIME_SERIES_EXP_PHASE_LENGTHS:
            print(MINOR_STRING)

            for index_usage_type in TIME_SERIES_EXP_INDEX_USAGE_TYPES:
                    print("> query_complexity: " + str(query_complexity) +
                            " write_ratio: " + str(write_ratio) +
                            " phase_length: " + str(phase_length) +
                            " index_usage_type: " + str(index_usage_type) )

                    # Get result file
                    result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity],
                                       str(phase_length)]

                    latency_result_file = get_result_file(TIME_SERIES_DIR,
                                                          result_dir_list,
                                                          TIME_SERIES_LATENCY_CSV)
                    index_result_file = get_result_file(TIME_SERIES_DIR,
                                                        result_dir_list,
                                                        TIME_SERIES_INDEX_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   index_usage_type=index_usage_type,
                                   write_ratio=write_ratio,
                                   index_count_threshold=TIME_SERIES_EXP_INDEX_COUNT_THRESHOLD,
                                   variability_threshold=TIME_SERIES_EXP_VARIABILITY_THRESHOLD,
                                   query_count=TIME_SERIES_EXP_QUERY_COUNT,
                                   query_complexity=query_complexity)

                    # Collect stat
                    stat_offset = -1
                    collect_stat(DEFAULT_QUERY_COUNT, latency_result_file, stat_offset)
                    stat_offset = -2
                    collect_stat(DEFAULT_QUERY_COUNT, index_result_file, stat_offset)

# VARIABILITY -- EVAL
def variability_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(VARIABILITY_DIR)
    print("VARIABILITY EVAL")

    for query_complexity in VARIABILITY_EXP_QUERY_COMPLEXITYS:
        print(MAJOR_STRING)

        for index_usage_type in VARIABILITY_EXP_INDEX_USAGE_TYPES:
            print(MINOR_STRING)

            for variability_threshold in VARIABILITY_EXP_VARIABILITY_THRESHOLDS:
                print("> query_complexity: " + str(query_complexity) +
                        " index_usage_type: " + str(index_usage_type) +
                        " variability_threshold: " + str(variability_threshold) )


                # Get result file
                result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(VARIABILITY_DIR, result_dir_list, VARIABILITY_CSV)

                # Run experiment
                run_experiment(phase_length=VARIABILITY_EXP_PHASE_LENGTH,
                               index_usage_type=index_usage_type,
                               write_ratio=VARIABILITY_EXP_WRITE_RATIO,
                               query_complexity=query_complexity,
                               variability_threshold=variability_threshold)

                # Collect stat
                collect_aggregate_stat(variability_threshold, result_file)

# INDEX COUNT -- EVAL
def index_count_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(INDEX_COUNT_DIR)
    print("INDEX COUNT EVAL")

    for index_count_threshold in INDEX_COUNT_EXP_INDEX_COUNT_THRESHOLDS:
            print("> index_count_threshold: " + str(index_count_threshold))

            # Get result file
            result_dir_list = [str(index_count_threshold)]

            latency_result_file = get_result_file(INDEX_COUNT_DIR,
                                                  result_dir_list,
                                                  INDEX_COUNT_LATENCY_CSV)
            index_result_file = get_result_file(INDEX_COUNT_DIR,
                                                result_dir_list,
                                                INDEX_COUNT_INDEX_CSV)

            # Run experiment
            run_experiment(phase_length=INDEX_COUNT_EXP_PHASE_LENGTH,
                           index_usage_type=INDEX_COUNT_EXP_INDEX_USAGE_TYPE,
                           write_ratio=INDEX_COUNT_EXP_WRITE_RATIO,
                           query_count=TIME_SERIES_EXP_QUERY_COUNT,
                           index_count_threshold=index_count_threshold,
                           variability_threshold=INDEX_COUNT_EXP_VARIABILITY_THRESHOLD)

            # Collect stat
            stat_offset = -1
            collect_stat(DEFAULT_QUERY_COUNT, latency_result_file, stat_offset)
            stat_offset = -2
            collect_stat(DEFAULT_QUERY_COUNT, index_result_file, stat_offset)

# SELECTIVITY -- EVAL
def selectivity_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(SELECTIVITY_DIR)
    print("SELECTIVITY EVAL")

    for selectivity in SELECTIVITY_EXP_SELECTIVITYS:
        print(MAJOR_STRING)

        for index_usage_type in SELECTIVITY_EXP_INDEX_USAGE_TYPES:
            print("> selectivity: " + str(selectivity) +
                    " index_usage_type: " + str(index_usage_type) )

            # Get result file
            result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type]]
            result_file = get_result_file(SELECTIVITY_DIR, result_dir_list, SELECTIVITY_CSV)

            # Run experiment
            run_experiment(write_ratio=SELECTIVITY_EXP_WRITE_RATIO,
                           query_complexity=SELECTIVITY_EXP_QUERY_COMPLEXITY,
                           index_usage_type=index_usage_type,
                           selectivity=selectivity,
                           phase_length=SELECTIVITY_EXP_PHASE_LENGTH)

            # Collect stat
            collect_aggregate_stat(selectivity, result_file)

# SCALE -- EVAL
def scale_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(SCALE_DIR)
    print("SCALE EVAL")

    for scale_factor in SCALE_EXP_SCALES:
        print(MAJOR_STRING)

        for index_usage_type in SCALE_EXP_INDEX_USAGE_TYPES:
            print("> scale_factor: " + str(scale_factor) +
                    " index_usage_type: " + str(index_usage_type))

            # Get result file
            result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type]]
            result_file = get_result_file(SCALE_DIR, result_dir_list, SCALE_CSV)

            # Run experiment
            run_experiment(write_ratio=SCALE_EXP_WRITE_RATIO,
                           query_complexity=SCALE_EXP_QUERY_COMPLEXITY,
                           index_usage_type=index_usage_type,
                           scale_factor=scale_factor,
                           phase_length=SCALE_EXP_PHASE_LENGTH)

            # Collect stat
            collect_aggregate_stat(scale_factor, result_file)

def layout_eval():
    # CLEAN UP RESULT DIR
    clean_up_dir(LAYOUT_DIR)

    phase_length = LAYOUT_EXP_PHASE_LENGTH

    for selectivity in LAYOUT_EXP_SELECTIVITIES:
        print(MAJOR_STRING)
        for projectivity in LAYOUT_EXP_PROJECTIVITIES:
            print(MINOR_STRING)
            for layout_mode in LAYOUT_EXP_LAYOUT_MODES:
                for index_usage_type in LAYOUT_EXP_INDEX_USAGES_TYPES:
                    print(MINOR_STRING)
                    print("> layout_mode: " + str(layout_mode) +
                    " index_usage_type: " + str(index_usage_type) +
                    " selectivity: " + str(selectivity) +
                    " projectivity: " + str(projectivity))

                    # Get result file
                    result_dir_list = [str(selectivity),
                                       str(projectivity)]

                    result_file = get_result_file(LAYOUT_DIR, result_dir_list, LAYOUT_CSV)

                    # Run experiment
                    start_time = time.time()

                    if projectivity > 0.5 and selectivity > 0.5:
                        phase_length = 400
                    else:
                        phase_length = 1000

                    run_experiment(layout_mode=layout_mode,
                                   index_usage_type=index_usage_type,
                                   write_ratio=LAYOUT_EXP_WRITE_RATIO,
                                   query_complexity=LAYOUT_EXP_QUERY_COMPLEXITY,
                                   selectivity=selectivity,
                                   projectivity=projectivity,
                                   column_count=LAYOUT_EXP_COLUMN_COUNT,
                                   phase_length=phase_length,
                                   query_count=LAYOUT_EXP_QUERY_COUNT,
                                   scale_factor=LAYOUT_EXP_SCALE_FACTOR,
                                   variability_threshold=LAYOUT_EXP_VARIABILITY_THRESHOLD)
                    print("Executed in", time.time() - start_time, "s")
                    # Collect stat
                    collect_aggregate_stat(phase_length, result_file)

# MOTIVATION -- EVAL
def motivation_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(MOTIVATION_DIR)
    print("MOTIVATION EVAL")

    num_graphs = len(MOTIVATION_EXP_WRITE_RATIOS)

    for graph_itr in range(0, num_graphs):
        print(MAJOR_STRING)

        # Pick parameters for time series graph set
        query_complexity = MOTIVATION_EXP_QUERY_COMPLEXITY
        write_ratio = MOTIVATION_EXP_WRITE_RATIOS[graph_itr]

        for phase_length in MOTIVATION_EXP_PHASE_LENGTHS:
            print(MINOR_STRING)

            for index_usage_type in MOTIVATION_EXP_INDEX_USAGE_TYPES:
                    print("> query_complexity: " + str(query_complexity) +
                            " write_ratio: " + str(write_ratio) +
                            " phase_length: " + str(phase_length) +
                            " index_usage_type: " + str(index_usage_type) )

                    # Get result file
                    result_dir_list = [INDEX_USAGE_TYPES_STRINGS[index_usage_type],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity],
                                       str(phase_length)]

                    latency_result_file = get_result_file(MOTIVATION_DIR,
                                                          result_dir_list,
                                                          MOTIVATION_LATENCY_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   index_usage_type=index_usage_type,
                                   write_ratio=write_ratio,
                                   index_count_threshold=MOTIVATION_EXP_INDEX_COUNT_THRESHOLD,
                                   query_count=MOTIVATION_EXP_QUERY_COUNT,
                                   scale_factor=MOTIVATION_EXP_SCALE_FACTOR,
                                   query_complexity=query_complexity,
                                   tile_groups_indexed_per_iteration=MOTIVATION_EXP_TILE_GROUPS_INDEXED_PER_ITERATION,
                                   duration_of_pause=MOTIVATION_EXP_DURATION_OF_PAUSE)

                    # Collect stat
                    stat_offset = -1
                    collect_stat(DEFAULT_QUERY_COUNT, latency_result_file, stat_offset)

# HOLISTIC -- EVAL
def holistic_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(HOLISTIC_DIR)
    print("HOLISTIC EVAL")

    for holistic_index_enabled in HOLISTIC_EXPERIMENT_HOLISTIC_INDEXING:
        print(MAJOR_STRING)

        # Get result file
        result_dir_list = [HOLISTIC_EXPERIMENT_HOLISTIC_INDEXING_STRINGS[holistic_index_enabled]]
        holistic_result_file = get_result_file(HOLISTIC_DIR, result_dir_list, HOLISTIC_CSV)

        # Pick appropriate index usage type (slow for adaptive)
        index_usage_type = INDEX_USAGE_TYPE_PARTIAL_FAST
        if holistic_index_enabled == 1:
            index_usage_type = INDEX_USAGE_TYPE_PARTIAL_SLOW

        # Holistic index enabled (skip for adaptive)
        holistic_mode = 0
        if holistic_index_enabled == 2:
            holistic_mode = 1

        print("> holistic: " + str(holistic_mode) +
                " multi_stage: " + str(HOLISTIC_EXPERIMENT_MULTI_STAGE) +
                " layout: " + LAYOUT_MODE_STRINGS[HOLISTIC_EXPERIMENT_LAYOUT_MODE])

        # Run experiment
        start = time.time()
        run_experiment(holistic_index_enabled=holistic_mode,
            multi_stage=HOLISTIC_EXPERIMENT_MULTI_STAGE,
            query_count=HOLISTIC_EXPERIMENT_QUERY_COUNT,
            phase_length=HOLISTIC_EXPERIMENT_PHASE_LENGTH,
            scale_factor=HOLISTIC_EXPERIMENT_SCALE_FACTOR,
            query_complexity=HOLISTIC_EXPERIMENT_QUERY_COMPLEXITY,
            index_usage_type=index_usage_type,
            write_complexity=HOLISTIC_EXPERIMENT_WRITE_COMPLEXITY,
            write_ratio_threshold=HOLISTIC_EXPERIMENT_WRITE_RATIO_THRESHOLD,
            column_count=HOLISTIC_EXPERIMENT_COLUMN_COUNT,
            layout_mode=HOLISTIC_EXPERIMENT_LAYOUT_MODE)
        print("Executed in", time.time() - start, "s")

        stat_offset = -1
        collect_stat(DEFAULT_QUERY_COUNT, holistic_result_file, stat_offset)


# MODEL -- EVAL
def model_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(MODEL_DIR)
    print("MODEL EVAL")

    num_graphs = len(MODEL_EXP_WRITE_RATIOS)

    for graph_itr in range(0, num_graphs):
        print(MAJOR_STRING)

        # Pick parameters for time series graph set
        query_complexity = MODEL_EXP_QUERY_COMPLEXITY
        write_ratio = MODEL_EXP_WRITE_RATIOS[graph_itr]
        index_usage_type = MODEL_EXP_INDEX_USAGE_TYPE

        for phase_length in MODEL_EXP_PHASE_LENGTHS:
            print(MINOR_STRING)

            for tuner_model_type in MODEL_EXP_TUNER_MODEL_TYPES:
                    print("> tuner_model_type: " + str(tuner_model_type) +
                            " write_ratio: " + str(write_ratio) +
                            " phase_length: " + str(phase_length) +
                            " index_usage_type: " + str(index_usage_type) )

                    # Get result file
                    result_dir_list = [TUNER_MODEL_TYPES_STRINGS[tuner_model_type],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity],
                                       str(phase_length)]

                    latency_result_file = get_result_file(MODEL_DIR,
                                                          result_dir_list,
                                                          MODEL_LATENCY_CSV)
                    index_result_file = get_result_file(MODEL_DIR,
                                                        result_dir_list,
                                                        MODEL_INDEX_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   index_usage_type=index_usage_type,
                                   tuner_model_type=tuner_model_type,
                                   write_ratio=write_ratio,
                                   index_count_threshold=MODEL_EXP_INDEX_COUNT_THRESHOLD,
                                   variability_threshold=MODEL_EXP_VARIABILITY_THRESHOLD,
                                   scale_factor=MODEL_EXP_SCALE_FACTOR,
                                   query_count=MODEL_EXP_QUERY_COUNT,
                                   query_complexity=query_complexity)

                    # Collect stat
                    stat_offset = -1
                    collect_stat(DEFAULT_QUERY_COUNT, latency_result_file, stat_offset)
                    stat_offset = -2
                    collect_stat(DEFAULT_QUERY_COUNT, index_result_file, stat_offset)


###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Incremental Experiments')

    parser.add_argument("-a", "--reflex_eval", help="eval reflex", action='store_true')
    parser.add_argument("-b", "--convergence_eval", help="eval convergence", action='store_true')
    parser.add_argument("-c", "--time_series_eval", help="eval time series", action='store_true')
    parser.add_argument("-d", "--variability_eval", help="eval variability", action='store_true')
    parser.add_argument("-e", "--selectivity_eval", help="eval selectivity", action='store_true')
    parser.add_argument("-f", "--scale_eval", help="eval scale", action='store_true')
    parser.add_argument("-g", "--index_count_eval", help="eval index_count", action='store_true')
    parser.add_argument("-i", "--layout_eval", help="eval layout", action="store_true")
    parser.add_argument("-j", "--motivation_eval", help="eval motivation", action="store_true")
    parser.add_argument("-k", "--holistic_eval", help="eval holistic index", action="store_true")
    parser.add_argument("-l", "--model_eval", help="eval model", action='store_true')

    parser.add_argument("-m", "--reflex_plot", help="plot query", action='store_true')
    parser.add_argument("-n", "--convergence_plot", help="plot convergence", action='store_true')
    parser.add_argument("-o", "--time_series_plot", help="plot time series", action='store_true')
    parser.add_argument("-p", "--variability_plot", help="plot variability", action='store_true')
    parser.add_argument("-q", "--selectivity_plot", help="plot selectivity", action='store_true')
    parser.add_argument("-r", "--scale_plot", help="plot scale", action='store_true')
    parser.add_argument("-s", "--index_count_plot", help="plot index_count", action='store_true')
    parser.add_argument("-t", "--layout_plot", help="plot layout", action='store_true')
    parser.add_argument("-u", "--motivation_plot", help="plot motivation", action='store_true')
    parser.add_argument("-v", "--holistic_plot", help="plot_holistic", action='store_true')
    parser.add_argument("-w", "--hybrid_plot", help="plot_hybrid", action='store_true')
    parser.add_argument("-x", "--model_plot", help="plot_model", action='store_true')

    parser.add_argument("-z", "--trend_plot", help="plot trend", action='store_true')

    args = parser.parse_args()

    ## EVAL

    if args.reflex_eval:
        reflex_eval()

    if args.convergence_eval:
        convergence_eval()

    if args.time_series_eval:
        time_series_eval()

    if args.variability_eval:
        variability_eval()

    if args.selectivity_eval:
        selectivity_eval()

    if args.scale_eval:
        scale_eval()

    if args.index_count_eval:
        index_count_eval()

    if args.layout_eval:
        layout_eval()

    if args.motivation_eval:
        motivation_eval()

    if args.holistic_eval:
        holistic_eval();

    if args.model_eval:
        model_eval();

    ## PLOT

    if args.reflex_plot:
        reflex_plot()

    if args.convergence_plot:
        convergence_plot()

    if args.time_series_plot:
        time_series_plot()

    if args.variability_plot:
        variability_plot()

    if args.selectivity_plot:
        selectivity_plot()

    if args.scale_plot:
        scale_plot()

    if args.index_count_plot:
        index_count_plot()

    if args.layout_plot:
        layout_plot()

    if args.trend_plot:
        trend_plot()

    if args.motivation_plot:
        motivation_plot()

    if args.holistic_plot:
        holistic_plot()

    if args.hybrid_plot:
        hybrid_plot()

    if args.model_plot:
        model_plot()

    #create_legend_index_usage_type()
    create_legend_motivation()
    #create_bar_legend_index_usage_type()
    #create_legend_trend()
    #create_legend_index_usage_type_subset()
    #create_legend_index_count()
    #create_legend_layout()
    create_legend_holistic()
    create_legend_hybrid()
    create_legend_model()
