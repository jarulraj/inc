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

MAJOR_STRING = "\"++++++++++++++++++++++++++++\n\n\""
MINOR_STRING = "\"----------------\n\""

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

PELOTON_BUILD_DIR = BASE_DIR + "/../peloton/build"
SDBENCH = PELOTON_BUILD_DIR + "/bin/sdbench"

OUTPUT_FILE = "outputfile.summary"

## TUNER MODE TYPES
TUNER_MODE_TYPE_AGG_FAST  = 1
TUNER_MODE_TYPE_AGG_SLOW = 2
TUNER_MODE_TYPE_CON_FAST = 3
TUNER_MODE_TYPE_CON_SLOW = 4
TUNER_MODE_TYPE_FULL = 5
TUNER_MODE_TYPE_NEVER = 6

TUNER_MODE_STRINGS = {
    1 : "agg-fast",
    2 : "agg-slow",
    3 : "con-fast",
    4 : "con-slow",
    5 : "full",
    6 : "never"
}


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

QUERY_COMPLEXITY_STRINGS = {
    1 : "simple",
    2 : "moderate",
    3 : "complex"
}

## WRITE COMPLEXITY TYPES
WRITE_COMPLEXITY_SIMPLE   = 1
WRITE_COMPLEXITY_COMPLEX  = 2

WRITE_COMPLEXITY_STRINGS = {
    1 : "simple",
    2 : "complex"
}

## WRITE RATIO TYPES
WRITE_RATIO_READ_ONLY   = 0.0
WRITE_RATIO_READ_HEAVY  = 0.1
WRITE_RATIO_BALANCED    = 0.5
WRITE_RATIO_WRITE_HEAVY = 0.9

WRITE_RATIO_STRINGS = {
    0.0 : "read-only",
    0.1 : "read-heavy",
    0.5 : "balanced",
    0.9 : "write-heavy"
}

DEFAULT_TUNER_MODE = TUNER_MODE_TYPE_AGG_FAST
DEFAULT_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
DEFAULT_WRITE_COMPLEXITY = WRITE_COMPLEXITY_COMPLEX
DEFAULT_SCALE_FACTOR = 100
DEFAULT_COLUMN_COUNT = 20
DEFAULT_WRITE_RATIO = WRITE_RATIO_READ_ONLY
DEFAULT_TUPLES_PER_TG = 1000
DEFAULT_PHASE_LENGTH = 100
DEFAULT_QUERY_COUNT = 2000
DEFAULT_SELECTIVITY = 0.001
DEFAULT_PROJECTIVITY = 0.1
DEFAULT_VERBOSITY = 0
DEFAULT_CONVERGENCE_MODE = 0
DEFAULT_CONVERGENCE_QUERY_THRESHOLD = 400
DEFAULT_VARIABILITY_THRESHOLD = 25
DEFAULT_INDEX_COUNT_THRESHOLD = 20
DEFAULT_INDEX_UTILITY_THRESHOLD = 0.25
DEFAULT_WRITE_RATIO_THRESHOLD = 1.0
DEFAULT_LAYOUT_MODE = LAYOUT_MODE_ROW

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5)

## EXPERIMENTS
QUERY_EXPERIMENT = 1
CONVERGENCE_EXPERIMENT = 2
TIME_SERIES_EXPERIMENT = 3
VARIABILITY_EXPERIMENT = 4
SELECTIVITY_EXPERIMENT = 5
INDEX_COUNT_EXPERIMENT = 6
INDEX_UTILITY_EXPERIMENT = 7
WRITE_RATIO_EXPERIMENT = 8
TREND_EXPERIMENT = 9
MOTIVATION_EXPERIMENT = 10

## DIRS
QUERY_DIR = BASE_DIR + "/results/query"
CONVERGENCE_DIR = BASE_DIR + "/results/convergence"
TIME_SERIES_DIR = BASE_DIR + "/results/time_series"
VARIABILITY_DIR = BASE_DIR + "/results/variability"
SELECTIVITY_DIR = BASE_DIR + "/results/selectivity"
INDEX_COUNT_DIR = BASE_DIR + "/results/index_count"
INDEX_UTILITY_DIR = BASE_DIR + "/results/index_utility"
WRITE_RATIO_DIR = BASE_DIR + "/results/write_ratio"
LAYOUT_DIR = BASE_DIR + "/results/layout"
TREND_DIR = BASE_DIR + "/results/trend"
MOTIVATION_DIR = BASE_DIR + "/results/motivation"

## TUNER MODES
TUNER_MODES_ALL = [TUNER_MODE_TYPE_AGG_FAST, TUNER_MODE_TYPE_AGG_SLOW, 
                   TUNER_MODE_TYPE_CON_FAST, TUNER_MODE_TYPE_CON_SLOW,
                   TUNER_MODE_TYPE_FULL, TUNER_MODE_TYPE_NEVER]
TUNER_MODES_SUBSET = TUNER_MODES_ALL[:-1]
TUNER_MODES_MOTIVATION = [TUNER_MODE_TYPE_AGG_FAST, TUNER_MODE_TYPE_FULL, 
                          TUNER_MODE_TYPE_NEVER]

## QUERY EXPERIMENT
QUERY_EXP_TUNER_MODES = TUNER_MODES_ALL
QUERY_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY]
QUERY_EXP_QUERY_COMPLEXITYS = [QUERY_COMPLEXITY_SIMPLE]
QUERY_EXP_PHASE_LENGTHS = [50, 100, 250, 500]
QUERY_CSV = "query.csv"

##  CONVERGENCE EXPERIMENT
CONVERGENCE_EXP_CONVERGENCE_MODE = 1
CONVERGENCE_EXP_VARIABILITY_THRESHOLD = 15
CONVERGENCE_EXP_PHASE_LENGTH = 5
CONVERGENCE_EXP_TUNER_MODES = TUNER_MODES_SUBSET
CONVERGENCE_EXP_WRITE_RATIOS = QUERY_EXP_WRITE_RATIOS
CONVERGENCE_EXP_QUERY_COMPLEXITYS = QUERY_EXP_QUERY_COMPLEXITYS
CONVERGENCE_CSV = "convergence.csv"

##  TIME SERIES EXPERIMENT
TIME_SERIES_EXP_PHASE_LENGTHS = [50]
TIME_SERIES_EXP_TUNER_MODES = TUNER_MODES_ALL
TIME_SERIES_EXP_INDEX_COUNT_THRESHOLD = 5
TIME_SERIES_EXP_QUERY_COUNT = 3000
TIME_SERIES_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_HEAVY, WRITE_RATIO_WRITE_HEAVY]
TIME_SERIES_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_COMPLEX
TIME_SERIES_LATENCY_MODE = 1
TIME_SERIES_INDEX_MODE = 2
TIME_SERIES_PLOT_MODES = [TIME_SERIES_LATENCY_MODE, TIME_SERIES_INDEX_MODE]
TIME_SERIES_LATENCY_CSV = "time_series_latency.csv"
TIME_SERIES_INDEX_CSV = "time_series_index.csv"

##  VARIABILITY EXPERIMENT
VARIABILITY_EXP_PHASE_LENGTH = 50
VARIABILITY_EXP_TUNER_MODES = QUERY_EXP_TUNER_MODES
VARIABILITY_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
VARIABILITY_EXP_QUERY_COMPLEXITYS = QUERY_EXP_QUERY_COMPLEXITYS
VARIABILITY_EXP_VARIABILITY_THRESHOLDS = [5, 10, 15, 25]
VARIABILITY_CSV = "variability.csv"

## SELECTIVITY EXPERIMENT
SELECTIVITY_EXP_TUNER_MODES = TUNER_MODES_ALL
SELECTIVITY_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
SELECTIVITY_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
SELECTIVITY_EXP_PHASE_LENGTHS = QUERY_EXP_PHASE_LENGTHS
SELECTIVITY_EXP_SELECTIVITYS = [0.1, 0.01]
SELECTIVITY_EXP_SCALE_FACTORS = [100, 1000]
SELECTIVITY_CSV = "selectivity.csv"

## INDEX_COUNT EXPERIMENT
INDEX_COUNT_EXP_TUNER_MODES = TUNER_MODES_ALL
INDEX_COUNT_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY, WRITE_RATIO_READ_HEAVY, WRITE_RATIO_BALANCED, WRITE_RATIO_WRITE_HEAVY]
INDEX_COUNT_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
INDEX_COUNT_EXP_INDEX_COUNT_THRESHOLDS = [5, 10, 15, 20]
INDEX_COUNT_EXP_PHASE_LENGTH = 50
INDEX_COUNT_CSV = "index_count.csv"

## INDEX_UTILITY EXPERIMENT
INDEX_UTILITY_EXP_TUNER_MODES = TUNER_MODES_ALL
INDEX_UTILITY_EXP_WRITE_RATIO = WRITE_RATIO_BALANCED
INDEX_UTILITY_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
INDEX_UTILITY_EXP_INDEX_UTILITY_THRESHOLDS = [0, 0.25, 0.5, 0.75]
INDEX_UTILITY_EXP_INDEX_COUNT_THRESHOLDS = [10, 20]
INDEX_UTILITY_EXP_PHASE_LENGTH = INDEX_COUNT_EXP_PHASE_LENGTH
INDEX_UTILITY_CSV = "index_utility.csv"

## WRITE_RATIO EXPERIMENT
WRITE_RATIO_EXP_TUNER_MODES = TUNER_MODES_ALL
WRITE_RATIO_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_HEAVY, WRITE_RATIO_WRITE_HEAVY]
WRITE_RATIO_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
WRITE_RATIO_EXP_WRITE_RATIO_THRESHOLDS = [0.75, 0.9]
WRITE_RATIO_EXP_PHASE_LENGTH = INDEX_COUNT_EXP_PHASE_LENGTH
WRITE_RATIO_CSV = "write_ratio.csv"

## LAYOUT EXPERIMENT
LAYOUT_EXP_LAYOUT_MODES = [LAYOUT_MODE_ROW, LAYOUT_MODE_COLUMN, LAYOUT_MODE_HYBRID]
LAYOUT_EXP_WRITE_RATIO = WRITE_RATIO_READ_HEAVY
LAYOUT_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_MODERATE
LAYOUT_EXP_SELECTIVITYS = [0.01, 0.9]
LAYOUT_EXP_PROJECTIVITYS = [0.01, 0.9]
LAYOUT_EXP_COLUMN_COUNT = 100 # COLUMN_COUNT * PROJECTIVITY should > 0
LAYOUT_EXP_TUNER_MODES = [TUNER_MODE_TYPE_AGG_FAST, TUNER_MODE_TYPE_NEVER]
LAYOUT_EXP_PHASE_LENGTH=250
LAYOUT_CSV = "layout.csv"

## TREND EXPERIMENT
TREND_EXP_TUNING_COUNT = 300
TREND_EXP_METHODS = ["Data", "Holt-Winters Forecast"]
TREND_CSV = "trend.csv"
TREND_LINE_COLORS = ( '#594F4F', '#45ADA8')

## MOTIVATION EXPERIMENT
MOTIVATION_EXP_PHASE_LENGTHS = [3000]
MOTIVATION_EXP_TUNER_MODES = TUNER_MODES_MOTIVATION
MOTIVATION_EXP_INDEX_COUNT_THRESHOLD = 5
MOTIVATION_EXP_QUERY_COUNT = 3000
MOTIVATION_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY]
MOTIVATION_EXP_QUERY_COMPLEXITY = QUERY_COMPLEXITY_SIMPLE
MOTIVATION_LATENCY_MODE = 1
MOTIVATION_INDEX_MODE = 2
MOTIVATION_PLOT_MODES = [MOTIVATION_LATENCY_MODE, MOTIVATION_INDEX_MODE]
MOTIVATION_LATENCY_CSV = "motivation_latency.csv"
MOTIVATION_INDEX_CSV = "motivation_index.csv"

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

def create_legend_tuner_mode():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_VALUES = TUNER_MODE_STRINGS.values()

    figlegend = pylab.figure(figsize=(15, 0.5))
    idx = 0
    lines = [None] * (len(LEGEND_VALUES) + 1)
    data = [1]
    x_values = [1]

    TITLE = "TUNER MODES:"
    LABELS = [TITLE, "AGG-FAST", "AGG-SLOW", "CON-FAST", "CON-SLOW", "FULL", "NEVER"]

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

    figlegend.savefig('legend_tuner_mode.pdf')
    
def create_bar_legend_tuner_mode():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(13, 0.5))

    LEGEND_VALUES = TUNER_MODE_STRINGS.values()
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
                              color=OPT_COLORS[idx - 1],
                              hatch=OPT_PATTERNS[idx - 1],
                              linewidth=BAR_LINEWIDTH)
        idx = idx + 1

    TITLE = "ADAPTATION MODES:"
    LABELS = [TITLE, "AGGRESSIVE", "BALANCED", "CONSERVATIVE"]

    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_bar_tuner_mode.pdf')
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
    
###################################################################################
# PLOT
###################################################################################

def create_query_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in QUERY_EXP_PHASE_LENGTHS]
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
    #ax1.set_ylim([YAXIS_MIN, YAXIS_MAX])
    #ax1.set_yscale('log', basey=10)

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
    M = len(TUNER_MODES_SUBSET)
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
                               color=OPT_COLORS[group],
                               hatch=OPT_PATTERNS[group],
                               linewidth=BAR_LINEWIDTH)

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Convergence time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Read-write ratio", fontproperties=LABEL_FP)
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
        ax1.set_ylabel("Query latency (ms)", fontproperties=LABEL_FP)
    # INDEX
    elif plot_mode == TIME_SERIES_INDEX_MODE:
        ax1.set_ylabel("Index count", fontproperties=LABEL_FP)

    #ax1.set_yscale('log', basey=10)

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
    #ax1.set_yscale('log', basey=10)

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

def create_index_count_bar_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in INDEX_COUNT_EXP_INDEX_COUNT_THRESHOLDS]
    N = len(x_values)
    ind = np.arange(N)
    M = len(TUNER_MODES_ALL)
    margin = 0.1
    width = (1.-2.*margin)/M
    bars = [None] * N

    idx = 0
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
                       color=OPT_COLORS[group],
                       hatch=OPT_PATTERNS[group],
                       linewidth=BAR_LINEWIDTH)
        idx = idx + 1

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    YAXIS_MIN = 0
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    ax1.set_ylim(bottom=YAXIS_MIN)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Index count threshold", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    #ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_index_utility_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in INDEX_UTILITY_EXP_INDEX_UTILITY_THRESHOLDS]
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
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Index utility threshold", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_write_ratio_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in WRITE_RATIO_EXP_WRITE_RATIO_THRESHOLDS]
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
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Write ratio threshold", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
    ax1.set_xlim([XAXIS_MIN, XAXIS_MAX])

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_layout_bar_chart(datasets, desc=""):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [LAYOUT_MODE_STRINGS[layout_mode] for layout_mode in LAYOUT_EXP_LAYOUT_MODES]
    N = len(x_values)
    ind = np.arange(N)
    M = 2
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
                               color=OPT_COLORS[group],
                               hatch=OPT_PATTERNS[group],
                               linewidth=BAR_LINEWIDTH)

    # GRID
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(np.arange(3) + 0.5)
    ax1.set_xlabel("Storage Layout" if desc == "" else "Storage Layout(" + desc + ")", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)
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
    #ax1.set_yscale('log', basey=10)

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
    x_values = [str(i) for i in range(1, MOTIVATION_EXP_QUERY_COUNT + 1)]
    N = len(x_values)
    ind = np.arange(N)
    
    MOTIVATION_OPT_LINE_WIDTH = 3.0
    MOTIVATION_OPT_MARKER_SIZE = 5.0
    MOTIVATION_OPT_MARKER_FREQUENCY = MOTIVATION_EXP_QUERY_COUNT/10

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

    # LATENCY
    if plot_mode == MOTIVATION_LATENCY_MODE:
        ax1.set_ylabel("Query latency (ms)", fontproperties=LABEL_FP)
    # INDEX
    elif plot_mode == MOTIVATION_INDEX_MODE:
        ax1.set_ylabel("Index count", fontproperties=LABEL_FP)

    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    #ax1.set_xticks(ind + 0.5)
    major_ticks = np.arange(0, MOTIVATION_EXP_QUERY_COUNT + 1, 
                            MOTIVATION_OPT_MARKER_FREQUENCY)
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

    for query_complexity in QUERY_EXP_QUERY_COMPLEXITYS:
        for write_ratio in QUERY_EXP_WRITE_RATIOS:

            datasets = []
            for tuner_mode in QUERY_EXP_TUNER_MODES:
                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   WRITE_RATIO_STRINGS[write_ratio],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(QUERY_DIR, result_dir_list, QUERY_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_query_line_chart(datasets)

            file_name = "query" + "-" + \
                        QUERY_COMPLEXITY_STRINGS[query_complexity] + "-" + \
                        WRITE_RATIO_STRINGS[write_ratio] + ".pdf"

            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# CONVERGENCE -- PLOT
def convergence_plot():

    for query_complexity in CONVERGENCE_EXP_QUERY_COMPLEXITYS:

            datasets = []
            for tuner_mode in CONVERGENCE_EXP_TUNER_MODES:
                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
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
                for tuner_mode in TIME_SERIES_EXP_TUNER_MODES:

                        # Get result file
                        result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
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

                saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 3.0, height=OPT_GRAPH_HEIGHT)

# VARIABILITY -- PLOT
def variability_plot():

    for query_complexity in VARIABILITY_EXP_QUERY_COMPLEXITYS:

            datasets = []
            for tuner_mode in VARIABILITY_EXP_TUNER_MODES:

                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
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

    for selectivity in SELECTIVITY_EXP_SELECTIVITYS:
        for scale_factor in SELECTIVITY_EXP_SCALE_FACTORS:

            datasets = []
            for tuner_mode in SELECTIVITY_EXP_TUNER_MODES:
                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   str(selectivity),
                                   str(scale_factor)]
                result_file = get_result_file(SELECTIVITY_DIR, result_dir_list, SELECTIVITY_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_query_line_chart(datasets)

            file_name = "selectivity" + "-" + \
                        str(selectivity) + "-" + \
                        str(scale_factor) + ".pdf"

            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# INDEX_COUNT -- PLOT
def index_count_plot():

    for write_ratio in INDEX_COUNT_EXP_WRITE_RATIOS:

        datasets = []
        for tuner_mode in INDEX_COUNT_EXP_TUNER_MODES:
            # Get result file
            result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                               WRITE_RATIO_STRINGS[write_ratio]]
            result_file = get_result_file(INDEX_COUNT_DIR, result_dir_list, INDEX_COUNT_CSV)

            dataset = loadDataFile(result_file)
            datasets.append(dataset)

        fig = create_index_count_bar_chart(datasets)

        file_name = "index-count" + "-" + \
                    WRITE_RATIO_STRINGS[write_ratio] + ".pdf"

        saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# INDEX_UTILITY -- PLOT
def index_utility_plot():

    for index_count_threshold in INDEX_UTILITY_EXP_INDEX_COUNT_THRESHOLDS:

        datasets = []
        for tuner_mode in INDEX_UTILITY_EXP_TUNER_MODES:
            # Get result file
            result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                               str(index_count_threshold)]
            result_file = get_result_file(INDEX_UTILITY_DIR, result_dir_list, INDEX_UTILITY_CSV)

            dataset = loadDataFile(result_file)
            datasets.append(dataset)

        fig = create_index_utility_line_chart(datasets)

        file_name = "index-utility" + "-" + \
                    str(index_count_threshold) + ".pdf"

        saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# WRITE_RATIO -- PLOT
def write_ratio_plot():

    for write_ratio in WRITE_RATIO_EXP_WRITE_RATIOS:

        datasets = []
        for tuner_mode in WRITE_RATIO_EXP_TUNER_MODES:
            # Get result file
            result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                               str(write_ratio)]
            result_file = get_result_file(WRITE_RATIO_DIR, result_dir_list, WRITE_RATIO_CSV)

            dataset = loadDataFile(result_file)
            datasets.append(dataset)

        fig = create_write_ratio_line_chart(datasets)

        file_name = "write-ratio" + "-" + \
                    str(write_ratio) + ".pdf"

        saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)
        
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

def layout_plot():
    print('plotting layout')
    for selectivity in LAYOUT_EXP_SELECTIVITYS:
        for projectivity in LAYOUT_EXP_PROJECTIVITYS:
            datasets = []
            for tuner_mode in LAYOUT_EXP_TUNER_MODES:
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   str(selectivity), str(projectivity)]

                result_file = get_result_file(LAYOUT_DIR, result_dir_list, LAYOUT_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_layout_bar_chart(datasets, "sel=" + str(selectivity) + " proj=" + str(projectivity))

            file_name = "layout" + '-' \
                        "sel=" + str(selectivity) + "-" + "proj=" + \
                        str(projectivity) + ".pdf"
            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

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
    tuner_mode=DEFAULT_TUNER_MODE,
    tuples_per_tg=DEFAULT_TUPLES_PER_TG,
    scale_factor=DEFAULT_SCALE_FACTOR,
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
    layout_mode=DEFAULT_LAYOUT_MODE):

    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)
    PROGRAM_OUTPUT_FILE_NAME = "program.txt"
    PROGRAM_OUTPUT_FILE = open(PROGRAM_OUTPUT_FILE_NAME, "w")
    arg_list = [program,
                     "-a", str(column_count),
                     "-b", str(convergence_query_threshold),
                     "-c", str(query_complexity),
                     "-d", str(variability_threshold),
                     "-e", str(tuner_mode),
                     "-g", str(tuples_per_tg),
                     "-k", str(scale_factor),
                     "-o", str(convergence_mode),
                     "-p", str(projectivity),
                     "-q", str(query_count),
                     "-s", str(selectivity),
                     "-t", str(phase_length),
                     "-v", str(verbosity),
                     "-u", str(write_complexity),
                     "-w", str(write_ratio),
                     "-x", str(index_count_threshold),
                     "-y", str(index_utility_threshold),
                     "-z", str(write_ratio_threshold),
                     "-l", str(layout_mode)
                     ]
    arg_string = ' '.join(arg_list[1:])
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
                # INDEX COUNT
                elif plot_mode == MOTIVATION_INDEX_MODE:
                    CSV_FILE = MOTIVATION_INDEX_CSV
                    OUTPUT_STRING = "index"

                datasets = []
                for tuner_mode in MOTIVATION_EXP_TUNER_MODES:

                        # Get result file
                        result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
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

                saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 3.0, height=OPT_GRAPH_HEIGHT)


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

    for query_complexity in QUERY_EXP_QUERY_COMPLEXITYS:
        print(MAJOR_STRING)

        for write_ratio in QUERY_EXP_WRITE_RATIOS:
            print(MINOR_STRING)

            for tuner_mode in QUERY_EXP_TUNER_MODES:
                for phase_length in QUERY_EXP_PHASE_LENGTHS:
                    print("> query_complexity: " + str(query_complexity) +
                            " write_ratio: " + str(write_ratio) +
                            " tuner_mode: " + str(tuner_mode) +
                            " phase_length: " + str(phase_length) )

                    # Get result file
                    result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(QUERY_DIR, result_dir_list, QUERY_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   tuner_mode=tuner_mode,
                                   write_ratio=write_ratio,
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

        for tuner_mode in CONVERGENCE_EXP_TUNER_MODES:
            print(MINOR_STRING)

            for write_ratio in CONVERGENCE_EXP_WRITE_RATIOS:
                    print("> query_complexity: " + str(query_complexity) +
                            " tuner_mode: " + str(tuner_mode) +
                            " write_ratio: " + str(write_ratio) )

                    # Get result file
                    result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(CONVERGENCE_DIR, result_dir_list, CONVERGENCE_CSV)

                    # Run experiment (only one phase)
                    run_experiment(phase_length=CONVERGENCE_EXP_PHASE_LENGTH,
                                   tuner_mode=tuner_mode,
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

            for tuner_mode in TIME_SERIES_EXP_TUNER_MODES:
                    print("> query_complexity: " + str(query_complexity) +
                            " write_ratio: " + str(write_ratio) +
                            " phase_length: " + str(phase_length) +
                            " tuner_mode: " + str(tuner_mode) )

                    # Get result file
                    result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
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
                                   tuner_mode=tuner_mode,
                                   write_ratio=write_ratio,
                                   index_count_threshold=TIME_SERIES_EXP_INDEX_COUNT_THRESHOLD,
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

        for tuner_mode in VARIABILITY_EXP_TUNER_MODES:
            print(MINOR_STRING)

            for variability_threshold in VARIABILITY_EXP_VARIABILITY_THRESHOLDS:
                print("> query_complexity: " + str(query_complexity) +
                        " tuner_mode: " + str(tuner_mode) +
                        " variability_threshold: " + str(variability_threshold) )


                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(VARIABILITY_DIR, result_dir_list, VARIABILITY_CSV)

                # Run experiment
                run_experiment(phase_length=VARIABILITY_EXP_PHASE_LENGTH,
                               tuner_mode=tuner_mode,
                               write_ratio=VARIABILITY_EXP_WRITE_RATIO,
                               query_complexity=query_complexity,
                               variability_threshold=variability_threshold)

                # Collect stat
                collect_aggregate_stat(variability_threshold, result_file)

# SELECTIVITY -- EVAL
def selectivity_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(SELECTIVITY_DIR)
    print("SELECTIVITY EVAL")

    for selectivity in SELECTIVITY_EXP_SELECTIVITYS:
        print(MAJOR_STRING)

        for scale_factor in SELECTIVITY_EXP_SCALE_FACTORS:
            print(MINOR_STRING)

            for tuner_mode in SELECTIVITY_EXP_TUNER_MODES:
                for phase_length in SELECTIVITY_EXP_PHASE_LENGTHS:
                    print("> selectivity: " + str(selectivity) +
                            " scale_factor: " + str(scale_factor) +
                            " tuner_mode: " + str(tuner_mode) +
                            " phase_length: " + str(phase_length) )

                    # Get result file
                    result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                       str(selectivity),
                                       str(scale_factor)]
                    result_file = get_result_file(SELECTIVITY_DIR, result_dir_list, SELECTIVITY_CSV)

                    # Run experiment
                    run_experiment(write_ratio=SELECTIVITY_EXP_WRITE_RATIO,
                                   query_complexity=SELECTIVITY_EXP_QUERY_COMPLEXITY,
                                   phase_length=phase_length,
                                   tuner_mode=tuner_mode,
                                   selectivity=selectivity,
                                   scale_factor=scale_factor)

                    # Collect stat
                    collect_aggregate_stat(phase_length, result_file)

# INDEX_COUNT -- EVAL
def index_count_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(INDEX_COUNT_DIR)
    print("INDEX COUNT EVAL")

    for write_ratio in INDEX_COUNT_EXP_WRITE_RATIOS:
        print(MAJOR_STRING)

        for tuner_mode in INDEX_COUNT_EXP_TUNER_MODES:
            print(MINOR_STRING)

            for index_count_threshold in INDEX_COUNT_EXP_INDEX_COUNT_THRESHOLDS:
                print("> write_ratio: " + str(write_ratio) +
                        " tuner_mode: " + str(tuner_mode) +
                        " index_count_threshold: " + str(index_count_threshold) )

                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   WRITE_RATIO_STRINGS[write_ratio]]
                result_file = get_result_file(INDEX_COUNT_DIR, result_dir_list, INDEX_COUNT_CSV)

                # Run experiment
                run_experiment(query_complexity=INDEX_COUNT_EXP_QUERY_COMPLEXITY,
                               phase_length=INDEX_COUNT_EXP_PHASE_LENGTH,
                               tuner_mode=tuner_mode,
                               write_ratio=write_ratio,
                               index_count_threshold=index_count_threshold)

                # Collect stat
                collect_aggregate_stat(index_count_threshold, result_file)

# INDEX_UTILITY -- EVAL
def index_utility_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(INDEX_UTILITY_DIR)
    print("INDEX UTILITY EVAL")

    for tuner_mode in INDEX_UTILITY_EXP_TUNER_MODES:
        print(MAJOR_STRING)

        for index_count_threshold in INDEX_UTILITY_EXP_INDEX_COUNT_THRESHOLDS:
            print(MINOR_STRING)

            for index_utility_threshold in INDEX_UTILITY_EXP_INDEX_UTILITY_THRESHOLDS:
                print("> tuner_mode: " + str(tuner_mode) +
                        " index_count_threshold: " + str(index_count_threshold) +
                        " index_utility_threshold: " + str(index_utility_threshold) )


                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   str(index_count_threshold)]
                result_file = get_result_file(INDEX_UTILITY_DIR, result_dir_list, INDEX_UTILITY_CSV)

                # Run experiment
                run_experiment(query_complexity=INDEX_UTILITY_EXP_QUERY_COMPLEXITY,
                               phase_length=INDEX_UTILITY_EXP_PHASE_LENGTH,
                               write_ratio=INDEX_UTILITY_EXP_WRITE_RATIO,
                               tuner_mode=tuner_mode,
                               index_count_threshold=index_count_threshold)

                # Collect stat
                collect_aggregate_stat(index_utility_threshold, result_file)

# wRITE_RATIO -- EVAL
def write_ratio_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(WRITE_RATIO_DIR)
    print("WRITE RATIO EVAL")

    for tuner_mode in WRITE_RATIO_EXP_TUNER_MODES:
        print(MAJOR_STRING)

        for write_ratio in WRITE_RATIO_EXP_WRITE_RATIOS:
            print(MINOR_STRING)

            for write_ratio_threshold in WRITE_RATIO_EXP_WRITE_RATIO_THRESHOLDS:
                print("> tuner_mode: " + str(tuner_mode) +
                              " write_ratio: " + str(write_ratio) +
                              " write_ratio_threshold: " + str(write_ratio_threshold) )

                # Get result file
                result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                   str(write_ratio)]
                result_file = get_result_file(WRITE_RATIO_DIR, result_dir_list, WRITE_RATIO_CSV)

                # Run experiment
                run_experiment(query_complexity=WRITE_RATIO_EXP_QUERY_COMPLEXITY,
                               phase_length=WRITE_RATIO_EXP_PHASE_LENGTH,
                               write_ratio=write_ratio,
                               tuner_mode=tuner_mode)

                # Collect stat
                collect_aggregate_stat(write_ratio_threshold, result_file)

def layout_eval():
    # CLEAN UP RESULT DIR
    clean_up_dir(LAYOUT_DIR)

    phase_length = LAYOUT_EXP_PHASE_LENGTH

    for layout_mode in LAYOUT_EXP_LAYOUT_MODES: # Different lines
        print(MAJOR_STRING)

        for tuner_mode in LAYOUT_EXP_TUNER_MODES:
            print(MINOR_STRING)

            for projectivity in LAYOUT_EXP_PROJECTIVITYS:
                print(MINOR_STRING)

                for selectivity in LAYOUT_EXP_SELECTIVITYS:
                    print(MINOR_STRING)

                    print("> layout_mode: " + str(layout_mode) +
                            " tuner_mode: " + str(tuner_mode) +
                            " selectivity: " + str(selectivity) +
                            " projectivity: " + str(projectivity))

                    # Get result file
                    result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                       str(selectivity),
                                       str(projectivity)]

                    result_file = get_result_file(LAYOUT_DIR, result_dir_list, LAYOUT_CSV)

                    # Run experiment
                    start_time = time.time()
                    run_experiment(
                                   layout_mode=layout_mode,
                                   tuner_mode=tuner_mode,
                                   write_ratio=LAYOUT_EXP_WRITE_RATIO,
                                   query_complexity=LAYOUT_EXP_QUERY_COMPLEXITY,
                                   selectivity=selectivity,
                                   projectivity=projectivity,
                                   column_count=LAYOUT_EXP_COLUMN_COUNT)
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

            for tuner_mode in MOTIVATION_EXP_TUNER_MODES:
                    print("> query_complexity: " + str(query_complexity) +
                            " write_ratio: " + str(write_ratio) +
                            " phase_length: " + str(phase_length) +
                            " tuner_mode: " + str(tuner_mode) )

                    # Get result file
                    result_dir_list = [TUNER_MODE_STRINGS[tuner_mode],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity],
                                       str(phase_length)]

                    latency_result_file = get_result_file(MOTIVATION_DIR,
                                                          result_dir_list,
                                                          MOTIVATION_LATENCY_CSV)
                    index_result_file = get_result_file(MOTIVATION_DIR,
                                                        result_dir_list,
                                                        MOTIVATION_INDEX_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   tuner_mode=tuner_mode,
                                   write_ratio=write_ratio,
                                   index_count_threshold=MOTIVATION_EXP_INDEX_COUNT_THRESHOLD,
                                   query_count=MOTIVATION_EXP_QUERY_COUNT,
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
    parser.add_argument("-f", "--index_count_eval", help="eval index_count", action='store_true')
    parser.add_argument("-g", "--index_utility_eval", help="eval index_utility", action='store_true')
    parser.add_argument("-i", "--write_ratio_eval", help="eval write_ratio", action='store_true')
    parser.add_argument("-j", "--layout_eval", help="eval layout", action="store_true")
    parser.add_argument("-k", "--motivation_eval", help="eval motivation", action="store_true")

    parser.add_argument("-m", "--reflex_plot", help="plot query", action='store_true')
    parser.add_argument("-n", "--convergence_plot", help="plot convergence", action='store_true')
    parser.add_argument("-o", "--time_series_plot", help="plot time series", action='store_true')
    parser.add_argument("-p", "--variability_plot", help="plot variability", action='store_true')
    parser.add_argument("-q", "--selectivity_plot", help="plot selectivity", action='store_true')
    parser.add_argument("-r", "--index_count_plot", help="plot index_count", action='store_true')
    parser.add_argument("-s", "--index_utility_plot", help="plot index_utility", action='store_true')
    parser.add_argument("-t", "--write_ratio_plot", help="plot write_ratio", action='store_true')
    parser.add_argument("-u", "--layout_plot", help="plot layout", action='store_true')
    parser.add_argument("-v", "--motivation_plot", help="plot motivation", action='store_true')

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

    if args.index_count_eval:
        index_count_eval()

    if args.index_utility_eval:
        index_utility_eval()

    if args.write_ratio_eval:
        write_ratio_eval()

    if args.layout_eval:
        layout_eval()

    if args.motivation_eval:
        motivation_eval()

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

    if args.index_count_plot:
        index_count_plot()

    if args.index_utility_plot:
        index_utility_plot()

    if args.write_ratio_plot:
        write_ratio_plot()

    if args.layout_plot:
        layout_plot()

    if args.trend_plot:
        trend_plot()

    if args.motivation_plot:
        motivation_plot()


    #create_legend_tuner_mode()
    #create_bar_legend_tuner_mode()
    create_legend_trend()
