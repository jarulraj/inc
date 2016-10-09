#!/usr/bin/env python

###################################################################################
# INCREMENTAL EXPERIMENTS
###################################################################################

from __future__ import print_function

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

###################################################################################
# OUTPUT CONFIGURATION
###################################################################################

BASE_DIR = os.path.dirname(__file__)
OPT_FONT_NAME = 'Helvetica'
OPT_GRAPH_HEIGHT = 300
OPT_GRAPH_WIDTH = 400

# Make a list by cycling through the colors you care about
# to match the length of your data.
NUM_COLORS = 5
COLOR_MAP = ( '#F58A87', '#80CA86', '#9EC9E9', '#FED113', '#D89761' )

OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = ([ "////", "////", "o", "o", "\\\\" , "\\\\" , "//////", "//////", ".", "." , "\\\\\\" , "\\\\\\" ])

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

## INDEX USAGE TYPES
INDEX_USAGE_AGGRESSIVE  = 1
INDEX_USAGE_BALANCED = 2
INDEX_USAGE_CONSERVATIVE = 3
INDEX_USAGE_NEVER = 4

INDEX_USAGE_STRINGS = {
    1 : "aggressive",
    2 : "balanced",
    3 : "conservative",
    4 : "never"
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

DEFAULT_INDEX_USAGE = INDEX_USAGE_AGGRESSIVE
DEFAULT_QUERY_COMLEXITY = QUERY_COMPLEXITY_SIMPLE
DEFAULT_SCALE_FACTOR = 100
DEFAULT_COLUMN_COUNT = 20
DEFAULT_WRITE_RATIO = WRITE_RATIO_READ_ONLY
DEFAULT_TUPLES_PER_TG = 1000
DEFAULT_PHASE_LENGTH = 100
DEFAULT_QUERY_COUNT = 1000
DEFAULT_SELECTIVITY = 0.01
DEFAULT_PROJECTIVITY = 0.1
DEFAULT_VERBOSITY = 0
DEFAULT_CONVERGENCE_MODE = 0
DEFAULT_CONVERGENCE_QUERY_THRESHOLD = 200
DEFAULT_VARIABILITY_THRESHOLD = 25

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5)

## EXPERIMENTS
QUERY_EXPERIMENT = 1
CONVERGENCE_EXPERIMENT = 2
TIME_SERIES_EXPERIMENT = 3
VARIABILITY_EXPERIMENT = 4

## DIRS
QUERY_DIR = BASE_DIR + "/results/query"
CONVERGENCE_DIR = BASE_DIR + "/results/convergence"
TIME_SERIES_DIR = BASE_DIR + "/results/time_series"
VARIABILITY_DIR = BASE_DIR + "/results/variability"

## QUERY EXPERIMENT
QUERY_EXP_INDEX_USAGES = [INDEX_USAGE_AGGRESSIVE, INDEX_USAGE_BALANCED, INDEX_USAGE_CONSERVATIVE, INDEX_USAGE_NEVER]
QUERY_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY, WRITE_RATIO_READ_HEAVY, WRITE_RATIO_BALANCED, WRITE_RATIO_WRITE_HEAVY]
QUERY_EXP_QUERY_COMPLEXITYS = [QUERY_COMPLEXITY_SIMPLE, QUERY_COMPLEXITY_MODERATE, QUERY_COMPLEXITY_COMPLEX]
QUERY_EXP_PHASE_LENGTHS = [50, 100, 250, 500]
QUERY_CSV = "query.csv"

##  CONVERGENCE EXPERIMENT
CONVERGENCE_EXP_CONVERGENCE_MODE = 1
CONVERGENCE_EXP_PHASE_LENGTH = DEFAULT_QUERY_COUNT
CONVERGENCE_EXP_INDEX_USAGES = QUERY_EXP_INDEX_USAGES
CONVERGENCE_EXP_WRITE_RATIOS = QUERY_EXP_WRITE_RATIOS
CONVERGENCE_EXP_QUERY_COMPLEXITYS = QUERY_EXP_QUERY_COMPLEXITYS
CONVERGENCE_EXP_CONVERGENCE_QUERY_THRESHOLDS = [50, 100]
CONVERGENCE_CSV = "convergence.csv"

##  TIME SERIES EXPERIMENT
TIME_SERIES_EXP_PHASE_LENGTHS = [50, 100]
TIME_SERIES_EXP_INDEX_USAGES = [INDEX_USAGE_AGGRESSIVE, INDEX_USAGE_BALANCED, INDEX_USAGE_CONSERVATIVE]

TIME_SERIES_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY, WRITE_RATIO_READ_HEAVY]
TIME_SERIES_EXP_QUERY_COMPLEXITYS = [QUERY_COMPLEXITY_SIMPLE, QUERY_COMPLEXITY_MODERATE]
TIME_SERIES_LATENCY_MODE = 1
TIME_SERIES_INDEX_MODE = 2
TIME_SERIES_PLOT_MODES = [TIME_SERIES_LATENCY_MODE, TIME_SERIES_INDEX_MODE]
TIME_SERIES_LATENCY_CSV = "time_series_latency.csv"
TIME_SERIES_INDEX_CSV = "time_series_index.csv"

##  VARIABILITY EXPERIMENT
VARIABILITY_EXP_PHASE_LENGTH = 50
VARIABILITY_EXP_INDEX_USAGES = QUERY_EXP_INDEX_USAGES
VARIABILITY_EXP_WRITE_RATIO = WRITE_RATIO_READ_ONLY
VARIABILITY_EXP_QUERY_COMPLEXITYS = QUERY_EXP_QUERY_COMPLEXITYS
VARIABILITY_EXP_VARIABILITY_THRESHOLDS = [5, 10, 25]
VARIABILITY_CSV = "variability.csv"

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

def create_legend_index_usage():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    LEGEND_VALUES = INDEX_USAGE_STRINGS.values()

    figlegend = pylab.figure(figsize=(9, 0.5))
    idx = 0
    lines = [None] * len(LEGEND_VALUES)

    LEGEND_VALUES_UPPER_CASE = [x.upper() for x in LEGEND_VALUES]

    for group in xrange(len(LEGEND_VALUES)):
        data = [1]
        x_values = [1]
        lines[idx], = ax1.plot(x_values, data, color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))
        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LEGEND_VALUES_UPPER_CASE, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1, handlelength=4)

    figlegend.savefig('legend_index_usage.pdf')

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
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Phase Lengths", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_convergence_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in CONVERGENCE_EXP_CONVERGENCE_QUERY_THRESHOLDS]
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
    ax1.set_ylabel("Convergence time (ms)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xticks(ind + 0.5)
    ax1.set_xlabel("Convergence query threshold", fontproperties=LABEL_FP)
    ax1.set_xticklabels(x_values)

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig


def create_time_series_line_chart(datasets, plot_mode):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = [str(i) for i in range(1, DEFAULT_QUERY_COUNT + 1)]
    N = len(x_values)
    ind = np.arange(N)

    TIME_SERIES_OPT_LINE_WIDTH = 3.0
    TIME_SERIES_OPT_MARKER_SIZE = 5.0
    TIME_SERIES_OPT_MARKER_FREQUENCY = DEFAULT_QUERY_COUNT/10

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
    major_ticks = np.arange(0, DEFAULT_QUERY_COUNT + 1, TIME_SERIES_OPT_MARKER_FREQUENCY)
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

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

###################################################################################
# PLOT HELPERS
###################################################################################

# QUERY -- PLOT
def query_plot():

    for query_complexity in QUERY_EXP_QUERY_COMPLEXITYS:
        for write_ratio in QUERY_EXP_WRITE_RATIOS:

            datasets = []
            for index_usage in QUERY_EXP_INDEX_USAGES:
                # Get result file
                result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
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
        for write_ratio in CONVERGENCE_EXP_WRITE_RATIOS:

            datasets = []
            for index_usage in CONVERGENCE_EXP_INDEX_USAGES:
                # Get result file
                result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                   WRITE_RATIO_STRINGS[write_ratio],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(CONVERGENCE_DIR, result_dir_list, CONVERGENCE_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_convergence_line_chart(datasets)

            file_name = "convergence" + "-" + \
                        QUERY_COMPLEXITY_STRINGS[query_complexity] + "-" + \
                        WRITE_RATIO_STRINGS[write_ratio] + ".pdf"

            saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

# TIME SERIES -- PLOT
def time_series_plot():

    num_graphs = len(TIME_SERIES_EXP_QUERY_COMPLEXITYS)

    for graph_itr in range(0, num_graphs):

        # Pick parameters for time series graph set
        query_complexity = TIME_SERIES_EXP_QUERY_COMPLEXITYS[graph_itr]
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
                for index_usage in TIME_SERIES_EXP_INDEX_USAGES:

                        # Get result file
                        result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
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

                saveGraph(fig, file_name, width=OPT_GRAPH_WIDTH * 2.0, height=OPT_GRAPH_HEIGHT/1.5)

# VARIABILITY -- PLOT
def variability_plot():

    for query_complexity in VARIABILITY_EXP_QUERY_COMPLEXITYS:

            datasets = []
            for index_usage in VARIABILITY_EXP_INDEX_USAGES:

                # Get result file
                result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                   QUERY_COMPLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(VARIABILITY_DIR, result_dir_list, VARIABILITY_CSV)

                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_variability_line_chart(datasets)

            file_name = "variability" + "-" + \
                        QUERY_COMPLEXITY_STRINGS[query_complexity] + ".pdf"

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
    index_usage=DEFAULT_INDEX_USAGE,
    query_complexity=DEFAULT_QUERY_COMLEXITY,
    scale_factor=DEFAULT_SCALE_FACTOR,
    column_count=DEFAULT_COLUMN_COUNT,
    write_ratio=DEFAULT_WRITE_RATIO,
    tuples_per_tg=DEFAULT_TUPLES_PER_TG,
    phase_length=DEFAULT_PHASE_LENGTH,
    query_count=DEFAULT_QUERY_COUNT,
    selectivity=DEFAULT_SELECTIVITY,
    projectivity=DEFAULT_PROJECTIVITY,
    verbosity=DEFAULT_VERBOSITY,
    convergence_mode=DEFAULT_CONVERGENCE_MODE,
    convergence_query_threshold=DEFAULT_CONVERGENCE_QUERY_THRESHOLD,
    variability_threshold=DEFAULT_VARIABILITY_THRESHOLD):

    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)
    subprocess.call([program,
                     "-v", str(verbosity),
                     "-f", str(index_usage),
                     "-c", str(query_complexity),
                     "-k", str(scale_factor),
                     "-a", str(column_count),
                     "-w", str(write_ratio),
                     "-g", str(tuples_per_tg),
                     "-t", str(phase_length),
                     "-q", str(query_count),
                     "-s", str(selectivity),
                     "-p", str(projectivity),
                     "-o", str(convergence_mode),
                     "-b", str(convergence_query_threshold),
                     "-d", str(variability_threshold)])

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
def query_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(QUERY_DIR)

    for query_complexity in QUERY_EXP_QUERY_COMPLEXITYS:
        for write_ratio in QUERY_EXP_WRITE_RATIOS:
            for index_usage in QUERY_EXP_INDEX_USAGES:
                for phase_length in QUERY_EXP_PHASE_LENGTHS:

                    # Get result file
                    result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(QUERY_DIR, result_dir_list, QUERY_CSV)

                    # Run experiment
                    run_experiment(phase_length=phase_length,
                                   index_usage=index_usage,
                                   write_ratio=write_ratio,
                                   query_complexity=query_complexity)

                    # Collect stat
                    collect_aggregate_stat(phase_length, result_file)

# CONVERGENCE -- EVAL
def convergence_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(CONVERGENCE_DIR)

    for query_complexity in CONVERGENCE_EXP_QUERY_COMPLEXITYS:
        for write_ratio in CONVERGENCE_EXP_WRITE_RATIOS:
            for index_usage in CONVERGENCE_EXP_INDEX_USAGES:
                for convergence_query_threshold in CONVERGENCE_EXP_CONVERGENCE_QUERY_THRESHOLDS :

                    # Get result file
                    result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(CONVERGENCE_DIR, result_dir_list, CONVERGENCE_CSV)

                    # Run experiment (only one phase)
                    run_experiment(phase_length=CONVERGENCE_EXP_PHASE_LENGTH,
                                   index_usage=index_usage,
                                   write_ratio=write_ratio,
                                   query_complexity=query_complexity,
                                   convergence_mode=CONVERGENCE_EXP_CONVERGENCE_MODE,
                                   convergence_query_threshold=convergence_query_threshold)

                    # Collect stat
                    collect_aggregate_stat(convergence_query_threshold, result_file)

# TIME SERIES -- EVAL
def time_series_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(TIME_SERIES_DIR)

    num_graphs = len(TIME_SERIES_EXP_QUERY_COMPLEXITYS)

    for graph_itr in range(0, num_graphs):

        # Pick parameters for time series graph set
        query_complexity = TIME_SERIES_EXP_QUERY_COMPLEXITYS[graph_itr]
        write_ratio = TIME_SERIES_EXP_WRITE_RATIOS[graph_itr]

        for phase_length in TIME_SERIES_EXP_PHASE_LENGTHS:
            for index_usage in TIME_SERIES_EXP_INDEX_USAGES:

                    # Get result file
                    result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
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
                                   index_usage=index_usage,
                                   write_ratio=write_ratio,
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

    for query_complexity in VARIABILITY_EXP_QUERY_COMPLEXITYS:
            for index_usage in VARIABILITY_EXP_INDEX_USAGES:
                for variability_threshold in VARIABILITY_EXP_VARIABILITY_THRESHOLDS:

                    # Get result file
                    result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                       QUERY_COMPLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(VARIABILITY_DIR, result_dir_list, VARIABILITY_CSV)

                    # Run experiment
                    run_experiment(phase_length=VARIABILITY_EXP_PHASE_LENGTH,
                                   index_usage=index_usage,
                                   write_ratio=VARIABILITY_EXP_WRITE_RATIO,
                                   query_complexity=query_complexity,
                                   variability_threshold=variability_threshold)

                    # Collect stat
                    collect_aggregate_stat(variability_threshold, result_file)

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Incremental Experiments')

    parser.add_argument("-a", "--query_eval", help="eval query", action='store_true')
    parser.add_argument("-b", "--convergence_eval", help="eval convergence", action='store_true')
    parser.add_argument("-c", "--time_series_eval", help="eval time series", action='store_true')
    parser.add_argument("-d", "--variability_eval", help="eval variability", action='store_true')

    parser.add_argument("-m", "--query_plot", help="plot query", action='store_true')
    parser.add_argument("-n", "--convergence_plot", help="plot convergence", action='store_true')
    parser.add_argument("-o", "--time_series_plot", help="plot time series", action='store_true')
    parser.add_argument("-p", "--variability_plot", help="plot variability", action='store_true')

    args = parser.parse_args()

    ## EVAL

    if args.query_eval:
        query_eval()

    if args.convergence_eval:
        convergence_eval()

    if args.time_series_eval:
        time_series_eval()

    if args.variability_eval:
        variability_eval()

    ## PLOT

    if args.query_plot:
        query_plot()

    if args.convergence_plot:
        convergence_plot()

    if args.time_series_plot:
        time_series_plot()

    if args.variability_plot:
        variability_plot()

    #create_legend_index_usage()


