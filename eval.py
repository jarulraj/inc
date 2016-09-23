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

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 3.0
OPT_MARKER_SIZE = 6.0

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

# SET FONT

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TINY_FONT_SIZE = 8
LEGEND_FONT_SIZE = 16

SMALL_LABEL_FONT_SIZE = 10
SMALL_LEGEND_FONT_SIZE = 10

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

# SET FONT

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TINY_FONT_SIZE = 8
LEGEND_FONT_SIZE = 16

SMALL_LABEL_FONT_SIZE = 10
SMALL_LEGEND_FONT_SIZE = 10

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

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
INDEX_USAGE_INCREMENTAL = 1
INDEX_USAGE_FULL  = 2
INDEX_USAGE_NEVER = 3

INDEX_USAGE_STRINGS = {
    1 : "incremental",
    2 : "full",
    3 : "never"
}

## QUERY COMPLEXITY TYPES
QUERY_COMLEXITY_SIMPLE   = 1
QUERY_COMLEXITY_MODERATE = 2
QUERY_COMLEXITY_COMPLEX  = 3

QUERY_COMLEXITY_STRINGS = {
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
    0.0 : "read_only",
    0.1 : "read_heavy",
    0.5 : "balanced",
    0.9 : "write_heavy"
}

DEFAULT_INDEX_USAGE = INDEX_USAGE_INCREMENTAL
DEFAULT_QUERY_COMLEXITY = QUERY_COMLEXITY_SIMPLE
DEFAULT_SCALE_FACTOR = 100.0
DEFAULT_COLUMN_COUNT = 10
DEFAULT_WRITE_RATIO = WRITE_RATIO_READ_ONLY
DEFAULT_TUPLES_PER_TG = 1000
DEFAULT_PHASE_LENGTH = 40
DEFAULT_PHASE_COUNT = 10
DEFAULT_SELECTIVITY = 0.01
DEFAULT_PROJECTIVITY = 0.1

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5)

## EXPERIMENTS
QUERY_EXPERIMENT = 1

## DIRS
QUERY_DIR = BASE_DIR + "/results/query"

## QUERY EXPERIMENT
QUERY_EXP_INDEX_USAGES = [INDEX_USAGE_INCREMENTAL, INDEX_USAGE_FULL, INDEX_USAGE_NEVER]
QUERY_EXP_WRITE_RATIOS = [WRITE_RATIO_READ_ONLY, WRITE_RATIO_READ_HEAVY]
QUERY_EXP_QUERY_COMPLEXITYS = [QUERY_COMLEXITY_SIMPLE]
QUERY_EXP_PHASE_LENGTHS = [10, 20]

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
        
    pprint.pprint(result_file_name)
    return result_file_name

###################################################################################
# PLOT
###################################################################################

def create_query_line_chart(datasets):
    fig = plot.figure()
    axe = fig.add_subplot(111)

    # Get X-axis values
    x_values = list(xrange(1, len(datasets[0])+1))

    for idx, dataset in enumerate(datasets):
        y_values = [value[1] for value in dataset]
        axe.plot(x_values, y_values, 
                 color=OPT_LINE_COLORS[idx],
                 linewidth= OPT_LINE_WIDTH, 
                 marker=OPT_MARKERS[idx],
                 markersize= OPT_MARKER_SIZE)

    axes = axe.get_axes()
    makeGrid(axes)

    # Y-AXIS
    axe.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    axe.minorticks_off()
    axe.set_ylabel("Execution time (ms)", fontproperties=LABEL_FP)
    #axe.set_yscale('log', basey=10)

    # X-AXIS
    axe.set_xlabel("Phase Lengths", fontproperties=LABEL_FP)

    # LABELS
    axe.legend(["Incremental", "Full", "Never"])

    for label in axe.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in axe.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

###################################################################################
# PLOT HELPERS
###################################################################################

# QUERY -- PLOT
def query_plot():
    
    for index_usage in QUERY_EXP_INDEX_USAGES:
        for write_ratio in QUERY_EXP_WRITE_RATIOS:
    
            datasets = []
            for query_complexity in QUERY_EXP_QUERY_COMPLEXITYS:
                # Get result file
                result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                   WRITE_RATIO_STRINGS[write_ratio],
                                   QUERY_COMLEXITY_STRINGS[query_complexity]]
                result_file = get_result_file(QUERY_DIR, result_dir_list, "query.csv")
        
                dataset = loadDataFile(result_file)
                datasets.append(dataset)

            fig = create_query_line_chart(datasets)

            file_name = "query" + "-" + \
                        INDEX_USAGE_STRINGS[query_complexity] + "-" + \
                        WRITE_RATIO_STRINGS[write_ratio] + ".pdf"

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
    phase_count=DEFAULT_PHASE_COUNT,
    selectivity=DEFAULT_SELECTIVITY,
    projectivity=DEFAULT_PROJECTIVITY):

    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)
    subprocess.call([program,
                     "-f", str(index_usage),
                     "-c", str(query_complexity),
                     "-k", str(scale_factor),
                     "-a", str(column_count),
                     "-w", str(write_ratio),
                     "-g", str(tuples_per_tg),
                     "-t", str(phase_length),
                     "-q", str(phase_length * phase_count),
                     "-s", str(selectivity),
                     "-p", str(projectivity)])

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

    pprint.pprint(stat)
    result_file.write(str(independent_variable) + " , " + str(stat) + "\n")
    result_file.close()

###################################################################################
# EVAL
###################################################################################

# QUERY -- EVAL
def query_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(QUERY_DIR)
    
    QUERY_EXP_PHASE_COUNT = 10
    
    for index_usage in QUERY_EXP_INDEX_USAGES:
        for write_ratio in QUERY_EXP_WRITE_RATIOS:
            for query_complexity in QUERY_EXP_QUERY_COMPLEXITYS:
                for phase_length in QUERY_EXP_PHASE_LENGTHS:

                    # Get result file
                    result_dir_list = [INDEX_USAGE_STRINGS[index_usage],
                                       WRITE_RATIO_STRINGS[write_ratio],
                                       QUERY_COMLEXITY_STRINGS[query_complexity]]
                    result_file = get_result_file(QUERY_DIR, result_dir_list, "query.csv")
                    
                    # Run experiment
                    run_experiment(phase_count=QUERY_EXP_PHASE_COUNT, 
                                   phase_length=phase_length,
                                   index_usage=index_usage, 
                                   write_ratio=write_ratio,
                                   query_complexity=query_complexity)
    
                    # Collect stat
                    collect_aggregate_stat(phase_length, result_file)

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Incremental Experiments')

    parser.add_argument("-a", "--query_eval", help="eval query", action='store_true')

    parser.add_argument("-m", "--query_plot", help="plot query", action='store_true')

    args = parser.parse_args()

    ## EVAL

    if args.query_eval:
        query_eval()

    ## PLOT

    if args.query_plot:
        query_plot()

    #create_legend()


