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

ADAPT_DIR = BASE_DIR + "/results/adapt/"
QUERY_DIR = BASE_DIR + "/results/query/"

LAYOUTS = ("hybrid")
OPERATORS = ("direct", "aggregate")

SCALE_FACTOR = 100.0

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5)

COLUMN_COUNTS = (50, 50)
WRITE_RATIOS = (0, 1)
TUPLES_PER_TILEGROUP = (100, 1000, 10000, 100000)
NUM_GROUPS = 5

TRANSACTION_COUNT = 3

NUM_ADAPT_TESTS = 20
REPEAT_ADAPT_TEST = 300
ADAPT_QUERY_COUNT = NUM_ADAPT_TESTS * REPEAT_ADAPT_TEST

ADAPT_EXPERIMENT = 1
QUERY_EXPERIMENT = 2

## INDEX USAGE TYPES
INDEX_USAGE_TYPE_INC   = 1
INDEX_USAGE_TYPE_FULL  = 2
INDEX_USAGE_TYPE_NEVER = 3

INDEX_USAGE_TYPE_STRINGS = {
    1 : "inc",
    2 : "full",
    3 : "never"
}

## QUERY COMPLEXITY TYPES
QUERY_COMLEXITY_TYPE_SIMPLE   = 1
QUERY_COMLEXITY_TYPE_MODERATE = 2
QUERY_COMLEXITY_TYPE_COMPLEX  = 3

QUERY_COMLEXITY_TYPE_STRINGS = {
    1 : "simpleq",
    2 : "moderateq",
    3 : "complexq"
}

## QUERY EXPERIMENT
QUERY_INDEX_USAGE_TYPES = [INDEX_USAGE_TYPE_INC, INDEX_USAGE_TYPE_FULL, INDEX_USAGE_TYPE_NEVER]
QUERY_WRITE_RATIOS = [0.0]
QUERY_QUERY_COMLEXITY_TYPES = [QUERY_COMLEXITY_TYPE_SIMPLE]

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

def make_query_result_file_name(index_usage_type, write_ratio, query_complexity_type):
    result_file_name = "query"
    result_file_name += "_" + INDEX_USAGE_TYPE_STRINGS[index_usage_type]
    result_file_name += "_" + str(write_ratio)
    result_file_name += "_" + QUERY_COMLEXITY_TYPE_STRINGS[query_complexity_type]
    result_file_name += '.csv'
    return result_file_name

###################################################################################
# PLOT
###################################################################################

def create_bar_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))

    num_items = len(LAYOUTS);
    ind = np.arange(1)
    margin = 0.10
    width = ((1.0 - 2 * margin) / num_items) * 2
    data = [1]

    bars = [None] * (len(LAYOUTS) + 1) * 2

    # TITLE
    idx = 0
    bars[idx] = ax1.bar(ind + margin + ((idx) * width), data, width,
                        color = 'w',
                        linewidth=0)

    idx = 0
    for group in xrange(len(LAYOUTS)):
        bars[idx + 1] = ax1.bar(ind + margin + ((idx + 1) * width), data, width,
                              color=OPT_COLORS[idx],
                              hatch=OPT_PATTERNS[idx * 2],
                              linewidth=BAR_LINEWIDTH)

        idx = idx + 1

    TITLE = "Storage Models : "
    LABELS = [TITLE, "NSM", "DSM", "FSM"]

    # LEGEND
    figlegend.legend(bars, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4,
                     mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0,
                     handleheight=1.5, handlelength=4)

    figlegend.savefig('legend_bar.pdf')

def create_legend():
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)

    figlegend = pylab.figure(figsize=(9, 0.5))
    idx = 0
    lines = [None] * (len(LAYOUTS) + 1)
    data = [1]
    x_values = [1]

    TITLE = "Storage Models : "
    LABELS = [TITLE, "NSM", "DSM", "FSM"]

    lines[idx], = ax1.plot(x_values, data, linewidth = 0)
    idx = 0

    for group in xrange(len(LAYOUTS)):
        lines[idx + 1], = ax1.plot(x_values, data,
                               color=OPT_LINE_COLORS[idx], linewidth=OPT_LINE_WIDTH,
                               marker=OPT_MARKERS[idx], markersize=OPT_MARKER_SIZE, label=str(group))

        idx = idx + 1

    # LEGEND
    figlegend.legend(lines, LABELS, prop=LEGEND_FP,
                     loc=1, ncol=4, mode="expand", shadow=OPT_LEGEND_SHADOW,
                     frameon=False, borderaxespad=0.0, handlelength=4)

    figlegend.savefig('legend.pdf')

def create_query_line_chart(datasets):
    fig = plot.figure()
    axe = fig.add_subplot(111)

    QUERY_OPT_LINE_WIDTH = 3.0
    QUERY_OPT_MARKER_SIZE = 5.0
    QUERY_OPT_MARKER_FREQUENCY = 10

    # Get X-axis values
    x_values = list(xrange(1, len(datasets[0])+1))

    miny = 9999
    maxy = 0
    for idx, dataset in enumerate(datasets):
        y_values = [value[1] for value in dataset]
        miny = min(min(y_values), miny)
        maxy = max(max(y_values), maxy)
        axe.plot(x_values, y_values, color=OPT_LINE_COLORS[idx],
                 linewidth=QUERY_OPT_LINE_WIDTH, marker=OPT_MARKERS[idx],
                 markersize=QUERY_OPT_MARKER_SIZE, markevery=QUERY_OPT_MARKER_FREQUENCY)

    axes = axe.get_axes()
    makeGrid(axes)

    # Y-AXIS
    axe.set_ylim([int(miny * 0.9), int(maxy * 1.1)])
    axe.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    axe.minorticks_off()
    axe.set_ylabel("Execution time (s)", fontproperties=LABEL_FP)
    #axe.set_yscale('log', basey=10)

    # X-AXIS
    axe.set_xlabel("Query Sequence", fontproperties=LABEL_FP)

    # LABELS
    axe.legend(["Inc", "Full", "Never"])

    for label in axe.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in axe.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return fig

def create_adapt_line_chart(datasets):
    fig = plot.figure()
    ax1 = fig.add_subplot(111)

    # X-AXIS
    x_values = list(xrange(1, ADAPT_QUERY_COUNT + 1))
    N = len(x_values)
    x_labels = x_values

    num_items = len(LAYOUTS);
    ind = np.arange(N)
    idx = 0

    ADAPT_OPT_LINE_WIDTH = 3.0
    ADAPT_OPT_MARKER_SIZE = 5.0
    ADAPT_OPT_MARKER_FREQUENCY = 10

    group_data = []
    # LINE
    for line_index, line in enumerate(x_values):
        group_data.append(datasets[0][line_index][1])

    LOG.info("data = %s ", str(group_data))

    ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=ADAPT_OPT_LINE_WIDTH,
             marker=OPT_MARKERS[idx], markersize=ADAPT_OPT_MARKER_SIZE,
             markevery=ADAPT_OPT_MARKER_FREQUENCY)


    # GRID
    axes = ax1.get_axes()
    makeGrid(ax1)

    # Y-AXIS
    ax1.yaxis.set_major_locator(LinearLocator(YAXIS_TICKS))
    ax1.minorticks_off()
    ax1.set_ylabel("Execution time (s)", fontproperties=LABEL_FP)
    #ax1.set_yscale('log', basey=10)

    # X-AXIS
    ax1.set_xlabel("Query Sequence", fontproperties=LABEL_FP)
    major_ticks = np.arange(0, ADAPT_QUERY_COUNT + 1, REPEAT_ADAPT_TEST)
    ax1.set_xticks(major_ticks)

    # LABELS
    y_mark = 0.9
    x_mark_count = 1.0/NUM_ADAPT_TESTS
    x_mark_offset = x_mark_count/2 - x_mark_count/4
    x_marks = np.arange(0, 1, x_mark_count)

    ADAPT_LABELS = (["Scan", "Insert"])

    for idx, x_mark in enumerate(x_marks):
            ax1.text(x_mark + x_mark_offset,
                     y_mark,
                     ADAPT_LABELS[idx%2],
                     transform=ax1.transAxes,
                     bbox=dict(facecolor='skyblue', alpha=0.5))

    for label in ax1.get_yticklabels() :
        label.set_fontproperties(TICK_FP)
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(TICK_FP)

    return (fig)

###################################################################################
# PLOT HELPERS
###################################################################################

# ADAPT -- PLOT
def adapt_plot():

    ADAPT_COLUMN_COUNT = COLUMN_COUNTS[1]
    datasets = []
    layout = "hybrid"

    data_file = ADAPT_DIR + "/" + str(ADAPT_COLUMN_COUNT) + "/" + str(layout) + "/" + "adapt.csv"

    dataset = loadDataFile(data_file)
    #random.shuffle(dataset)
    datasets.append(dataset)

    fig = create_adapt_line_chart(datasets)

    fileName = "adapt.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH * 3, height=OPT_GRAPH_HEIGHT/1.5)

# QUERY -- PLOT
def query_plot():
    
    for query_complexity_type in QUERY_QUERY_COMLEXITY_TYPES:
        for write_ratio in QUERY_WRITE_RATIOS:
            datasets = []

            for index_usage_type in QUERY_INDEX_USAGE_TYPES:
                result_file_name = make_query_result_file_name(index_usage_type, write_ratio, query_complexity_type)
                result_file_path = QUERY_DIR + '/' + result_file_name
            
                if os.path.exists(result_file_path):
                    dataset = loadDataFile(QUERY_DIR + '/' + result_file_name)
                    datasets.append(dataset)

            fig = create_query_line_chart(datasets)

            fileName = "query" + "_" + INDEX_USAGE_TYPE_STRINGS[query_complexity_type] + \
                       "_" + str(write_ratio) + ".pdf"

            saveGraph(fig, fileName, width=OPT_GRAPH_WIDTH, height=OPT_GRAPH_HEIGHT)

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
    index_usage_type=INDEX_USAGE_TYPE_INC,
    query_complexity_type=QUERY_COMLEXITY_TYPE_SIMPLE,
    scale_factor = 100.0,
    column_count = 10,
    write_ratio = 0.0,
    tuples_per_tg = 1000,
    phase_length = 40,
    phase_count = 10,
    selectivity = 0.1,
    projectivity = 0.1):

    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)
    subprocess.call([program,
                     "-f", str(index_usage_type),
                     "-c", str(query_complexity_type),
                     "-k", str(scale_factor),
                     "-a", str(column_count),
                     "-w", str(write_ratio),
                     "-g", str(tuples_per_tg),
                     "-t", str(phase_length),
                     "-q", str(phase_length * phase_count),
                     "-s", str(selectivity),
                     "-p", str(projectivity)])

# COLLECT STATS

# Collect result to a given file
def collect_stats2(result_dir, 
                   result_file_name):
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_f = open(result_dir + "/" + result_file_name, "w")

    itr = 1
    with open(OUTPUT_FILE) as fp:
        for line in fp:
            line = line.strip()
            data = line.split(" ")
            duration = data[-1]
            result_f.write(str(itr) + " , " + str(duration) + "\n")
            itr += 1

    result_f.close()

def collect_stats(result_dir,
                  result_file_name,
                  category):

    fp = open(OUTPUT_FILE)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        data = line.split()

        # Collect info
        layout = data[0]
        operator = data[1]

        selectivity = data[2]
        projectivity = data[3]
        txn_itr = data[4]

        write_ratio = data[5]
        scale_factor = data[6]

        column_count = data[7]
        tuples_per_tg = data[8]

        stat = data[9]

        if(layout == "1"):
            layout = "row"
        elif(layout == "2"):
            layout = "column"
        elif(layout == "3"):
            layout = "hybrid"

        if(operator == "1"):
            operator = "direct"

        # MAKE RESULTS FILE DIR
        if category == ADAPT_EXPERIMENT:
            result_directory = result_dir + "/" + column_count + "/" + layout

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")

        # WRITE OUT STATS
        if category == ADAPT_EXPERIMENT:
            result_file.write(str(txn_itr) + " , " + str(stat) + "\n")

        result_file.close()

###################################################################################
# EVAL
###################################################################################

# ADAPT -- EVAL
def adapt_eval():

    # CLEAN UP RESULT DIR
    clean_up_dir(ADAPT_DIR)

    # RUN EXPERIMENT
    run_experiment(SDBENCH, SCALE_FACTOR,
                   TRANSACTION_COUNT, ADAPT_EXPERIMENT)

    # COLLECT STATS
    collect_stats(ADAPT_DIR, "adapt.csv", ADAPT_EXPERIMENT)

def query_eval():
    
    for index_usage_type in QUERY_INDEX_USAGE_TYPES:
        for write_ratio in QUERY_WRITE_RATIOS:
            for query_complexity_type in QUERY_QUERY_COMLEXITY_TYPES:

                result_file_name = make_query_result_file_name(index_usage_type, write_ratio, query_complexity_type)

                run_experiment(phase_count=10, phase_length=10,
                               index_usage_type=index_usage_type, write_ratio=write_ratio,
                               query_complexity_type=query_complexity_type)

                collect_stats2(QUERY_DIR, result_file_name)

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Incremental Experiments')

    parser.add_argument("-a", "--adapt", help='eval adapt', action='store_true')
    parser.add_argument("-b", "--query", help="eval query", action='store_true')

    parser.add_argument("-m", "--adapt_plot", help='plot adapt', action='store_true')
    parser.add_argument("-n", "--query_plot", help="plot query", action='store_true')

    args = parser.parse_args()

    ## EVAL

    if args.adapt:
        adapt_eval()

    if args.query:
        query_eval()

    ## PLOT

    if args.adapt_plot:
        adapt_plot()

    if args.query_plot:
        query_plot()

    #create_legend()


