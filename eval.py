#!/usr/bin/env python

###################################################################################
# INCREMENTAL EXPERIMENTS
###################################################################################

from __future__ import print_function

import argparse
import csv
import logging
import math
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LinearLocator
import os
import pylab
import subprocess

import matplotlib.pyplot as plot
import numpy as np

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

LAYOUTS = ("row", "column", "hybrid")
OPERATORS = ("direct", "aggregate")

SCALE_FACTOR = 10.0

SELECTIVITY = (0.2, 0.4, 0.6, 0.8, 1.0)
PROJECTIVITY = (0.01, 0.1, 0.5)

COLUMN_COUNTS = (50, 500)
WRITE_RATIOS = (0, 1)
TUPLES_PER_TILEGROUP = (100, 1000, 10000, 100000)
NUM_GROUPS = 5

TRANSACTION_COUNT = 3

NUM_ADAPT_TESTS = 4
REPEAT_ADAPT_TEST = 25
ADAPT_QUERY_COUNT = NUM_ADAPT_TESTS * REPEAT_ADAPT_TEST

ADAPT_EXPERIMENT = 1

###################################################################################
# UTILS
###################################################################################

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def loadDataFile(n_rows, n_cols, path):
    file = open(path, "r")
    reader = csv.reader(file)

    data = [[0 for x in xrange(n_cols)] for y in xrange(n_rows)]

    row_num = 0
    for row in reader:
        column_num = 0
        for col in row:
            data[row_num][column_num] = float(col)
            column_num += 1
        row_num += 1

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

    # GROUP
    for group_index, group in enumerate(LAYOUTS):
        group_data = []

        # LINE
        for line_index, line in enumerate(x_values):
            group_data.append(datasets[group_index][line_index][1])

        LOG.info("%s group_data = %s ", group, str(group_data))

        ax1.plot(x_values, group_data, color=OPT_LINE_COLORS[idx], linewidth=ADAPT_OPT_LINE_WIDTH,
                 marker=OPT_MARKERS[idx], markersize=ADAPT_OPT_MARKER_SIZE,
                 markevery=ADAPT_OPT_MARKER_FREQUENCY, label=str(group))

        idx = idx + 1

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

    #for major_tick in major_ticks[1:-1]:
    #    ax1.axvline(major_tick, color='0.5', linestyle='dashed', linewidth=ADAPT_OPT_LINE_WIDTH)

    # LABELS
    y_mark = 0.9
    x_mark_count = 1.0/NUM_ADAPT_TESTS
    x_mark_offset = x_mark_count/2 - x_mark_count/4
    x_marks = np.arange(0, 1, x_mark_count)

    ADAPT_LABELS = (["Scan", "Insert", "Scan", "Insert"])

    for idx, x_mark in enumerate(x_marks):
            ax1.text(x_mark + x_mark_offset,
                     y_mark,
                     ADAPT_LABELS[idx],
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
    #ADAPT_SEED = 0
    #random.seed(ADAPT_SEED)
    datasets = []

    for layout in LAYOUTS:
        data_file = ADAPT_DIR + "/" + str(ADAPT_COLUMN_COUNT) + "/" + layout + "/" + "adapt.csv"

        dataset = loadDataFile(ADAPT_QUERY_COUNT, 2, data_file)
        #random.shuffle(dataset)
        datasets.append(dataset)

    fig = create_adapt_line_chart(datasets)

    fileName = "adapt.pdf"

    saveGraph(fig, fileName, width= OPT_GRAPH_WIDTH * 3, height=OPT_GRAPH_HEIGHT/1.5)

###################################################################################
# EVAL HELPERS
###################################################################################

# CLEAN UP RESULT DIR
def clean_up_dir(result_directory):

    subprocess.call(['rm', '-rf', result_directory])
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

# RUN EXPERIMENT
def run_experiment(program,
                   scale_factor,
                   transaction_count,
                   experiment_type):

    # cleanup
    subprocess.call(["rm -f " + OUTPUT_FILE], shell=True)

    subprocess.call([program,
                     "-e", str(experiment_type),
                     "-k", str(scale_factor),
                     "-t", str(transaction_count)])


# COLLECT STATS
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
        elif(operator == "2"):
            operator = "aggregate"
        elif(operator == "3"):
            operator = "arithmetic"
        elif(operator == "4"):
            operator = "join"

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

# COLLECT STATS
def collect_ycsb_stats(result_dir,
                       result_file_name):

    fp = open(OUTPUT_FILE)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        data = line.split()

        # Collect info
        layout = data[0]
        operator = data[1]
        column_count = data[2]
        stat = data[3]

        if(layout == "0"):
            layout = "row"
        elif(layout == "1"):
            layout = "column"
        elif(layout == "2"):
            layout = "hybrid"

        result_directory = result_dir + "/" + layout + "/" + column_count

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        file_name = result_directory + "/" + result_file_name

        result_file = open(file_name, "a")
        result_file.write(str(operator) + " , " + str(stat) + "\n")
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

###################################################################################
# MAIN
###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Tilegroup Experiments')

    parser.add_argument("-a", "--adapt", help='eval adapt', action='store_true')
    
    parser.add_argument("-m", "--adapt_plot", help='plot adapt', action='store_true')

    args = parser.parse_args()

    ## EVAL

    if args.adapt:
        adapt_eval()

    ## PLOT

    if args.adapt_plot:
        adapt_plot()

    #create_legend()


