# =============================================================================
# draw_graph.py — Matplotlib helpers for plotting classifier results
# =============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Generate figures without Type-3 fonts (needed for IEEE/ACM submissions)
matplotlib.rcParams['ps.useafm']          = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex']        = False
plt.rcParams['axes.unicode_minus']        = False

LABEL_SIZE  = 18
LEGEND_SIZE = 18
TICK_SIZE   = 16


# ── Data containers ───────────────────────────────────────────────────────────

class DataArray:
    """
    Container for multi-series plot data.
    
    Supports up to 4 Y-series with flexible axis assignment:
        - axis_num=1: all series on single Y-axis  
        - axis_num=2: y1/y2 on left axis, y3/y4 on right axis
    
    Pass 0 for any unused series.
    """
    def __init__(self, x, y1, y2=0, y3=0, y4=0):
        self.x  = np.array(x)
        self.y1 = np.array(y1)
        self.y2 = np.array(y2) if not isinstance(y2, int) else y2
        self.y3 = np.array(y3) if not isinstance(y3, int) else y3
        self.y4 = np.array(y4) if not isinstance(y4, int) else y4


class Context:
    """Chart configuration: axis labels, file path, limits, ticks, legend labels."""
    def __init__(self, xLabel, y1Label, y2Label='', file2save='',
                 y1Lim=0, y2Lim=0,
                 label1='', label2='', label3='', label4='',
                 xTicks=0, y1Ticks=0, y2Ticks=0):
        self.xLabel    = xLabel
        self.y1Label   = y1Label
        self.y2Label   = y2Label
        self.file2save = file2save
        self.y1Lim     = y1Lim
        self.y2Lim     = y2Lim
        self.label1    = label1
        self.label2    = label2
        self.label3    = label3
        self.label4    = label4
        self.xTicks    = xTicks
        self.y1Ticks   = y1Ticks
        self.y2Ticks   = y2Ticks


# ── Drawing functions ─────────────────────────────────────────────────────────

def draw_results(data, axis_num, context):
    """Draw line chart: axis_num=1 for single Y-axis, 2 for dual Y-axis; save to disk."""
    if axis_num == 1:
        _draw_single_axis(data, context)
    elif axis_num == 2:
        _draw_dual_axis(data, context)
    else:
        raise ValueError(f'axis_num must be 1 or 2, got {axis_num}')


def _draw_single_axis(data, ctx):
    fig, ax1 = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    lines = ax1.plot(data.x, data.y1, marker='+', markersize=12, color='k', label=ctx.label1)

    if not isinstance(data.y2, int):
        lines += ax1.plot(data.x, data.y2, marker='*', markersize=12, color='r', label=ctx.label2)
    if not isinstance(data.y3, int):
        lines += ax1.plot(data.x, data.y3, marker='o', markersize=12, color='g', label=ctx.label3)
    if not isinstance(data.y4, int):
        lines += ax1.plot(data.x, data.y4, marker='^', markersize=12, color='b', label=ctx.label4)

    _apply_axis_style(ax1, ctx.xLabel, ctx.y1Label, ctx.xTicks, ctx.y1Ticks, ctx.y1Lim)
    legend = ax1.legend(lines, [l.get_label() for l in lines], loc=0, fontsize='x-large')
    legend.get_title().set_fontsize(LEGEND_SIZE)

    plt.savefig(ctx.file2save)
    plt.close('all')
    print(f'Figure saved: {ctx.file2save}')


def _draw_dual_axis(data, ctx):
    fig, ax1 = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    lines  = ax1.plot(data.x, data.y1, marker='*', markersize=12, color='g', label=ctx.label1)
    if not isinstance(data.y2, int):
        lines += ax1.plot(data.x, data.y2, marker='v', markersize=12, color='k', label=ctx.label2)

    ax2    = ax1.twinx()
    if not isinstance(data.y3, int):
        lines += ax2.plot(data.x, data.y3, marker='+', markersize=12, color='b', label=ctx.label3)
    if not isinstance(data.y4, int):
        lines += ax2.plot(data.x, data.y4, marker='o', markersize=12, color='r', label=ctx.label4)

    _apply_axis_style(ax1, ctx.xLabel, ctx.y1Label, ctx.xTicks, ctx.y1Ticks, ctx.y1Lim)
    _apply_axis_style(ax2, None,       ctx.y2Label, None,       ctx.y2Ticks, ctx.y2Lim)

    legend = ax1.legend(lines, [l.get_label() for l in lines], loc=0, fontsize='x-large')
    legend.get_title().set_fontsize(LEGEND_SIZE)

    plt.savefig(ctx.file2save)
    plt.close('all')
    print(f'Figure saved: {ctx.file2save}')


def _apply_axis_style(ax, xlabel, ylabel, xticks, yticks, ylim):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    if xticks is not None and not isinstance(xticks, int) and len(xticks) > 0:
        ax.set_xticks(xticks)
    if yticks is not None and not isinstance(yticks, int) and len(yticks) > 0:
        ax.set_yticks(yticks)
    if ylim != 0:
        ax.set_ylim(ylim[0], ylim[1])
