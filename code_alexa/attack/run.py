# =============================================================================
# run.py — Main entry point: run all experiments and generate figures
#
# To choose which experiment to run, set the RUN_MODE variable below.
# All paths are set in config.py.
# =============================================================================

import os
import numpy as np

import config
import nFoldCrossValidation
import draw_graph


# ── Choose which experiment to run ───────────────────────────────────────────
# Options: 'round_test' | 'pad_size_test' | 'epoch_test' |
#          'vector_size_test' | 'trade_off_test' | 'mini_time_test' | 'all'
# Use 'all' to run every experiment in sequence

RUN_MODE = 'all'

# ── Helpers ───────────────────────────────────────────────────────────────────
def _run_model(model_name, data_dir, interval, word2vec_file, result_file):

    """Convenience wrapper around run_cross_validation."""
    return nFoldCrossValidation.run_cross_validation(
        data_dir      = data_dir,
        model_name    = model_name,
        n_folds       = config.N_FOLDS,
        interval      = interval,
        result_file   = result_file,
        word2vec_file = word2vec_file,
    )


def _tmp_result_file(label):
    """Return a unique temp result filepath."""
    return os.path.join(config.TEMP_DIR, f'result_{label}.txt')


# ── Experiment: Rounding Parameter Impact ────────────────────────────────────

def round_test():
    """
    Sweep the histogram interval (rounding parameter) for Bayes and VNG++.
    Evaluates BOTH Yahoo and Quora models separately to show consistency (like paper Figures 4-5).
    Plots accuracy and normalized semantic distance vs interval.
    """
    data_dir         = config.DATA_DIR
    bayes_intervals  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    vngpp_intervals  = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # ── Bayes sweep with BOTH Yahoo and Quora models ──────────────────────────
    bayes_acc = []
    bayes_rank_yahoo = []
    bayes_rank_quora = []

    for itv in bayes_intervals:
        # Run with Quora model
        rank_quora, acc = _run_model('Bayes', data_dir, itv,
                                      config.QUORA_E100_V300,
                                      _tmp_result_file(f'bayes_{itv}_quora'))
        bayes_acc.append(acc)
        bayes_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # Run with Yahoo model  
        rank_yahoo, _ = _run_model('Bayes', data_dir, itv,
                                    config.YAHOO_E100_V300,
                                    _tmp_result_file(f'bayes_{itv}_yahoo'))
        bayes_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)
    
    # Debug output
    print(f'\n=== Bayes Results (Rounding Test) ===')
    print(f'Accuracy values: {bayes_acc}')
    print(f'Rank values (Yahoo): {bayes_rank_yahoo}')
    print(f'Rank values (Quora): {bayes_rank_quora}')

    fig_path = os.path.join(config.FIGURES_DIR, 'Bayes_round_test.png')
    
    # Combine both rank lists to find appropriate y2Lim
    all_ranks = bayes_rank_yahoo + bayes_rank_quora
    rank_min, rank_max = min(all_ranks), max(all_ranks)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    y2_lower = max(0, rank_min - rank_buffer)
    y2_upper = rank_max + rank_buffer
    
    ctx = draw_graph.Context(
        xLabel='Rounding Parameter', y1Label='Accuracy',
        y2Label='Normalized Semantic Distance',
        file2save=fig_path,
        y1Lim=[0.1, 0.5], y2Lim=[y2_lower, y2_upper],
        label1='Accuracy',
        label3='Normalized SD (Yahoo)',
        label4='Normalized SD (Quora)',
        xTicks=bayes_intervals,
        y1Ticks=[0.1, 0.2, 0.3, 0.4, 0.5],
        y2Ticks=None,  # Auto-calculate ticks
    )
    draw_graph.draw_results(
        draw_graph.DataArray(bayes_intervals, bayes_acc, 0, bayes_rank_yahoo, bayes_rank_quora),
        axis_num=2, context=ctx
    )

    # ── VNG++ sweep with BOTH Yahoo and Quora models ──────────────────────────
    vng_acc = []
    vng_rank_yahoo = []
    vng_rank_quora = []

    for itv in vngpp_intervals:
        # Run with Quora model
        rank_quora, acc = _run_model('VNGpp', data_dir, itv,
                                      config.QUORA_E100_V300,
                                      _tmp_result_file(f'vng_{itv}_quora'))
        vng_acc.append(acc)
        vng_rank_quora.append(rank_quora if rank_quora is not None else 0.0)

        # Run with Yahoo model
        rank_yahoo, _ = _run_model('VNGpp', data_dir, itv,
                                    config.YAHOO_E100_V300,
                                    _tmp_result_file(f'vng_{itv}_yahoo'))
        vng_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)
    
    # Debug output
    print(f'\n=== VNG++ Results (Rounding Test) ===')
    print(f'Accuracy values: {vng_acc}')
    print(f'Rank values (Yahoo): {vng_rank_yahoo}')
    print(f'Rank values (Quora): {vng_rank_quora}')

    fig_path = os.path.join(config.FIGURES_DIR, 'VNGpp_round_test.png')
    
    # Combine both rank lists to find appropriate y2Lim
    all_ranks = vng_rank_yahoo + vng_rank_quora
    rank_min, rank_max = min(all_ranks), max(all_ranks)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    y2_lower = max(0, rank_min - rank_buffer)
    y2_upper = rank_max + rank_buffer
    
    ctx = draw_graph.Context(
        xLabel='Rounding Parameter', y1Label='Accuracy',
        y2Label='Normalized Semantic Distance',
        file2save=fig_path,
        y1Lim=[0.1, 0.5], y2Lim=[y2_lower, y2_upper],
        label1='Accuracy',
        label3='Normalized SD (Yahoo)',
        label4='Normalized SD (Quora)',
        xTicks=[2000, 4000, 6000, 8000, 10000],
        y1Ticks=[0.1, 0.2, 0.3, 0.4, 0.5],
        y2Ticks=None,  # Auto-calculate ticks
    )
    draw_graph.draw_results(
        draw_graph.DataArray(vngpp_intervals, vng_acc, 0, vng_rank_yahoo, vng_rank_quora),
        axis_num=2, context=ctx
    )


# ── Experiment: Padding Size Impact ──────────────────────────────────────────

def pad_size_test():
    """
    Test classification performance across different fixed padding sizes.
    Evaluates BOTH Yahoo and Quora models to show consistency.
    """
    # Base directory for obfuscated data — subdirectories named like '1000/1000_50_20'
    base_dir    = config.BUFLO_DIR    # directory with BuFLO-obfuscated traffic CSV files
    pad_sizes   = [1000, 1100, 1200, 1300, 1400, 1500]

    ada_acc = []
    ada_rank_yahoo, ada_rank_quora = [], []
    vng_acc = []
    vng_rank_yahoo, vng_rank_quora = [], []

    for ps in pad_sizes:
        sub_path = os.path.join(base_dir, str(ps), f'{ps}_50_20')

        # AdaBoost with Quora
        rank_quora, acc = _run_model('AdaBoost', sub_path, config.VNGPP_INTERVAL,
                                      config.QUORA_E100_V300,
                                      _tmp_result_file(f'pad_ada_quora_{ps}'))
        ada_acc.append(acc)
        ada_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # AdaBoost with Yahoo
        rank_yahoo, _ = _run_model('AdaBoost', sub_path, config.VNGPP_INTERVAL,
                                    config.YAHOO_E100_V300,
                                    _tmp_result_file(f'pad_ada_yahoo_{ps}'))
        ada_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)

        # VNG++ with Quora
        rank_quora, acc = _run_model('VNGpp', sub_path, config.VNGPP_INTERVAL,
                                      config.QUORA_E100_V300,
                                      _tmp_result_file(f'pad_vng_quora_{ps}'))
        vng_acc.append(acc)
        vng_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # VNG++ with Yahoo
        rank_yahoo, _ = _run_model('VNGpp', sub_path, config.VNGPP_INTERVAL,
                                    config.YAHOO_E100_V300,
                                    _tmp_result_file(f'pad_vng_yahoo_{ps}'))
        vng_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)

    print(f'\n=== Padding Size Test Results ===')
    print(f'Accuracy (AdaBoost): {ada_acc}')
    print(f'Accuracy (VNG++): {vng_acc}')
    print(f'Rank (AdaBoost Yahoo): {ada_rank_yahoo}')
    print(f'Rank (AdaBoost Quora): {ada_rank_quora}')
    print(f'Rank (VNG++ Yahoo): {vng_rank_yahoo}')
    print(f'Rank (VNG++ Quora): {vng_rank_quora}')

    fig_path = os.path.join(config.FIGURES_DIR, 'padSizeTest.png')
    ctx = draw_graph.Context(
        xLabel='Fixed Package Size', y1Label='Accuracy',
        y2Label='Normalized Semantic Distance',
        file2save=fig_path,
        y1Lim=[0, 0.2], y2Lim=[30, 60],
        label1='Accuracy AdaBoost',  label2='Accuracy VNG++',
        label3='Norm SD AdaBoost (Yahoo)',   label4='Norm SD AdaBoost (Quora)',
        label5='Norm SD VNG++ (Yahoo)',      label6='Norm SD VNG++ (Quora)',
        xTicks=pad_sizes,
        y1Ticks=[0, 0.1, 0.2],
        y2Ticks=[30, 40, 50, 60],
    )
    draw_graph.draw_results(
        draw_graph.DataArray(pad_sizes, ada_acc, vng_acc, ada_rank_yahoo, ada_rank_quora, vng_rank_yahoo, vng_rank_quora),
        axis_num=2, context=ctx
    )


# ── Experiment: Epoch Count Impact ───────────────────────────────────────────

def epoch_test():
    """
    Test how the number of doc2vec training epochs affects semantic distance.
    Evaluates BOTH Yahoo and Quora models to show consistency (like paper Figures 6-7).
    Uses epochs: 100, 125, 150, 175 with vector_size=300
    """
    data_dir  = config.DATA_DIR
    epochs    = [100, 125, 150, 175]
    quora_files = [
        config.QUORA_E100_V300,
        config.QUORA_E125_V300,
        config.QUORA_E150_V300,
        config.QUORA_E175_V300,
    ]
    yahoo_files = [
        config.YAHOO_E100_V300,
        config.YAHOO_E125_V300,
        config.YAHOO_E150_V300,
        config.YAHOO_E175_V300,
    ]

    nb_rank_yahoo, nb_rank_quora = [], []
    vng_rank_yahoo, vng_rank_quora = [], []

    for e, quora_f, yahoo_f in zip(epochs, quora_files, yahoo_files):
        # Bayes with Quora
        rank_quora, _ = _run_model('Bayes', data_dir, config.BAYES_INTERVAL,
                                    quora_f, _tmp_result_file(f'epoch_nb_quora_{e}'))
        nb_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # Bayes with Yahoo
        rank_yahoo, _ = _run_model('Bayes', data_dir, config.BAYES_INTERVAL,
                                    yahoo_f, _tmp_result_file(f'epoch_nb_yahoo_{e}'))
        nb_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)
        
        # VNG++ with Quora
        rank_quora, _ = _run_model('VNGpp', data_dir, config.VNGPP_INTERVAL,
                                    quora_f, _tmp_result_file(f'epoch_vng_quora_{e}'))
        vng_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # VNG++ with Yahoo
        rank_yahoo, _ = _run_model('VNGpp', data_dir, config.VNGPP_INTERVAL,
                                    yahoo_f, _tmp_result_file(f'epoch_vng_yahoo_{e}'))
        vng_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)
    
    # Debug output
    print(f'\n=== Epoch Test Results (Bayes) ===')
    print(f'Rank values (Yahoo): {nb_rank_yahoo}')
    print(f'Rank values (Quora): {nb_rank_quora}')
    
    print(f'\n=== Epoch Test Results (VNG++) ===')
    print(f'Rank values (Yahoo): {vng_rank_yahoo}')
    print(f'Rank values (Quora): {vng_rank_quora}')

    # Bayes plot
    all_ranks = nb_rank_yahoo + nb_rank_quora
    rank_min, rank_max = min(all_ranks), max(all_ranks)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    y1_lower = max(0, rank_min - rank_buffer)
    y1_upper = rank_max + rank_buffer
    
    ctx = dict(xLabel='Number of Epochs',
               y1Label='Normalized Semantic Distance',
               y1Lim=[y1_lower, y1_upper], xTicks=epochs, y1Ticks=None,
               label1='Norm SD (Yahoo)', label2='Norm SD (Quora)')

    draw_graph.draw_results(
        draw_graph.DataArray(epochs, nb_rank_yahoo, nb_rank_quora),
        axis_num=1,
        context=draw_graph.Context(file2save=os.path.join(config.FIGURES_DIR, 'epochNB.png'),
                                    **ctx)
    )
    
    # VNG++ plot
    all_ranks = vng_rank_yahoo + vng_rank_quora
    rank_min, rank_max = min(all_ranks), max(all_ranks)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    y1_lower = max(0, rank_min - rank_buffer)
    y1_upper = rank_max + rank_buffer
    
    ctx = dict(xLabel='Number of Epochs',
               y1Label='Normalized Semantic Distance',
               y1Lim=[y1_lower, y1_upper], xTicks=epochs, y1Ticks=None,
               label1='Norm SD (Yahoo)', label2='Norm SD (Quora)')
    
    draw_graph.draw_results(
        draw_graph.DataArray(epochs, vng_rank_yahoo, vng_rank_quora),
        axis_num=1,
        context=draw_graph.Context(file2save=os.path.join(config.FIGURES_DIR, 'epochVNG.png'),
                                    **ctx)
    )


# ── Experiment: Vector Size Impact ───────────────────────────────────────────

def vector_size_test():
    """
    Test how the doc2vec vector dimensionality affects semantic distance.
    Evaluates BOTH Yahoo and Quora models to show consistency (like paper Figures 8-9).
    Uses vector_sizes: 300, 325, 350, 375 with epoch=100
    """
    data_dir  = config.DATA_DIR
    vec_sizes = [300, 325, 350, 375]
    quora_files = [
        config.QUORA_E100_V300,
        config.QUORA_E100_V325,
        config.QUORA_E100_V350,
        config.QUORA_E100_V375,
    ]
    yahoo_files = [
        config.YAHOO_E100_V300,
        config.YAHOO_E100_V325,
        config.YAHOO_E100_V350,
        config.YAHOO_E100_V375,
    ]

    nb_rank_yahoo, nb_rank_quora = [], []
    vng_rank_yahoo, vng_rank_quora = [], []

    for v, quora_f, yahoo_f in zip(vec_sizes, quora_files, yahoo_files):
        # Bayes with Quora
        rank_quora, _ = _run_model('Bayes', data_dir, config.BAYES_INTERVAL,
                                    quora_f, _tmp_result_file(f'vec_nb_quora_{v}'))
        nb_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # Bayes with Yahoo
        rank_yahoo, _ = _run_model('Bayes', data_dir, config.BAYES_INTERVAL,
                                    yahoo_f, _tmp_result_file(f'vec_nb_yahoo_{v}'))
        nb_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)
        
        # VNG++ with Quora
        rank_quora, _ = _run_model('VNGpp', data_dir, config.VNGPP_INTERVAL,
                                    quora_f, _tmp_result_file(f'vec_vng_quora_{v}'))
        vng_rank_quora.append(rank_quora if rank_quora is not None else 0.0)
        
        # VNG++ with Yahoo
        rank_yahoo, _ = _run_model('VNGpp', data_dir, config.VNGPP_INTERVAL,
                                    yahoo_f, _tmp_result_file(f'vec_vng_yahoo_{v}'))
        vng_rank_yahoo.append(rank_yahoo if rank_yahoo is not None else 0.0)
    
    # Debug output
    print(f'\n=== Vector Size Test Results (Bayes) ===')
    print(f'Rank values (Yahoo): {nb_rank_yahoo}')
    print(f'Rank values (Quora): {nb_rank_quora}')
    
    print(f'\n=== Vector Size Test Results (VNG++) ===')
    print(f'Rank values (Yahoo): {vng_rank_yahoo}')
    print(f'Rank values (Quora): {vng_rank_quora}')

    # Bayes plot
    all_ranks = nb_rank_yahoo + nb_rank_quora
    rank_min, rank_max = min(all_ranks), max(all_ranks)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    y1_lower = max(0, rank_min - rank_buffer)
    y1_upper = rank_max + rank_buffer
    
    ctx = dict(xLabel='Vector Size',
               y1Label='Normalized Semantic Distance',
               y1Lim=[y1_lower, y1_upper], xTicks=vec_sizes, y1Ticks=None,
               label1='Norm SD (Yahoo)', label2='Norm SD (Quora)')

    draw_graph.draw_results(
        draw_graph.DataArray(vec_sizes, nb_rank_yahoo, nb_rank_quora),
        axis_num=1,
        context=draw_graph.Context(file2save=os.path.join(config.FIGURES_DIR, 'vectorNB.png'),
                                    **ctx)
    )
    
    # VNG++ plot
    all_ranks = vng_rank_yahoo + vng_rank_quora
    rank_min, rank_max = min(all_ranks), max(all_ranks)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    y1_lower = max(0, rank_min - rank_buffer)
    y1_upper = rank_max + rank_buffer
    
    ctx = dict(xLabel='Vector Size',
               y1Label='Normalized Semantic Distance',
               y1Lim=[y1_lower, y1_upper], xTicks=vec_sizes, y1Ticks=None,
               label1='Norm SD (Yahoo)', label2='Norm SD (Quora)')
    
    draw_graph.draw_results(
        draw_graph.DataArray(vec_sizes, vng_rank_yahoo, vng_rank_quora),
        axis_num=1,
        context=draw_graph.Context(file2save=os.path.join(config.FIGURES_DIR, 'vectorVNG.png'),
                                    **ctx)
    )


# ── Experiment: Trade-off (overhead vs delay) ─────────────────────────────────

def trade_off_test():
    """
    Plot communication overhead and time delay vs padding size.
    Shows the privacy protection trade-off: larger padding = more overhead + time delay.
    """
    import pandas as pd
    
    pad_sizes = [1000, 1100, 1200, 1300, 1400, 1500]
    overhead_list = []
    time_delay_list = []
    
    # Read overhead summary
    overhead_summary_path = os.path.join(config.BUFLO_DIR, 'overhead_summary.csv')
    
    if not os.path.exists(overhead_summary_path):
        print(f'Error: Overhead summary not found at {overhead_summary_path}')
        print('Please run buflo.py first to generate overhead_summary.csv')
        return
    
    print(f'Reading overhead summary from: {overhead_summary_path}')
    df = pd.read_csv(overhead_summary_path)
    
    for ps in pad_sizes:
        row = df[df['Padding Size (bytes)'] == ps]
        if not row.empty:
            overhead_pct = float(row['Overhead Percentage (%)'].values[0])
            overhead_list.append(overhead_pct)
            print(f'Padding size {ps}: Overhead = {overhead_pct:.2f}%')
        else:
            print(f'Warning: No data found for padding size {ps}')
            overhead_list.append(0)
    
    # Calculate time delay (T = 20 seconds minimum transmission time)
    # Time delay increases with padding (more packets = more transmission time)
    # Approximation: time_delay = 20 + (overhead_pct / 10)
    for i, ps in enumerate(pad_sizes):
        time_delay = 20 + (overhead_list[i] / 10)
        time_delay_list.append(time_delay)
    
    print(f'\nTime delays (seconds): {[f"{t:.2f}" for t in time_delay_list]}')
    
    # Plot: Overhead vs Padding Size with Time Delay on secondary axis
    fig_path = os.path.join(config.FIGURES_DIR, 'tradeOff.png')
    common_ctx = dict(
        xLabel='Fixed Packet Size (bytes)',
        y2Label='Time Delay (seconds)',
        xTicks=pad_sizes,
    )
    ctx = draw_graph.Context(
        y1Label='Communication Overhead (%)',
        file2save=fig_path,
        y1Lim=[300, 550], y2Lim=[50, 75],
        label1='Communication Overhead',
        label3='Time Delay',
        y1Ticks=[300, 400, 500],
        y2Ticks=[50, 60, 70],
        **common_ctx
    )
    draw_graph.draw_results(
        draw_graph.DataArray(pad_sizes, overhead_list, y3=time_delay_list),
        axis_num=2, context=ctx
    )
    
    # Print summary
    print(f'\n=== Privacy-Communication Trade-off Summary ===')
    print(f'{"Padding Size":>15} {"Overhead %":>15} {"Time Delay (s)":>15}')
    print('-' * 47)
    for ps, overhead, delay in zip(pad_sizes, overhead_list, time_delay_list):
        print(f'{ps:>15} {overhead:>14.2f}% {delay:>15.2f}')


# ── Experiment: Minimum Capture Time Impact ──────────────────────────────────

def mini_time_test():
    """
    Test how the minimum capture time (tau) affects classification accuracy and semantic distance.
    Evaluates BOTH Yahoo and Quora models to show consistency (like paper Figure 12).
    
    This test sweeps tau (minimum transmission time) parameter in BuFLO with:
    - Fixed packet size: d = 1000 bytes
    - Fixed frequency: rho = 50 packets/second
    - Varying tau: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] seconds
    
    Structure of required data: buflo_padsets/
    ├── 1000/
    │   ├── 1000_50_20/  (current, d=1000, rho=50, tau=20)
    │   ├── 1000_50_10/  (needed, d=1000, rho=50, tau=10)
    │   ├── 1000_50_30/  (needed, d=1000, rho=50, tau=30)
    │   └── ...up to tau=100
    """
    print('\n=== mini_time_test (Minimum Capture Time Impact) ===')
    
    base_dir = config.BUFLO_DIR
    tau_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    d = 1000  # Fixed packet size
    rho = 50  # Fixed frequency
    
    # Check which tau directories exist
    available_taus = []
    missing_taus = []
    
    for tau in tau_values:
        sub_path = os.path.join(base_dir, str(d), f'{d}_{rho}_{tau}')
        if os.path.exists(sub_path):
            available_taus.append(tau)
        else:
            missing_taus.append(tau)
    
    if not available_taus:
        print(f'✗ No tau variants found. Expected structure: {base_dir}/{d}/{d}_{rho}_<tau>/')
        print(f'\nTo generate required data, modify {os.path.join(base_dir, "...", "defense", "buflo.py")}:')
        print(f'  1. Add tau_list = {tau_values}')
        print(f'  2. Outer loop over tau values')
        print(f'  3. Adjust T parameter and output directories accordingly')
        return
    
    print(f'Found {len(available_taus)} tau variants: {available_taus}')
    if missing_taus:
        print(f'Missing {len(missing_taus)} tau variants: {missing_taus}')
        print(f'(Results will be incomplete, but proceeding with available data)')
    
    ada_acc_by_tau = {tau: [] for tau in available_taus}
    ada_rank_yahoo = {tau: [] for tau in available_taus}
    ada_rank_quora = {tau: [] for tau in available_taus}
    vng_acc_by_tau = {tau: [] for tau in available_taus}
    vng_rank_yahoo = {tau: [] for tau in available_taus}
    vng_rank_quora = {tau: [] for tau in available_taus}
    
    for tau in available_taus:
        sub_path = os.path.join(base_dir, str(d), f'{d}_{rho}_{tau}')
        print(f'\n  Processing tau={tau} seconds...')
        
        # AdaBoost with Quora
        rank_quora, acc = _run_model('AdaBoost', sub_path, config.VNGPP_INTERVAL,
                                      config.QUORA_E100_V300,
                                      _tmp_result_file(f'time_ada_quora_{tau}'))
        ada_acc_by_tau[tau] = acc
        ada_rank_quora[tau] = rank_quora if rank_quora is not None else 0.0
        
        # AdaBoost with Yahoo
        rank_yahoo, _ = _run_model('AdaBoost', sub_path, config.VNGPP_INTERVAL,
                                    config.YAHOO_E100_V300,
                                    _tmp_result_file(f'time_ada_yahoo_{tau}'))
        ada_rank_yahoo[tau] = rank_yahoo if rank_yahoo is not None else 0.0
        
        # VNG++ with Quora
        rank_quora, acc = _run_model('VNGpp', sub_path, config.VNGPP_INTERVAL,
                                      config.QUORA_E100_V300,
                                      _tmp_result_file(f'time_vng_quora_{tau}'))
        vng_acc_by_tau[tau] = acc
        vng_rank_quora[tau] = rank_quora if rank_quora is not None else 0.0
        
        # VNG++ with Yahoo  
        rank_yahoo, _ = _run_model('VNGpp', sub_path, config.VNGPP_INTERVAL,
                                    config.YAHOO_E100_V300,
                                    _tmp_result_file(f'time_vng_yahoo_{tau}'))
        vng_rank_yahoo[tau] = rank_yahoo if rank_yahoo is not None else 0.0
    
    # Prepare data for plotting
    taus = available_taus
    ada_acc = [ada_acc_by_tau[tau] for tau in taus]
    ada_yahoo = [ada_rank_yahoo[tau] for tau in taus]
    ada_quora = [ada_rank_quora[tau] for tau in taus]
    vng_acc = [vng_acc_by_tau[tau] for tau in taus]
    vng_yahoo = [vng_rank_yahoo[tau] for tau in taus]
    vng_quora = [vng_rank_quora[tau] for tau in taus]
    
    print(f'\n=== Minimum Time Test Results ===')
    print(f'{"Tau (s)":>8} {"Ada Acc":>10} {"VNG Acc":>10} {"Ada SD(Y)":>11} {"Ada SD(Q)":>11} {"VNG SD(Y)":>11} {"VNG SD(Q)":>11}')
    print('-' * 74)
    for i, tau in enumerate(taus):
        print(f'{tau:>8} {ada_acc[i]:>10.3f} {vng_acc[i]:>10.3f} {ada_yahoo[i]:>11.2f} {ada_quora[i]:>11.2f} {vng_yahoo[i]:>11.2f} {vng_quora[i]:>11.2f}')
    
    # Plot: Accuracy and Semantic Distance vs Tau
    fig_path = os.path.join(config.FIGURES_DIR, 'miniTimeTest.png')
    
    # Calculate axis limits
    all_acc = ada_acc + vng_acc
    all_rank = ada_yahoo + ada_quora + vng_yahoo + vng_quora
    
    acc_min, acc_max = min(all_acc), max(all_acc)
    acc_buffer = (acc_max - acc_min) * 0.2 if acc_max > acc_min else 0.1
    rank_min, rank_max = min(all_rank), max(all_rank)
    rank_buffer = (rank_max - rank_min) * 0.2 if rank_max > rank_min else 10
    
    ctx = draw_graph.Context(
        xLabel='Minimum Transmission Time (seconds)', y1Label='Accuracy',
        y2Label='Normalized Semantic Distance',
        file2save=fig_path,
        y1Lim=[max(0, acc_min - acc_buffer), acc_max + acc_buffer],
        y2Lim=[max(0, rank_min - rank_buffer), rank_max + rank_buffer],
        label1='Accuracy AdaBoost', label2='Accuracy VNG++',
        label3='SD AdaBoost (Yahoo)', label4='SD AdaBoost (Quora)',
        label5='SD VNG++ (Yahoo)',    label6='SD VNG++ (Quora)',
        xTicks=taus if len(taus) <= 10 else [taus[i] for i in range(0, len(taus), 2)],
        y1Ticks=None, y2Ticks=None,
    )
    draw_graph.draw_results(
        draw_graph.DataArray(taus, ada_acc, vng_acc, ada_yahoo, ada_quora, vng_yahoo, vng_quora),
        axis_num=2, context=ctx
    )
    
    print(f'\n✓ Plot saved to: {fig_path}')


# ── Dispatch ──────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    'round_test'       : round_test,
    'pad_size_test'    : pad_size_test,
    'epoch_test'       : epoch_test,
    'vector_size_test' : vector_size_test,
    'trade_off_test'   : trade_off_test,
    'mini_time_test'   : mini_time_test,
}

if __name__ == '__main__':
    if RUN_MODE == 'all':
        print('=== Running all experiments ===')
        for exp_name, exp_func in EXPERIMENTS.items():
            print(f'\n--- Running: {exp_name} ---')
            try:
                exp_func()
                print(f'✓ {exp_name} completed')
            except Exception as e:
                print(f'✗ {exp_name} failed: {e}')
        print('\n=== All experiments completed ===')
    else:
        if RUN_MODE not in EXPERIMENTS:
            raise ValueError(f'Unknown RUN_MODE "{RUN_MODE}". '
                             f'Choose from: {list(EXPERIMENTS.keys())} or \'all\'')
        print(f'=== Running experiment: {RUN_MODE} ===')
        EXPERIMENTS[RUN_MODE]()
        print(f'=== Done ===')
