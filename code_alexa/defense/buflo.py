# =============================================================================
# buflo.py — Apply BuFLO (Buffered Fixed-Length Obfuscation) countermeasure
#
# Reads all CSV traffic files from INPUT_DIR, applies BuFLO obfuscation
# at different padding sizes, and writes obfuscated CSVs to separate directories.
# Generates overhead statistics CSV for each padding size.
# =============================================================================

import os
import csv
import random
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# ── Hardcoded settings — edit these before running ───────────────────────────
# =============================================================================

# Folder containing your original traffic CSV files
INPUT_DIR    = r'E:\Research work\wenggang\paper\code_alexa\data\trace_csv'

# Base folder where obfuscated CSVs will be saved (subdirs: 1000/, 1100/, etc.)
BASE_OUTPUT_DIR = r'E:\Research work\wenggang\paper\code_alexa\data\buflo_padsets'

# Padding sizes to generate (bytes)
PADDING_SIZES = [1000, 1100, 1200, 1300, 1400, 1500]

# ── BuFLO parameters (match paper: f=50 packets/sec, t=20 min seconds) ──────
F = 50   # Transmission frequency in packets per second
T = 20   # Minimum transmission time in seconds
#          Total minimum packets = T × F = 20 × 50 = 1000 packets

# =============================================================================


def load_csv(csv_path):
    """Read a traffic CSV file into a list of rows."""
    reader = csv.reader(open(csv_path, 'r'), delimiter=',')
    rows   = list(reader)

    try:
        float(rows[0][0])
    except ValueError:
        rows.pop(0)

    return [list(map(float, row)) for row in rows]


def apply_buflo(csv_path, query_data, d, f, t, output_dir):
    """
    Apply BuFLO obfuscation with padding size d.
    Writes to output_dir/<tracename>_buflo.csv
    """
    trace_name      = Path(csv_path).stem
    start_time      = query_data[0][1]
    end_time        = query_data[-1][1]
    total_min_packs = t * f
    total_overhead  = 0
    original_size   = sum(p[2] for p in query_data)
    index           = 0

    # Make a copy to avoid modifying original
    query_data = [list(row) for row in query_data]

    # ── Step 1 & 2: fix packet sizes ─────────────────────────────────────────
    i = 0
    while i < len(query_data):
        p = query_data[i]

        if len(p) == 4 and p[2] <= d:
            # ── Pad: packet is smaller than d ─────────────────────────────────
            overhead        = d - p[2]
            total_overhead += overhead
            p[2]            = d
            p[1]            = round(start_time + index * (1 / f), 2)
            p[0]            = index
            p.append(overhead)
            p.append('padded')
            index += 1

        elif len(p) == 4 and p[2] > d:
            # ── Chop: packet is larger than d ─────────────────────────────────
            remaining = p[2] - d
            p[2]      = d
            p[1]      = round(start_time + index * (1 / f), 2)
            p[0]      = index
            p.append(0)
            p.append('chopped')

            final_leftover = remaining % d
            if final_leftover == 0:
                n_new = int(remaining / d)
            else:
                n_new = int(remaining / d) + 1

            extra_index = i + 1
            while n_new > 0:
                index += 1
                if n_new == 1 and final_leftover != 0:
                    pad        = d - final_leftover
                    new_packet = [index,
                                  round(start_time + index * (1 / f), 2),
                                  d, p[3], pad, 'new']
                    total_overhead += pad
                else:
                    new_packet = [index,
                                  round(start_time + index * (1 / f), 2),
                                  d, p[3], 0, 'new']
                query_data.insert(extra_index, new_packet)
                extra_index += 1
                n_new       -= 1

            index += 1

        i += 1

    # ── Step 3: fill up to minimum duration with dummy packets ───────────────
    if index < total_min_packs:
        for i in range(index, int(total_min_packs)):
            direction  = float(np.sign(-1 + 2 * random.random()))
            dummy      = [i + 1,
                          round(start_time + (i + 1) * (1 / f), 2),
                          d, direction, d, 'dummy']
            total_overhead += d
            query_data.append(dummy)

    # ── Step 4: compute time delay ────────────────────────────────────────────
    time_delay = query_data[-1][1] - end_time

    # ── Step 5: save obfuscated CSV ───────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, trace_name + '_buflo.csv')
    df = pd.DataFrame(query_data,
                      columns=['index', 'time', 'size',
                               'direction', 'overhead', 'status'])
    df.to_csv(out_path, index=False)

    return out_path, total_overhead, original_size


def run_buflo_for_all_sizes():
    """Generate BuFLO obfuscations for all padding sizes."""
    
    csv_files = [
        os.path.join(INPUT_DIR, fname)
        for fname in os.listdir(INPUT_DIR)
        if fname.endswith('.csv')
    ]

    if not csv_files:
        print(f'No CSV files found in {INPUT_DIR}')
        return

    print(f'Found {len(csv_files)} source CSV files')
    print(f'Generating for padding sizes: {PADDING_SIZES}')
    print()

    # Track overhead stats for each padding size
    overhead_stats = {}  # {padding_size: {'total': X, 'avg': Y, 'num_files': Z, 'original_total': W}}

    for d in PADDING_SIZES:
        output_dir = os.path.join(BASE_OUTPUT_DIR, str(d), f'{d}_50_20')
        print(f'=== Padding size D={d} bytes ===')
        print(f'Output: {output_dir}')
        
        count = 0
        total_overhead_for_size = 0
        total_original_size = 0
        
        for csv_path in sorted(csv_files):
            query_data = load_csv(csv_path)
            out_path, file_overhead, original_size = apply_buflo(csv_path, query_data, d, F, T, output_dir)
            total_overhead_for_size += file_overhead
            total_original_size += original_size
            count += 1
            
            if count % 50 == 0:
                print(f'  Processed {count}/{len(csv_files)} files...')
        
        avg_overhead = total_overhead_for_size / count if count > 0 else 0
        overhead_percentage = (total_overhead_for_size / total_original_size * 100) if total_original_size > 0 else 0
        
        overhead_stats[d] = {
            'total': total_overhead_for_size,
            'avg': avg_overhead,
            'num_files': count,
            'original_total': total_original_size,
            'percentage': overhead_percentage
        }
        
        print(f'  ✓ Completed {count} files')
        print(f'  Communication Overhead Summary:')
        print(f'    Total overhead bytes: {total_overhead_for_size:,}')
        print(f'    Average overhead per file: {avg_overhead:,.2f} bytes')
        print(f'    Total original data: {total_original_size:,} bytes')
        print(f'    Overhead percentage: {overhead_percentage:.2f}%')
        print()

    print('=== All padding sizes generated ===')
    print()
    print('=== Overhead Summary Across All Padding Sizes ===')
    print(f'{"Padding Size":>15} {"Total Overhead":>20} {"Avg per File":>20} {"Overhead %":>15}')
    print('-' * 70)
    for d in sorted(overhead_stats.keys()):
        stats = overhead_stats[d]
        print(f'{d:>15} {stats["total"]:>20,} {stats["avg"]:>20,.2f} {stats["percentage"]:>14.2f}%')
    
    # Save overhead statistics to CSV file
    summary_output = os.path.join(BASE_OUTPUT_DIR, 'overhead_summary.csv')
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    with open(summary_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Padding Size (bytes)', 'Total Overhead (bytes)', 
                        'Avg Overhead per File (bytes)', 'Total Original Data (bytes)', 
                        'Overhead Percentage (%)'])
        
        for d in sorted(overhead_stats.keys()):
            stats = overhead_stats[d]
            writer.writerow([
                d,
                int(stats['total']),
                f'{stats["avg"]:.2f}',
                int(stats['original_total']),
                f'{stats["percentage"]:.2f}'
            ])
    
    print()
    print(f'✓ Overhead statistics saved to: {summary_output}')


if __name__ == '__main__':
    run_buflo_for_all_sizes()
