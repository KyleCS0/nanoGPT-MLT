"""
Parse Nsight Compute reports and extract roofline metrics.

Usage:
    python parse_ncu_results.py --report ncu_reports/v0_T128.ncu-rep
"""
import argparse
import subprocess
import csv
import sys
import json
import io

def parse_ncu_report(report_path):
    """Extract key metrics from ncu report using CLI."""

    # Export to CSV (Standard "Import" which produces the table view)
    # But we need RAW counters. The previous attempt used --page raw.
    # We will use exactly that command again but parse correctly.
    cmd = [
        '/local/kyle0/conda_envs/nanogpt/bin/ncu', '--import', report_path,
        '--csv',
        '--page', 'raw'
    ]
    # Increase field size limit for huge ncu lines
    csv.field_size_limit(sys.maxsize)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ncu: {e}")
        print(e.stderr)
        return {}

    metrics = {
        'dram_bytes': 0.0,
        'fadd': 0.0,
        'fmul': 0.0,
        'ffma': 0.0,
        'time_ns': 0.0
    }

    # Parse CSV output
    f = io.StringIO(result.stdout)
    reader = csv.DictReader(f)
    
    # Read/Skip unit row (Line 2)
    # We can peek or just check if "ID" is empty/numeric. 
    # ncu raw csv unit row usually has empty ID or distinct content.
    # But DictReader consumes the header (Line 1). The next detail is Line 2.
    # We'll simple skip the first row if it looks like units.
    
    # Iterate and sum
    for i, row in enumerate(reader):
        # Skip unit row. Usually ID is empty or non-numeric for unit row
        if i == 0:
            val = row.get('ID', '')
            if not val or not val.isdigit():
                continue
        
        # Depending on ncu version, sometimes unit row is not present in --page raw?
        # Let's check ID. If ID is a number, it's a kernel.
        if not row.get('ID', '').isdigit():
            continue

        # Extract DRAM Bytes
        # dram__bytes_read.sum + dram__bytes_write.sum
        try:
            # Helper to get float
            def get_val(key):
                v = row.get(key, '0')
                return float(v) if v else 0.0

            # DRAM Bytes: Unit is Mbyte
            d_read = get_val('dram__bytes_read.sum')
            d_write = get_val('dram__bytes_write.sum')
            metrics['dram_bytes'] += (d_read + d_write) * 1_000_000 # Convert MB to Bytes

            # Extract Time: Unit is us
            metrics['time_ns'] += get_val('gpu__time_duration.sum') * 1_000 # Convert us to ns

            # Extract FLOPs (Derived from SMSP rates * elapsed cycles)
            # Rate is "inst / cycle" (per device elapsed cycle)
            
            # Use sm__cycles_elapsed.max as duration in cycles
            duration_cycles = get_val('sm__cycles_elapsed.max')
            
            fadd_rate = get_val('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed')
            fmul_rate = get_val('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed')
            ffma_rate = get_val('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed')

            metrics['fadd'] += (fadd_rate * duration_cycles)
            metrics['fmul'] += (fmul_rate * duration_cycles)
            metrics['ffma'] += (ffma_rate * duration_cycles)

        except ValueError:
            continue

    # Calculate derived metrics
    metrics['total_flops'] = metrics['fadd'] + metrics['fmul'] + 2 * metrics['ffma']

    if metrics['dram_bytes'] > 0:
        metrics['arithmetic_intensity'] = metrics['total_flops'] / metrics['dram_bytes']
    
    if metrics['time_ns'] > 0:
        metrics['achieved_flops'] = metrics['total_flops'] / (metrics['time_ns'] * 1e-9)

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    metrics = parse_ncu_report(args.report)

    print(f"Report: {args.report}")
    print(f"  DRAM Bytes: {metrics.get('dram_bytes', 0):,.0f}")
    print(f"  Total FLOPs: {metrics.get('total_flops', 0):,.0f}")
    print(f"    (FADD: {metrics.get('fadd',0):,.0f}, FMUL: {metrics.get('fmul',0):,.0f}, FFMA: {metrics.get('ffma',0):,.0f})")
    print(f"  GPU Time (ns): {metrics.get('time_ns', 0):,.0f}")
    print(f"  Arithmetic Intensity: {metrics.get('arithmetic_intensity', 0):.2f} FLOPs/Byte")
    print(f"  Achieved FLOP/s: {metrics.get('achieved_flops', 0):.2e}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
