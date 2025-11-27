#!/usr/bin/env python3
import argparse, sys, time, json, os, struct
import numpy as np
import pandas as pd
from collections import Counter
import serial

# Helpers ....
def read_csv_windows(path_windows):
    arr = pd.read_csv(path_windows, header=None).values.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path_windows} must be 2D; got {arr.shape}")
    return arr

def read_labels(path_labels):
    df = pd.read_csv(path_labels)
    if "label" not in df.columns:
        raise ValueError(f"{path_labels} must contain a 'label' column.")
    return df["label"].values.astype(int)

def load_label_map(path_json):
    with open(path_json, "r") as f:
        lm = json.load(f)
    lm2 = {int(k): v for k, v in lm.items()}
    inv = {v: k for k, v in lm2.items()}
    return lm2, inv

# Serial communication ....
def open_serial(port, baud, boot_wait=2.0, drain_secs=2.0, verbose=True):
    ser = serial.Serial(port=port, baudrate=baud, timeout=1.0)
    time.sleep(boot_wait)

    t0 = time.time()
    if verbose: print("Reading board banner")
    while time.time() - t0 < drain_secs:
        line = ser.readline()
        if not line:
            break
        try:
            s = line.decode(errors="ignore").strip()
        except Exception:
            s = repr(line)
        if verbose and s:
            print(s)
    if verbose: print("End banner")
    return ser

def send_csv_window(ser, row, delay_ms=20):
    line = ",".join(f"{v:.6f}" for v in row)
    ser.write(line.encode("utf-8") + b"\n")
    ser.flush()
    if delay_ms > 0:
        time.sleep(delay_ms/1000.0)

def send_binary_window(ser, row, delay_ms=20):
    buf = struct.pack("<" + "f"*len(row), *row)
    ser.write(buf)
    ser.flush()
    if delay_ms > 0:
        time.sleep(delay_ms/1000.0)

def read_pred_line(ser, timeout_s=2.0, verbose=False):
    t0 = time.time()
    best = None
    while time.time() - t0 < timeout_s:
        raw = ser.readline()
        if not raw:
            continue
        try:
            s = raw.decode(errors="ignore").strip()
        except Exception:
            s = repr(raw)
        if verbose and s:
            print("<<", s)
        if "Pred:" in s or "Prediction:" in s or s.isdigit():
            best = s
            break
    return best

def parse_pred_text(s, inv_label_map):
    if s is None:
        return None
    s2 = s.strip()
    if "Pred:" in s2:
        try:
            after = s2.split("Pred:", 1)[1].strip()
            before_scores = after.split("|", 1)[0].strip()
            return int(before_scores)
        except Exception:
            pass
    if "Prediction:" in s2:
        try:
            return int(s2.split("Prediction:", 1)[1].strip())
        except Exception:
            pass
    if s2.isdigit():
        return int(s2)
    return inv_label_map.get(s2, None)

# Metrics
def f1_from_counts(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)

def calc_window_metrics(y_true, y_pred, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_true == y_pred))
    TP = Counter(); FP = Counter(); FN = Counter()
    for gt, pr in zip(y_true, y_pred):
        if pr == -1:  # missing
            FN[gt] += 1
        elif pr == gt:
            TP[gt] += 1
        else:
            FP[pr] += 1
            FN[gt] += 1
    f1s = [f1_from_counts(TP[c], FP[c], FN[c]) for c in classes]
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    return acc, macro_f1

# Main function
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_csv", default="output/arduino_test_data/test_windows_serial.csv")
    ap.add_argument("--labels_csv",  default="output/arduino_test_data/test_labels.csv")
    ap.add_argument("--label_map",   default="output/arduino_test_data/label_map.json")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud",        type=int, default=115200)
    ap.add_argument("--delay_ms",    type=int, default=20, help="Delay after writing one window")
    ap.add_argument("--max_n",       type=int, default=None)
    ap.add_argument("--protocol",    choices=["csv","binary"], default="csv",
                    help="csv = one line of 160 floats; binary = raw float32*160")
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    X = read_csv_windows(args.windows_csv)   # (N, 160)
    y = read_labels(args.labels_csv)         # (N,)
    label_map, inv_label_map = load_label_map(args.label_map)
    classes = sorted(label_map.keys())

    N = min(len(X), len(y))
    if args.max_n is not None:
        N = min(N, args.max_n)
    X = X[:N]; y = y[:N]

    print(f"Loaded {N} windows from {args.windows_csv}")
    print(f"Protocol: {args.protocol.upper()}   Port: {args.port}   Baud: {args.baud}")

    ser = open_serial(args.port, args.baud, boot_wait=2.0, drain_secs=3.0, verbose=True)

    # Sender selected
    sender = send_csv_window if args.protocol == "csv" else send_binary_window

    preds = np.full((N,), -1, dtype=int)
    for i in range(N):
        sender(ser, X[i], delay_ms=args.delay_ms)
        resp = read_pred_line(ser, timeout_s=2.5, verbose=args.verbose)
        pred_id = parse_pred_text(resp, inv_label_map)
        if pred_id is None:
            print(f"[{i+1:>5}/{N}] No prediction from board: (resp={resp})")
        else:
            preds[i] = pred_id
            if args.verbose:
                gt_name = label_map[y[i]]
                pr_name = label_map.get(pred_id, str(pred_id))
                print(f"[{i+1:>5}/{N}] GT={gt_name}  PRED={pr_name}")

    ser.close()

    # Metrics
    acc, macro_f1 = calc_window_metrics(y, preds, classes)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"F1 score:  {macro_f1:.4f}")


if __name__ == "__main__":
    main()