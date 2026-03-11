# src/data_loader.py
import os
import glob
import re
import numpy as np
import pandas as pd

def extract_user_ids(data_dir):
    acc_files = glob.glob(os.path.join(data_dir, "*_PocketPhone_Accelerometer_(Samsung_S6).csv"))
    user_ids = []

    for f in acc_files:
        fname = os.path.basename(f)
        match = re.match(r"(\d+)_PocketPhone_Accelerometer_", fname)
        if match:
            user_ids.append(int(match.group(1)))

    return sorted(user_ids)

def load_user_data(data_dir, user_id):
    acc_path = os.path.join(data_dir, f"{user_id}_PocketPhone_Accelerometer_(Samsung_S6).csv")
    gyro_path = os.path.join(data_dir, f"{user_id}_PocketPhone_Gyroscope_(Samsung_S6).csv")

    acc = pd.read_csv(acc_path)
    gyro = pd.read_csv(gyro_path)

    # keep only needed columns
    acc = acc[["Xvalue", "Yvalue", "Zvalue", "time"]].copy()
    gyro = gyro[["Xvalue", "Yvalue", "Zvalue", "time"]].copy()

    # rename so they do not clash
    acc.columns = ["acc_x", "acc_y", "acc_z", "time"]
    gyro.columns = ["gyro_x", "gyro_y", "gyro_z", "time"]

    # convert time to datetime
    acc["time"] = pd.to_datetime(acc["time"])
    gyro["time"] = pd.to_datetime(gyro["time"])

    # sort by time just in case
    acc = acc.sort_values("time").reset_index(drop=True)
    gyro = gyro.sort_values("time").reset_index(drop=True)

    # merge by nearest timestamp
    merged = pd.merge_asof(
        acc,
        gyro,
        on="time",
        direction="nearest"
    )

    # drop rows with missing matches
    merged = merged.dropna().reset_index(drop=True)

    # final 6-channel signal
    signal = merged[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values

    return signal, merged

def load_all_users(data_dir):
    user_ids = extract_user_ids(data_dir)

    user_signals = {}
    user_tables = {}

    for user_id in user_ids:
        signal, merged = load_user_data(data_dir, user_id)
        user_signals[user_id] = signal
        user_tables[user_id] = merged

    return user_ids, user_signals, user_tables