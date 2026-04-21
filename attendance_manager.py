# attendance_manager.py
import pandas as pd
import os
from datetime import datetime

CSV_FILE = "attendance.csv"

def mark_attendance(names):

    # If file doesn't exist or is empty → create fresh CSV
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        try:
            df = pd.read_csv(CSV_FILE)
        except:
            # If file is corrupt → recreate
            df = pd.DataFrame(columns=["Name", "Date", "Time"])

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    for name in names:
        # Avoid duplicate attendance for same day
        exists = df[(df["Name"] == name) & (df["Date"] == date)]
        if exists.empty:
            df.loc[len(df)] = [name, date, time]

    df.to_csv(CSV_FILE, index=False)
    return df
