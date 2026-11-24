"""Manifest (resumability) + parquet/json storage."""
import glob
import json
import os
import sqlite3
import threading
import time

import pandas as pd


class Store:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(os.path.join(root, "manifest.db"), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks(key TEXT PRIMARY KEY, kind TEXT, status TEXT, "
                        "rows INTEGER, info TEXT, ts REAL)")
        self.db.commit()

    def path(self, *p):
        return os.path.join(self.root, *p)

    def done(self, key):
        with self._lock:
            r = self.db.execute("SELECT status FROM tasks WHERE key=?", (key,)).fetchone()
        return bool(r and r[0] == "done")

    def mark(self, key, kind, status, rows=0, info=""):
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?)",
                            (key, kind, status, int(rows), str(info)[:500], time.time()))
            self.db.commit()

    def write_df(self, df, *p):
        path = self.path(*p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

    def read_glob(self, *pattern):
        files = sorted(glob.glob(self.path(*pattern)))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def json_get(self, *p, max_age=None):
        path = self.path(*p)
        if not os.path.exists(path):
            return None
        if max_age is not None and time.time() - os.path.getmtime(path) > max_age:
            return None
        with open(path) as f:
            return json.load(f)

    def json_put(self, obj, *p):
        path = self.path(*p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f)

    def stats(self):
        with self._lock:
            return self.db.execute("SELECT kind, status, COUNT(*), SUM(rows) FROM tasks GROUP BY kind, status").fetchall()

    def failures(self, limit=20):
        with self._lock:
            return self.db.execute("SELECT key, info FROM tasks WHERE status='failed' LIMIT ?", (limit,)).fetchall()

    def disk_mb(self):
        total = 0
        for r, _, fs in os.walk(self.root):
            total += sum(os.path.getsize(os.path.join(r, f)) for f in fs)
        return total / 1e6
