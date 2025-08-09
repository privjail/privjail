# Copyright 2025 TOYOTA MOTOR CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
import privjail as pj
from privjail import pandas as ppd

from bokeh.plotting import figure, show
from bokeh.models import LogColorMapper, ColorBar, LogTicker
from bokeh.palettes import Magma256

# LAT_MIN, LAT_MAX = 39.5, 40.5
# LON_MIN, LON_MAX = 116.0, 117.0
LAT_MIN, LAT_MAX = 39.7, 40.2
LON_MIN, LON_MAX = 116.2, 116.7

# G = 0.04
# G = 0.01
G = 0.004
# G = 0.001

M = 4
# M = 10

# eps = 1.0
eps = 2.0
# eps = 4.0
# eps = 10.0
# eps = 1000.0

def main():
    df = ppd.read_csv("data/tdrive.csv", "schema/tdrive.json")

    print("Dataset loaded.")

    # binning for lat, lon
    lat_bins = np.arange(np.floor(LAT_MIN / G), np.floor(LAT_MAX / G) + 1) * G
    df["lat_bin"] = ppd.cut(df["latitude"], bins=list(lat_bins), labels=list(lat_bins[:-1]))

    lon_bins = np.arange(np.floor(LON_MIN / G), np.floor(LON_MAX / G) + 1) * G
    df["lon_bin"] = ppd.cut(df["longitude"], bins=list(lon_bins), labels=list(lon_bins[:-1]))

    df = df.dropna()

    print("Binning done.")

    # bound user contribution
    df = df.sample(frac=1).groupby("taxi_id").head(M)

    print("User contribution bounded.")

    # groupby-count
    counts = df.groupby(["lat_bin", "lon_bin"]).size()
    print("Counts calculated.")

    # add noise
    noisy_counts = counts.reveal(eps=eps)
    print("Counts revealed.")

    visualize(noisy_counts)

    print()
    print("Consumed Privacy Budget:")
    print(pj.consumed_privacy_budget())

def visualize(ser):
    df = ser.reset_index(name="count")

    counts_2d = df.pivot(index="lat_bin", columns="lon_bin", values="count").values

    mapper = LogColorMapper(palette=Magma256, low=1, high=counts_2d.max())

    p = figure(
        title="Traffic Density Heatmap",
        x_axis_label="Longitude",
        y_axis_label="Latitude",
        x_range=(LON_MIN, LON_MAX),
        y_range=(LAT_MIN, LAT_MAX),
        frame_width=800,
        frame_height=800,
        toolbar_location="above",
    )

    p.image(
        image=[counts_2d],
        x=LON_MIN,
        y=LAT_MIN,
        dw=LON_MAX - LON_MIN,
        dh=LAT_MAX - LAT_MIN,
        color_mapper=mapper
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=LogTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0)
    )

    p.add_layout(color_bar, 'right')

    show(p)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        mode = "local"
    else:
        mode = sys.argv[1]

    if mode in ("server", "client"):
        if len(sys.argv) < 4:
            raise ValueError(f"Usage: {sys.argv[0]} {mode} <host> <port>")

        host = sys.argv[2]
        port = int(sys.argv[3])

    if mode == "local":
        main()

    elif mode == "server":
        # print(pj.proto_file_content())
        pj.serve(port)

    elif mode == "client":
        pj.connect(host, port)
        main()

    else:
        raise ValueError(f"Usage: {sys.argv[0]} [local|server|client] [host] [port]")
