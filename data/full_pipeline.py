from generate_hourly_graphml import csv_to_graphml
import os
import subprocess
from utils import *

# This script runs the full pipeline to get from the raw data to the animated manifold.
# The first step is to get the data from RIPE


def full_pipeline_hourly_manifold(ip_type, region, date = '2023-08-29', e_value=4):
    # first fetch the probes
    # check if the file already exists and if it does, don't run the command
    if not os.path.exists(f"{ip_type}/probes_{ip_type}_{region}.csv"):
        command = ["python", "get_probes.py", "-i", ip_type, "-r", region]
        subprocess.run(command)
    # then get the measurements
    # check if the folder for that date exists and if it doesn't, make it
    if not os.path.exists(f"{ip_type}/{date}/graph_{region}/"):
        os.makedirs(f"{ip_type}/{date}/graph_{region}")
    #check if the file already exists and if it does, don't run the command
    # if not os.path.exists(f"{ip_type}/{date}/graph_Europe_hourly/latencies_23.csv"):
    command = ["python", "get_measurements.py", "-i", ip_type, "-d", date, "-inter", "12", "-o",
                   f"{ip_type}/{date}/graph_{region}/", "-r", region]
    subprocess.run(command)
    if not os.path.exists(f"{ip_type}/{date}/graph_{region}/{e_value}/"):
        os.makedirs(f"{ip_type}/{date}/graph_{region}/{e_value}/")
    # if not os.path.exists(f"{ip_type}/{date}/graph_Europe_hourly/{e_value}/graph_23_{e_value}.graphml"):
    for i in range(24):
        csv_to_graphml(date,i, ip_type, e_value)
    if not os.path.exists(f"{ip_type}/{date}/out_{region}_hourly/graph_23_{e_value}/"):
        command = ["python", "../src/generate_hourly_manifold.py"]
        subprocess.run(command)
    command = ["python", "../src/animate_hourly_manifold.py"]
    subprocess.run(command)

if __name__ == "__main__":
    full_pipeline_hourly_manifold(ip_type=ip_type, e_value=threshold, region = 'US', date = date)