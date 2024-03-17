import os
import subprocess
import sys
import re
from datetime import datetime
sys.path[0] = '../src/'
from utils import *

def csv_to_graphml(date,hour, ip_type = 'ipv4', e_value=4):
    base_dir = f"{ip_type}/{date}/graph_{region}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Regular expression to match the filenames and extract the dates
    filename_regex = r'latencies_(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}_(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}_US\.csv'

    # List to hold filenames and their start dates
    files_with_dates = []

    # List files in the directory
    for filename in os.listdir(base_dir):
        if filename.startswith('latencies'):
            # Extract dates from the filename
            match = re.match(filename_regex, filename)
            if match:
                start_date_str, _ = match.groups()
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                files_with_dates.append((filename, start_date))

    # Sort the list by start dates
    files_with_dates.sort(key=lambda x: x[1])

    # Extract the sorted filenames
    sorted_filenames = [filename for filename, _ in files_with_dates]

    latencies_file = os.path.join(base_dir, sorted_filenames[hour])
    output_file = os.path.join(base_dir, str(e_value) ,f"graph_{sorted_filenames[hour].split('latencies_')[1].split('.csv')[0]}_{e_value}")

    command = ["python", "csv_to_graphml.py", "-l", latencies_file, "-i", ip_type, "-o", output_file, "-e",
               str(e_value), "-r", region]
    subprocess.run(command)



def main():
    for i in range(24):
        csv_to_graphml(date,i, e_value=8)


if __name__ == "__main__":
    main()