### contains all the meta information needed to run the script
import os

ip_type = 'ipv4'
project_dir = '/Users/loqmansalamatian/Documents/GitHub/linear-geodesic-optimization/'
date = '2023-08-15'
region = 'US'
directory = os.path.join(project_dir,'data', ip_type, date, f'out_{region}_hourly')

sides = 50
width = height = sides
leaveout_proportion = 1.
max_iterations = 200
lambda_curvature = 1.
lambda_smooth = 0.004
lambda_geodesic = 0.
initial_radius = 20.
scale = 1.
subdirectory_name = f'{lambda_curvature}_{lambda_smooth}_{lambda_geodesic}_{initial_radius}_{width}_{height}_{scale}'
threshold = 6
manifold_count = 24
fps = 24

