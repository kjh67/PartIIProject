import argparse
import os

from evaluation.plotting import plot_violin

from colmap.scripts.python.read_write_model import read_points3D_binary


def read_reprojection_errors(directory_path):
    """Accepts a path to a COLMAP output directory, and returns a list containing the reprojection errors of 3D points. 
    
    Will return an empty list if the directory does not exist, or if an error occurs when reading COLMAP data"""
    full_path = os.path.join(directory_path, 'sparse', '0')
    errors = []
    if os.path.exists(full_path):
        points = read_points3D_binary(os.path.join(full_path, "points3D.bin"))
        errors = [points[point].error for point in points]
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Given a path to one or more colmap output folders, retrieves and plots the reprojection errors of extracted 3D points."""
    )

    parser.add_argument('--colmap_dir_paths', nargs='+')
    parser.add_argument('--plot_labels', nargs='+')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    data = []
    for colmap_dataset in args.colmap_dir_paths:
        data.append(read_reprojection_errors(colmap_dataset))

    plot_violin(data, args.plot_labels, "Reprojection Error (px)", args.output_file)
