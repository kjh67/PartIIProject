import os
import numpy as np
from scipy.stats import t as t_distribution


class ModelMetrics():
    def __init__(self, model, iteration, psnrs, ssims, mses):
        self.modeltype = model
        self.iteration = iteration
        self.psnrs = psnrs
        self.ssims = ssims
        self.mses = mses

    def save_metrics(self, output_folder):
        filename = f"{self.modeltype}{self.iteration}_metrics.txt"
        with open(os.path.join(output_folder, filename), "w") as f:
            f.write(self.modeltype + '\n')
            f.write(self.iteration + '\n')
            f.write(",".join(str(psnr) for psnr in self.psnrs) + '\n')
            f.write(",".join(str(ssim) for ssim in self.ssims) + '\n')
            f.write(",".join(str(mse) for mse in self.mses))

    @staticmethod
    def load_metrics(source_folder):
        """Returns an array of ModelMetrics objects, containing all metrics saved in the source folder"""
        metrics = []
        for file in os.listdir(source_folder):
            if file.endswith(".txt"):
                with open(file, "r") as f:
                    lines = f.readlines()
                    psnrs = np.array([float(psnr) for psnr in lines[2].split(",")])
                    ssims = np.array([float(ssim) for ssim in lines[3].split(",")])
                    mses = np.array([float(mse) for mse in lines[4].split(",")])
                    modelmetrics = ModelMetrics(lines[0], int(lines[1]), psnrs, ssims, mses)
                metrics.append(modelmetrics)
        return metrics 

def get_means_and_confidence(data):
    """Returns the mean and 95% confidence intervals for the provided array of data"""

    mean = np.mean(data)
    standard_deviation = np.std(data, ddof=1)
    n = len(data)
    lower, upper = t_distribution.interval(0.95, n, mean, standard_deviation/np.sqrt(n))

    return mean, lower, upper
