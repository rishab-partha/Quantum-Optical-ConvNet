import matplotlib.pyplot as plt
import csv
import os
import numpy as np

DIR = "Training_CSV"

def generate_plot(metric_name, algorithms = ["ComplexNet", "CNN"], colors = ["orange", "blue"]):
    fig = plt.figure()
    file_ending = metric_name.lower().replace(" ", "-") + ".csv"
    
    cntr = 0
    xtickmarks = [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000]
    ytickmarks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for algorithm in algorithms:
        file_name = algorithm.lower() + "-" + file_ending
        file_path = os.path.join(DIR, file_name)
        batch_numbers = []
        metric = []
        with open(file_path) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                batch_numbers.append(row[1])
                metric.append(row[2])
                print(row[1], row[2])
        plt.plot(batch_numbers, metric, color = colors[cntr])
        cntr += 1
    
    plt.legend(algorithms)
    plt.xlabel("Batch Number")
    plt.ylabel("metric_name")
    plt.xticks(np.arange(0, 1.05, step = 0.1))
    plt.xticks(np.arange(11), xtickmarks)
    plt.yticks(np.arange(0, 1.05, step=0.1))
    plt.yticks(np.arange(11), ytickmarks)
    plt.show()

generate_plot("Training Accuracy")