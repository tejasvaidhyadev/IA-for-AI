import numpy as np
import argparse
import sys,json

parser = argparse.ArgumentParser()
parser.add_argument("--preds_path", type=str, required=True)
params = parser.parse_args()

labelid2name = {int(x[0]): x[2].strip()
             for x in np.loadtxt('class_labels_indices.csv', delimiter=',', dtype='str', skiprows=1)
           }

preds = json.load(open(params.preds_path + "/predictions.json"))

output = ['filename,emotion'] + [x + ',' + labelid2name[y] for x,y in preds.items()]

open(params.preds_path + "/sample_submission.csv", 'w+').write('\n'.join(output))


