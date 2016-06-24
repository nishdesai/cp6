import numpy as np
import re, os, sys

def parse(label_file, label_ind, log_file, output_file):
    labels = np.loadtxt(label_file, delimiter=' ')
    with open(log_file, 'r') as log:
        while True:
            line = log.readline()
            if line == '':
                break
            #expected format of this line is: ">> query : ImageLabel(Image[0])"
            if "ImageLabel" in line:
                ind = int(re.search(r'\d+', line).group())
                mline = log.readline()
                #expected format of this line is: "Mean = 0.00000000   Var = 0.00000000"
                mean = float(mline.split()[2])
                if mean < 0.5:
                    labels[ind, label_ind+1] = 0
                else:
                    labels[ind, label_ind+1] = 1

    np.savetxt(label_file, labels, delimiter=' ')
    np.savetxt(output_file, labels, delimiter=' ', fmt='%d')


if __name__ == "__main__":
    log_file = sys.argv[1]
    label_file = sys.argv[2]
    label_ind = int(sys.argv[3])
    output_dir = sys.argv[4]
    parse(label_file, label_ind, log_file, os.path.join(output_dir, 'label_vectors_results.txt'))
