import os
import sys

import matplotlib
import numpy as np
import sklearn.metrics
import tensorflow as tf

# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
# define some configurable parameter
tf.flags.DEFINE_string('dn', 'nyt', 'dataset name')
result_dir = os.path.join('test_result', FLAGS.dn)


def main():
    models = sys.argv[1:]
    # markers = ['D', '*', 's', 'p', 'o', '^', '+', 'x']
    for i, model in enumerate(models):
        x = np.load(os.path.join(result_dir, model + '_x' + '.npy'))
        y = np.load(os.path.join(result_dir, model + '_y' + '.npy'))
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x=x, y=y)
        # plt.plot(x, y, lw=2, label=model + '-auc='+str(auc))
        plt.plot(x, y, lw=2, label=model)
        # plt.plot(x[::10000], y[::10000], markers[i], label=model)
        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
        print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300],
                                                                        (y[100] + y[200] + y[300]) / 3))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall')
    # plt.legend(loc="upper right", fontsize='small')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, FLAGS.dn + '_pr_curve'))


if __name__ == "__main__":
    main()
