
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt    
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, precision_recall_curve, precision_score, auc,
                         roc_curve, recall_score, classification_report, f1_score,
                         precision_recall_fscore_support)


###################################################
################ confusion_matrix #################
###################################################
def get_confusion_matrix(y_test, pred, title):
    ''' returns a confusion matrix '''
    conf_matrix = confusion_matrix(y_test, pred)
    RANDOM_SEED = 42
    LABELS = ["Normal", "Fraud"]
    plt.figure(figsize=(10,6))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix - "+str(title))
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    return plt.show()


def get_evaluation_stats(y_test, pred):
    ''' returns roc_auc score, recall and precision '''
    print('##########################')
    print('####### Evaluation #######')
    print('##########################')
    fpr, tpr, thres = metrics.roc_curve(y_test, pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('\n','Test AUC Score: %.3f' % roc_auc)
    print('\n','Test Recall Score: %.3f' % recall_score(pred, y_test))
    print('\n','Test Precision Score: %.3f' % precision_score(pred, y_test))
    print('\n','##########################')
    #print('\n','Test F2 Score: %.3f' % fbeta_score(y_pred=pred, y_true=y_test, beta=2),'\n')
    
def get_plot_ROC(test_y, test_pred, title):
    ''' returns Receiver Operating Characteristic plot '''
    fpr, tpr, thres = metrics.roc_curve(test_y, test_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(15,5))
    plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - '+str(title))
    plt.legend(loc="lower right")
    plt.show()


###################################################
################ ROC  WITH  THRES #################
###################################################
import numbers
import six
import numpy
import matplotlib.collections
from matplotlib import pyplot

#Code adapted from plot_roc.py by podshumok 
#Source: https://gist.github.com/podshumok/c1d1c9394335d86255b8

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates,
    in the correct format for LineCollection:
    an array of the form
    numlines x (points per line) x 2 (x and y) array
    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, axes=None,
              cmap=pyplot.get_cmap('coolwarm'),
              norm=pyplot.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,
              **kwargs):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if isinstance(z, numbers.Real):
        z = numpy.array([z])

    z = numpy.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, alpha=alpha, **kwargs
    )

    if axes is None:
        axes = pyplot.gca()

    axes.add_collection(lc)
    axes.autoscale()

    return lc


def get_plot_roc_thres(tpr, fpr, thresholds, subplots_kwargs=None,
             label_every=None, label_kwargs=None,
             fpr_label='False Positive Rate',
             tpr_label='True Positive Rate',
             luck_label='Luck',
             title='Receiver operating characteristic',
             **kwargs):

    roc_auc = metrics.auc(fpr, tpr)

    if subplots_kwargs is None:
        subplots_kwargs = {}

    figure, axes = pyplot.subplots(1, 1, **subplots_kwargs, figsize=(20,8) )
    axes.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc), color='w')
    if 'lw' not in kwargs:
        kwargs['lw'] = 1

    axes.plot(fpr, tpr, **kwargs)

    if label_every is not None:
        if label_kwargs is None:
            label_kwargs = {}

        if 'bbox' not in label_kwargs:
            label_kwargs['bbox'] = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.5,
            )

        for k in six.moves.range(len(tpr)):
            if k % label_every != 0:
                continue

            threshold = str(numpy.round(thresholds[k], 2))
            x = fpr[k]
            y = tpr[k]
            axes.annotate(threshold, (x, y), **label_kwargs)

    if luck_label is not None:
        axes.plot((0, 1), (0, 1), '--', color='black')

    lc = colorline(fpr, tpr, thresholds, axes=axes)

    figure.colorbar(lc)
    
    axes.set_xlim([-0.05, 1.05])
    axes.set_ylim([-0.05, 1.05])

    axes.set_xlabel(fpr_label)
    axes.set_ylabel(tpr_label)

    axes.set_title(title)

    axes.legend(loc="lower right")

    return figure, axes