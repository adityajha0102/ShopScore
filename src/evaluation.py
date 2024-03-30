import logging
from abc import ABC, abstractmethod

from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating our models
    """

    @abstractmethod
    def calculate_scores(self, **kwargs):
        """
        Calculates the scores for the model evaluation
        Args:
            **kwargs: Keyword arguments specific to each evaluation method
        Returns:
            scores: Calculated scores
        """
        pass


'''class ConfusionMatrix(Evaluation):
    """
    Evaluation strategy that uses Confusion Matrix
    """

    def calculate_scores(self, y_test: np.ndarray, pred_label_test: np.ndarray):
        try:
            cf_matrix_test = confusion_matrix(y_test, pred_label_test)
            logging.info("ConfusionMatrixDone")
            return cf_matrix_test

        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
'''

'''class AUC_ROC(Evaluation):
    """
    Evaluation strategy that uses AUC ROC
    """

    def calculate_scores(self, y_train: np.ndarray, y_test: np.ndarray, y_pred_train: np.ndarray,
                         y_pred_test: np.ndarray):
        try:
            fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
            fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

            train_auc = round(auc(fpr_train, tpr_train), 3)
            test_auc = round(auc(fpr_test, tpr_test), 3)

            plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
            plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
            plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
            plt.xlabel('False Positive Rate(FPR)')
            plt.ylabel('True Positive Rate(TPR)')
            plt.title('ROC curve')
            plt.legend()

            logging.info("ROC_AUC: {}".format(plt))
            return plt

        except Exception as e:
            logging.error("Error in calculating ROC AUC: {}".format(e))
            raise e
'''

class FScore(Evaluation):
    """
    Evaluation strategy that uses F1 Score
    """

    def calculate_scores(self, y_train: np.ndarray, y_test: np.ndarray, pred_label_train: np.ndarray,
                         pred_label_test: np.ndarray):
        try:
            f_train = round(f1_score(y_train, pred_label_train), 4)
            f_test = round(f1_score(y_test, pred_label_test), 4)
            logging.info("F1 Score for Train: {} and F1 Score for Test:{}".format(f_train, f_test))
            return f_test

        except Exception as e:
            logging.error("Error in calculating F1 Score: {}".format(e))
            raise e


class Accuracy(Evaluation):
    """
    Evaluation strategy that uses Accuracy Score
    """

    def calculate_scores(self, y_train: np.ndarray, y_test: np.ndarray, pred_label_train: np.ndarray,
                         pred_label_test: np.ndarray):
        try:
            accuracy_train = round(accuracy_score(y_train, pred_label_train), 4)
            accuracy_test = round(accuracy_score(y_test, pred_label_test), 4)
            logging.info(
                "Accuracy Score for Train: {} and Accuracy Score for Test:{}".format(accuracy_train, accuracy_test))
            return accuracy_test

        except Exception as e:
            logging.error("Error in calculating Accuracy Score: {}".format(e))
            raise e

