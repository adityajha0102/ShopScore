import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training Data
            y_train: Training Labels
        Returns:
             None
        """
        pass


class LogisticRegressionModel(Model):
    """
    Logistic Regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train:
            y_train:
        Returns:
             None
        """
        try:
            logistic = LogisticRegression(max_iter=1000, solver='lbfgs')
            param = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30]}

            clf = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs')
            clf.fit(X_train, y_train)
            logging.info("Model training completed")
            return clf
        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e


class NaiveBayesModel(Model):
    """
    Naive Bayes model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train:
            y_train:
        Returns:
             None
        """
        try:
            clf = MultinomialNB(alpha=0.0001, class_prior=[0.5, 0.5])
            clf.fit(X_train, y_train)
            logging.info("Model training completed")
            return clf

        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e


class DecisionTreeModel(Model):
    """
    Decision Tree model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train:
            y_train:
        Returns:
             None
        """
        try:
            clf = DecisionTreeClassifier(class_weight='balanced', max_depth=20, min_samples_split=300)
            clf.fit(X_train, y_train)
            logging.info("Model training completed")
            return clf

        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e


class RandomForestModel(Model):
    """
    Random Forest model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train:
            y_train:
        Returns:
             None
        """
        try:
            clf = RandomForestClassifier(class_weight='balanced', max_depth=10, min_samples_split=5)
            clf.fit(X_train, y_train)
            logging.info("Model training completed")
            return clf

        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e


class GradienBoostModel(Model):
    """
    Gradient Boost model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train:
            y_train:
        Returns:
             None
        """
        try:

            clf = GradientBoostingClassifier(max_depth=8, min_samples_split=10)
            clf.fit(X_train, y_train)
            logging.info("Model training completed")
            return clf

        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e
