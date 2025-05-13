from pypekit import Task
import pandas as pd


class IrisLoader(Task):
    output_category = "dataset"

    def run(self, input_path=None, output_base_path=None):
        from sklearn.datasets import load_iris
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target
        iris_df.to_csv(f"{output_base_path}.csv", index=False)
        return f"{output_base_path}.csv"

class Scaler(Task):
    input_category = "dataset"
    output_category = "dataset"

    def run(self, input_path=None, output_base_path=None):
        iris_df = pd.read_csv(input_path)
        X = iris_df.drop(columns=['target'])

        scaler = self.get_scaler()
        X_scaled = scaler.fit_transform(X)

        scaled_df = pd.DataFrame(data=X_scaled, columns=iris_df.columns[:-1])
        scaled_df['target'] = iris_df['target']
        scaled_df.to_csv(f"{output_base_path}.csv", index=False)
        return f"{output_base_path}.csv"

    def get_scaler(self):
        raise NotImplementedError("Subclasses should implement this method.")


class MinMaxScaler(Scaler):
    def get_scaler(self):
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()


class StandardScaler(Scaler):
    def get_scaler(self):
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()


class PCA(Task):
    input_category = "dataset"
    output_category = "dataset"

    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, input_path=None, output_base_path=None):
        iris_df = pd.read_csv(input_path)
        X = iris_df.drop(columns=['target'])

        from sklearn.decomposition import PCA
        pca = PCA(**self.kwargs)
        X_pca = pca.fit_transform(X)

        pca_df = pd.DataFrame(data=X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        pca_df['target'] = iris_df['target']
        pca_df.to_csv(f"{output_base_path}.csv", index=False)

        return f"{output_base_path}.csv"


class Classifier(Task):
    input_category = "dataset"

    def run(self, input_path=None, output_base_path=None):
        iris_df = pd.read_csv(input_path)
        X = iris_df.drop(columns=['target'])
        y = iris_df['target']

        classifier = self.get_classifier()
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        X['y_pred'] = y_pred
        X['target'] = y
        X.to_csv(f"{output_base_path}.csv", index=False)

        return f"{output_base_path}.csv"


class LogisticRegression(Classifier):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_classifier(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**self.kwargs)


class RandomForestClassifier(Classifier):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**self.kwargs)


class SVC(Classifier):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_classifier(self):
        from sklearn.svm import SVC
        return SVC(**self.kwargs)
    