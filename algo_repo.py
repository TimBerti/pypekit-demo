from pypekit import Task
import pandas as pd


class DataLoader(Task):
    input_types = {"source"}
    output_types = {"raw"}

    def run(self, _=None):
        load_data = self.get_data_loader()
        data = load_data()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

    def get_data_loader(self):
        raise NotImplementedError("Subclasses should implement this method.")


class MeltPoolNetLoader(DataLoader):
    def run(self, _=None):
        df = pd.read_csv('meltpoolnet_classification.csv')
        df = df[df['Process'] == 'PBF'][['Power', 'Velocity', 'beam D',
                                         'density', 'Cp', 'k', 'melting T', 'meltpool shape']]
        df = df.dropna()
        df['target'] = df['meltpool shape'].astype('category').cat.codes
        df = df.drop(columns=['meltpool shape'])
        return df


class IrisLoader(DataLoader):
    def get_data_loader(self):
        from sklearn.datasets import load_iris
        return load_iris


class WineLoader(DataLoader):
    def get_data_loader(self):
        from sklearn.datasets import load_wine
        return load_wine


class TrainTestSplitter(Task):
    input_types = {"raw"}
    output_types = {"split"}

    def run(self, df):
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_df['train'] = 1
        test_df['train'] = 0
        df = pd.concat([train_df, test_df], ignore_index=True)
        return df


class Scaler(Task):
    input_types = {"split"}
    output_types = {"processed"}

    def run(self, df):
        X = df.drop(columns=['target', 'train'])
        X_train = X[df['train'] == 1]

        scaler = self.get_scaler()
        scaler.fit(X_train)

        X_scaled = scaler.transform(X)
        scaled_df = pd.DataFrame(data=X_scaled, columns=X.columns)
        scaled_df['target'] = df['target']
        scaled_df['train'] = df['train']

        return scaled_df

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
    input_types = {"split", "processed"}
    output_types = {"processed"}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, df):
        X = df.drop(columns=['target', 'train'])
        X_train = X[df['train'] == 1]

        from sklearn.decomposition import PCA
        pca = PCA(**self.kwargs)
        pca.fit(X_train)

        X_pca = pca.transform(X)
        pca_df = pd.DataFrame(data=X_pca, columns=[
                              f'PC{i+1}' for i in range(X_pca.shape[1])])
        pca_df['target'] = df['target']
        pca_df['train'] = df['train']

        return pca_df


class Classifier(Task):
    input_types = {"split", "processed"}
    output_types = {"predicted"}

    def run(self, df):
        X = df.drop(columns=['target', 'train'])
        y = df['target']
        X_train = X[df['train'] == 1]
        y_train = y[df['train'] == 1]

        classifier = self.get_classifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X)
        df['predicted'] = y_pred

        return df


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


class Evaluator(Task):
    input_types = {"predicted"}
    output_types = {"sink"}

    def run(self, df):
        df_test = df[df['train'] == 0]
        return (df_test['target'] == df_test['predicted']).mean()


ALGORITHMS = {
    "Data Loader": {
        "MeltPoolNet Loader": MeltPoolNetLoader,
        "Iris Loader": IrisLoader,
        "Wine Loader": WineLoader,
    },
    "Train-Test Split (required)": {
        "Train-Test Split": TrainTestSplitter,
    },
    "Scaler": {
        "Min-Max Scaler": MinMaxScaler,
        "Standard Scaler": StandardScaler,
    },
    "Processing": {
        "PCA": PCA,
    },
    "Classifier (required)": {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "SVC": SVC,
    },
    "Evaluator (required)": {
        "Accuracy Evaluator": Evaluator,
    },
}
