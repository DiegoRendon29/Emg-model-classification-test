from TrainingModelLocally import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_randomForest(train_data, train_label,val_data,val_labels):

    n_estimators_list = [75]
    max_depth_list = [25]
    min_samples_split_list = [2, 5, 10]

    best_accuracy = 0
    best_params = {}

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:

                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_split=min_samples_split, random_state=42)
                multi_rf = MultiOutputClassifier(rf)

                multi_rf.fit(train_data, train_label)

                # Predecir
                y_pred = multi_rf.predict(val_data)

                # Evaluar precisión

                y_pred_one_hot = np.zeros_like(y_pred)
                y_pred_one_hot[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
                overall_accuracy = np.mean(np.all(y_pred_one_hot == val_labels, axis=1))

                print(overall_accuracy)
                if overall_accuracy > best_accuracy:
                    best_accuracy = overall_accuracy
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split
                    }
                print(
                    f"n_estimators={n_estimators}, max_depth={max_depth},"
                    f" min_samples_split={min_samples_split} → Accuracy: {overall_accuracy:.4f}")

                metrics = [
                    "Census", "mean", "std", "kurtosis",
                    "skewness", "entropy", "median",
                    "percentile 25", "percentile 75"
                ]

                columns = [f"{metric} channel {i}" for i in range(10) for metric in metrics]



                avg_importances = np.zeros(90)
                for estimator in multi_rf.estimators_:
                    avg_importances += estimator.feature_importances_

                avg_importances /= len(multi_rf.estimators_)

                # Ordenar características por importancia promedio
                sorted_features = sorted(zip(columns, avg_importances), key=lambda x: x[1], reverse=True)

                # Imprimir el ranking global
                print("\nRanking global de importancia de características:")
                for feature, importance in sorted_features:
                    print(f"{feature}: {importance:.4f}")

                # Ordenar por importancia
                importance_df = importance_df.sort_values(by='Importance', ascending=False)

                print(importance_df)


    print("\nMejores hiperparámetros:", best_params)
    print(f"Mejor precisión global: {best_accuracy:.4f}")
    # Obtener el mejor modelo


def train_svc(train_data, train_label,val_data,val_labels):
    svm_model = SVC(kernel='poly', probability=True)
    svm_model.fit(train_data, np.argmax(train_label, axis=1))
    y_pred_labels = svm_model.predict(val_data)
    y_pred_one_hot = np.zeros_like(val_labels)
    y_pred_one_hot[np.arange(len(y_pred_labels)), y_pred_labels] = 1
    accuracy = np.mean(np.all(y_pred_one_hot == val_labels, axis=1))
    print(f"Accuracy: {accuracy:.4f}")


def train_knn(train_data, train_label,val_data,val_labels):
    train_labels_idx = np.argmax(train_label, axis=1)
    val_labels_idx = np.argmax(val_labels, axis=1)

    knn = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2)
    knn.fit(train_data, train_labels_idx)

    y_pred_labels = knn.predict(val_data)
    accuracy = np.mean(y_pred_labels == val_labels_idx)

    print(f"Accuracy: {accuracy:.4f}")
if __name__ == "__main__":
    frame_size = 31
    (train_data, train_labels), (val_data, val_labels) = load_data(frame_size=frame_size)
    train_knn(train_data, train_labels, val_data, val_labels)
    train_svc(train_data, train_labels, val_data, val_labels)
    train_randomForest(train_data,train_labels,val_data,val_labels)