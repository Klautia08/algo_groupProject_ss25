
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Index des verwendeten Merkmals
        self.threshold = threshold  # Trennungsschwellwert
        self.left = left  # Linker Kindknoten (<= Schwellwert)
        self.right = right  # Rechter Kindknoten (> Schwellwert)
        self.value = value  # Nur in Blattknoten: Vorhergesagte Klasse

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # Mindestanzahl für Teilung
        self.max_depth = max_depth  # Maximale Baumtiefe
        self.n_features = n_features  # Anzahl verwendeter Merkmale
        self.root = None  # Wurzelknoten

    def fit(self, X, y):
        # Verwende nur 3 Merkmale gemäß Aufgabenstellung
        self.n_features = 3
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Abbruchkriterium (maximale Tiefe, reine Klasse oder zu wenig Samples)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Verwende alle Merkmale (n_features=3)
        feat_idxs = np.arange(n_feats)

        # Finde beste Aufteilung
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Aufteilung in linke/rechte Teilmenge
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # Rekursiver Aufbau der Teilbäume
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Entropie des Elternknotens
        parent_entropy = self._entropy(y)

        # Kindknoten erstellen
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Gewichtete Entropie der Kinder
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Informationsgewinn berechnen
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        # Berechnet Entropie: H(S) = -Σ p_i * log2(p_i)
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    # Neue Methode für Entscheidungspfad
    def predict_with_path(self, x):
        return self._traverse_tree_with_path(x, self.root, [])

    def _traverse_tree_with_path(self, x, node, path):
        if node.is_leaf_node():
            path.append(f"Leaf: Klasse {node.value}")
            return node.value, path

        decision = "links" if x[node.feature] <= node.threshold else "rechts"
        feature_info = (
            f"Merkmal {node.feature} <= {node.threshold:.4f}? "
            f"({x[node.feature]:.4f} → {decision})"
        )
        path.append(feature_info)

        if decision == "links":
            return self._traverse_tree_with_path(x, node.left, path)
        else:
            return self._traverse_tree_with_path(x, node.right, path)

    # Neue Methode für textuelle Baumausgabe
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "  " * depth

        if node.is_leaf_node():
            print(f"{indent}Leaf: Klasse {node.value}")
            return

        print(f"{indent}Merkmal {node.feature} <= {node.threshold:.4f}?")
        print(f"{indent}--> True:")
        self.print_tree(node.left, depth + 1)
        print(f"{indent}--> False:")
        self.print_tree(node.right, depth + 1)


# Datenvorverarbeitung
def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Auswahl der drei Merkmale: [0] = mean radius, [1] = mean texture, [4] = mean smoothness
    selected_indices = [0, 1, 4]
    X_selected = X[:, selected_indices]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    return X_selected, y, selected_feature_names


# Hauptprogramm
if __name__ == "__main__":
    # Daten laden
    X, y, feature_names = load_data()

    # Entscheidungsbaum trainieren (max_depth=3)
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)

    # Beispielpatientinnen
    patient_A = np.array([14.2, 20.5, 0.095])
    patient_B = np.array([18.7, 17.0, 0.104])

    # Vorhersagen mit Entscheidungspfad
    pred_A, path_A = clf.predict_with_path(patient_A)
    pred_B, path_B = clf.predict_with_path(patient_B)

    print("\n--- Patient A ---")
    print(f"Vorhersage: {'benign' if pred_A == 1 else 'malignant'}")
    print("Entscheidungspfad:")
    for step in path_A:
        print(f"- {step}")

    print("\n--- Patient B ---")
    print(f"Vorhersage: {'benign' if pred_B == 1 else 'malignant'}")
    print("Entscheidungspfad:")
    for step in path_B:
        print(f"- {step}")

    # Baumstruktur ausgeben
    print("\nBaumstruktur:")
    clf.print_tree()
