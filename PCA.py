import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join

path = "datasets/"
output_path = "processed/"

filenames = [f for f in listdir(path) if isfile(join(path, f))]
print(filenames)

pca = PCA(n_components=13)
for filename in filenames:
    df = pd.read_csv(path + filename)

    if filename == "AMP_dataset_features.csv":
        y = df["Labels"]
        X = pca.fit_transform(df.drop(columns = ["AASequences", "Labels"]))
    else:
        y = df["label"]
        X = pca.fit_transform(df.drop(columns = ["sequence", "label"]))
    
    # realno u ovoj skripti nista nije bitno osim ovog dijela
    # bitno je samo da je format datoteke koje program moze citati sljedeci
    # BROJ_KLASA BROJ_ZNACAJKI BROJ_REDAKA
    # zatim u odvojenim retcima minimalna i maksimalna vrijednost svake znacajke, te broj 0 ako je znacajka kontinuirana odnosno 1 ako je diskretna
    # nakon toga svi podaci gdje je oznaka koju trazimo prva vrijednost u redu
    with open(output_path + filename[:-4], "w+") as f:
        f.write("%d %d %d\n" % (len(y.values), len(X[0]), len(set(y.values))))
        for m, M in zip(X.min(axis=0), X.max(axis=0)):
            f.write("%f %f 0\n" % (m, M))
        for label, features in zip(y.values, X):
            f.write(str(label))
            for feature in features:
                f.write(" "+str(feature))
            f.write("\n")