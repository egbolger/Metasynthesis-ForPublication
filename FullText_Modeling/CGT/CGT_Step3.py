# DOING THE SAME AS FOR THE ABSTRACTS
## Point of this step: Implement Supervised ML model to assess human coded labels. 

# We compare the performance of using SMOTE and SMOTETomek for imbalanced classes
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
plt.rcParams.update({'font.size': 14})

random_state = 1733
    
################ MULTI CLASS LOGISTIC REGRESSION ###############
        #  "multi-class logistic regression applies the softmax function to assign a probability to each possible answer, ensuring the probabilities add up to 1. softmax creates a probability distribution across multiple classes, ensuring the sum is always 1.  The goal is to minimize the cross-entropy loss, which measures the difference between the predicted probabilities and the actual classes."
        # https://medium.com/@jshaik2452/multi-class-logistic-regression-a-friendly-guide-to-classifying-the-many-4a590c2e6c26
    # Use stratify option for imbalanced classes - https://medium.com/@aymuosmukherjee/why-do-we-use-stratify-in-train-test-split-e3eb296a5494

def run_logistic_regression(X_data, y_data, save_str):
    # Define a pipeline to pass to the fitting routine
    pipeline = Pipeline([
    ('smote', SMOTETomek( random_state=random_state, n_jobs=-1, smote=SMOTE(k_neighbors=3))), 
    ('classifier', LogisticRegressionCV(solver='lbfgs', max_iter=500, random_state=random_state)) # tests a range of parameters
    ])

    # Define Stratified K Folds - cross_val_predict can't take repeats and we need that function for metrics and confusion matrix
    rskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Do cross validation 
    y_pred = cross_val_predict(pipeline, X_data, y_data, cv=rskf, n_jobs=-1)

    # Return Model Outputs/Parameters
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    print(classification_report(y_data, y_pred))
    print(cm)
    print(cm_normalized)

    build_confusion_matrix(cm_normalized, labels, "Logistic Regression", save_str+"logisticregression.png")

################ MULTI CLASS SVMs ###############
# Okay, doing the same process for Multi-Class SVMs - holdout, implementing cv, metrics, confusion matrix
    # Technically, SVC for support vector classifier
def run_svc(X_data, y_data, save_str):
    pipeline = Pipeline([
        ('smote', SMOTETomek(random_state=random_state, smote=SMOTE(k_neighbors=3))),  #  ensures all features contribute equally
        ('svc', SVC(class_weight='balanced', probability=True)) # handle imbalanced classes
    ])
    param_grid = {
        'svc__kernel': ['linear', 'rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto'],  
        'svc__decision_function_shape': ['ovo', 'ovr'] # only relevant for RBF kernel
            # https://wadhwatanya1234.medium.com/multi-class-classification-one-vs-all-one-vs-one-993dd23ae7ca
    }
    rskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=rskf, 
        scoring='f1_weighted', 
        n_jobs=-1,
        # verbose=2,
        refit=True 
    )
    grid_search.fit(X_data, y_data)
    print("Best parameters:", grid_search.best_params_) #{'svc__C': 0.1, 'svc__decision_function_shape': 'ovo', 'svc__gamma': 'scale', 'svc__kernel': 'linear'}
    print("Best cross-validation score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred = cross_val_predict(best_model, X_data, y_data, cv=rskf, n_jobs=-1)

    # Return Model Outputs/Parameters
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    print(classification_report(y_data, y_pred))
    print(cm)
    print(cm_normalized)
    print("y_pred:", np.unique(y_pred))

    build_confusion_matrix(cm_normalized, labels, "Support Vector Classifier", save_str+"svc.png")
   

   
 ### CONFUSION MATRIX 
        # row wise normalization to see which classes the model struggles with (since classes imbalances)
    # Calculating by hand so i can make the plot better
def build_confusion_matrix(cm_normalized, labels, model_str, save_str):
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)
    disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # remove default values
    for txt in ax.texts:
        txt.set_visible(False)

    # Annotate manually to remove decimals for zeros
    n_classes = len(labels)
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_normalized[i, j]
            if value == 0:
                text_str = "0"
            else:
                text_str = f"{value:.2f}"
            color = "white" if value >= 0.5 else "black"

            ax.text(j, i, text_str,
                    ha="center", va="center", color=color, fontsize=10)
    plt.tight_layout()
    # plt.title("Multi-Class " + model_str + "\n" + data_str + " Dataset", fontsize = 20, fontweight='bold')
    plt.title("Multi-Class " + model_str + "\n" + "Full Text" + " Dataset", fontsize = 20, fontweight='bold')

    plt.xlabel("Predicted Label", fontsize = 17, fontweight='bold')
    plt.ylabel("True Label", fontsize = 17, fontweight='bold')
    plt.savefig(save_str, dpi=500, bbox_inches='tight', pad_inches=0.3)
    plt.close()

 

def main():
    # READ in Final Chosen Data Set for Full Text data - Pink (Removed - Parameters 15_0.2_5_3)
    data_path = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/datafilesforCGT/removed/finalset_Pink/"     # DATA PATH - has cluster data and human codes data
    fig_path = "/mnt/research/NLP-Lit-Review/bolger/metasyn/figures/removed/finalset_Pink/"     # FIGURE PATH
    
    data_file = pd.read_parquet(data_path + "ftremoved_embedding15_0.2_5_3.parquet")
    # Add in description column 
        # dictionary, create new column, add data based on dictionary 
    descriptions_dict = {-1: "Noise", 0: "Developmental Classes / Pathways", 1: "Pedagogical Practices to Promote Equity and Inclusion", 2: "Applied Engineering Educational Approaches", 3: "Faculty Roles for Institutional Change", 4: "TA Training", 5: "Educational Technologies", 6: "Active Learning Strategies in Biology", 7:"Evaluating Active Learning Strategies", 8 : "Assessment and Outcomes", 9 : "Scaling Evidence-Based Pedagogies"}
    data_file["description"] = data_file["labels"].map(descriptions_dict)

    
    # Prepping data for supervised learning 
    nonoise_data = data_file[data_file["labels"]!=-1] # remove noise (labels assigned by CGT)
    X_data = np.vstack(nonoise_data["ftremoved_embedding"].values) # shape 174,1024
    y_data = nonoise_data["description"] # Using labels assigned by humans, sklearn does encoding to ints for me


    # RUN LR
    # run_logistic_regression(X_data, y_data, fig_path)


    # RUN SVC
    run_svc(X_data, y_data, fig_path)
  

if __name__ == '__main__':
    main()


        
    