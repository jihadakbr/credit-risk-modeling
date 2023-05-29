## warning
import warnings
warnings.filterwarnings("ignore")

## random state
random_state_ = 42

## for data
import numpy as np
import pandas as pd
import math

## for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid")

## for statistical tests
from scipy import stats

## for a pipeline
from imblearn.pipeline import Pipeline

## for machine learning
from sklearn import model_selection, feature_selection, metrics, set_config, base, linear_model, tree, naive_bayes, ensemble, discriminant_analysis
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

#########################################################################################################################
#########################################################################################################################

def column_check(df):
    unique_values = df.nunique()
    dtypes = df.dtypes
    examples = df.apply(lambda x: ", ".join(x.sample(n=5, random_state=random_state_).astype(str)))

    info = pd.DataFrame({"Unique Values": unique_values, "Data Types": dtypes, "Examples": examples})
    return info

def col_to_drop(df, columns_list):
    df.drop(columns = columns_list, inplace = True) 

def object_input_check(df, columns):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        print(f"Processing column: {col}")
        rows_with_number = []

        if col in df.columns:
            for index, value in enumerate(df[col]):
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                    rows_with_number.append((index, str(value)))
            if rows_with_number:
                print("Number found in the 'object' column.")
                print("Rows with number values: (index from 0)")
                for row in rows_with_number:
                    print(f"Row {row[0]}: {row[1]}")
            else:
                print("No number found in the 'object' column.")
        else:
            print(f"Column '{col}' not found in the DataFrame.")
        print("")

def emp_length_converter(df, column):
    df[column] = df[column].str.replace("\+ years", "")
    df[column] = df[column].str.replace("< 1 year", str(0))
    df[column] = df[column].str.replace(" years", "")
    df[column] = df[column].str.replace(" year", "")
    df[column] = pd.to_numeric(df[column])
    df[column].fillna(value = 0, inplace = True)

# convert date columns to datetime format and create a new column as a difference between today and the respective date
def date_columns(df, column):
    # store current date
    today_date = pd.to_datetime("2023-05-20")
    
    # convert to datetime format
    df[column] = pd.to_datetime(df[column], format = "%b-%y")
    
    # calculate the difference in months and add to a new column
    df["mths_since_" + column] = round(pd.to_numeric((today_date - df[column]) / np.timedelta64(1, "M")))
    
    # make any resulting -ve values to be equal to the max date
    df["mths_since_" + column] = df["mths_since_" + column].apply(lambda x: df["mths_since_" + column].max() if x < 0 else x)
    
    # drop the original date column
    df.drop(columns = [column], inplace = True)    

# function to remove "months" string from the "term" column and convert it to numeric
def loan_term_converter(df, column):
    df[column] = pd.to_numeric(df[column].str.replace(" months", ""))    
    
def count_outliers(data, columns):
    outlier_percentage = {}
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        upper_limit = q3 + 1.5 * (q3-q1)
        lower_limit = q1 - 1.5 * (q3-q1)

        col_outliers = (data[col] < lower_limit) | (data[col] > upper_limit)
        outlier_percentage[col] = f"{col_outliers.mean() * 100:.2f}%"

    return outlier_percentage

def outliers_graph(data, cols):
    num_cols = len(cols)
    num_rows = math.ceil(num_cols / 8)
    
    fig, axs = plt.subplots(num_rows, 8, figsize=(24, 6*num_rows))
    fig.tight_layout(w_pad=5.0, h_pad=3.0)
    axs = axs.flatten()
    
    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted")
    
    for i, col in enumerate(cols):
        sns.boxplot(y=data[col], ax=axs[i], color=custom_palette[i % len(custom_palette)])
        axs[i].set_ylabel(col, fontsize=14)
        axs[i].grid(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
    
    for j in range(num_cols, num_rows*8):
        fig.delaxes(axs[j])
    
    return fig, axs

def plot_histograms(df, selected_cols):
    num_rows = math.ceil(len(selected_cols) / 4)
    fig, axs = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 4*num_rows))
    axs = axs.ravel()
    for i, col in enumerate(selected_cols):
        axs[i].hist(df[col], bins=50)
        axs[i].set_title(None)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel("Frequency")
        axs[i].grid(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)

    for i in range(len(selected_cols), len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.show()

def plot_kdeplots(df, selected_cols, target, x_label=None, titles=None):
    num_rows = math.ceil(len(selected_cols) / 4)
    fig, axs = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 4*num_rows))
    axs = axs.ravel()

    x_limits = []

    for i, col in enumerate(selected_cols):
        
        sns.kdeplot(data=df.loc[df[target] == 0, col], ax=axs[i], label="Defaulter", alpha=0.5, shade=True)
        sns.kdeplot(data=df.loc[df[target] == 1, col], ax=axs[i], label="Non-Defaulter", alpha=0.5, shade=True)

        if titles:
            axs[i].set_title(titles[i])
        
        if x_label:
            axs[i].set_xlabel(x_label[i])
            
        sns.despine(top=True, right=True)
        axs[i].set_ylabel('Density')
        axs[i].grid(False)
        axs[i].legend()
        
        x_limits.append(axs[i].get_xlim())

    for i in range(len(selected_cols), len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.show()

def cat_dist(data, col_x, col_target, target_0_label, target_1_label, colour_number, label_x_offset=0.05, legend_loc="upper right", x_label_rotation=0, custom=None):
    palette = ["husl", "hls", "Spectral", "coolwarm", "viridis"]
    sns.set_style("whitegrid")

    sorted_data = data[col_x].value_counts().sort_values(ascending=False).index
    hue_order = sorted(data[col_target].unique(), reverse=True)
    order = sorted_data.tolist()

    if custom == "vertical":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    # First bar chart
    sns.countplot(data=data, x=col_x, hue=col_target, order=order, hue_order=hue_order, palette=palette[colour_number], ax=ax1)
    sns.despine(top=True, right=True)
    ax1.set_xlabel(f"{col_x}", fontsize=15)
    ax1.set_ylabel("Count", fontsize=15)
    ax1.legend([target_1_label, target_0_label], loc=legend_loc)
    ax1.grid(False)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=x_label_rotation)

    for i in ax1.patches:
        ax1.text(i.get_x()+label_x_offset, i.get_height()+1000, int(i.get_height()), fontsize=11)

    # Second bar chart
    if custom == "vertical":
        ax2 = plt.subplot(212)
    else:
        ax2 = plt.subplot(122)
    
    order2 = data[col_x].value_counts().sort_values(ascending=False).index
    (data[col_x][data[col_target] == 1].value_counts() / data[col_x][data[col_target] == 0].value_counts()).reindex(order2).plot(kind="bar", ax=ax2, color="black")

    sns.despine(top=True, right=True)
    ax2.set_xlabel(f"{col_x}", fontsize=15)
    ax2.set_ylabel("Ratio", fontsize=15)
    ax2.grid(False)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=x_label_rotation)

    for i, v in enumerate(data[col_x][data[col_target] == 1].value_counts() / data[col_x][data[col_target] == 0].value_counts().reindex(order2)):
        ax2.text(i, v + 0.05, str(round(v,2)), ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.show()
    
def target_dist(data, col_target, target_0_label, target_1_label):
    mpl.rcParams["font.size"] = 11
    r = data.groupby(col_target)[col_target].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[target_0_label, target_1_label], radius=1.5, autopct="%1.1f%%", shadow=True, startangle=45,
           colors=["#ff9999", "#66b3ff"])
    ax.set_aspect("equal")
    ax.set_frame_on(False)

def chi_square_test(X_train_cat, y_train):
    chi2_check = {}

    for column in X_train_cat:
        chi, p, dof, ex = stats.chi2_contingency(pd.crosstab(y_train, X_train_cat[column]))

        # interpret test-statistic
        prob = 0.95
        critical = stats.chi2.ppf(prob, dof)
        if abs(chi) >= critical:
            result_stat = "Dependent (reject H0)"
        else:
            result_stat = "Independent (fail to reject H0)"

        # interpret p-value
        alpha = 1.0 - prob
        if p <= alpha:
            result_p = "Dependent (reject H0)"
        else:
            result_p = "Independent (fail to reject H0)"

        chi2_check.setdefault("Feature", []).append(column)
        chi2_check.setdefault("p-value", []).append(round(p, 10))
        chi2_check.setdefault("interpret test-statistic", []).append(result_stat)
        chi2_check.setdefault("interpret p-value", []).append(result_p)

    chi2_result = pd.DataFrame(data=chi2_check)
    chi2_result.sort_values(by=["p-value"], ascending=True, ignore_index=True, inplace=True)
    
    return chi2_result    
    
def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], drop_first=True, prefix = col, prefix_sep = ":"))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df

def calculate_ANOVA_F_table(X_train_num, y_train):
    F_statistic, p_values = feature_selection.f_classif(X_train_num, y_train)

    ANOVA_F_table = pd.DataFrame(data={"Numerical_Feature": X_train_num.columns.values,
                                       "F-Score": F_statistic,
                                       "p values": p_values.round(decimals=10)})
    
    ANOVA_F_table.sort_values(by=["F-Score"], ascending=False, ignore_index=True, inplace=True)

    alpha=0.05
    ANOVA_F_table["Result"] = ANOVA_F_table["p values"].apply(lambda p: "Dependent (reject H0)" if p <= alpha \
                                                              else "Independent (fail to reject H0)")
    return ANOVA_F_table

def corrr(data):
    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(10,10))
    sns.heatmap(data, annot=True, mask=mask, cmap='coolwarm', annot_kws={"size": 7})
    sns.despine(left=True, bottom=True)
    plt.grid(False)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Correlation Matrix", fontsize=15, fontweight='bold')
    plt.show()

def model_val(model, model_df, X_train, y_train, numerical_ix, categorical_ix):
    
    t = [("num", RobustScaler(), numerical_ix),
         ("cat", "passthrough", categorical_ix)]
    
    col_transform = ColumnTransformer(transformers=t)
    
    steps = [("prep", col_transform), ("model", model)]
    pipeline = Pipeline(steps=steps)
    
    cv = model_selection.RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=random_state_)

    scores = model_selection.cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, error_score="raise")
    pipeline.fit(X_train, y_train)
    
    print(f"{model} - X_train, y_train - AUROC: %.2f" % np.mean(scores))
    print(f"{model} - X_train, y_train - GINI: %.2f" % (np.mean(scores)*2 - 1))
    
    model_df[model] = round(np.mean(scores), 2)
    display(pipeline)
    
    return pipeline
    
def evaluation(df, threshold, y_actual, y_predicted, y_proba, recall, precision):
    
    results = {
        "Threshold": threshold,
        "Accuracy": metrics.accuracy_score(df[y_actual], df[y_predicted]),
        "Precision": metrics.precision_score(df[y_actual], df[y_predicted]),
        "Recall": metrics.recall_score(df[y_actual], df[y_predicted]),
        "F1": metrics.f1_score(df[y_actual], df[y_predicted]),
        "AUROC": metrics.roc_auc_score(df[y_actual], df[y_proba]),
        "Gini": metrics.roc_auc_score(df[y_actual], df[y_proba]) * 2 - 1,
        "AUCPR": metrics.auc(recall, precision),
    }
    return results

def confusion_matrix_mod(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2%", ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title('Confusion Matrix')

    True_Pos = cm[1, 1]
    True_Neg = cm[0, 0]
    False_Pos = cm[0, 1]
    False_Neg = cm[1, 0]

    plt.show()

    return True_Pos, True_Neg, False_Pos, False_Neg
    