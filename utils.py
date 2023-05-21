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

## for statistical tests
from scipy import stats

## for machine learning
from sklearn import model_selection, pipeline, feature_selection, metrics, linear_model, base

#########################################################################################################################
#########################################################################################################################

def column_check(df):
    unique_values = df.nunique()
    dtypes = df.dtypes
    examples = df.apply(lambda x: ", ".join(x.sample(n=5, random_state=random_state_).astype(str)))

    info = pd.DataFrame({"Unique Values": unique_values, "Data Types": dtypes, "Examples": examples})
    return info

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

def plot_histograms(df, selected_cols):
    num_rows = math.ceil(len(selected_cols) / 4)
    fig, axs = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, 4*num_rows))
    axs = axs.ravel()
    for i, col in enumerate(selected_cols):
        axs[i].hist(df[col], bins=50)
        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].grid()

    for i in range(len(selected_cols), len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.show()
    
def cat_dist(data, col_x, col_target, target_0_label, target_1_label, colour_number, label_x_offset=0.05, legend_loc="upper right", x_label_rotation=0):
    palette = ["husl", "hls", "Spectral", "coolwarm", "viridis"]
    sns.set_style("whitegrid")
    
    sorted_data = data[col_x].value_counts().sort_values(ascending=False).index
    hue_order = sorted(data[col_target].unique(), reverse=True)
    order = sorted_data.tolist()
    
    ax = sns.countplot(data=data, x=col_x, hue=col_target, order=order, hue_order=hue_order, palette=palette[colour_number])
    sns.despine(top=True, right=True)
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([target_1_label, target_0_label], loc=legend_loc)
    plt.grid(False)
    plt.xticks(rotation=x_label_rotation)
    
    for i in ax.patches:
        ax.text(i.get_x()+label_x_offset, i.get_height()+1000, int(i.get_height()), fontsize=11)

def cat_dist_top5(data, col_x, col_target, target_0_label, target_1_label, colour_number, label_x_offset=0.05, legend_loc="upper right", x_label_rotation=0):
    palette = ["husl","hls","Spectral","coolwarm", "viridis"]
    sns.set_style("whitegrid")
    
    sorted_data = data[col_x].value_counts().sort_values(ascending=False).index
    hue_order = sorted(data[col_target].unique(), reverse=True)
    order = sorted_data.tolist()
    
    top5 = data[col_x].value_counts().index[:5]
    ax = sns.countplot(data=data[data[col_x].isin(top5)], x=col_x, hue=col_target, order=top5, hue_order=hue_order,
                       palette=palette[colour_number])
    sns.despine(top=True, right=True)
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([target_1_label, target_0_label], loc=legend_loc)
    plt.grid(False)
    plt.title(f"Top 5 {col_x} categories", fontsize=18, pad=20)
    plt.xticks(rotation=x_label_rotation)

    for i in ax.patches:
        ax.text(i.get_x()+label_x_offset, i.get_height()+1000, int(i.get_height()), fontsize=11)
    
def target_dist(data, col_target, target_0_label, target_1_label):
    mpl.rcParams["font.size"] = 11
    r = data.groupby(col_target)[col_target].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[target_0_label, target_1_label], radius=1.5, autopct="%1.1f%%", shadow=True, startangle=45,
           colors=["#ff9999", "#66b3ff"])
    ax.set_aspect("equal")
    ax.set_frame_on(False)     

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

def col_to_drop(df, columns_list):
    df.drop(columns = columns_list, inplace = True)    

def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ":"))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df
    
def woe_discrete(df, cat_variabe_name, y_df):
    df = pd.concat([df[cat_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], "n_obs", "prop_good"]
    df["prop_n_obs"] = df["n_obs"] / df["n_obs"].sum()
    df["n_good"] = df["prop_good"] * df["n_obs"]
    df["n_bad"] = (1 - df["prop_good"]) * df["n_obs"]
    df["prop_n_good"] = df["n_good"] / df["n_good"].sum()
    df["prop_n_bad"] = df["n_bad"] / df["n_bad"].sum()
    df["WoE"] = np.log(df["prop_n_good"] / df["prop_n_bad"])
    df = df.sort_values(["WoE"])
    df = df.reset_index(drop = True)
    df["diff_prop_good"] = df["prop_good"].diff().abs()
    df["diff_WoE"] = df["WoE"].diff().abs()
    df["IV"] = (df["prop_n_good"] - df["prop_n_bad"]) * df["WoE"]
    df["IV"] = df["IV"].sum()
    return df

def woe_ordered_continuous(df, continuous_variabe_name, y_df):
    df = pd.concat([df[continuous_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], "n_obs", "prop_good"]
    df["prop_n_obs"] = df["n_obs"] / df["n_obs"].sum()
    df["n_good"] = df["prop_good"] * df["n_obs"]
    df["n_bad"] = (1 - df["prop_good"]) * df["n_obs"]
    df["prop_n_good"] = df["n_good"] / df["n_good"].sum()
    df["prop_n_bad"] = df["n_bad"] / df["n_bad"].sum()
    df["WoE"] = np.log(df["prop_n_good"] / df["prop_n_bad"])
    #df = df.sort_values(["WoE"])
    #df = df.reset_index(drop = True)
    df["diff_prop_good"] = df["prop_good"].diff().abs()
    df["diff_WoE"] = df["WoE"].diff().abs()
    df["IV"] = (df["prop_n_good"] - df["prop_n_bad"]) * df["WoE"]
    df["IV"] = df["IV"].sum()
    return df

def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    sns.set()
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE["WoE"]
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker = "o", linestyle = "--", color = "k")
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel("Weight of Evidence")
    plt.title(str("Weight of Evidence by " + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)

ref_categories = ["grade:G","home_ownership:MORTGAGE","verification_status:Not Verified","purpose:bus_edu_mov_ho_re_me_we_va",
                  "total_pymnt:>57777.58","int_rate:>26.06","out_prncp:>32160.38","mths_since_last_credit_pull_d:>182.6",
                  "mths_since_issue_d:>187.5","inq_last_6mths:>8","term:36","revol_util:>141.8","dti:>33.992","tot_cur_bal:>1597300.0",
                  "mths_since_earliest_cr_line:>524.5","total_rev_hi_lim:>290106.667","total_rec_int:>17409.168"]
                  
class WoE_Binning(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, X):
        self.X = X
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_new = X.loc[:, "grade:A": "grade:G"]
        
        X_new["home_ownership:OTHER_NONE_RENT"] = sum([X["home_ownership:OTHER"], X["home_ownership:NONE"], X["home_ownership:RENT"]])
        X_new["home_ownership:OWN"] = X.loc[:,"home_ownership:OWN"]
        X_new["home_ownership:MORTGAGE"] = X.loc[:,"home_ownership:MORTGAGE"]
          
        X_new = pd.concat([X_new, X.loc[:, "verification_status:Not Verified":"verification_status:Verified"]], axis = 1)
        
        X_new["purpose:bus_edu_mov_ho_re_me_we_va"] = sum([X["purpose:small_business"],X["purpose:educational"],X["purpose:moving"],
                                                           X["purpose:house"],X["purpose:renewable_energy"],X["purpose:medical"],
                                                           X["purpose:wedding"],X["purpose:vacation"]])
        X_new["purpose:other"] = X.loc[:,"purpose:other"]
        X_new["purpose:debt_consolidation"] = X.loc[:,"purpose:debt_consolidation"]
        X_new["purpose:home_impr__major_purch__car"] = sum([X["purpose:home_improvement"],X["purpose:major_purchase"],X["purpose:car"]])
        X_new["purpose:credit_card"] = X.loc[:,"purpose:credit_card"]

        X_new["total_pymnt:<=11555.516"] = np.where((X["total_pymnt"] <= 11555.516), 1, 0)
        X_new["total_pymnt:11555.516–23111.032"] = np.where((X["total_pymnt"] > 11555.516) & (X["total_pymnt"] <= 23111.032), 1, 0)
        X_new["total_pymnt:23111.032–57777.58"] = np.where((X["total_pymnt"] > 23111.032) & (X["total_pymnt"] <= 57777.58), 1, 0)
        X_new["total_pymnt:>57777.58"] = np.where((X["total_pymnt"] > 57777.58), 1, 0)
        
        X_new["int_rate:<=7.484"] = np.where((X["int_rate"] <= 7.484), 1, 0)
        X_new["int_rate:7.484–8.172"] = np.where((X["int_rate"] > 7.484) & (X["int_rate"] <= 8.172), 1, 0)
        X_new["int_rate:8.172–10.924"] = np.where((X["int_rate"] > 8.172) & (X["int_rate"] <= 10.924), 1, 0)
        X_new["int_rate:10.924–13.676"] = np.where((X["int_rate"] > 10.924) & (X["int_rate"] <= 13.676), 1, 0)
        X_new["int_rate:13.676–17.116"] = np.where((X["int_rate"] > 13.676) & (X["int_rate"] <= 17.116), 1, 0)
        X_new["int_rate:17.116–20.556"] = np.where((X["int_rate"] > 17.116) & (X["int_rate"] <= 20.556), 1, 0)
        X_new["int_rate:20.556–26.06"] = np.where((X["int_rate"] > 20.556) & (X["int_rate"] <= 26.06), 1, 0)
        X_new["int_rate:>26.06"] = np.where((X["int_rate"] > 26.06), 1, 0)
        
        X_new["out_prncp:<=3216.038"] = np.where((X["out_prncp"] <= 3216.038), 1, 0)
        X_new["out_prncp:3216.038–9648.114"] = np.where((X["out_prncp"] > 3216.038) & (X["out_prncp"] <= 9648.114), 1, 0)
        X_new["out_prncp:9648.114–16080.19"] = np.where((X["out_prncp"] > 9648.114) & (X["out_prncp"] <= 16080.19), 1, 0)
        X_new["out_prncp:16080.19–32160.38"] = np.where((X["out_prncp"] > 16080.19) & (X["out_prncp"] <= 32160.38), 1, 0)
        X_new["out_prncp:>32160.38"] = np.where((X["out_prncp"] > 32160.38), 1, 0)
        
        X_new["mths_since_last_credit_pull_d:missing"] = np.where(X["mths_since_last_credit_pull_d"].isnull(), 1, 0)
        X_new["mths_since_last_credit_pull_d:<=99.4"] = np.where((X["mths_since_last_credit_pull_d"] <= 99.4), 1, 0)
        X_new["mths_since_last_credit_pull_d:99.4–109.8"] = np.where((X["mths_since_last_credit_pull_d"] > 99.4) & (X["mths_since_last_credit_pull_d"] <= 109.8), 1, 0)
        X_new["mths_since_last_credit_pull_d:109.8–182.6"] = np.where((X["mths_since_last_credit_pull_d"] > 109.8) & (X["mths_since_last_credit_pull_d"] <= 182.6), 1, 0)
        X_new["mths_since_last_credit_pull_d:>182.6"] = np.where((X["mths_since_last_credit_pull_d"] > 182.6), 1, 0)
        
        X_new["mths_since_issue_d:<=106.5"] = np.where((X["mths_since_issue_d"] <= 106.5), 1, 0)
        X_new["mths_since_issue_d:106.5–111.0"] = np.where((X["mths_since_issue_d"] > 106.5) & (X["mths_since_issue_d"] <= 111.0), 1, 0)
        X_new["mths_since_issue_d:111.0–115.5"] = np.where((X["mths_since_issue_d"] > 111.0) & (X["mths_since_issue_d"] <= 115.5), 1, 0)
        X_new["mths_since_issue_d:115.5–120.0"] = np.where((X["mths_since_issue_d"] > 115.5) & (X["mths_since_issue_d"] <= 120.0), 1, 0)
        X_new["mths_since_issue_d:120.0–129.0"] = np.where((X["mths_since_issue_d"] > 120.0) & (X["mths_since_issue_d"] <= 129.0), 1, 0)
        X_new["mths_since_issue_d:129.0–142.5"] = np.where((X["mths_since_issue_d"] > 129.0) & (X["mths_since_issue_d"] <= 142.5), 1, 0)
        X_new["mths_since_issue_d:142.5–187.5"] = np.where((X["mths_since_issue_d"] > 142.5) & (X["mths_since_issue_d"] <= 187.5), 1, 0)
        X_new["mths_since_issue_d:>187.5"] = np.where((X["mths_since_issue_d"] > 187.5), 1, 0)

        X_new["inq_last_6mths:missing"] = np.where(X["inq_last_6mths"].isnull(), 1, 0)
        X_new["inq_last_6mths:0"] = np.where((X["inq_last_6mths"] == 0), 1, 0)
        X_new["inq_last_6mths:1"] = np.where((X["inq_last_6mths"] == 1), 1, 0)
        X_new["inq_last_6mths:2"] = np.where((X["inq_last_6mths"] == 2), 1, 0)
        X_new["inq_last_6mths:3-8"] = np.where((X["inq_last_6mths"] >= 3) & (X["inq_last_6mths"] <= 8), 1, 0)
        X_new["inq_last_6mths:>8"] = np.where((X["inq_last_6mths"] > 8), 1, 0)

        X_new["term:36"] = np.where((X["term"] == 36), 1, 0)
        X_new["term:60"] = np.where((X["term"] == 60), 1, 0)

        X_new["revol_util:missing"] = np.where(X["revol_util"].isnull(), 1, 0)
        X_new["revol_util:<=28.36"] = np.where((X["revol_util"] <= 28.36), 1, 0)
        X_new["revol_util:28.36–56.72"] = np.where((X["revol_util"] > 28.36) & (X["revol_util"] <= 56.72), 1, 0)
        X_new["revol_util:56.72–85.08"] = np.where((X["revol_util"] > 56.72) & (X["revol_util"] <= 85.08), 1, 0)
        X_new["revol_util:85.08–141.8"] = np.where((X["revol_util"] > 85.08) & (X["revol_util"] <= 141.8), 1, 0)
        X_new["revol_util:>141.8"] = np.where((X["revol_util"] > 141.8), 1, 0)

        X_new["dti:<=5.998"] = np.where((X["dti"] <= 5.998), 1, 0)
        X_new["dti:5.998–11.997"] = np.where((X["dti"] > 5.998) & (X["dti"] <= 11.997), 1, 0)
        X_new["dti:11.997–15.996"] = np.where((X["dti"] > 11.997) & (X["dti"] <= 15.996), 1, 0)
        X_new["dti:15.996–21.995"] = np.where((X["dti"] > 15.996) & (X["dti"] <= 21.995), 1, 0)
        X_new["dti:21.995–27.993"] = np.where((X["dti"] > 21.995) & (X["dti"] <= 27.993), 1, 0)
        X_new["dti:27.993–33.992"] = np.where((X["dti"] > 27.993) & (X["dti"] <= 33.992), 1, 0)
        X_new["dti:>33.992"] = np.where((X["dti"] > 33.992), 1, 0)        
        
        X_new["tot_cur_bal:missing"] = np.where(X["tot_cur_bal"].isnull(), 1, 0)
        X_new["tot_cur_bal:<=159730.0"] = np.where((X["tot_cur_bal"] <= 159730.0), 1, 0)
        X_new["tot_cur_bal:159730.0–319460.0"] = np.where((X["tot_cur_bal"] > 159730.0) & (X["tot_cur_bal"] <= 319460.0), 1, 0)
        X_new["tot_cur_bal:319460.0–1597300.0"] = np.where((X["tot_cur_bal"] > 319460.0) & (X["tot_cur_bal"] <= 1597300.0), 1, 0)
        X_new["tot_cur_bal:>1597300.0"] = np.where((X["tot_cur_bal"] > 1597300.0), 1, 0)

        X_new["mths_since_earliest_cr_line:missing"] = np.where(X["mths_since_earliest_cr_line"].isnull(), 1, 0)
        X_new["mths_since_earliest_cr_line:<=216.1"] = np.where((X["mths_since_earliest_cr_line"] <= 216.1), 1, 0)
        X_new["mths_since_earliest_cr_line:216.1–267.5"] = np.where((X["mths_since_earliest_cr_line"] > 216.1) & (X["mths_since_earliest_cr_line"] <= 267.5), 1, 0)
        X_new["mths_since_earliest_cr_line:267.5–318.9"] = np.where((X["mths_since_earliest_cr_line"] > 267.5) & (X["mths_since_earliest_cr_line"] <= 318.9), 1, 0)
        X_new["mths_since_earliest_cr_line:318.9–370.3"] = np.where((X["mths_since_earliest_cr_line"] > 318.9) & (X["mths_since_earliest_cr_line"] <= 370.3), 1, 0)
        X_new["mths_since_earliest_cr_line:370.3–421.7"] = np.where((X["mths_since_earliest_cr_line"] > 370.3) & (X["mths_since_earliest_cr_line"] <= 421.7), 1, 0)
        X_new["mths_since_earliest_cr_line:421.7–524.5"] = np.where((X["mths_since_earliest_cr_line"] > 421.7) & (X["mths_since_earliest_cr_line"] <= 524.5), 1, 0)
        X_new["mths_since_earliest_cr_line:>524.5"] = np.where((X["mths_since_earliest_cr_line"] > 524.5), 1, 0)
        
        X_new["total_rev_hi_lim:missing"] = np.where(X["total_rev_hi_lim"].isnull(), 1, 0)
        X_new["total_rev_hi_lim:<=26373.333"] = np.where((X["total_rev_hi_lim"] <= 26373.333), 1, 0)
        X_new["total_rev_hi_lim:26373.333–52746.667"] = np.where((X["total_rev_hi_lim"] > 26373.333) & (X["total_rev_hi_lim"] <= 52746.667), 1, 0)
        X_new["total_rev_hi_lim:52746.667–79120.0"] = np.where((X["total_rev_hi_lim"] > 52746.667) & (X["total_rev_hi_lim"] <= 79120.0), 1, 0)
        X_new["total_rev_hi_lim:79120.0–290106.667"] = np.where((X["total_rev_hi_lim"] > 79120.0) & (X["total_rev_hi_lim"] <= 290106.667), 1, 0)
        X_new["total_rev_hi_lim:>290106.667"] = np.where((X["total_rev_hi_lim"] > 290106.667), 1, 0)
        
        X_new["total_rec_int:<=725.382"] = np.where((X["total_rec_int"] <= 725.382), 1, 0)
        X_new["total_rec_int:725.382–1450.764"] = np.where((X["total_rec_int"] > 725.382) & (X["total_rec_int"] <= 1450.764), 1, 0)
        X_new["total_rec_int:1450.764–2176.146"] = np.where((X["total_rec_int"] > 1450.764) & (X["total_rec_int"] <= 2176.146), 1, 0)
        X_new["total_rec_int:2176.146–2901.528"] = np.where((X["total_rec_int"] > 2176.146) & (X["total_rec_int"] <= 2901.528), 1, 0)
        X_new["total_rec_int:2901.528–3626.91"] = np.where((X["total_rec_int"] > 2901.528) & (X["total_rec_int"] <= 3626.91), 1, 0)
        X_new["total_rec_int:3626.91–4352.292"] = np.where((X["total_rec_int"] > 3626.91) & (X["total_rec_int"] <= 4352.292), 1, 0)
        X_new["total_rec_int:4352.292–5803.056"] = np.where((X["total_rec_int"] > 4352.292) & (X["total_rec_int"] <= 5803.056), 1, 0)
        X_new["total_rec_int:5803.056–17409.168"] = np.where((X["total_rec_int"] > 5803.056) & (X["total_rec_int"] <= 17409.168), 1, 0)
        X_new["total_rec_int:>17409.168"] = np.where((X["total_rec_int"] > 17409.168), 1, 0)
        
        X_new.drop(columns = ref_categories, inplace = True)
        return X_new
    
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
    