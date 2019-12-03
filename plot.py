import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def equalize_y(left_axis, right_axis):
    if left_axis.get_ylim()[1] > right_axis.get_ylim()[1]:
        right_axis.set_ylim(left_axis.get_ylim())
    else:
        left_axis.set_ylim(right_axis.get_ylim())

def plot_binary(left_axis, right_axis, parameter):
    left_axis.bar(
        [0, 1],
        [
            churn_df[
                (churn_df["Churn"] == "No") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][0])
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "No") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][1])
            ].shape[0]
        ],
        color="green",
        alpha=.8
    )
    right_axis.bar(
        [0, 1],
        [
            churn_df[
                (churn_df["Churn"] == "Yes") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][0])
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "Yes") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][1])
            ].shape[0]
        ],
        color="red",
        alpha=.8
    )
    equalize_y(left_axis, right_axis)

    for axis in [left_axis, right_axis]:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.set_ylabel(
            parameter_dict[parameter]["ylabel"],
            fontdict={
                'fontsize': 15,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            },
            labelpad=15
        )
        axis.tick_params(
            axis='y',
            which='both',
            labelsize=12
        )
        axis.tick_params(
            axis='x',
            which='both',
            labelsize=12
        )
        labels = [item.get_text() for item in axis.get_xticklabels()]
        labels[1] = parameter_dict[parameter]["tick_labels"][0]
        labels[3] = parameter_dict[parameter]["tick_labels"][1]
        axis.set_xticklabels(labels)
        axis.tick_params(
            axis='y',
            which='both',
            labelsize=12
        )
        axis.tick_params(
            axis='x',
            which='both',
            length=0,
            labelsize=12
        )

def plot_tertiary(left_axis, right_axis, parameter):
    left_axis.bar(
        [0, 1, 2],
        [
            churn_df[
                (churn_df["Churn"] == "No") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][0])
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "No") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][1])
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "No") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][2])
            ].shape[0]
        ],
        color="green",
        alpha=.8
    )
    right_axis.bar(
        [0, 1, 2],
        [
            churn_df[
                (churn_df["Churn"] == "Yes") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][0])
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "Yes") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][1])
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "Yes") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][2])
            ].shape[0]
        ],
        color="red",
        alpha=.8
    )
    equalize_y(left_axis, right_axis)

    for axis in [left_axis, right_axis]:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.set_ylabel(
            parameter_dict[parameter]["ylabel"],
            fontdict={
                'fontsize': 15,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            },
            labelpad=15
        )
        axis.tick_params(
            axis='y',
            which='both',
            labelsize=12
        )
        axis.tick_params(
            axis='x',
            which='both',
            labelsize=12
        )
        labels = [item.get_text() for item in axis.get_xticklabels()]
        labels[2] = parameter_dict[parameter]["tick_labels"][0]
        labels[4] = parameter_dict[parameter]["tick_labels"][1]
        labels[6] = parameter_dict[parameter]["tick_labels"][2]
        axis.set_xticklabels(labels)
        axis.tick_params(
            axis='y',
            which='both',
            labelsize=12
        )
        axis.tick_params(
            axis='x',
            which='both',
            length=0,
            labelsize=12
        )

def plot_linear(left_axis, right_axis, parameter):
    churn_df[churn_df["Churn"] == "No"][parameter].plot(
        kind='hist',
        color="green",
        alpha=.8,
        ax=left_axis
    )
    churn_df[churn_df["Churn"] == "Yes"][parameter].plot(
        kind='hist',
        color="red",
        alpha=.8,
        ax=right_axis
    )
    equalize_y(left_axis, right_axis)

    for axis in [left_axis, right_axis]:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.set_ylabel(
            parameter_dict[parameter]["ylabel"],
            fontdict={
                'fontsize': 15,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            },
            labelpad=15
        )
        axis.tick_params(
            axis='y',
            which='both',
            labelsize=12
        )
        axis.tick_params(
            axis='x',
            which='both',
            labelsize=12
        )

def plot_demographics():
    fig, ax = plt.subplots(5, 2, figsize=(10, 3*5))
    ax[0][0].set_title(
        "No Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    ax[0][1].set_title(
        "Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    plot_binary(
        left_axis=ax[0][0],
        right_axis=ax[0][1],
        parameter="gender"
    )
    plot_binary(
        left_axis=ax[1][0],
        right_axis=ax[1][1],
        parameter="SeniorCitizen"
    )
    plot_binary(
        left_axis=ax[2][0],
        right_axis=ax[2][1],
        parameter="Partner"
    )
    plot_binary(
        left_axis=ax[3][0],
        right_axis=ax[3][1],
        parameter="Dependents"
    )
    plot_linear(
        left_axis=ax[4][0],
        right_axis=ax[4][1],
        parameter="tenure"
    )
    plt.tight_layout()
    plt.savefig(
        "Demographics.png", 
        bbox_inches="tight"
    )
    plt.close()

def plot_phone():
    fig, ax = plt.subplots(2, 2, figsize=(10, 3*2))
    ax[0][0].set_title(
        "No Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    ax[0][1].set_title(
        "Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    plot_binary(
        left_axis=ax[0][0],
        right_axis=ax[0][1],
        parameter="PhoneService"
    )
    plot_tertiary(
        left_axis=ax[1][0],
        right_axis=ax[1][1],
        parameter="MultipleLines"
    )
    plt.tight_layout()
    plt.savefig(
        "PhoneService.png", 
        bbox_inches="tight"
    )
    plt.close()

def plot_internet():
    fig, ax = plt.subplots(3, 2, figsize=(10, 3*3))
    ax[0][0].set_title(
        "No Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    ax[0][1].set_title(
        "Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    plot_tertiary(
        left_axis=ax[0][0],
        right_axis=ax[0][1],
        parameter="InternetService"
    )
    plot_binary(
        left_axis=ax[1][0],
        right_axis=ax[1][1],
        parameter="OnlineSecurity"
    )
    plot_binary(
        left_axis=ax[2][0],
        right_axis=ax[2][1],
        parameter="OnlineBackup"
    )
    plt.tight_layout()
    plt.savefig(
        "InternetService.png", 
        bbox_inches="tight"
    )
    plt.close()

def plot_support():
    fig, ax = plt.subplots(2, 2, figsize=(10, 3*2))
    ax[0][0].set_title(
        "No Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    ax[0][1].set_title(
        "Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    plot_binary(
        left_axis=ax[0][0],
        right_axis=ax[0][1],
        parameter="DeviceProtection"
    )
    plot_binary(
        left_axis=ax[1][0],
        right_axis=ax[1][1],
        parameter="TechSupport"
    )
    plt.tight_layout()
    plt.savefig(
        "Support.png", 
        bbox_inches="tight"
    )
    plt.close()

def plot_streaming():
    fig, ax = plt.subplots(2, 2, figsize=(10, 3*2))
    ax[0][0].set_title(
        "No Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    ax[0][1].set_title(
        "Churn",
        fontdict={
                'fontsize': 22,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            }
    )
    plot_binary(
        left_axis=ax[0][0],
        right_axis=ax[0][1],
        parameter="StreamingTV"
    )
    plot_binary(
        left_axis=ax[1][0],
        right_axis=ax[1][1],
        parameter="StreamingMovies"
    )
    plt.tight_layout()
    plt.savefig(
        "Streaming.png", 
        bbox_inches="tight"
    )
    plt.close()


os.environ['KMP_DUPLICATE_LIB_OK']='True'

parameter_dict = {
    "gender": {
        "df":["Female", "Male"],
        "tick_labels":["Female", "Male"],
        "ylabel":"Gender"
        },
    "SeniorCitizen": {
        "df":[0, 1],
        "tick_labels":["Not Senior Citizen", "Senior Citizen"],
        "ylabel":"Senior Citizen"
        },
    "Partner": {
        "df":["No", "Yes"],
        "tick_labels":["No Partner", "Has Partner"],
        "ylabel":"Partner"
    },
    "Dependents": {
        "df":["No", "Yes"],
        "tick_labels":["No Dependents", "Has Dependents"],
        "ylabel":"Dependents"
    },
    "tenure": {
        "ylabel":"Tenure"
    },
    "PhoneService": {
        "df":["No", "Yes"],
        "tick_labels":["No Phone\nService", "Has Phone\nService"],
        "ylabel":"Phone Service"
    },
    "MultipleLines": {
        "df":['No phone service', 'No', 'Yes'],
        "tick_labels":["No Phone\nService", "1 Line", "Multiple Lines"],
        "ylabel":"Multiple Lines"
    },
    "InternetService": {
        "df":["No", 'DSL', 'Fiber optic'],
        "tick_labels":["No Internet\nService", "DSL", "Fiber Optic"],
        "ylabel":"Internet Service"
    },
    "OnlineSecurity": {
        "df":["No", "Yes"],
        "tick_labels":["No Online\nSecurity", "Has Online\nSecurity"],
        "ylabel":"Online Security"
    },
    "OnlineBackup": {
        "df":["No", "Yes"],
        "tick_labels":["No Online\nBackup", "Has Online\nBackup"],
        "ylabel":"Online Backup"
    },
    "DeviceProtection": {
        "df":["No", "Yes"],
        "tick_labels":["No Device\nProtection", "Has Device\nProtection"],
        "ylabel":"Device\nProtection"
    },
    "TechSupport": {
        "df":["No", "Yes"],
        "tick_labels":["No Tech\nSupport", "Has Had Tech\nSupport"],
        "ylabel":"Tech Support"
    },
    "StreamingTV": {
        "df":["No", "Yes"],
        "tick_labels":["No TV Streaming", "Has Movie Streaming"],
        "ylabel":"TV Streaming"
    },
    "StreamingMovies": {
        "df":["No", "Yes"],
        "tick_labels":["No Movie Streaming", "Has Movie Streaming"],
        "ylabel":"Movie Streaming"
    },
}

churn_df = pd.read_csv(
    "Customer_Churn_Dataset.csv",
    index_col=0
)
churn_df["TotalCharges"] = churn_df["TotalCharges"].replace([" "], "0").astype("float")
plot_demographics()
plot_phone()
plot_internet()
plot_support()
plot_streaming()

#gender SeniorCitizen   Partner Dependents




