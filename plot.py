import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers

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

def plot_quaternary(left_axis, right_axis, parameter):
    left_axis.bar(
        [0, 1, 2, 3],
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
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "No") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][3])
            ].shape[0]
        ],
        color="green",
        alpha=.8
    )
    right_axis.bar(
        [0, 1, 2, 3],
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
            ].shape[0],
            churn_df[
                (churn_df["Churn"] == "Yes") & 
                (churn_df[parameter] == parameter_dict[parameter]["df"][3])
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
        labels[2] = parameter_dict[parameter]["tick_labels"][1]
        labels[3] = parameter_dict[parameter]["tick_labels"][2]
        labels[4] = parameter_dict[parameter]["tick_labels"][3]
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

def plot_billing():
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
    plot_tertiary(
        left_axis=ax[0][0],
        right_axis=ax[0][1],
        parameter="Contract"
    )
    plot_binary(
        left_axis=ax[1][0],
        right_axis=ax[1][1],
        parameter="PaperlessBilling"
    )
    plot_quaternary(
        left_axis=ax[2][0],
        right_axis=ax[2][1],
        parameter="PaymentMethod"
    )
    plot_linear(
        left_axis=ax[3][0],
        right_axis=ax[3][1],
        parameter="MonthlyCharges"
    )
    plot_linear(
        left_axis=ax[4][0],
        right_axis=ax[4][1],
        parameter="TotalCharges"
    )
    plt.tight_layout()
    plt.savefig(
        "Billing.png", 
        bbox_inches="tight"
    )
    plt.close()

def train_and_plot_model():
    churn_df = pd.read_csv(
        "Customer_Churn_Dataset.csv",
        index_col=0
    )
    churn_df["gender"] = churn_df["gender"].astype("category").cat.codes.astype("float")
    churn_df["SeniorCitizen"] = churn_df["SeniorCitizen"].astype("category").cat.codes.astype("float")
    churn_df["Partner"] = churn_df["Partner"].astype("category").cat.codes.astype("float")
    churn_df["Dependents"] = churn_df["Dependents"].astype("category").cat.codes.astype("float")
    churn_df["tenure"] = churn_df["tenure"]/72
    churn_df["PhoneService"] = churn_df["PhoneService"].astype("category").cat.codes.astype("float")
    churn_df["MultipleLines"] = churn_df["MultipleLines"].astype("category").cat.codes/2
    churn_df["InternetService"] = churn_df["InternetService"].astype("category").cat.codes/2
    churn_df["OnlineSecurity"] = churn_df["OnlineSecurity"].astype("category").cat.codes/2
    churn_df["OnlineBackup"] = churn_df["OnlineBackup"].astype("category").cat.codes/2
    churn_df["DeviceProtection"] = churn_df["DeviceProtection"].astype("category").cat.codes/2
    churn_df["TechSupport"] = churn_df["TechSupport"].astype("category").cat.codes/2
    churn_df["StreamingTV"] = churn_df["StreamingTV"].astype("category").cat.codes/2
    churn_df["StreamingMovies"] = churn_df["StreamingMovies"].astype("category").cat.codes/2 
    churn_df["Contract"] = churn_df["Contract"].astype("category").cat.codes/2
    churn_df["PaperlessBilling"] = churn_df["PaperlessBilling"].astype("category").cat.codes.astype("float")
    churn_df["PaymentMethod"] = churn_df["PaymentMethod"].astype("category").cat.codes/3
    churn_df["MonthlyCharges"] = churn_df["MonthlyCharges"]/118.75
    churn_df["TotalCharges"] = churn_df["TotalCharges"].replace([" "], "0")
    churn_df["TotalCharges"] = churn_df["TotalCharges"].astype(float)/999.9

    # Target
    churn_df["Churn"] = churn_df["Churn"].astype("category").cat.codes.astype("float")
    x = churn_df[
        [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
        ]
    ].values
    Y = churn_df["Churn"].values
    max_epochs = 20
    batch = 32
    percent_split = .7
    train_idx = round(x.shape[0]*percent_split)
    x_train = x[:train_idx]
    x_val = x[train_idx:]
    y_train = Y[:train_idx]
    y_val = Y[train_idx:]
    model_1 = models.Sequential()
    model_1.add(layers.Dense(8, activation = 'relu', input_shape = (19,)))
    model_1.add(layers.Dense(1, activation = 'sigmoid'))
    model_1.compile(
        optimizer = optimizers.RMSprop(lr = 0.001), 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy']
    )
    history_1 = model_1.fit(
        x_train, 
        y_train, 
        epochs = max_epochs, 
        batch_size = batch, 
        validation_data = (x_val, y_val), 
        verbose = 0
    )
    epochs = range(1, max_epochs + 1)
    train_loss = history_1.history['loss']
    val_loss = history_1.history['val_loss']
    train_accuracy = history_1.history['accuracy']
    val_accuracy = history_1.history['val_accuracy']
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(epochs, train_loss, 'o', color="black", label = "Training Loss")
    ax[0].plot(epochs, val_loss, color="red", label="Validation Loss")
    ax[0].set_title(
        "Training/Validation Loss",
        fontdict={
            'fontsize': 22,
            'verticalalignment': 'baseline',
            'horizontalalignment': "center"
        }
    )
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, train_accuracy, 'o', color="black", label = "Training Accuracy")
    ax[1].plot(epochs, val_accuracy, color="green", label = "Validation Accuracy")
    ax[1].set_title(
        "Training/Validation Accuracy",
        fontdict={
            'fontsize': 22,
            'verticalalignment': 'baseline',
            'horizontalalignment': "center"
        }
    )
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    for axis in ax:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
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
        axis.set_xlabel(
            axis.get_xlabel(),
            fontdict={
                'fontsize': 15,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            },
            labelpad=15
        )
        axis.set_ylabel(
            axis.get_ylabel(),
            fontdict={
                'fontsize': 15,
                'verticalalignment': 'baseline',
                'horizontalalignment': "center"
            },
            labelpad=15
        )
    plt.tight_layout()
    plt.savefig(
        "TrainAndValidate.png", 
        bbox_inches="tight"
    )
    plt.close()
    predictions_df = pd.DataFrame(
        {
            "Y":Y,
            "predictions":model_1.predict(x).flatten().round()
        }
    )
    font_size = 20
    churn_correct = predictions_df[
        (predictions_df["Y"] == 1) &
        (predictions_df["predictions"] == 1)
    ].shape[0] / predictions_df[
        (predictions_df["Y"] == 1)
    ].shape[0]

    nochurn_correct = predictions_df[
        (predictions_df["Y"] == 0) &
        (predictions_df["predictions"] == 0)
    ].shape[0] / predictions_df[
        (predictions_df["Y"] == 0)
    ].shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    color_map = plt.imshow([
            [churn_correct, 1-nochurn_correct],
            [1-churn_correct, nochurn_correct]
        ])
    color_map.set_cmap("Blues")
    # plt.colorbar()

    ax.set_ylabel(
        "Predicted",
        fontdict={
            'fontsize': 20,
            'verticalalignment': 'baseline',
            'horizontalalignment': "center"
        },
        labelpad=15
    )
    ax.set_xlabel(
        "Actual",
        fontdict={
            'fontsize': 20,
            'verticalalignment': 'baseline',
            'horizontalalignment': "center"
        },
        labelpad=15
    )
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[2] = "Churn"
    labels[6] = "No Churn"
    ax.set_xticklabels(labels)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[2] = "Churn"
    labels[6] = "No Churn"
    ax.set_yticklabels(labels)

    ax.text(
        .0,
        .04,
        str(
            round(predictions_df[
                (predictions_df["Y"] == 1) &(predictions_df["predictions"] == 1)
            ].shape[0] / predictions_df[(predictions_df["Y"] == 1)
            ].shape[0]*100)
        ) + "%",
        ha='center',
        va='bottom',
        fontsize=font_size,
        color="white"
    ) 
    ax.text(
        0,
        1.04,
        str(
            round(predictions_df[
                (predictions_df["Y"] == 1) &
                (predictions_df["predictions"] == 0)
            ].shape[0] / predictions_df[(predictions_df["Y"] == 1)
            ].shape[0]*100) 
        )+ "%",
        ha='center',
        va='bottom',
        fontsize=font_size,
        color="white"
    ) 
    ax.text(
        1.0,
        .04,
        str(
            round(predictions_df[
                (predictions_df["Y"] == 0) &
                (predictions_df["predictions"] == 1)
            ].shape[0] / predictions_df[(predictions_df["Y"] == 0)
            ].shape[0]*100) 
        ) + "%",
        ha='center',
        va='bottom',
        fontsize=font_size,
        color="black"
    )
    ax.text(
        1.0,
        1.04,
        str(
            round(predictions_df[
                (predictions_df["Y"] == 0) &
                (predictions_df["predictions"] == 0)
            ].shape[0] / predictions_df[(predictions_df["Y"] == 0)
            ].shape[0]*100) 
        ) + "%",
        ha='center',
        va='bottom',
        fontsize=font_size,
        color="white"
    )

    ax.tick_params(
        axis='y',
        which='both',
        length=0,
        labelsize=15
    )
    ax.tick_params(
        axis='x',
        which='both',
        length=0,
        labelsize=15
    )
    plt.savefig(
        "Confusion.png", 
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
    "Contract": {
        "df":['Month-to-month', 'One year', 'Two year'],
        "tick_labels":['Month-to-month', '1 year', '2 year'],
        "ylabel":"Contract"
    },
    "PaperlessBilling": {
        "df":["No", "Yes"],
        "tick_labels":["No Paperless Billing", "Has Paperless Billing"],
        "ylabel":"Paperless Billing"
    },
    "PaymentMethod": {
        "df":['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'],
        "tick_labels":["Electronic\nCheck", "Mailed\nCheck", "Bank\nTransfer", "Credit\nCard"],
        "ylabel":"Payment Method"
    },
    "MonthlyCharges": {
        "ylabel":"Monthly Charge"
    },
    "TotalCharges": {
        "ylabel":"Total Charge"
    }
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
plot_billing()
train_and_plot_model()




