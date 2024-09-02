#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import inspect


# ## prepare Data

# In[ ]:


client =MongoClient(host="localhost", port=27017)
db = client["air-quality"]
nairobi = db["nairobi"]


# In[ ]:


# Check our work
assert any([isinstance(df, pd.DataFrame), isinstance(df, pd.Series)])
assert len(df) <= 32907
assert isinstance(df.index, pd.DatetimeIndex)
assert df.index.tzinfo == pytz.timezone("Africa/Nairobi")


# ## Explore

# In[ ]:


results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

df = pd.DataFrame(results).set_index("timestamp")


# Localize time zone

# In[ ]:


df.index=df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")


# Create a boxplot of the "P2" readings in df and readings above 500 are dropped from the dataset

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind="box", vert= False, title="Distribution of P2", ax=ax);


# In[ ]:


# Check our work
assert len(df) <= 32906


# Creating a time series plot of the "P2" readings in df

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Time",ylabel="PM2.5", title="Distribution of PM2.5 Timesseries",  ax=ax);


#  Resampling df to provide the mean "P2" reading for each hour. Used a forward fill to impute any missing values

# In[ ]:


df["P2"].resample("1H").mean().fillna(method="ffill").to_frame()


# ## Wrangle function for reproducability

# In[ ]:


def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    df = pd.DataFrame(results).set_index("timestamp")
    #localize time zone
    df.index=df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")
    # drop outlaiers
    df=df[df["P2"]<500]
    # resaple "1H" intervals and ffill missing values
    df=df["P2"].resample("1H").mean().fillna(method="ffill").to_frame()
    # Creating new column with lag and 1 step
    df["P2.L1"]= df["P2"].shift(1)
    #drop nan values
    df.dropna(inplace=True)
    
    return df


# In[ ]:


y =wrangle (nairobi)
y.head()


# Checking the Auoto Correlation

# In[ ]:


fig , ax= plt.subplots(figsize=(15,6))
plot_pacf(y,ax=ax)
plt.xlabel("Lags[Hours]")
plt.ylabel("Correlation Coefficient");


# ## Split

# In[ ]:


target = "P2"
y = df[target]
X = df.drop(columns=target)


# In[ ]:


cutoff = int(len(X)* 0.8)
X_train=X.iloc[:cutoff]
y_train=y.iloc[:cutoff]
X_test=X.iloc[cutoff:]
y_test=y.iloc[cutoff:]


# ## Build Model

# ### Baseline model

# In[ ]:


y_mean=y_train.mean()
y_pred_baseline =[y_mean]* len(y_train)
mae_baseline = mean_absolute_error(y_pred_baseline, y_train)

print("Mean P2 Reading:", round(y_train.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2))


# ### Iterate

# Considering 2 hyperparameters that  $p$ is  the number of lagged observations included the model and $q$ is the  error lag

# In[ ]:


p_params = range(0,26,8)
q_params = range(0,3,1)


# In[ ]:


# Create dictionary to store MAEs
mae_grid = dict()
# Outer loop: Iterate through possible values for `p`
for p in p_params:
    # Create key-value pair in dict. Key is `p`, value is empty list.
    mae_grid[p] = list()
    # Inner loop: Iterate through possible values for `q`
    for q in q_params:
        # Combination of hyperparameters for model
        order = (p, 0, q)
        # Note start time
        start_time = time.time()
        # Train model
        model =ARIMA(y_train, order=order).fit()
        # Calculate model training time
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict()
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        print(mae)
        # Append MAE to list in dictionary
        mae_grid[p].append(mae)
        


# In[ ]:


print(mae_grid)


# In[ ]:


mae_df =pd.DataFrame(mae_grid)
mae_df.round(4)


# Creating Heatmap to find the best hyperparameters

# In[ ]:


sns.heatmap(mae_df, cmap="Blues")
plt.xlabel("p values")
plt.ylabel("q value")
plt.title(" ARMA model Drid Search (Criterion:MAE)")


# The best hyperparameters: $p=8$ and $q=1$

# Evaluate

# In[ ]:


y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = ARIMA(history, order=(8,0,1))
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])


# ## Communicate Results

# In[ ]:


df_predictions =pd,DataFrame({"y_test": y_test, "y_pred_wfv": y_pred_wfv})
fig = px.line(df_predictions, labels={" values": "PM2.5"})
fig.show()

