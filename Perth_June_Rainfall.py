import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        I am currently organising my wedding for June, due to cheaper prices for vendors and venues (since winter is not a popular season for weddings). However, I would like to predict  how likely it is for there to be rain on my wedding day.

        On this project I will be using various types of ML algortihms from the SciKJit library including Random Forest, Linear Regression and Support Vector Regressor.
        """
    )
    return


@app.cell
def __():
    import os
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    import datetime as dt
    from datetime import date
    import plotly.express as px
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression 
    from sklearn.svm import SVR
    return (
        GridSearchCV,
        LinearRegression,
        RandomForestRegressor,
        SVR,
        date,
        dt,
        go,
        make_regression,
        make_subplots,
        mean_squared_error,
        mo,
        np,
        os,
        pd,
        pl,
        plt,
        px,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        The data to be used was collected from the Bureau of Meteorology (Australian Government).

        http://www.bom.gov.au/climate/data/index.shtml
        """
    )
    return


@app.cell
def __(__file__, os, pl):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    df_t = pl.read_csv(os.path.join(script_dir, "PA_MAX_TEMP.csv"))
    df_r = pl.read_csv(os.path.join(script_dir, "PA_RAINFALL.csv"))
    df_s = pl.read_csv(os.path.join(script_dir, "PA_SOLAR_EXPOSURE.csv"))
    return df_r, df_s, df_t, script_dir


@app.cell
def __(df_t):
    df_t.describe()
    return


@app.cell
def __(df_r):
    df_r.describe()
    return


@app.cell
def __(df_s):
    df_s.describe()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Looking at the imported dataframes, we will not be interested in data such as station number, product code, etc...

        Instead, we will be just focusing on the raw readings (solar exposure, rainfall and max temperature) and dates. Then, we will join all of the dataframes according to the dates.

        Considering that solar exposure only has data from 1990, we will be excluding all data from solar and rainfall measures prior 1990.
        """
    )
    return


@app.cell
def __(df_r, df_s, df_t, pl):
    _df_s = df_s.with_columns(
        (pl.datetime(pl.col('Year'),pl.col('Month'),pl.col('Day'))).dt.strftime('%Y-%m-%d').str.to_date().alias('DATE')
    ).select(
        ['DATE','Daily global solar exposure (MJ/m*m)']
    )

    _df_r = df_r.with_columns(
        (pl.datetime(pl.col('Year'),pl.col('Month'),pl.col('Day'))).dt.strftime('%Y-%m-%d').str.to_date().alias('DATE')
    ).select(
        ['DATE','Rainfall amount (millimetres)']
    )

    _df_t = df_t.with_columns(
        (pl.datetime(pl.col('Year'),pl.col('Month'),pl.col('Day'))).dt.strftime('%Y-%m-%d').str.to_date().alias('DATE')
    ).select(
        ['DATE','Maximum temperature (Degree C)']
    )

    _df = _df_s.join(_df_r, on="DATE")
    df_temp = _df.join(_df_t, on='DATE').rename({
        "Daily global solar exposure (MJ/m*m)":"SOLAR_EXP",
        "Rainfall amount (millimetres)":"RAINFALL",
        "Maximum temperature (Degree C)":"MAX_TEMP"
    })
    df_temp
    return (df_temp,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Observing the data it can be seen that there are several issues:

        - Rainfall and temperature are in string format.
        - Columns have null values.

        Additionally, we will be adding another columns to categorise the rain according to its intensity: 

        - 0: No rain
        - 1: Light rain
        - 2: Moderate rain
        - 3: Heavy rain
        """
    )
    return


@app.cell
def __(df_temp, pl):
    _df = df_temp.with_columns(
        (pl.col('RAINFALL').cast(pl.Float64).alias('RAINFALL')),
        ((pl.col('MAX_TEMP').cast(pl.Float64).alias('MAX_TEMP'))),
    ).fill_null(0).fill_nan(0)

    df = _df.with_columns(
        pl.col('DATE').dt.year().alias('YEAR'),
        pl.col('DATE').dt.day().alias('DAY'),
        pl.col('DATE').dt.month().alias('MONTH'),
        pl.when((pl.col("RAINFALL") < 1))
        .then(0)
        .when((pl.col("RAINFALL") > 1) & (pl.col("RAINFALL") <= 10))
        .then(1)
        .when((pl.col("RAINFALL") > 10) & (pl.col("RAINFALL") <= 30))
        .then(2)
        .otherwise(3)
        .alias("RAINFALL_CAT")
    )

    df
    return (df,)


@app.cell
def __(df, px):
    _df = df.select(
        ['DATE','RAINFALL','SOLAR_EXP','MAX_TEMP']
    )
    _fig = px.imshow(_df.corr(),
                    x=['Date', 'Solar Exposure', 'Rainfall', 'Maximum Temp'],
                    y=['Date', 'Solar Exposure', 'Rainfall', 'Maximum Temp']
                     ,color_continuous_scale='RdBu',range_color=[-1, 1],)
    _fig.show()
    return


@app.cell
def __(mo):
    mo.md(r"""There seems to be that the highest correlation for rainfall is a negative one with solar exposure and maximum temperature. This makes sense since both solar exposure and maximum temperature have a high positive correlation.""")
    return


@app.cell
def __(df, go, make_subplots, pl):
    _df = df.filter(pl.col('MONTH')==6)

    _fig = make_subplots(rows=2, cols=1)

    _dfm = _df.group_by("DAY").agg(pl.mean("RAINFALL_CAT").alias("Mean_Rainfall"))
    _fig.add_trace(
        go.Bar(x=_dfm['DAY'], y=_dfm['Mean_Rainfall'], name='Mean rain category'),
        row=1, col=1
    )

    _dfmx = _df.group_by("DAY").agg(pl.max("RAINFALL_CAT").alias("Max_Rainfall"))
    _fig.add_trace(
        go.Bar(x=_dfmx['DAY'], y=_dfmx['Max_Rainfall'], name='Max recorded rain category'),
        row=2, col=1
    )

    _fig.update_layout(
        title="June's Rainfall Category Analysis",
        xaxis_title='Day',
        yaxis_title='Rainfall Category',
        barmode='overlay',
        annotations=[
            dict(
                text="Rainfall Categories:<br>0: No rain<br>1: Light rain<br>2: Moderate rain<br>3: Heavy rain",
                xref="paper", yref="paper",
                x=1.3, y=-0.01,
                showarrow=False,
                align="left",
                bgcolor="black",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    # Show the plot
    _fig.show()
    return


@app.cell
def __(df, go, make_subplots, pl):
    _df = df.filter(pl.col('MONTH')==6)

    _fig = make_subplots(rows=2, cols=1)

    _dfm = _df.group_by("DAY").agg(pl.mean("RAINFALL").alias("Mean_Rainfall"))
    _fig.add_trace(
        go.Bar(x=_dfm['DAY'], y=_dfm['Mean_Rainfall'], name='Mean Rainfall'),
        row=1, col=1
    )

    _dfmx = _df.group_by("DAY").agg(pl.max("RAINFALL").alias("Max_Rainfall"))
    _fig.add_trace(
        go.Bar(x=_dfmx['DAY'], y=_dfmx['Max_Rainfall'], name='Max Rainfall'),
        row=2, col=1
    )

    _fig.update_layout(
        title="June's Rainfall Analysis",
        xaxis_title='Day',
        yaxis_title='Rainfall (mm)',
        barmode='overlay'
    )

    # Show the plot
    _fig.show()
    return


@app.cell
def __(df, pl, px):
    _df = df.with_columns(
        pl.col('DATE').dt.year().alias('YEAR')
        ,pl.col('DATE').dt.day().alias('DAY')
        ,pl.col('DATE').dt.month().alias('MONTH')
    ).filter(
        (pl.col('DAY')==14) & (pl.col('MONTH')==6)
    )

    px.bar(_df,x='YEAR',y='RAINFALL')
    return


@app.cell
def __(df, go, make_subplots, pl):
    _df = df.filter(
        (pl.col('DAY')==14) & (pl.col('MONTH')==6)
    )

    _fig = make_subplots(rows=2, cols=1)

    _fig.add_trace(
        go.Histogram(x=_df['RAINFALL_CAT'], name='Rainfall Category (%)', histnorm='percent'),
        row=1, col=1
    )

    # Add a bar chart for Rainfall Category
    _fig.add_trace(
        go.Histogram(x=_df['RAINFALL_CAT'], name='Rainfall Category'),
        row=2, col=1
    )

    # Update layout
    _fig.update_layout(
        title='Historical distribution of June 14th Rainfall category',
        xaxis_title='Rainfall Category',
        yaxis_title='Count',
        barmode='overlay',
        annotations=[
            dict(
                text="Rainfall Categories:<br>0: No rain<br>1: Light rain<br>2: Moderate rain<br>3: Heavy rain",
                xref="paper", yref="paper",
                x=1.2, y=-0.01,
                showarrow=False,
                align="left",
                bgcolor="black",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )

    # Show the plot
    _fig.show()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Machine Learning  Algorithms
        -
        The dependable variable will be Rainfall whilst the independent variables will be date, day, temperature and solar exposure.
        """
    )
    return


@app.cell
def __(df):
    df
    return


@app.cell
def __(df, pl, train_test_split):
    _df = df.filter(
        (pl.col('DAY')==14) & (pl.col('MONTH')==6)
    )

    x = _df.select(
        ['SOLAR_EXP', 'MAX_TEMP', 'DAY','DATE']  
    )
    y = _df.select(['RAINFALL'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    return x, x_test, x_train, y, y_test, y_train


@app.cell
def __(
    LinearRegression,
    RandomForestRegressor,
    SVR,
    mean_squared_error,
    r2_score,
    x,
    x_test,
    x_train,
    y_test,
    y_train,
):
    # Linear Regression Model 
    model_LR = LinearRegression() 
    model_LR.fit(x_train, y_train) 

    # LR predictions 
    y_pred_LR = model_LR.predict(x_test) 

    # LR Evaluation 
    mse_LR = mean_squared_error(y_test, y_pred_LR) 
    r2_LR = r2_score(y_test, y_pred_LR) 

    print('Linear Regression') 
    print(f'Mean Squared Error: {mse_LR}') 
    print(f'R^2 Score: {r2_LR}') 
    print('Coefficients:', model_LR.coef_) 
    print('Intercept:', model_LR.intercept_) 
    print() 

    # Random Forest Regressor Model 
    model_RF = RandomForestRegressor(n_estimators=100, random_state=11) 
    model_RF.fit(x_train, y_train) 

    # RFR predictions 
    y_pred_RF = model_RF.predict(x_test) 

    # RFR evaluations 
    mse_RF = mean_squared_error(y_test, y_pred_RF) 
    r2_RF = r2_score(y_test, y_pred_RF) 

    print('Random Forest') 
    print(f'Mean Squared Error: {mse_RF}') 
    print(f'R^2 Score: {r2_RF}') 
    importances = model_RF.feature_importances_ 
    for feature, importance in zip(x.columns, importances): 
        print(f'Feature: {feature}, Importance: {importance}') 

    max_depths = [tree.tree_.max_depth for tree in model_RF.estimators_] 
    overall_max_depth = max(max_depths) 

    print(f'Maximum Depth of Trees: {overall_max_depth}') 

    # SVR model 
    model_SVR = SVR(kernel='rbf')  
    model_SVR.fit(x_train, y_train) 

    # SVR predictions 
    y_pred_SVR = model_SVR.predict(x_test) 

    # SVR evaluation 
    mse_SVR = mean_squared_error(y_test, y_pred_SVR) 
    r2_SVR = r2_score(y_test, y_pred_SVR) 
    print() 
    print('SVR') 
    print(f'Mean Squared Error: {mse_SVR}') 
    print(f'R^2 Score: {r2_SVR}')
    return (
        feature,
        importance,
        importances,
        max_depths,
        model_LR,
        model_RF,
        model_SVR,
        mse_LR,
        mse_RF,
        mse_SVR,
        overall_max_depth,
        r2_LR,
        r2_RF,
        r2_SVR,
        y_pred_LR,
        y_pred_RF,
        y_pred_SVR,
    )


@app.cell
def __(GridSearchCV, RandomForestRegressor, SVR, x_train, y_train):
    # SVR 
    svr = SVR() 

    # Parameters 
    param_grid = { 
        'C': [0.1, 1, 10], 
        'epsilon': [0.01, 0.1, 1], 
        'kernel': ['linear', 'poly', 'rbf'] 
    } 

    #Grid Search 
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error') 
    grid_search.fit(x_train, y_train) 

    print("Best parameters:", grid_search.best_params_) 


    #RFR 
    rf = RandomForestRegressor(random_state=42) 

    # Parameters 
    param_grid_rf = { 
        'n_estimators': [100, 200, 300], 
        'max_depth': [10, 20, None], 
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4] 
    } 

    # Grid Search 
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, 
                                   scoring={'MSE': 'neg_mean_squared_error', 'R2': 'r2'}, 
                                   refit='MSE')  # You can choose which metric to optimize by default 
    grid_search_rf.fit(x_train, y_train) 

    print("Best parameters:", grid_search_rf.best_params_) 
    print("Best MSE:", -grid_search_rf.best_score_)  
    print("Best R²:", grid_search_rf.cv_results_['mean_test_R2'][grid_search_rf.best_index_])
    return grid_search, grid_search_rf, param_grid, param_grid_rf, rf, svr


@app.cell
def __(
    RandomForestRegressor,
    SVR,
    mean_squared_error,
    model_RF,
    overall_max_depth,
    r2_score,
    x,
    x_test,
    x_train,
    y_pred_RF,
    y_test,
    y_train,
):
    # Random Forest Regressor Model 
    _model_RF = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split = 2, min_samples_leaf=1) 
    _model_RF.fit(x_train, y_train) 

    # RFR predictions 
    _y_pred_RF = model_RF.predict(x_test) 

    # RFR evaluations 
    _mse_RF = mean_squared_error(y_test, _y_pred_RF) 
    _r2_RF = r2_score(y_test, y_pred_RF) 

    print('Random Forest') 
    print(f'Mean Squared Error: {_mse_RF}') 
    print(f'R^2 Score: {_r2_RF}') 
    _importances = model_RF.feature_importances_ 
    for _feature, _importance in zip(x.columns, _importances): 
        print(f'Feature: {_feature}, Importance: {_importance}') 

    _max_depths = [tree.tree_.max_depth for tree in _model_RF.estimators_] 
    _overall_max_depth = max(_max_depths) 

    print(f'Maximum Depth of Trees: {overall_max_depth}') 

    # SVR model 
    _model_SVR = SVR(epsilon = 0.01, kernel = 'linear',)  
    _model_SVR.fit(x_train, y_train) 

    # SVR predictions 
    _y_pred_SVR = _model_SVR.predict(x_test) 

    # SVR evaluation 
    _mse_SVR = mean_squared_error(y_test, _y_pred_SVR) 
    _r2_SVR = r2_score(y_test, _y_pred_SVR) 
    print() 
    print('SVR') 
    print(f'Mean Squared Error: {_mse_SVR}') 
    print(f'R^2 Score: {_r2_SVR}')
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
