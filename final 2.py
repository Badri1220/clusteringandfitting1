import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import t
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def load_and_clean_data(file_path):
    """
    Load and clean the data.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - original_data (pd.DataFrame): Original data.
    - cleaned_transposed_data (pd.DataFrame): Cleaned and transposed data.
    """
    # Load data
    data = pd.read_csv(file_path)

    # Perform data cleaning and selection of relevant columns
    selected_columns = ["Forest area (% of land area) [AG.LND.FRST.ZS]" ,
                        "Access to clean fuels and technologies for cooking (% of population) [EG.CFT.ACCS.ZS]" ,
                        "Access to electricity (% of population) [EG.ELC.ACCS.ZS]" ,
                        "Agricultural land (% of land area) [AG.LND.AGRI.ZS]"]
    data[selected_columns] = data[selected_columns].apply(pd.to_numeric , errors='coerce')

    # Use SimpleImputer to replace NaN values with the mode
    imputer = SimpleImputer(strategy='most_frequent')
    data[selected_columns] = imputer.fit_transform(data[selected_columns])

    # Transpose the cleaned data
    cleaned_transposed_data = data[selected_columns].T

    return data , cleaned_transposed_data


def kmeans_clustering(data , num_clusters=3):
    """
    Perform K-means clustering on the normalized data.

    Parameters:
    - data (pd.DataFrame): Original data.
    - num_clusters (int): Number of clusters for K-means.

    Returns:
    - silhouette_avg (float): Silhouette score indicating clustering quality.
    - cluster_labels (pd.Series): Cluster labels assigned to each data point.
    """
    # Select relevant columns for clustering
    selected_columns = ["Forest area (% of land area) [AG.LND.FRST.ZS]" ,
                        "Access to clean fuels and technologies for cooking (% of population) [EG.CFT.ACCS.ZS]" ,
                        "Access to electricity (% of population) [EG.ELC.ACCS.ZS]" ,
                        "Agricultural land (% of land area) [AG.LND.AGRI.ZS]"]

    # Extract selected columns and convert to numeric
    X = data[selected_columns].apply(pd.to_numeric , errors='coerce').values

    # Normalize the data
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters , random_state=42)
    cluster_labels = kmeans.fit_predict(X_normalized)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_normalized , cluster_labels)

    return silhouette_avg , pd.Series(cluster_labels , name='Cluster')


def exponential_growth(x , a , b):
    """
    Exponential growth model function.

    Parameters:
    - x (array): Independent variable.
    - a (float): Parameter representing the amplitude.
    - b (float): Parameter representing the growth rate.

    Returns:
    - array: Dependent variable values based on the exponential growth model.
    """
    return a * np.exp(b * x)


def err_ranges(func , x_values , params , cov_matrix , alpha=0.05):
    """
    Calculate lower and upper bounds for confidence intervals.

    Parameters:
    - func (callable): Function to calculate confidence intervals for.
    - x_values (array): Independent variable values.
    - params (array): Fitted parameters of the function.
    - cov_matrix (2D array): Covariance matrix of the fitted parameters.
    - alpha (float): Significance level for confidence intervals.

    Returns:
    - lower_bound (array): Lower bounds for confidence intervals.
    - upper_bound (array): Upper bounds for confidence intervals.
    """
    p = len(params)
    dof = max(0 , len(x_values) - p)  # degrees of freedom

    t_value = abs(np.percentile(t.ppf(1 - alpha / 2 , dof) , 50))
    stderr = np.sqrt(np.diag(cov_matrix))

    lower_bound = func(x_values , *(params - t_value * stderr))
    upper_bound = func(x_values , *(params + t_value * stderr))

    return lower_bound , upper_bound


def plot_clustering_results(data , cluster_labels):
    """
    Visualize clustering results.

    Parameters:
    - data (pd.DataFrame): Original data.
    - cluster_labels (pd.Series): Cluster labels assigned to each data point.
    """
    plt.scatter(data["Forest area (% of land area) [AG.LND.FRST.ZS]"] ,
                data["Agricultural land (% of land area) [AG.LND.AGRI.ZS]"] ,
                c=cluster_labels , cmap='viridis' , s=50 , alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.xlabel("Forest area (% of land area)")
    plt.ylabel("Agricultural land (% of land area)")
    plt.title('Clustering Results')
    plt.show()


def plot_curve_fitting(data , country_name='India'):
    """
    Visualize curve fitting results.

    Parameters:
    - data (pd.DataFrame): Original data.
    - country_name (str): Name of the country for curve fitting.
    """
    # Select relevant time series data
    time_series_data = \
        data[data['Country Name'] == country_name]['Forest area (% of land area) [AG.LND.FRST.ZS]']

    # Fit the model
    popt , pcov = curve_fit(exponential_growth , range(len(time_series_data)) , time_series_data)

    # Prediction and Confidence Intervals
    x_values = np.arange(len(time_series_data) + 10)  # 10 years into the future
    y_pred = exponential_growth(x_values , *popt)
    lower_bound , upper_bound = err_ranges(exponential_growth , x_values , popt , pcov)

    # Visualization of Curve Fitting
    plt.plot(range(len(time_series_data)) , time_series_data , 'o' , label='Original Data')
    plt.plot(x_values , y_pred , label='Fitted Curve')
    plt.fill_between(x_values , lower_bound , upper_bound , color='gray' ,
                     alpha=0.2 , label='Confidence Interval')
    plt.xlabel('Years')
    plt.ylabel('Forest area (% of land area)')
    plt.title(f'Exponential Growth Model for {country_name} and Confidence Interval')
    plt.legend()
    plt.show()


# Example usage:
file_path = "9d28d133-d728-4cd5-bb11-d1883cc03d72_Data.csv"
original_data , cleaned_transposed_data = load_and_clean_data(file_path)

silhouette_avg , cluster_labels = kmeans_clustering(original_data)
print("silhouette score",silhouette_avg)
plot_clustering_results(original_data , cluster_labels)

plot_curve_fitting(original_data , country_name='India')
plot_curve_fitting(original_data , country_name='Australia')
