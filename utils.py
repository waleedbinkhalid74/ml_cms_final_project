import pandas as pd
from tqdm import tqdm 
import numpy as np
from EDMD import EDMD


def make_eigenfunction(X: np.array, Y: np.array, model: EDMD, eigvec_log: int) -> np.array:
    """Calculates the eigenfunction at given points for a fitted EDMD model at a specific eigenvector

    Args:
        X (np.array): X coordinates
        Y (np.array): Y coordinates
        model (EDMD): fitted EDMD model
        eigvec_log (int): eigenvector position at which to calculate the eigenfunction

    Returns:
        np.array: eigenfunction
    """
    eigenfunction = []
    for i in range(Y.shape[1]):
        temp = pd.DataFrame(np.array([X[:,i], Y[:,i]]).T, columns=['x', 'y'])
        temp.insert(0, 'ID', 0)
        temp.insert(1, 'time', np.arange(0, Y.shape[1]))
        a = model.calculate_eigenfunction(temp, eigvec_log)
        eigenfunction.append(a)
    eigenfunction = np.array(eigenfunction).T
    return eigenfunction.real

def plot_data(data: pd.DataFrame, ax):
    """Plots the trajectories for all snapshot pair datapoints

    Args:
        data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2
        
        ax ([type]): [description]
    """
    for id in data['ID'].unique():
        data_id = data[data['ID'] == id]
        ax.plot(data_id.iloc[:,2].to_numpy(), data_id.iloc[:,3].to_numpy(), c='black')
        add_arrow(data_id, ax)

def add_arrow(data: pd.DataFrame, ax):
    """Adds arrows to the trajectory plot to show direction of flow

    Args:
        data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2      ...     Var_n
        
        ax: Axes object on which to plot
    """
    data_size = len(data)
    arrow_pos = 3
    ax.arrow(
    data.iloc[arrow_pos,2], 
    data.iloc[arrow_pos,3], 
    data.iloc[arrow_pos+1,2] - data.iloc[arrow_pos,2], 
    data.iloc[arrow_pos+1,3] - data.iloc[arrow_pos,3],
    color="black",
    head_width=0.1,
)
    
    
def name_col_changer(col_name: str) -> str:
    """Maps the input column names to more meaningful column names

    Args:
        col_name (str): input column names

    Returns:
        str: mapped column names
    """
    if 'pedestrianId' in col_name:
        return 'ID'
    elif 'startX' in col_name:
        return 'x1'
    elif 'startY' in col_name:
        return 'x2'
    elif 'endX' in col_name:
        return 'y1'
    elif 'endY' in col_name:
        return 'y2'
    elif 'simTime' in col_name:
        return 'time'
    else:
        return 'delete'
    
def format_traj_df(df: pd.DataFrame) -> pd.DataFrame:
    """format dataframe such that it only has ID and x and y columns

    Args:
        df (pd.DataFrame): unformated pedestrian trajectories

    Returns:
        pd.DataFrame: formatted dataframe
    """
    df = df.rename(name_col_changer, axis='columns')
    df.drop(['delete'], axis=1, inplace=True)
    return df

def find_center_of_gravity(traj_df:pd.DataFrame) -> np.array:
    """Calculates the center of gravity of a crowd at each timestep

    Args:
        traj_df (pd.DataFrame): trajectory data

    Returns:
        np.array: center of gravity at each time-step of the simulation
    """
    max_steps = traj_df.groupby('ID').count().iloc[:,1].max()
    ids = list(traj_df.ID.unique())
    center_of_gravity = []
    for step in tqdm(range(max_steps)):
        total_peds = 0
        x1_sum = 0
        x2_sum = 0
        for id in ids:
            df = traj_df[traj_df.ID == id]
            if len(df) < step+1:
                continue
            x1_sum += df.iloc[step, 2]
            x2_sum += df.iloc[step, 3]
            total_peds += 1
        x1_mean = x1_sum / total_peds     
        x2_mean = x2_sum / total_peds
        center_of_gravity.append([x1_mean, x2_mean])
    center_of_gravity = np.array(center_of_gravity)
    return center_of_gravity


def plot_trajectories(traj_df: pd.DataFrame, no_peds: int, ax):
    """Plots the trajectories of a given number of pedestrians

    Args:
        traj_df (pd.DataFrame): DataFrame containing the trajectory data in the format
        no_peds (int): Number of trajectories to draw
        ax ([type]): axis on which to draw
    """
    for ped in range(no_peds+1):
        df = traj_df[traj_df.ID == ped]
        ax.plot(df.x1.to_numpy(), df.x2.to_numpy())
    ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_title(f"Trajectories for {no_peds} pedestrians")

def remove_extrapolated_data(pred_data: pd.DataFrame, org_data: pd.DataFrame) -> pd.DataFrame:
    """Removes predictions beyond provided time stamp in the original dataset for trajectories

    Args:
        pred_data (pd.DataFrame): Predicted dataframe
        org_data (pd.DataFrame): Actual data

    Returns:
        pd.DataFrame: Predicted data with prediction beyond timestamp of respective trajectory removed
    """
    pred_df_list = []
    for id in tqdm(org_data.ID.unique()):
        df_org = org_data[org_data.ID == id]
        df_pred = pred_data[pred_data.ID == id].iloc[:len(df_org)]
        pred_df_list.append(df_pred)
    return pd.concat(pred_df_list)


def mse_calc(pred: np.array, target: pd.DataFrame) -> float:
    """Calculates the MSE between two arrays

    Args:
        pred (np.array): predicted data
        target (pd.DataFrame): target data

    Returns:
        float: MSE
    """
    diff = pred - target
    return np.sum(diff**2) / pred.shape[0]

def mse_all_traj(df_1: pd.DataFrame, df_2: pd.DataFrame) -> float:
    """Calculates the mean of MSE for all trajectories in a dataframe

    Args:
        df_1 (pd.DataFrame): predicted dataframe
        df_2 (pd.DataFrame): target dataframe

    Returns:
        float: mean of MSE across all trajectories
    """
    
    mse_error = 0
    for i, traj in enumerate(df_1.ID.unique()):
        df1 = df_1[df_1.ID == traj]
        df2 = df_2[df_2.ID == traj]
        mse_error += mse_calc(df1.iloc[:,2:4].to_numpy(), df2.iloc[:,2:4].to_numpy())
    i += 1
    mse_error = mse_error / i
    return mse_error

def resample_trajectory(traj_df: pd.DataFrame, delta_t: float) -> pd.DataFrame:
    """Resamples the pedestrian trajectories so that all pedestrians have data at each n*delta_t

    Args:
        traj_df (pd.DataFrame): Original trajectory data from Vadere
        delta_t (float): Time steps at which data should be available

    Returns:
        pd.DataFrame: Resampled dataframe
    """
    resampled_list = []
    for ped in tqdm(traj_df.pedestrianId.unique()):
        df = traj_df[traj_df.pedestrianId == ped]
        t_span = np.arange(0.4, df.simTime.iloc[-1], delta_t)
        resampled_df = pd.DataFrame()
        resampled_df['time'] = t_span
        resampled_df['x1'] = 0
        resampled_df['x2'] = 0
        resampled_df.insert(0, 'ID', df.pedestrianId.iloc[0])
        start = df.iloc[0]
        actual_delta_t = np.diff(df.simTime.to_numpy())[0]

        for i, time in enumerate(t_span):
            if df[df['simTime'] >= time].empty:
                continue
            else:
                second_last = df[df['simTime'] <= time].iloc[-1]
                last = df[df['simTime'] >= time].iloc[0]
                dt = time - second_last.simTime
                resampled_df['x1'].iloc[i] = second_last['startX-PID1'] + (last['startX-PID1'] - second_last['startX-PID1']) * dt / actual_delta_t 
                resampled_df['x2'].iloc[i] = second_last['startY-PID1'] + (last['startY-PID1'] - second_last['startY-PID1']) * dt / actual_delta_t 
        resampled_list.append(resampled_df)
    
    resampled_trajectories = pd.concat(resampled_list)
    resampled_trajectories.reset_index(drop=True, inplace=True)
    return resampled_trajectories
