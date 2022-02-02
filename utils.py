import pandas as pd
from tqdm import tqdm 
import numpy as np

def plot_data(data: pd.DataFrame, ax):
    """Plots the trajectories for all snapshot pair datapoints

    Args:
        data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2
        
        ax ([type]): [description]
    """
    for id in data['ID'].unique():
        data_id = data[data['ID'] == id]
        ax.plot(data_id.iloc[:,1].to_numpy(), data_id.iloc[:,2].to_numpy(), c='black')
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
    data.iloc[arrow_pos,1], 
    data.iloc[arrow_pos,2], 
    data.iloc[arrow_pos+1,1] - data.iloc[arrow_pos,1], 
    data.iloc[arrow_pos+1,2] - data.iloc[arrow_pos,2],
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
            x1_sum += df.iloc[step, 1]
            x2_sum += df.iloc[step, 2]
            total_peds += 1
        x1_mean = x1_sum / total_peds     
        x2_mean = x2_sum / total_peds
        center_of_gravity.append([x1_mean, x2_mean])
    center_of_gravity = np.array(center_of_gravity)
    return center_of_gravity

# TODO: Delete later
# def append_end_loc_to_ped_traj(traj_df_formatted: pd.DataFrame):
#     max_steps = traj_df_formatted.groupby('ID').count().iloc[:,1].max()
#     for ped in list(traj_df_formatted.ID.unique()):
#         df = traj_df_formatted[traj_df_formatted.ID == ped]
#         curr_steps = len(df)
#         if curr_steps < max_steps:
#             # second_last_loc = df.iloc[-2,:].to_numpy()            
#             last_loc = df.iloc[-1,:].to_numpy()
#             # difference = last_loc - second_last_loc
#             # difference[0] = ped

#             df_to_append = pd.DataFrame(np.tile(last_loc, (max_steps - curr_steps, 1)), columns=list(traj_df_formatted.columns))
#             # df_to_append = pd.DataFrame(np.tile(difference, (max_steps - curr_steps, 1)), columns=list(traj_df_formatted.columns))
#             traj_df_formatted = traj_df_formatted.append(df_to_append)
#     traj_df_formatted.reset_index(drop=True, inplace=True)
#     return traj_df_formatted

# def remove_end_loc_to_ped_traj(traj_df: pd.DataFrame):
#     min_steps = traj_df.groupby('ID').count().iloc[:,1].min()
#     traj_df_formatted = pd.DataFrame()
#     for ped in list(traj_df.ID.unique()):
#         df = traj_df[traj_df.ID == ped]
#         curr_steps = len(df)
#         if curr_steps > min_steps:
#             df.reset_index(drop=True, inplace=True)
#             df = df.drop(range(min_steps, len(df)))
#             traj_df_formatted = traj_df_formatted.append(df)
#     traj_df_formatted.reset_index(drop=True, inplace=True)
#     return traj_df_formatted