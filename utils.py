import pandas as pd

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
    
    
def name_col_changer(col_name):
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
    
def format_traj_df(df: pd.DataFrame):
    df = df.rename(name_col_changer, axis='columns')
    df.drop(['delete'], axis=1, inplace=True)
    return df