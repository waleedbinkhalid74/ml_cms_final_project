import matplotlib.pyplot as plt

def plot_data(data, ax):
    for id in data['ID'].unique():
        data_id = data[data['ID'] == id]
        ax.plot(data_id['x1'].to_numpy(), data_id['x2'].to_numpy(), c='black')
        add_arrow(data_id, ax)

def add_arrow(data, ax):
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