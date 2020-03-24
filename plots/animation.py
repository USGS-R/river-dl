import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


dfq = pd.read_csv('../data/obs_flow_full.csv')
gdf = gpd.read_file('../data/gis/Segments_subset.shp')

start_date = '1980-01-01'
dfq['date'] = pd.to_datetime(dfq['date'])
dfq = dfq[dfq['date'] > start_date]
dfqp = dfq.pivot(values='discharge_cms', index='seg_id_nat', columns='date')
dfqp.fillna(0, inplace=True)
dfqp.columns = dfqp.columns.astype(str)
dates = dfqp.columns

gdf.set_index('seg_id_nat', inplace=True)
gdf_comb = gdf.join(dfqp, how='outer')

fig, ax = plt.subplots()


def animate(date):
    ax.clear()
    my_ax = gdf_comb.plot(linewidth=0.5, color='lightgray', ax=ax)
    my_ax = gdf_comb.plot(linewidth=gdf_comb[date]**0.25, ax=my_ax)
    my_ax.text(0, 0, date, transform=ax.transAxes)

anim = FuncAnimation(fig, animate, frames=dates[:100])
anim.save('flow.gif', writer='imagemagick')

