import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


dfq = pd.read_csv('../data/obs_flow_full.csv')
gdf = gpd.read_file('../data/gis/Segments_subset.shp')

dfq['date'] = pd.to_datetime(dfq['date'])
dfqp = dfq.pivot(values='discharge_cms', index='seg_id_nat', columns='date')
dfqp.fillna(0, inplace=True)
dfqp.columns = dfqp.columns.astype(str)
dates = dfqp.columns

gdf.set_index('seg_id_nat', inplace=True)
gdf_comb = gdf.join(dfqp, how='outer')

fig, ax = plt.subplots()


def animate(date):
    ax.clear()
    my_ax = gdf_comb.plot(linewidth=0.5, color='gray', ax=ax)
    my_ax = gdf_comb.plot(linewidth=gdf_comb[date]**0.25, ax=my_ax)
    my_ax.text(0, 0, date, transform=ax.transAxes)

ani_dates = dates[500:525]
anim = FuncAnimation(fig, animate, frames=ani_dates)
anim.save('flow.gif', writer='imagemagick')

