import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(date, gdf, ax):
    ax = gdf.plot(linewidth=gdf[date]**0.25, ax=ax)
    ax.text(0, 0, date, transform=ax.transAxes)
    return ax


dfq = pd.read_csv('../data/obs_flow_full.csv')
gdf = gpd.read_file('../data/gis/Segments_subset.shp')

dfq['date'] = pd.to_datetime(dfq['date'])
# dfq.set_index('seg_id_nat', inplace=True)
dfqp = dfq.pivot(values='discharge_cms', index='seg_id_nat', columns='date')
dfqp.fillna(0, inplace=True)
dfqp.columns = dfqp.columns.astype(str)
dates = dfqp.columns

gdf.set_index('seg_id_nat', inplace=True)
gdf_comb = gdf.join(dfqp, how='outer')

fig, ax = plt.subplots()
ax = gdf_comb.plot(linewidth=0.5, color='gray', ax=ax)

ani_dates = dates[500:525]
anim = FuncAnimation(fig, animate, frames=ani_dates, fargs=(gdf_comb, ax))
anim.save('flow.gif', writer='imagemagick')


