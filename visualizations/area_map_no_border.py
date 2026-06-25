import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1. 定义区域范围 [Lon_min, Lon_max, Lat_min, Lat_max]
area1_extent = [122, 123, 29, 33]      # Area 1: 29N~33N, 122E~123E
area2_extent = [114, 124.5, 20.5, 26.5] # Area 2: 20.5N~26.5N, 114E~124.5E
ENABLE_BASEMAP_FEATURES = True  # True: 联网自动下载并使用 10m Natural Earth 高精度底图
FIG_DPI = 180
EXPORT_DPI = 600

def resolve_best_gshhs_l1_path():
    """选择本地可用的最高精度 GSHHS 海岸线（L1: 海岸线主轮廓）。"""
    search_roots = [
        Path("/root/.local/share/cartopy/shapefiles/gshhs"),
        Path("/root/miniconda3/lib/python3.8/site-packages/cartopy/data/shapefiles/gshhs"),
    ]
    # 按精度从高到低尝试：f > h > i > l > c
    for scale in ["f", "h", "i", "l", "c"]:
        filename = f"GSHHS_{scale}_L1.shp"
        for root in search_roots:
            candidate = root / scale / filename
            if candidate.exists():
                return candidate
    return None

GSHHS_COAST_PATH = resolve_best_gshhs_l1_path()

def draw_features(ax):
    """为地图添加基础地理要素（不绘制陆上国境线）"""
    # 纯色海洋背景（轻量、离线、不会像栅格底图那样吃内存）
    ax.set_facecolor('#f0f8ff')
    # 联网优先：自动下载并缓存 10m 高精度海岸线/陆地（首次运行会下载，后续走本地缓存）
    if ENABLE_BASEMAP_FEATURES:
        try:
            ax.add_feature(
                cfeature.NaturalEarthFeature("physical", "land", "10m"),
                facecolor="#e0e0e0",
                edgecolor="none",
                zorder=1,
            )
            ax.add_feature(
                cfeature.NaturalEarthFeature("physical", "ocean", "10m"),
                facecolor="#f0f8ff",
                edgecolor="none",
                zorder=0,
            )
            ax.add_feature(
                cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
                facecolor="none",
                edgecolor="#333333",
                linewidth=0.6,
                zorder=2,
            )
            return
        except Exception:
            # 下载失败时自动回退到本地离线数据
            pass

    # 直接读取本地 L1 海岸线（全球主海岸线），避免 Cartopy 在绘制时触发在线下载
    if GSHHS_COAST_PATH and GSHHS_COAST_PATH.exists():
        geoms = shapereader.Reader(str(GSHHS_COAST_PATH)).geometries()
        ax.add_geometries(
            geoms,
            ccrs.PlateCarree(),
            facecolor='#e0e0e0',
            edgecolor='#333333',
            linewidth=0.6,
            zorder=2,
        )
        return

    # 最后兜底：低精度内置要素，确保至少可见轮廓（不含国境线）
    ax.add_feature(cfeature.LAND, facecolor="#e0e0e0", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#f0f8ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#333333", zorder=2)

def connect_inset_to_extent(ax_main, ax_inset, extent, color, side):
    """在混合投影下稳定绘制主图区域框与放大图的连接线。"""
    lon_min, lon_max, lat_min, lat_max = extent
    if side == "left":
        src_lonlat = [(lon_min, lat_min), (lon_min, lat_max)]
        dst_axes = [(1.0, 0.0), (1.0, 1.0)]
    else:
        src_lonlat = [(lon_max, lat_min), (lon_max, lat_max)]
        dst_axes = [(0.0, 0.0), (0.0, 1.0)]

    for (lon, lat), (u, v) in zip(src_lonlat, dst_axes):
        x, y = ax_main.projection.transform_point(lon, lat, ccrs.PlateCarree())
        conn = ConnectionPatch(
            xyA=(x, y),
            coordsA=ax_main.transData,
            xyB=(u, v),
            coordsB=ax_inset.transAxes,
            axesA=ax_main,
            axesB=ax_inset,
            color=color,
            lw=0.8,
            zorder=10,
        )
        ax_main.figure.add_artist(conn)

# 2. 创建画布
fig = plt.figure(figsize=(14, 8), dpi=FIG_DPI)

# ----------------- 主图：全球范围 (Global View) -----------------
# 使用 Robinson 投影看起来更具顶会感（比 PlateCarree 更专业）
main_projection = ccrs.Robinson(central_longitude=150)
ax_main = fig.add_subplot(1, 1, 1, projection=main_projection)
ax_main.set_global()
draw_features(ax_main)

# 在主图上绘制两个区域的矩形框（需要转换坐标系）
data_crs = ccrs.PlateCarree()

# 绘制 Area 1 矩形
rect1 = Rectangle((area1_extent[0], area1_extent[2]), 
                  area1_extent[1]-area1_extent[0], area1_extent[3]-area1_extent[2],
                  linewidth=1.5, edgecolor='red', facecolor='none', 
                  transform=data_crs, zorder=5)
ax_main.add_patch(rect1)

# 绘制 Area 2 矩形
rect2 = Rectangle((area2_extent[0], area2_extent[2]), 
                  area2_extent[1]-area2_extent[0], area2_extent[3]-area2_extent[2],
                  linewidth=1.5, edgecolor='blue', facecolor='none', 
                  transform=data_crs, zorder=5)
ax_main.add_patch(rect2)

# ----------------- 子图 1: Area 1 放大 -----------------
ax_ins1 = inset_axes(ax_main, width="25%", height="25%", loc='lower left',
                     bbox_to_anchor=(0.05, 0.1, 1, 1),
                     bbox_transform=ax_main.transAxes,
                     axes_class=GeoAxes,
                     axes_kwargs=dict(projection=ccrs.PlateCarree()))

ax_ins1.set_extent(area1_extent, crs=data_crs)
draw_features(ax_ins1)
ax_ins1.set_title("Area 1", fontsize=15, color='red')
# 网格线只用于辅助，标签使用轴刻度强制显示两端
gl1 = ax_ins1.gridlines(draw_labels=False, x_inline=False, y_inline=False, linewidth=0.3)
gl1.xlocator = mticker.FixedLocator([area1_extent[0], area1_extent[1]])
gl1.ylocator = mticker.FixedLocator([area1_extent[2], area1_extent[3]])
ax_ins1.set_xticks([area1_extent[0], area1_extent[1]], crs=data_crs)
ax_ins1.set_yticks([area1_extent[2], area1_extent[3]], crs=data_crs)
ax_ins1.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
ax_ins1.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
ax_ins1.tick_params(axis='both', labelsize=8, pad=1)

# ----------------- 子图 2: Area 2 放大 -----------------
ax_ins2 = inset_axes(ax_main, width="25%", height="25%", loc='lower right',
                     bbox_to_anchor=(-0.05, 0.1, 1, 1),
                     bbox_transform=ax_main.transAxes,
                     axes_class=GeoAxes,
                     axes_kwargs=dict(projection=ccrs.PlateCarree()))

ax_ins2.set_extent(area2_extent, crs=data_crs)
draw_features(ax_ins2)
ax_ins2.set_title("Area 2", fontsize=15, color='blue')
gl2 = ax_ins2.gridlines(draw_labels=False, x_inline=False, y_inline=False, linewidth=0.3)
gl2.xlocator = mticker.FixedLocator([area2_extent[0], area2_extent[1]])
gl2.ylocator = mticker.FixedLocator([area2_extent[2], area2_extent[3]])
ax_ins2.set_xticks([area2_extent[0], area2_extent[1]], crs=data_crs)
ax_ins2.set_yticks([area2_extent[2], area2_extent[3]], crs=data_crs)
ax_ins2.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol='°'))
ax_ins2.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol='°'))
ax_ins2.tick_params(axis='both', labelsize=8, pad=1)

# 3. 添加连接线 (连接主图矩形框与放大图)
connect_inset_to_extent(ax_main, ax_ins1, area1_extent, color="red", side="left")
connect_inset_to_extent(ax_main, ax_ins2, area2_extent, color="blue", side="right")

# inset_axes 与 tight_layout 不兼容，改为手动边距
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92)
plt.savefig("UHSM_Study_Area_no_border.pdf", dpi=EXPORT_DPI) # 导出为矢量图
plt.show()
