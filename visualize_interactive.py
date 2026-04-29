import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 读取数据
df = pd.read_csv('/root/autodl-tmp/swh_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])

# 仅使用 vo 的第一个数据列
data_cols = [c for c in df.columns if c.lower() != 'date']
col = data_cols[0] if data_cols else 'value'

fig = make_subplots(rows=1, cols=1, subplot_titles=(f'{col} (swh 第1列)',), shared_xaxes=True)
fig.add_trace(
    go.Scatter(x=df['date'], y=df[col], name=col, mode='lines', line=dict(width=1, color='#1f77b4')),
    row=1, col=1
)
fig.update_yaxes(title_text=f'{col} (m/s)', row=1, col=1)
xaxis_title_row = 1

# 通用布局
fig.update_layout(
    title='swh 第1列',
    hovermode='x unified',
    height=900,
    template='plotly_white',
    showlegend=True
)
fig.update_xaxes(title_text='时间', row=xaxis_title_row, col=1)

# 添加基于滑动窗口的 slider（默认窗口7天，步长7天，可调整）
window_size_days = 30
step_days = 14

t0 = df['date'].min()
t1 = df['date'].max()

windows = []
start = t0
while start + pd.Timedelta(days=window_size_days) <= t1:
    end = start + pd.Timedelta(days=window_size_days)
    windows.append((start, end))
    start = start + pd.Timedelta(days=step_days)

if not windows:
    windows = [(t0, t1)]

steps = []
for i, (s, e) in enumerate(windows):
    label = s.strftime('%Y-%m-%d')
    args = [{
        'xaxis.range': [s.strftime('%Y-%m-%d %H:%M:%S'), e.strftime('%Y-%m-%d %H:%M:%S')]
    }]
    step = {'label': label, 'method': 'relayout', 'args': args}
    steps.append(step)

slider = {'active': len(steps) - 1, 'currentvalue': {'prefix': '窗口开始: '}, 'pad': {'t': 50}, 'steps': steps}
fig.update_layout(sliders=[slider])

# 保存并说明
output_file = 'swh_col0_interactive.html'
fig.write_html(output_file)
print(f"\n交互式图表已生成: {output_file}")
print("支持滑动窗口放大。若想调整窗口大小，请修改脚本中的 window_size_days / step_days。")

fig.show()
