import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager

ORDER = ['positive','negative','neutral']

def plot_top_tokens(lexicon,path,n=15,font_path = None):
    fp = font_manager.FontProperties(fname=font_path) if font_path else None
    pos = lexicon[lexicon['log_odds']>0]
    neg = lexicon[lexicon['log_odds']<0]
    panels = (
        (pos.nlargest(n,'log_odds'), 'green','Top Positive Tokens'),
        (neg.nsmallest(n,'log_odds'), 'red','Top Negative Tokens')
    )
    fig,axes = plt.subplots(1,2,figsize=(12,6))
    for ax,(data,color,title) in zip(axes,panels):
        ax.barh(data['token'],data['log_odds'],color=color)
        ax.invert_yaxis()
        ax.set_xlabel('Log-Odds')
        ax.set_title(title)

        if fp:
            for label in ax.get_yticklabels():
                label.set_fontproperties(fp)
    fig.tight_layout()
    fig.savefig(path,dpi = 150)
    plt.close(fig)
def plot_violin_and_confusion(df,path):
    fig,(ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
    sns.violinplot(data=df,x='label',y='token_count',order=ORDER,ax=ax1)
    sns.despine(ax=ax1)
    cm = pd.crosstab(df['label'],df['predicted_polarity']).reindex(index=ORDER,columns=ORDER,fill_value=0)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    fig.tight_layout()
    fig.savefig(path,dpi = 150)
    plt.close(fig)
def write_dashboard(lexicon, path):
    fig = px.scatter(lexicon, x='freq', y ='log_odds',hover_name ='token',hover_data = {'pos': True,'neg': True,'neu': True},color='log_odds',color_continuous_scale='RdYlGn',title = 'Malayalam sentiment lexicon explorer')
    fig.write_html(path, include_plotlyjs=True)