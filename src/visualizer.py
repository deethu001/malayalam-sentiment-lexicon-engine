import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager
import os
# sentiment label order
ORDER = ['positive','negative','neutral']

def plot_top_tokens(lexicon, output_path, n=15, font_path = None):
    required_columns = {'token','log_odds'}
    missing_columns = required_columns - set(lexicon.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    font_properties =(
        font_manager.FontProperties(fname=font_path)
        if font_path else None
    )
    
    positive_tokens = lexicon[lexicon['log_odds']>0]
    negative_tokens = lexicon[lexicon['log_odds']<0]

    panels = (
        (positive_tokens.nlargest(n,'log_odds'), 'green','Top Positive Tokens'),
        (negative_tokens.nsmallest(n,'log_odds'), 'red','Top Negative Tokens')
    )

    figure,axes = plt.subplots(1,2,figsize=(12,6))
    
    for axis,(token_data,color,title) in zip(axes,panels):

        axis.barh(token_data['token'],
                  token_data['log_odds'],
                  color=color)
        
        axis.invert_yaxis()
        axis.set_xlabel('Log-Odds')
        axis.set_title(title,fontweight='bold')

        if font_properties:
            for tick_label in axis.get_yticklabels():
                tick_label.set_fontproperties(font_properties)

    figure.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    figure.savefig(
        output_path,
        dpi = 300,
        bbox_inches='tight')
    
    plt.close(figure)

def plot_violin_and_confusion(sentiment_df,output_path):

    required_columns = {
        'label',
        'token_count',
        'predicted_polarity'
    }

    missing_columns = required_columns - set(sentiment_df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    figure,(axis1, axis2) = plt.subplots(1,2,figsize=(14,5))

    #Violin plot:Review length by sentiment
    sns.violinplot(
        data=sentiment_df,
        x='label',
        y='token_count',
        order=ORDER,
        ax=axis1
    )

    axis1.set_title("Review Length Distribution by Sentiment",fontweight='bold')

    axis1.set_xlabel(' Actual Sentiment')
    axis1.set_ylabel('Token Count')

    sns.despine(ax=axis1)

    #Confusion Matrix
    cm = pd.crosstab(
        sentiment_df['label'],
        sentiment_df['predicted_polarity']
    ).reindex(index=ORDER,columns=ORDER,fill_value=0)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axis2)

    axis2.set_title('Sentiment Confusion Matrix',fontweight='bold')

    axis2.set_xlabel('Predicted Sentiment')
    axis2.set_ylabel('Actual Sentiment')

    figure.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    figure.savefig(
        output_path,
        dpi = 300,
        bbox_inches='tight')

    plt.close(plt.figure)

def plot_correlation_heatmap(sentiment_df, output_path):
    numerical_features = sentiment_df.select_dtypes(
        include= 'number'
    )
    if numerical_features.shape[1] < 2:
        raise ValueError("At least two numerical features are required.")
    correlation_matrix = numerical_features.corr(
        method='pearson'
    )
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        ax=axis
    )
    axis.set_title(
        'Correlation Heatmap of Numerical Features', 
        fontweight='bold'
    )
    figure.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(figure)

def write_dashboard(lexicon, output_path):
    #Create an interactive Malayalam sentiment lexicon deshboard
    required_columns = {'token','freq','log_odds','pos','neg','neu'}

    missing_columns = required_columns - set(lexicon.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    figure = px.scatter(
        lexicon,
        x='freq',
        y ='log_odds',
        hover_name ='token',
        hover_data = {'pos': True,'neg': True,'neu': True},
        color='log_odds',
        color_continuous_scale='RdYlGn',
        title = 'Malayalam sentiment lexicon explorer'
    )

    figure.update_layout(
        xaxis_title='Token Frequency',
        yaxis_title='Log-Odds Sentiment Score'
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    figure.write_html(
        output_path,
        include_plotlyjs=True
    )