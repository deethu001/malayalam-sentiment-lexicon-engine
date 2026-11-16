import matplotlib.pyplot as plt
from pathlib import Path
def plot_top_tokens(pos_tokens,neg_tokens):
    top_pos = dict(sorted(pos_tokens.items(),key = lambda item: item[1],reverse=True)[:15])
    top_neg = dict(sorted(neg_tokens.items(),key = lambda item: item[1],reverse=True)[:15])
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    ax1.barh(list(top_pos.keys()),list(top_pos.values()),color = 'green')
    ax1.set_title("Top 15 Positive Malayalm Tokens")
    ax1.set_xlabel("Log-Odds Score")
    ax1.invert_yaxis()

    ax2.barh(list(top_neg.keys()),list(top_neg.values()),color = 'red')
    ax2.set_title("Top 15 Negative Malayalm Tokens")
    ax2.set_xlabel("Log-Odds Score")
    ax2.invert_yaxis()

    plt.tight_layout()
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root/"reports"/"figures"
    output_dir.mkdir(parents = True,exist_ok=True)

    output_file = output_dir / "primary_enveope.png"
    plt.savefig(output_file, dpi = 300,bbox_inches ="tight")
    print("graph successfully saved!")
    print("Image location:",output_file)
    plt.close()
    pass

if __name__ == "__main__":
    pos_tokens ={
        "nalla" : 5.2,
        "santhosham":4.8,
        "super":4.5
     }
    neg_tokens = {
        "mosham":5.1,
        "veruppu":3.7,
        "kashtam":1.3
    }
    plot_top_tokens(pos_tokens,neg_tokens)