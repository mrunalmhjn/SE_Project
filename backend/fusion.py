def fuse_results(visual, text):

    TEXT_W = 0.6
    VIS_W = 0.4

    sentiment = TEXT_W * text["sentiment"] + VIS_W * visual["sentiment"]
    virality = TEXT_W * text["virality"] + VIS_W * visual["virality"]

    sent_label = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
    vir_label = "Viral" if virality > 0.45 else "Non-Viral"

    return {
        "sentiment_score": round(sentiment, 3),
        "sentiment_label": sent_label,
        "virality_score": round(virality, 3),
        "virality_label": vir_label
    }