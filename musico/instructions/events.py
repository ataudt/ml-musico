import pandas as pd

def draw_event(ax, time, event):
    """Draw the event on the plot."""

    if event is None:
        return ax

    ## Add new instruction
    ax.text(
        time, 1, event['key'],
        {'color': event['color'], 'fontsize': 10},
        horizontalalignment='left', verticalalignment='top',
        rotation=90,
    )
    return ax


def _return_event(event, props):

    props['key'] = event
    return props


def evaluate_events(history_emo, EVENTS, EMOTIONS):

    # Percentage-Change-Emotion
    new_event = process_pct_emotion_events(
        history_emo = history_emo,
        events = EVENTS.get('Percentage-Change-Emotion')
    )
    if new_event is not None:
        return new_event

    # Percentage-Change-Emotionrank
    new_event = process_pct_emotionrank_events(
        history_emo = history_emo,
        events = EVENTS.get('Percentage-Change-Emotionrank'),
        EMOTIONS = EMOTIONS,
    )
    if new_event is not None:
        return new_event

    # Rank-Change
    new_event = process_rankchange_events(
        history_emo = history_emo,
        events = EVENTS.get('Rank-Change'),
        EMOTIONS = EMOTIONS,
    )
    if new_event is not None:
        return new_event

def process_pct_emotion_events(history_emo, events):
    """Process percentage change events."""

    # Directly return if events are missing
    if events is None:
        return None

    # Loop over all the possible events
    for event, props in events.items():

        ## Select appropriate window
        histwindow = history_emo[history_emo['time'] >= history_emo['time'].iloc[-1] - props['time_in_seconds']].copy()
        if len(histwindow) == 0:
            ### In case all values are NaN
            return None

        ## Loop over all emotions in event
        for emotion in props['emotions']:

            start_emotion = histwindow[emotion].iloc[0]
            end_emotion = histwindow[emotion].iloc[-1]
            pct_change = (end_emotion - start_emotion) / start_emotion * 100
            if props['pct_change'] >= 0:
                if pct_change >= props['pct_change']:
                    return _return_event(event, props)
            elif props['pct_change'] < 0:
                if pct_change <= props['pct_change']:
                    return _return_event(event, props)

    return None


def process_pct_emotionrank_events(history_emo, events, EMOTIONS):
    """Process percentage change events."""

    # Directly return if events are missing
    if events is None:
        return None

    # Loop over all the possible events
    for event, props in events.items():

        ## Select appropriate window
        histwindow = history_emo[history_emo['time'] >= history_emo['time'].iloc[-1] - props['time_in_seconds']].copy()
        if len(histwindow) == 0:
            ### In case all values are NaN
            return None

        ## Assign ranks to emotions
        histwindow_rank = histwindow.copy()
        histwindow_rank.loc[:, EMOTIONS] = histwindow.loc[:, EMOTIONS].rank(axis=1, method='dense', ascending=False)
        ## Rename emotion columns to the rank at the start of the window
        rename_map = pd.Series(histwindow_rank.iloc[0,:][EMOTIONS], index=EMOTIONS)
        histwindow.rename(columns=rename_map.to_dict(), inplace=True)

        ## Loop over all ranks in event
        for rank in props['ranks']:

            start_emotion = histwindow[rank].iloc[0]
            end_emotion = histwindow[rank].iloc[-1]
            pct_change = (end_emotion - start_emotion) / start_emotion * 100
            if props['pct_change'] >= 0:
                if pct_change >= props['pct_change']:
                    return _return_event(event, props)
            elif props['pct_change'] < 0:
                if pct_change <= props['pct_change']:
                    return _return_event(event, props)


def process_rankchange_events(history_emo, events, EMOTIONS):
    """Process rank change events."""

    # Directly return if events are missing
    if events is None:
        return None

    # Loop over all the possible events
    for event, props in events.items():

        ## Select appropriate window
        histwindow = history_emo[history_emo['time'] >= history_emo['time'].iloc[-1] - props['time_in_seconds']].copy()
        if len(histwindow) == 0:
            ### In case all values are NaN
            return None

        ## Assign ranks to emotions
        histwindow_rank = histwindow.copy()
        histwindow_rank.loc[:, EMOTIONS] = histwindow.loc[:, EMOTIONS].rank(axis=1, method='dense', ascending=False)
        ## Rename emotion columns to the rank at the start of the window
        rename_map = pd.Series(histwindow_rank.iloc[0,:][EMOTIONS], index=EMOTIONS)
        histwindow_rank.rename(columns=rename_map.to_dict(), inplace=True)

        ## Loop over all ranks in event
        for rank in props['ranks']:

            start_rank = histwindow_rank[rank].iloc[0]
            end_rank = histwindow_rank[rank].iloc[-1]
            rank_change = (end_rank - start_rank) / start_rank
            if props['rank_change'] >= 0:
                if rank_change >= props['rank_change']:
                    return _return_event(event, props)
            elif props['rank_change'] < 0:
                if rank_change <= props['rank_change']:
                    return _return_event(event, props)