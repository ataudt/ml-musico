import pandas as pd


def check_validity(instruction_new, history, rules):

    key = instruction_new['key']

    ## Check if we are over 'max_occurrence'
    if len(history[history['instruction-key'] == key]) >= instruction_new['max_occurrence']:
        return False

    ## Check if we are over 'max_repeats_per_instruction'
    last_n_keys = history.tail(rules['max_repeats_per_instruction'])['instruction-key']
    if (last_n_keys == key).all():
        return False

    return True


def give_random_instruction(instructions, rules, history):
    """Give a random instruction."""

    instruction_valid = False
    valid_counter = -1
    while not instruction_valid:
        valid_counter += 1
        # Sample random instruction
        instruction_new = instructions.sample(1)
        instruction_with_key = instruction_new[0]
        instruction_with_key['key'] = instruction_new.keys()[0]

        # Check validity
        if history is None:
            instruction_valid = True
        else:
            instruction_valid = check_validity(
                instruction_new=instruction_with_key,
                history=history,
                rules=rules,
            )

        if valid_counter >= 100:
            raise ValueError("Impossible definition of constraints.")

    return instruction_with_key


def draw_instruction(ax, instruction):
    """Draw the instruction on the axis."""

    if len(ax.texts) > 0:
        ## Delete old instruction in plot
        del ax.texts[-1]

    ## Add new instruction
    ax.text(
        0.5, 0.5, instruction['name'],
        {'color': instruction['color'], 'fontsize': instruction['size']},
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes
    )
    return ax

def update_history(history, frame, time, key):
    """Update the history dataframe for instructions."""

    if history is None:
        ## Initialize history dataframe
        history = pd.DataFrame(
            data={'frame': frame, 'time': time, 'instruction-key': key, 'occurrence': 1},
            index=pd.Index([frame], name='iframe')
        )
    else:
        ## Append new entry
        history.loc[frame, ['frame','time','instruction-key','occurrence']] = [
            frame, 
            time, 
            key, 
            len(history[history['instruction-key'] == key]) + 1
        ]

    return history