import pandas as pd
from pprint import pprint


def check_validity(instruction_new, history_instr, rules):

    key = instruction_new['key']

    ## First instruction given always True (history_instr == None), except for END
    if history_instr is None:
        if key == 'END':
            return False
        else:
            return True

    ## Check if we are over 'max_occurrence'
    if len(history_instr[history_instr['instruction-key'] == key]) >= instruction_new['max_occurrence']:
        print(f"Exceeded 'max_occurrence'.")
        return False

    ## Check if we are over 'max_repeats_per_instruction'
    last_n_keys = history_instr.tail(rules['max_repeats_per_instruction'])['instruction-key']
    if (last_n_keys == key).all():
        print("Exceeded 'max_repeats_per_instruction'.")
        return False

    ## Check if END instruction was selected
    if key == 'END':
        return False

    return True


def give_random_instruction(instructions, rules, history_instr):
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
        instruction_valid = check_validity(
            instruction_new=instruction_with_key,
            history_instr=history_instr,
            rules=rules,
        )

        if valid_counter >= 50:
            message = "Impossible definition of rules. Available instructions: "
            print(message)
            pprint(instructions.to_dict())
            print("Instruction history: ")
            print(history_instr)
            instruction_valid = True

    return instruction_with_key


def draw_instruction(ax, instruction):
    """Draw the instruction on the axis."""

    if instruction is None:
        return ax
        
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

def update_history(history_instr, frame, time, instruction_key, event_key):
    """Update the history_instr dataframe for instructions."""

    if history_instr is None:
        ## Initialize history_instr dataframe
        history_instr = pd.DataFrame(
            data={
                'frame': frame, 
                'time': time, 
                'instruction-key': instruction_key, 
                'instruction-occurrence': 1,
                'event-key': event_key,
                'event-occurrence': None,
            },
            index=pd.Index([frame], name='iframe')
        )
    else:
        ## Append new entry
        history_instr.loc[frame, ['frame','time','instruction-key','instruction-occurrence','event-key','event-occurrence']] = [
            frame, 
            time, 
            instruction_key, 
            len(history_instr[history_instr['instruction-key'] == instruction_key]) + 1,
            event_key, 
            len(history_instr[history_instr['event-key'] == event_key]) + 1
        ]

    return history_instr