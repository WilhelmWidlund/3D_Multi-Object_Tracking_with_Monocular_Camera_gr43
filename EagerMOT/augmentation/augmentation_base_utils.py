"""
Utility functions for augmentation
"""

from os import path, makedirs

def get_generic_folder(default_path: str, query: str, prompt: str, create: bool):
    print(query + " [y/n]")
    savechoice = str(input())
    if savechoice not in ['y', 'Y', 'yes', 'YES', 'Yes', '1', 'default', 'DEFAULT', 'Default']:
        print(prompt)
        # Take input from user to their folder of choice, staerting from project base folder
        userpath = str(input(default_path))
        custom_folder_name = default_path + userpath
        if create and not path.exists(custom_folder_name):
            makedirs(custom_folder_name)
        # return user chosen folder path
        return custom_folder_name
    else:
        return default_path


def print_available_choices(choices: dict):
    """
    Print options in choices: a dictionary with
    'Type' = "the parameter to choose an option for"
    integer(choice identifier) = ["First descstring", "Second descstring"/False, <More content as desired>]
    Provide False as the second element if one descriptive string is enough.
    """
    print("Choose which " + choices['Type'] + " should be used:")
    for key in choices:
        if key == 'Type':
            continue
        namestring = choices[key][0]
        # Check for further description
        if choices[key][1]:
            namestring += ":" + choices[key][1]
        print(str(key) + ". " + namestring + ".")


def choose_hyper_parameter(choices: dict):
    """
    Allow the user to choose a parameter with the options defined in choices: a dictionary with
    'Type' = "the parameter to choose an option for"
    integer(choice identifier) = ["First descstring", "Second descstring"/False, <More content as desired>]
    Provide False as the second element if one descriptive string is enough.
    """
    print_available_choices(choices)
    while True:
        try:
            choice = int(input())
        except ValueError:
            print("Please print an integer corresponding to your choice.")
            continue
        else:
            if choice not in choices.keys():
                print("Invalid choice. Please choose again.")
                continue
            return choice


def adapt_normalized_score(score: float) -> float:
    """
    Adapt a normalized score in [0, 1] where 1 = best and 0 = worst to the EagerMOT framework of 0 = best and 1 = worst
    """
    return 1 - score
