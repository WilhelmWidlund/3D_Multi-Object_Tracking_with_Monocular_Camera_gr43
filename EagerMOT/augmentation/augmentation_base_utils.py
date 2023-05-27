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
    print("Choose which " + choices['Type'] + " should be used:")
    for key in choices:
        namestring = choices[key][0]
        # Check for further description
        if len(choices[key]) > 2:
            namestring += choices[key][2]
        print(key + ". " + namestring + ".\n")


def choose_sub_parameter(choices: dict):
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
