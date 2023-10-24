from colored import fg, attr
"""
This file contains the colors that are used in the project for printing messages on the console.
"""

# Defining some colors
red = fg('red')
green = fg('green')
yellow = fg('yellow')
blue = fg('blue')
magenta = fg('magenta')
cyan = fg('cyan')

# Resetting the color
reset = attr('reset')

__all__ = ["red", "green", "yellow", "blue", "magenta", "cyan", "reset"]
