import re

string = 'abc123def453'

def find_numbers(string):
    if re.findall(r'\d{6}', string):
        return True
    return False

print(find_numbers(string))