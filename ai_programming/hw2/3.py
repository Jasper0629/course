import re

def isNumLeters(s):
    s = str(s)
    if s == '':
        return False

    if re.match('^[0-9a-zA-Z]{6,18}$', s):
        return True
    else:
        return False
        
s = "1234"
print(isNumLeters(s))