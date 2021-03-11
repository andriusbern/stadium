# Functions for editing config files
import os, random
# import config
import re

def parse_line(line, delim=':', sep='"'):
    comps = line.split(delim)
    if len(comps) == 2:
        arg, val = [x.strip().strip(',').strip(sep) for x in comps]
        return arg, val
    else:
        return None

def modify(filename, args_dict={}, ext='json', **kwargs):

    delims = {
        'json': ':',
        'conf': '=',
        'yml':  ':',
        'html': '='}
    seps   = {
        'json': '"',
        'conf': '',
        'yml' : '',
        'html': '"'}

    with open(filename, 'r') as f:
        original = f.readlines()
        to_change = {**args_dict, **kwargs}
        arg_list = list(to_change.keys())
        for i, line in enumerate(original):
            parsed = parse_line(line, delim=delims[ext], sep=seps[ext])
            if parsed is not None:
                arg, val = parsed
                if arg in arg_list:
                    original[i] = line.replace(val, str(to_change[arg]))
                    print('{}: "{}" | {} ==> {}'.format(
                        filename, arg, val, to_change[arg]))
    f = open(filename, 'w')
    f.writelines(original)
    f.close()

def count_cols(filename):
    with open(filename, 'r') as f:
        text = f.read()
        codes = re.findall('^#(([0-9a-fA-F]{2}){3}|([0-9a-fA-F]){3})$', text)
        print(text)
        print(codes)


def replace_color(filename, new):
    with open(filename, 'r') as f:
        original = f.read()
        # print(original)
        ind = original.find('"#') + 2
        print(ind)
        print(original[ind:ind+6])
        old = original[ind:ind+6]
        print(old)
        original = original.replace(old, new)
        # original[ind:ind+6] = new
    f = open(filename, 'w')
    f.writelines(original)
    f.close()



if __name__ == "__main__":
    # modify('tips.md', test=50)
    import json
    # modify('tips.md', ext='conf', test=60)
    # modify('json_test.json', args_dict={"workbench.colorTheme": "Eva Dark Bold"})
    # filename = '/home/andrius/.config/Code/User/settings.json'
    # themef = '/home/andrius/.vscode/extensions/fisheva.eva-theme-0.7.9/themes/Eva-Dark.json'
    # themes = ["Material Theme", 'Eva Dark Bold', 'Monokai']
    # modify(filename=filename, args_dict={"workbench.colorTheme": random.sample(themes, 1)[0]})
    # js = json.load(open('json_test.json'))
    # print(js)
    # modify('Classifier.svg', ext='html', fill='#000000')
    import glob
    ls = glob.glob(os.path.abspath(os.curdir)+'/*.svg')
    print(ls)
    # F90000
    col = 'F90000'
    [replace_color(x, col) for x in ls]
    # replace_color('Classifier.svg', 'F90000')
    # count_cols(themef)