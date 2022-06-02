
# -------------- read in data ---------------------------------
import uproot4 as up
import pandas as pd
from root_numpy import root2array, rec2array

def open_root(fname, tname, branch_names, label, norm, selection):

    root_file = up.open(fname)
    root_tree = root_file[tname]
    df = root_tree.arrays(library='pd')

    if (selection is not None): 
        for i in selection.keys():
            if (i in df.columns.values):
                df = df.loc[~((df[i] > selection[i][0] ) & (df[i] < selection[i][1] ))]

    for ib in branch_names:
        if ib not in df.columns.values:
            branch_names.remove(ib)

    df = df[branch_names]
    df['y'] = label
    df['normalization'] = norm

    return df
