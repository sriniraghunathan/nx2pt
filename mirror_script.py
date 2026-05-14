import sys,os
fname = sys.argv[1:-1]
machine = sys.argv[-1]

if machine == 'cc':
    parentfolder = '/u/srinirag/projects_caps/'
elif machine == 'anl':
    parentfolder = '/home/ac.srinirag/'
elif machine == 'hive':
    parentfolder = '/home/srinirag/projects_leknoxgrp/'
elif machine == 'pm':
    parentfolder = '/global/homes/s/srinirag/'
else:
    parentfolder = '/home/sri/'
cmd = 'rsync -trvz --max-size 20mb %s %s:/%s/analysis/git/nx2pt/' %(' '.join(fname), machine, parentfolder)
print(cmd)
os.system(cmd)
