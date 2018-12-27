import re
test_annotations = ["cw:sz",
 "MD:sz",
 "T: SZ",
 'sz',
 'SEIZURE ***'
 'SPASM',
 'absence',
 'SZ START',
 'SZ END']


rg1 = r"""sz|seizure|absence|spasm"""
rgx = re.compile(rg1, re.IGNORECASE)

for ss in test_annotations:
    print(rgx.search(ss))
