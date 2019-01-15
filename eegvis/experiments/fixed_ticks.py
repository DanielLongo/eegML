import pandas as pd
from bokeh.charts import Bar, output_file, show
from bokeh.models import FuncTickFormatter

skills_list = ['cheese making', 'squanching', 'leaving harsh criticisms']
pct_counts = [25, 40, 1]
df = pd.DataFrame({'skill':skills_list, 'pct jobs with skill':pct_counts})
p = Bar(df, 'index', values='pct jobs with skill', title="Top skills for ___ jobs", legend=False)
label_dict = {}
for i, s in enumerate(skills_list):
    label_dict[i] = s

print(label_dict)

p.xaxis.formatter = FuncTickFormatter(code="""
    var labels = %s;
    return labels[tick];
""" % label_dict)

output_file("bar.html")
show(p)