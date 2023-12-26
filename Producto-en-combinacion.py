import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data = pd.read_csv("pedidos por mesa.csv", encoding='latin1')
data1 = (data.groupby(["command_id", "product_name"])["qty"]
        .sum().unstack().reset_index().fillna(0).set_index("command_id"))
def hot_encode(n):
    return 1 if n >= 1 else 0
data2 = data1.applymap(hot_encode)
frq_items = apriori(data2, min_support=0.05, max_len=2, use_colnames=True, low_memory=True)
data3 = association_rules(frq_items,  metric="lift",min_threshold=1)#[["antecedents", "consequents", "antecedent support", "consequent support", "support"]]
data3.head(60)