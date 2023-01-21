# Import knihoven
import pandas as pd
from cleantext import clean
import re

# Načtení surových dat
df = pd.read_excel(r"C:\Users\jirih\Desktop\Dulezite\PostprodukceTwitter\TwitterData-original.xlsx")
df['NEW TEXT'] = 'Placeholder'

# Průchod skrz všechny příspěvky
for ind in df.index:
    df['NEW TEXT'][ind] = ((re.sub(r"(?:\@|\(http?\://|http?\://|https?\://|\(https?\://|www|\#|(\[).*?([\]]))\S+", "", clean(df["text"][ind], no_emoji=True))))

# Uložení začištěných dat
df.to_excel("C:\\Users\\jirih\\Desktop\\Dulezite\\PostprodukceTwitter\\dataSeZpracovanymTextem.xlsx")