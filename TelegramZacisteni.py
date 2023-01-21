# Import knihoven
import pandas as pd
from cleantext import clean
import re

# Načtení surových dat
df = pd.read_excel("C:\\Users\\jirih\Desktop\\Dulezite\\22a.PostProdukceTelegram\\Raw.xlsx")
df['NEW TEXT'] = 'Placeholder'

# Průchod skrz všechny příspěvky
for ind in df.index:
    # Odstraneni nežádoucích elementů
    df['NEW TEXT'][ind] = ((re.sub(r"(?:\@|\(http?\://|http?\://|https?\://|\(https?\://|www|\#)\S+", "", clean(df["text"][ind], no_emoji=True)))
                           .replace("*", "") #Odstraneni znaku *
                           .replace("[read full]", "") #odkaz na clanek
                           .replace("[A]", "") #odkaz na clanek
                           .replace("[", "") #Odstraneni znaku [
                           .replace("]", "") #Odstraneni znaku ]
                           .replace("(", "") #Odstraneni znaku (
                           .replace(")", "") #Odstraneni znaku )
                           .replace("__", "") #Odstraneni znaku __
                           .replace('Join Slavyangrad chat. Your opinion matters.', "") #Odstraneni reklamy
                           .replace('join slavyangrad', "") #Odstraneni reklamy
                           .replace('join slg intelligence briefings, strategy and analysis, expert community', "") #Odstraneni reklamy
                           )

# Uložení začištěných dat
df.to_excel("C:\\Users\\jirih\\Desktop\\Dulezite\\22a.PostProdukceTelegram\\Zpracovana.xlsx")


