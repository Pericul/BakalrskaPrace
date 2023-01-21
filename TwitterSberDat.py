# Import knihoven
import snscrape.modules.twitter as sc
import pandas as pd

# Vytvoreni struktur pro uchovani ziskanych dat
df = pd.DataFrame()

# Sběr příspěvků podle filtru
filtr = "(Russia OR Ukraine OR Ukraine War OR Russian War OR Putin OR Zelensky) until:2022-12-05 since:2022-12-04 lang:en"
for t in sc.TwitterSearchScraper(filtr).get_items():

    # Získání všech hastagů
    h = ""
    if t.hashtags != None:
        for ele in t.hashtags:
            h += ele
            h += ";"
    # Získání informací o datu, hastagu, autorovi a textu příspěvku
    data = {"sender": t.user.username, "text": t.content, "date": t.date, "hashtags": h}

    # Uložení dat
    temp_df = pd.DataFrame(data, index=[1])
    df = pd.concat([df, temp_df])

# Export dat do tabulky
df['date'] = df['date'].dt.tz_localize(None)
df.to_excel("C:\\Users\\jirih\\Desktop\\Datasety\\Twitter\TwitterData.xlsx", index=False)




