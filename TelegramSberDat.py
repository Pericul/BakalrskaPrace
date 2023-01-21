# Import knihoven
from telethon.sync import TelegramClient
import pandas as pd
import datetime
from telethon.errors import ChannelPrivateError

# Prihlasovaci udaje do telegram API
api_id = '************'
api_hash = '************'
phone = '************'

# Definování jmen kanálů
groups = ['WorldNews', 'live_news_times',
          'theverytruestory', 'latestukraine', 'KyivIndependent_official',
          'SpecialQForces', 'mod_russia_en', 'DonbassDevushka', 'Slavyangrad']

# Navázani spojeni
client = TelegramClient('test', api_id, api_hash)
client.connect()

# Vytvoření struktur pro uchovani ziskanych dat
df = pd.DataFrame()
forwards_list = []
forwards_df = pd.DataFrame(
    forwards_list,
    columns=[
        "From",
        "Username",
    ],
)

# Hledáni skrz všechny skupiny
for group in groups:
    with TelegramClient(None, api_id, api_hash) as client:
        # Vybrani poslednich tisic zprav od aktualniho data.
        for message in client.iter_messages(group, 1000, reverse=False):
            print(message)
            
            # Získání informací o datu, hastagu, autorovi textu a forwafdnutí příspěvku
            data = {"group": group, "sender": message.sender_id, "text": message.text, "date": message.date,
                    "forward": message.forward}
            
            # Uložení dat
            temp_df = pd.DataFrame(data, index=[1])
            df = df.append(temp_df)

            # Pokud se jedna o forwardnutou zprávu, uloži se z jakeho kanalu byla forvardnuta
            if message.forward is not None:
                try:
                    f_from_id = message.forward.original_fwd.from_id
                    if f_from_id is not None:
                        ent = client.get_entity(f_from_id)
                        username = ent.username
                        forwards_df = pd.DataFrame(
                            forwards_list,
                            columns=[
                                "From",
                                "Username",
                            ],
                        )
                        forwards_list.append(
                            [
                                group,
                                username,
                            ]
                        )
                except Exception as e:
                    if e is ChannelPrivateError:
                        print("Private channel")
                    continue

# Uložení získaných dat
forwards_df.to_excel("C:\\Users\\jirih\\Desktop\\Datasety\\Telegram\\forwards.xlsx", index=False)
df['date'] = df['date'].dt.tz_localize(None)
client.disconnect()
df.to_excel("C:\\Users\\jirih\\Desktop\\Datasety\\Telegram\\data_{}.xlsx".format(datetime.date.today()), index=False)

