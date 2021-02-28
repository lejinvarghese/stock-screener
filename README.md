# Description
A stock screener that can evaluate your watchlist and send you a technical analysis of selected stocks to a telegram channel.

# Features
..more to come..stay tuned!

# Sample
![Ticker Trend](docs/sample_L.TO.png)

# Curl samples
curl --request POST --url https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook --header 'content-type: application/json' --data '{"url": "https://2j48cpk83h.execute-api.us-east-1.amazonaws.com/dev"}'
#https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo