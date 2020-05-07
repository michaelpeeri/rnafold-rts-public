import sys
import json
import requests
import config


"""
Send simple notification via Push-over.
Note: Limited to 7500 messages/month.
TODO: Add rate limiting, better error handling.
"""
def notify(message):
    payload = { 'token': config.computation_monitor_app, 'user': config.computation_monitor_group, 'message': message }
    r = requests.post('https://api.pushover.net/1/messages.json', data=payload)
    if( r.status_code != 200 ):
        raise Exception("Pushover returned code %d, content %s" % r.status_code, r.json())
    return (r.status_code, r.json())


def defaultNotification():
    from socket import gethostname
    notify("Notification (%s)" % gethostname())


if __name__=="__main__":
    defaultNotification()
    sys.exit(0)
